"""
Stable DQN Agent for Snake Game
================================
Goal: Reliably outperform/meet tabular Q-Learning through:
- Proper input normalization with fixed statistics
- Conservative hyperparameters
- Prioritized Experience Replay (PER)
- Double DQN with soft target updates
- Comprehensive monitoring and diagnostics

Architecture:
    Input(34) → Dense(128, ReLU) → Dense(128, ReLU) → Dueling Heads
    - Advantage head: Dense(64, ReLU) → Dense(|A|)
    - Value head: Dense(64, ReLU) → Dense(1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import json
from collections import deque, namedtuple
from datetime import datetime

from constants import *
from game_engine import GameEngine

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Stable DQN using device: {device}")

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SumTree:
    """
    Sum Tree for efficient Prioritized Experience Replay.
    Binary tree where parent = sum of children.
    Leaf nodes store priorities, enabling O(log n) sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Update tree upwards after priority change."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find leaf node for given value s."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Total priority sum."""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add new experience with priority."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """Update priority for specific index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get experience for priority value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    
    Args:
        capacity: Maximum buffer size (500k)
        alpha: Priority exponent (0.6)
        beta: Importance sampling exponent (0.4 → 1.0)
        eps: Small constant to prevent zero priority (1e-6)
    """
    def __init__(self, capacity=500000, alpha=0.6, beta=0.4, eps=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # Anneal beta to 1.0 over training
        self.eps = eps
        self.max_priority = 1.0
    
    def __len__(self):
        return self.tree.n_entries
    
    def add(self, state, action, reward, next_state, done):
        """Add experience with max priority (for new experiences)."""
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """
        Sample batch with priorities.
        Returns: (states, actions, rewards, next_states, dones, indices, weights)
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        # Calculate importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (len(self) * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Unpack batch
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(device)
        actions = torch.LongTensor([e.action for e in batch]).to(device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class DuelingNetwork(nn.Module):
    """
    Dueling DQN Architecture:
    - Shared feature layers
    - Separate value and advantage streams
    - Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
    """
    def __init__(self, state_size=34, action_size=3, hidden_size=128):
        super(DuelingNetwork, self).__init__()
        
        # Shared feature layers
        self.feature1 = nn.Linear(state_size, hidden_size)
        self.feature2 = nn.Linear(hidden_size, hidden_size)
        
        # Value stream
        self.value1 = nn.Linear(hidden_size, 64)
        self.value2 = nn.Linear(64, 1)
        
        # Advantage stream
        self.advantage1 = nn.Linear(hidden_size, 64)
        self.advantage2 = nn.Linear(64, action_size)
        
        # Initialize weights (Glorot/Xavier uniform)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Glorot uniform, zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through dueling network."""
        # Shared features
        x = F.relu(self.feature1(x))
        x = F.relu(self.feature2(x))
        
        # Value stream
        value = F.relu(self.value1(x))
        value = self.value2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage1(x))
        advantage = self.advantage2(advantage)
        
        # Combine streams: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class InputNormalizer:
    """
    Fixed input normalization using statistics from random policy buffer.
    Z-score normalization: (x - μ) / σ
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, states):
        """Compute statistics from buffer of states."""
        # Convert PyTorch tensors to numpy if needed
        import torch
        if len(states) > 0 and isinstance(states[0], torch.Tensor):
            states = [s.cpu().detach().numpy() if s.is_cuda else s.detach().numpy() for s in states]
        
        states = np.array(states)
        self.mean = states.mean(axis=0)
        self.std = states.std(axis=0) + 1e-8  # Avoid division by zero
        self.is_fitted = True
        print(f"[Normalization] Fitted on {len(states)} states")
        print(f"  Mean: min={self.mean.min():.3f}, max={self.mean.max():.3f}")
        print(f"  Std:  min={self.std.min():.3f}, max={self.std.max():.3f}")
    
    def normalize(self, state):
        """Normalize single state."""
        if not self.is_fitted:
            return state
        return (state - self.mean) / self.std
    
    def save(self, filepath):
        """Save normalization stats."""
        if self.is_fitted:
            np.savez(filepath, mean=self.mean, std=self.std)
    
    def load(self, filepath):
        """Load normalization stats."""
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.mean = data['mean']
            self.std = data['std']
            self.is_fitted = True
            return True
        return False


class StableDQNAgent:
    """
    Stable DQN Agent with comprehensive monitoring and conservative hyperparameters.
    """
    def __init__(self, game_engine, state_size=34, action_size=3):
        self.game_engine = game_engine
        self.state_size = state_size
        self.action_size = action_size
        
        # Networks
        self.online_net = DuelingNetwork(state_size, action_size).to(device)
        self.target_net = DuelingNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer (Adam with standard settings)
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Replay buffer (PER)
        self.memory = PrioritizedReplayBuffer(
            capacity=500000,
            alpha=0.6,
            beta=0.4
        )
        
        # Input normalization
        self.normalizer = InputNormalizer()
        
        # Hyperparameters
        self.gamma = 0.99
        self.batch_size = 128
        self.tau = 0.005  # Soft update rate
        self.gradient_clip = 10.0
        
        # Exploration (epsilon-greedy)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay_steps = 200000  # Linear decay over 200k steps
        self.epsilon_decay = (1.0 - 0.05) / self.epsilon_decay_steps
        
        # Training state
        self.total_steps = 0
        self.update_steps = 0
        self.warmup_steps = 20000  # Collect 20k transitions before learning
        
        # Monitoring
        self.stats = {
            'td_errors': [],
            'losses': [],
            'q_values': [],
            'q_target_values': [],
            'grad_norms_pre': [],
            'grad_norms_post': [],
            'target_lag': [],
            'epsilon_history': [],
            'buffer_size': [],
            'eval_scores': [],
            'eval_episodes': []
        }
        
        # State representation from Enhanced DQN
        from enhanced_dqn import EnhancedStateRepresentation
        self.state_repr = EnhancedStateRepresentation
    
    def get_state(self):
        """Get enhanced state representation (34 features)."""
        raw_state = self.state_repr.get_enhanced_state(self.game_engine)
        normalized_state = self.normalizer.normalize(raw_state)
        return normalized_state
    
    def collect_warmup_data(self, num_transitions=20000):
        """
        Collect random policy data for normalization and initial buffer fill.
        """
        print(f"\n{'='*70}")
        print(f"WARMUP: Collecting {num_transitions} random transitions")
        print(f"{'='*70}")
        
        states_for_normalization = []
        transitions_collected = 0
        episode = 0
        
        while transitions_collected < num_transitions:
            self.game_engine.reset_game()
            episode_states = []
            steps = 0
            
            while not self.game_engine.game_over and steps < 1000:
                # Get state
                state = self.state_repr.get_enhanced_state(self.game_engine)
                episode_states.append(state)
                
                # Random action
                action = random.randint(0, self.action_size - 1)
                
                # Take action
                old_score = self.game_engine.score
                self.game_engine.set_direction_from_algorithm(
                    self._action_to_direction(action)
                )
                self.game_engine.move_snake()
                
                # Get next state
                next_state = self.state_repr.get_enhanced_state(self.game_engine)
                done = self.game_engine.game_over
                ate_food = self.game_engine.score > old_score
                
                # Calculate reward (same as Q-Learning)
                reward = self._calculate_reward(done, ate_food)
                
                # Store in buffer (unnormalized)
                self.memory.add(state, action, reward, next_state, done)
                
                transitions_collected += 1
                steps += 1
                
                if transitions_collected >= num_transitions:
                    break
            
            states_for_normalization.extend(episode_states)
            episode += 1
            
            if episode % 50 == 0:
                print(f"  Episode {episode} | Transitions: {transitions_collected}/{num_transitions}")
        
        # Fit normalizer on collected states
        self.normalizer.fit(states_for_normalization)
        
        print(f"[OK] Warmup complete: {transitions_collected} transitions collected")
        print(f"[OK] Buffer size: {len(self.memory)}")
        print(f"{'='*70}\n")
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.online_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        """
        Single training step with Double DQN and PER.
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from PER
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # TD errors for PER
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        
        # Smooth L1 loss (Huber) with importance sampling weights
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        
        # Monitor gradients
        grad_norm_pre = torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            float('inf')
        )
        grad_norm_post = torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            self.gradient_clip
        )
        
        self.optimizer.step()
        
        # Update priorities in PER
        self.memory.update_priorities(indices, td_errors)
        
        # Soft update target network
        for target_param, online_param in zip(self.target_net.parameters(), 
                                               self.online_net.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Update stats
        self.update_steps += 1
        self.stats['td_errors'].append(np.abs(td_errors).mean())
        self.stats['losses'].append(loss.item())
        self.stats['q_values'].append(current_q_values.mean().item())
        self.stats['q_target_values'].append(target_q_values.mean().item())
        self.stats['grad_norms_pre'].append(grad_norm_pre.item())
        self.stats['grad_norms_post'].append(grad_norm_post.item())
        
        # Target lag
        with torch.no_grad():
            target_q = self.target_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            lag = torch.abs(current_q_values - target_q).mean().item()
            self.stats['target_lag'].append(lag)
        
        return {
            'loss': loss.item(),
            'td_error': np.abs(td_errors).mean(),
            'q_mean': current_q_values.mean().item(),
            'grad_norm': grad_norm_post.item()
        }
    
    def train_episode(self):
        """
        Train for one episode.
        """
        self.game_engine.reset_game()
        episode_reward = 0
        steps = 0
        
        while not self.game_engine.game_over and steps < 1000:
            # Get state
            state = self.get_state()
            
            # Select action
            action = self.select_action(state, training=True)
            
            # Take action
            old_score = self.game_engine.score
            self.game_engine.set_direction_from_algorithm(
                self._action_to_direction(action)
            )
            self.game_engine.move_snake()
            
            # Get next state
            next_state = self.get_state()
            done = self.game_engine.game_over
            ate_food = self.game_engine.score > old_score
            
            # Calculate reward
            reward = self._calculate_reward(done, ate_food)
            episode_reward += reward
            
            # Store transition
            # Convert normalized states back to raw for storage
            raw_state = self.state_repr.get_enhanced_state(self.game_engine)
            self.memory.add(state, action, reward, next_state, done)
            
            # Train if past warmup
            if self.total_steps >= self.warmup_steps:
                self.train_step()
            
            # Update exploration
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)
            
            self.total_steps += 1
            steps += 1
        
        # Log epsilon
        self.stats['epsilon_history'].append(self.epsilon)
        self.stats['buffer_size'].append(len(self.memory))
        
        return {
            'score': self.game_engine.score,
            'steps': steps,
            'reward': episode_reward
        }
    
    def evaluate(self, num_episodes=20, epsilon=0.05):
        """
        Evaluate agent with low exploration.
        """
        scores = []
        
        for _ in range(num_episodes):
            self.game_engine.reset_game()
            steps = 0
            
            # Temporarily set epsilon
            old_epsilon = self.epsilon
            self.epsilon = epsilon
            
            while not self.game_engine.game_over and steps < 1000:
                state = self.get_state()
                action = self.select_action(state, training=False)
                self.game_engine.set_direction_from_algorithm(
                    self._action_to_direction(action)
                )
                self.game_engine.move_snake()
                steps += 1
            
            scores.append(self.game_engine.score)
            self.epsilon = old_epsilon
        
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'max': np.max(scores),
            'min': np.min(scores),
            'std': np.std(scores)
        }
    
    def _action_to_direction(self, action):
        """Convert action index to direction (relative to current)."""
        direction = self.game_engine.direction
        
        if action == 0:  # Turn left
            direction_map = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}
        elif action == 1:  # Straight
            return direction
        else:  # Turn right
            direction_map = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
        
        return direction_map.get(direction, direction)
    
    def _calculate_reward(self, done, ate_food):
        """Calculate reward (identical to Q-Learning)."""
        if done:
            return REWARD_DEATH
        elif ate_food:
            return REWARD_FOOD
        else:
            return REWARD_SURVIVAL
    
    def save_model(self, filepath):
        """Save model, normalizer, and stats."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'update_steps': self.update_steps,
            'stats': self.stats
        }, filepath)
        
        # Save normalizer
        norm_path = filepath.replace('.pth', '_normalizer.npz')
        self.normalizer.save(norm_path)
        
        # Save readable stats
        stats_path = filepath.replace('.pth', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump({k: v[-1000:] for k, v in self.stats.items()}, f, indent=2)
        
        print(f"[SAVE] Model saved to {filepath}")
    
    def load_model(self, filepath, silent=False, for_gameplay=False):
        """Load model, normalizer, and stats."""
        if not os.path.exists(filepath):
            if not silent:
                print(f"[ERROR] Model file not found: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            self.online_net.load_state_dict(checkpoint['online_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # For gameplay, always use epsilon=0 (pure exploitation)
            if for_gameplay:
                self.epsilon = 0.0
            else:
                self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            
            self.total_steps = checkpoint.get('total_steps', 0)
            self.update_steps = checkpoint.get('update_steps', 0)
            self.stats = checkpoint.get('stats', self.stats)
            
            # Load normalizer
            norm_path = filepath.replace('.pth', '_normalizer.npz')
            if os.path.exists(norm_path):
                self.normalizer.load(norm_path)
                if not silent:
                    print(f"[LOAD] Normalizer loaded from {norm_path}")
            else:
                if not silent:
                    print(f"[WARNING] Normalizer file not found: {norm_path}")
            
            if not silent:
                print(f"[LOAD] Model loaded from {filepath}")
                print(f"  Total steps: {self.total_steps}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                if for_gameplay:
                    print(f"  Mode: Gameplay (epsilon=0.0, pure exploitation)")
            
            return True
            
        except Exception as e:
            if not silent:
                print(f"[ERROR] Failed to load model: {e}")
            return False
