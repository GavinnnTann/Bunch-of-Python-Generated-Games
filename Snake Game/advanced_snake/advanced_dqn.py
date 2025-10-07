"""
Advanced DQN module for the Snake Game.
Implements a combination of DQN, Double DQN, Dueling DQN, and Prioritized Experience Replay 
to learn optimal snake movement strategy.
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from constants import *

# Check for GPU availability and configure PyTorch accordingly
is_gpu_available = torch.cuda.is_available() and USE_CUDA
device = torch.device("cuda" if is_gpu_available else "cpu")

# Print CUDA information if available
if torch.cuda.is_available():
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    
    # Set memory management options for better performance
    if USE_CUDA:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode enabled for faster training")
        optimal_batch_size = GPU_BATCH_SIZE
    else:
        print("CUDA is available but USE_CUDA is set to False in constants.py")
        optimal_batch_size = CPU_BATCH_SIZE
else:
    print("CUDA is not available. Using CPU for training.")
    optimal_batch_size = CPU_BATCH_SIZE

# Print CUDA information if available
if torch.cuda.is_available():
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
else:
    print("CUDA is not available. Using CPU for training.")

# Define experience tuple
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class SumTree:
    """
    Sum Tree data structure for efficient prioritized experience replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (experiences)
        self.tree = np.zeros(2 * capacity - 1)  # Tree array
        self.data = np.zeros(capacity, dtype=object)  # Data array
        self.data_pointer = 0  # Current position in data array
        self.n_entries = 0  # Current number of entries
        
    def add(self, priority, data):
        """Add a new experience with given priority."""
        # Find index in tree
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Store data
        self.data[self.data_pointer] = data
        
        # Update tree
        self.update(tree_idx, priority)
        
        # Move pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Increase count of entries if not full yet
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, tree_idx, priority):
        """Update the priority of an experience."""
        # Change = new priority - old priority
        change = priority - self.tree[tree_idx]
        
        # Update tree value
        self.tree[tree_idx] = priority
        
        # Propagate change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
            
    def get_leaf(self, v):
        """Get experience based on priority value v."""
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            # Otherwise search left or right
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - (self.capacity - 1)
        
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def total_priority(self):
        """Return total priority (sum of all priorities)."""
        return self.tree[0]
        

class PrioritizedReplayBuffer:
    """
    Experience replay buffer using prioritized sampling.
    """
    def __init__(self, capacity, alpha=DQN_PRIORITIZED_ALPHA):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # How much to prioritize (0 = uniform sampling)
        self.e = 0.01  # Small epsilon to avoid zero priority
        self.max_priority = 1.0  # Max priority for new experiences (in raw TD-error space)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority."""
        # Store tensors on CPU to save GPU memory
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu()
            
        experience = Experience(state, action, reward, next_state, done)
        # Apply alpha power only when writing to SumTree
        self.tree.add((self.max_priority + self.e) ** self.alpha, experience)
        
    def sample(self, batch_size, beta=DQN_PRIORITIZED_BETA):
        """Sample a batch of experiences based on priority."""
        batch = []
        indices = []
        priorities = []
        weights = []
        
        # Calculate segment size
        segment = self.tree.total_priority() / batch_size
        
        # Increase beta over time (annealing)
        beta = min(1.0, beta)
        
        # Sample from each segment
        for i in range(batch_size):
            # Get a value from each segment
            a = segment * i
            b = segment * (i + 1)
            
            # Sample uniformly from segment
            v = np.random.uniform(a, b)
            
            # Get experience
            idx, priority, experience = self.tree.get_leaf(v)
            
            indices.append(idx)
            batch.append(experience)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        weights = np.power(self.capacity * sampling_probabilities, -beta)
        weights = weights / weights.max()  # Normalize weights
        
        return batch, indices, torch.tensor(weights, dtype=torch.float32, device=device)
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions using raw TD errors."""
        for idx, priority in zip(indices, priorities):
            # Store raw priority value for max_priority tracking
            # Add small epsilon to avoid zero priority
            raw_priority = float(priority) + self.e
            
            # Update max_priority in raw space (before alpha power)
            self.max_priority = max(self.max_priority, raw_priority)
            
            # Apply alpha power only when writing to SumTree
            self.tree.update(idx, raw_priority ** self.alpha)
            
    def __len__(self):
        """Return the current size of memory."""
        return self.tree.n_entries


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.
    Uses ReLU activations for better performance and training stability.
    """
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Determine hidden size based on hardware capability
        hidden_size = DQN_HIDDEN_SIZE * 2 if is_gpu_available else DQN_HIDDEN_SIZE
        
        # Feature extraction layers without batch normalization
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the dueling network
        x can be of shape [batch_size, features] or [batch_size, 1, features] or just [features]
        """
        # Handle single state case (no batch dimension)
        if len(x.shape) == 1:
            # If input is [features], add batch dimension to make it [1, features]
            x = x.unsqueeze(0)
            
        # Handle case with extra dimension
        if len(x.shape) > 2:
            # If input is [batch_size, 1, features], reshape to [batch_size, features]
            x = x.view(x.shape[0], -1)
            
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Use the last dimension for mean regardless of tensor shape
        q_values = value + advantages - advantages.mean(dim=-1, keepdim=True)
        
        return q_values


class AdvancedDQNAgent:
    """
    Advanced DQN Agent combining Double DQN, Dueling DQN, and Prioritized Experience Replay.
    Uses relative actions (turn left/straight/right) instead of absolute directions.
    """
    def __init__(self, game_engine, state_size=11, action_size=3):  # 3 actions: left, straight, right
        self.game_engine = game_engine
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN parameters
        self.gamma = DQN_GAMMA
        self.epsilon = DQN_EPSILON
        self.epsilon_min = DQN_EPSILON_MIN
        self.epsilon_decay = DQN_EPSILON_DECAY
        self.learning_rate = DQN_LEARNING_RATE
        # Use the optimal batch size based on available hardware
        self.batch_size = optimal_batch_size if is_gpu_available else DQN_BATCH_SIZE
        self.beta = DQN_PRIORITIZED_BETA
        self.beta_increment = DQN_BETA_INCREMENT
        
        # Polyak averaging parameter for soft target updates
        self.tau = 1e-3  # Small value for gradual target net updates
        
        # Neural networks (policy and target)
        self.policy_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is used for evaluation only
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(DQN_MEMORY_SIZE)
        self.learn_step_counter = 0
        
        # Relative actions: 0=left, 1=straight, 2=right
        self.actions = ["LEFT", "STRAIGHT", "RIGHT"]
        
        # Direction mappings
        self.direction_vectors = {
            "UP": UP,
            "DOWN": DOWN,
            "LEFT": LEFT,
            "RIGHT": RIGHT
        }
        
        # Define the relative direction mapping
        # For each current direction, what are the resulting directions
        # when turning left, going straight, or turning right
        self.relative_direction_mapping = {
            "UP": ["LEFT", "UP", "RIGHT"],       # When facing UP: left=LEFT, straight=UP, right=RIGHT
            "RIGHT": ["UP", "RIGHT", "DOWN"],    # When facing RIGHT: left=UP, straight=RIGHT, right=DOWN
            "DOWN": ["RIGHT", "DOWN", "LEFT"],   # When facing DOWN: left=RIGHT, straight=DOWN, right=LEFT
            "LEFT": ["DOWN", "LEFT", "UP"]       # When facing LEFT: left=DOWN, straight=LEFT, right=UP
        }
        
        # Statistics for visualization - use bounded deques to limit memory growth
        # Store last 10000 items for each stat
        max_stats_size = 10000
        self.stats = {
            'losses': deque(maxlen=max_stats_size),
            'rewards': deque(maxlen=max_stats_size),
            'q_values': deque(maxlen=max_stats_size),
            'epsilon': deque(maxlen=max_stats_size)
        }
    
    def get_state(self):
        """
        Convert the current game state to a tensor representation for the neural network.
        """
        snake = self.game_engine.snake
        head = snake[0]
        food = self.game_engine.food
        
        # Detect danger in each direction
        # 1 = danger, 0 = safe
        danger_straight = 0
        danger_right = 0
        danger_left = 0
        
        # Get the current direction
        current_direction = self.game_engine.direction
        
        # Check danger straight
        next_pos_straight = (head[0] + current_direction[0], head[1] + current_direction[1])
        if (next_pos_straight[0] < 0 or next_pos_straight[0] >= GRID_HEIGHT or
            next_pos_straight[1] < 0 or next_pos_straight[1] >= GRID_WIDTH or
            next_pos_straight in list(snake)[:-1]):  # Skip checking collision with tail
            danger_straight = 1
        
        # Get right and left directions relative to current direction
        if current_direction == UP:
            right_dir = RIGHT
            left_dir = LEFT
        elif current_direction == RIGHT:
            right_dir = DOWN
            left_dir = UP
        elif current_direction == DOWN:
            right_dir = LEFT
            left_dir = RIGHT
        else:  # LEFT
            right_dir = UP
            left_dir = DOWN
        
        # Check danger right
        next_pos_right = (head[0] + right_dir[0], head[1] + right_dir[1])
        if (next_pos_right[0] < 0 or next_pos_right[0] >= GRID_HEIGHT or
            next_pos_right[1] < 0 or next_pos_right[1] >= GRID_WIDTH or
            next_pos_right in list(snake)[:-1]):
            danger_right = 1
        
        # Check danger left
        next_pos_left = (head[0] + left_dir[0], head[1] + left_dir[1])
        if (next_pos_left[0] < 0 or next_pos_left[0] >= GRID_HEIGHT or
            next_pos_left[1] < 0 or next_pos_left[1] >= GRID_WIDTH or
            next_pos_left in list(snake)[:-1]):
            danger_left = 1
        
        # Food direction relative to snake head
        food_left = int(food[1] < head[1])
        food_right = int(food[1] > head[1])
        food_up = int(food[0] < head[0])
        food_down = int(food[0] > head[0])
        
        # Current snake direction
        dir_up = int(current_direction == UP)
        dir_right = int(current_direction == RIGHT)
        dir_down = int(current_direction == DOWN)
        dir_left = int(current_direction == LEFT)
        
        # Create state representation as a tensor - shape [11]
        state = torch.tensor([
            danger_straight,
            danger_right,
            danger_left,
            food_up,
            food_right,
            food_down,
            food_left,
            dir_up,
            dir_right,
            dir_down,
            dir_left
        ], dtype=torch.float32, device=device)
        
        return state
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        Returns the selected action index.
        """
        if training and random.random() < self.epsilon:
            # Exploration: select a random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: select the action with the highest Q-value
            with torch.no_grad():
                q_values = self.policy_net(state)
                self.stats['q_values'].append(float(q_values.max().item()))
                return q_values.argmax().item()
    
    def calculate_reward(self, old_score, game_over, old_distance, new_distance):
        """
        Calculate reward based on game state transition.
        """
        # Get current game state
        current_score = self.game_engine.score
        reward = 0
        
        # Major rewards/penalties
        if game_over:
            reward += REWARD_DEATH
        elif current_score > old_score:  # Ate food
            reward += REWARD_FOOD
        else:
            reward += REWARD_SURVIVAL  # Small reward for surviving
        
        # Reward for moving closer to food
        if new_distance < old_distance:
            reward += REWARD_MOVE_TOWARDS_FOOD
        elif new_distance > old_distance:
            reward += REWARD_MOVE_AWAY_FROM_FOOD
        
        return reward
    
    def optimize_model(self):
        """
        Perform a single optimization step on the model using prioritized experience replay.
        Optimized for GPU acceleration when available.
        """
        # Check if we have enough samples in memory
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample from memory with priorities
        experiences, indices, weights = self.memory.sample(self.batch_size, beta=self.beta)
        
        # Unpack experiences
        batch = Experience(*zip(*experiences))
        
        # Create tensors for batch training - convert CPU stored tensors back to device
        states = []
        next_states = []
        
        for s in batch.state:
            if isinstance(s, torch.Tensor):
                states.append(s.to(device))
            else:
                states.append(torch.tensor(s, dtype=torch.float32, device=device))
                
        for ns in batch.next_state:
            if ns is not None:  # Handle potential None values for terminal states
                if isinstance(ns, torch.Tensor):
                    next_states.append(ns.to(device))
                else:
                    next_states.append(torch.tensor(ns, dtype=torch.float32, device=device))
            else:
                # Use a zero tensor for terminal states
                next_states.append(torch.zeros_like(states[0], device=device))
        
        # Stack tensors for efficient batch processing
        # Ensure consistent dimensions by reshaping if needed
        try:
            state_batch = torch.stack(states)
            # Check if reshaping is needed (if states are [11] rather than [1, 11])
            if len(state_batch.shape) == 2 and state_batch.shape[1] == 11:  # [batch, 11]
                state_batch = state_batch.unsqueeze(1)  # Make [batch, 1, 11]
        except RuntimeError as e:
            # print(f"Error stacking states: {e}")
            # Print out shapes of the first few states to diagnose (commented out to reduce terminal output)
            # for i, s in enumerate(states[:5]):
            #     print(f"State {i} shape: {s.shape}")
            # If states have inconsistent shapes, try to normalize them
            normalized_states = []
            for s in states:
                if len(s.shape) == 1:  # [features]
                    normalized_states.append(s.unsqueeze(0))  # [1, features]
                else:
                    normalized_states.append(s)
            state_batch = torch.stack(normalized_states)
            
        # Same for next states
        next_state_batch = torch.stack(next_states)
        if len(next_state_batch.shape) == 2 and next_state_batch.shape[1] == 11:
            next_state_batch = next_state_batch.unsqueeze(1)
        
        # Create action, reward and done tensors
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Debug the shapes (commented out to reduce terminal output)
        # print(f"Debug - State batch shape: {state_batch.shape}")
        # print(f"Debug - Action batch shape: {action_batch.shape}")
        
        # Forward pass through the network
        q_values = self.policy_net(state_batch)
        # print(f"Debug - Q values shape: {q_values.shape}")
        
        # Ensure q_values and action_batch are compatible for gather operation
        if len(q_values.shape) > 2 and q_values.shape[1] == 1:  # If q_values is [batch, 1, actions]
            q_values = q_values.squeeze(1)  # Make [batch, actions]
            # print(f"Debug - Squeezed Q values shape: {q_values.shape}")
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        try:
            state_action_values = q_values.gather(1, action_batch)
        except RuntimeError as e:
            # Comment out debug prints to reduce terminal output
            # print(f"Error in gather operation: {e}")
            # print(f"Final shapes - Q values: {q_values.shape}, Actions: {action_batch.shape}")
            
            # Try a different approach
            try:
                # Flatten q_values if needed
                if len(q_values.shape) > 2:
                    q_values = q_values.view(self.batch_size, -1)
                    # print(f"Reshaped q_values to: {q_values.shape}")
                
                # Ensure action_batch is properly shaped for indexing
                flat_actions = action_batch.view(-1)
                # print(f"Flat actions shape: {flat_actions.shape}")
                
                # Use a list comprehension to gather the values
                values = [q_values[i, a.item()] for i, a in enumerate(flat_actions)]
                state_action_values = torch.stack(values).unsqueeze(1)
                # print(f"Manually gathered values shape: {state_action_values.shape}")
            except Exception as e2:
                print(f"Alternative gather failed: {e2}")
                # As a last resort, create a tensor of zeros
                state_action_values = torch.zeros(self.batch_size, 1, device=device)
                print("Falling back to zero tensor")
                # Return early without optimization
                return 0
        
        # Double DQN:
        with torch.no_grad():
            # Get the actions from the policy network for the next state
            next_q_values = self.policy_net(next_state_batch)
            
            # Handle dimension mismatch for next_q_values if needed
            if len(next_q_values.shape) > 2 and next_q_values.shape[1] == 1:
                next_q_values = next_q_values.squeeze(1)
                
            next_actions = next_q_values.argmax(dim=1, keepdim=True)
            
            # Get the Q-values from the target network
            target_q_values = self.target_net(next_state_batch)
            
            # Handle dimension mismatch for target_q_values if needed
            if len(target_q_values.shape) > 2 and target_q_values.shape[1] == 1:
                target_q_values = target_q_values.squeeze(1)
                
            next_state_values = target_q_values.gather(1, next_actions)
            
            # Mask for terminal states
            next_state_values = next_state_values * (1 - done_batch)
            
            # Expected Q values
            expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute the TD errors for priority updates
        td_errors = torch.abs(state_action_values - expected_state_action_values).detach()
        
        # Update priorities in the replay buffer (convert to CPU for numpy operations)
        self.memory.update_priorities(indices, td_errors.cpu().numpy())
        
        # Calculate weighted loss
        loss = (weights * F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')).mean()
        
        # Print the loss value for debugging (commented out to reduce terminal output)
        # print(f"Debug - Loss: {loss.item()}")
        
        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
        loss.backward()
        
        # Clip gradients to prevent exploding gradients (with None check)
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # Track loss for visualization
        self.stats['losses'].append(loss.item())
        
        return loss.item()
    
    def get_absolute_direction_from_relative(self, relative_action):
        """
        Convert a relative action (left/straight/right) to an absolute direction
        based on the current direction of the snake.
        """
        # Get current direction as string
        current_dir_vector = self.game_engine.direction
        current_dir = None
        for dir_name, vector in self.direction_vectors.items():
            if vector == current_dir_vector:
                current_dir = dir_name
                break
                
        if current_dir is None:
            # Fallback if direction is not recognized
            return RIGHT
            
        # Get the new absolute direction based on the relative action
        new_dir_name = self.relative_direction_mapping[current_dir][relative_action]
        return self.direction_vectors[new_dir_name]
        
    def train_step(self):
        """
        Execute one training step:
        1. Get current state
        2. Choose action
        3. Take action and observe reward and next state
        4. Remember experience
        5. Update model
        Returns (reward, done, ate_food)
        """
        # Get current state
        state = self.get_state()
        
        # Get current head position and calculate distance to food
        head = self.game_engine.snake[0]
        food = self.game_engine.food
        old_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
        old_score = self.game_engine.score
        
        # Choose relative action (left/straight/right)
        action_idx = self.select_action(state)
        
        # Convert to absolute direction
        absolute_direction = self.get_absolute_direction_from_relative(action_idx)
        
        # Take action
        self.game_engine.set_direction_from_algorithm(absolute_direction)
        self.game_engine.move_snake()
        
        # Get new state, calculate reward
        next_state = self.get_state()
        game_over = self.game_engine.game_over
        
        # Get new head position and calculate distance to food
        new_head = self.game_engine.snake[0]
        food = self.game_engine.food
        new_distance = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
        
        # Calculate reward
        reward = self.calculate_reward(old_score, game_over, old_distance, new_distance)
        ate_food = self.game_engine.score > old_score
        
        # Store transition in memory
        self.memory.add(state, action_idx, reward, next_state, game_over)
        
        # Track reward
        self.stats['rewards'].append(reward)
        
        # Soft target network update with Polyak averaging
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Update beta value for prioritized replay
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Optimize model
        loss = self.optimize_model()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.stats['epsilon'].append(self.epsilon)
        
        # Increment counter
        self.learn_step_counter += 1
        
        return reward, game_over, ate_food
    
    def save_model(self, filepath):
        """Save the model to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert deque stats to lists for serialization
        serializable_stats = {
            'losses': list(self.stats['losses']),
            'rewards': list(self.stats['rewards']),
            'q_values': list(self.stats['q_values']),
            'epsilon': list(self.stats['epsilon'])
        }
        
        # Save the model and related info
        # Move models to CPU before saving to ensure compatibility across devices
        torch.save({
            'policy_net_state_dict': self.policy_net.cpu().state_dict(),
            'target_net_state_dict': self.target_net.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': serializable_stats,
            'learn_step_counter': self.learn_step_counter,
            'beta': self.beta,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'device': str(device),
            'is_gpu_available': is_gpu_available
        }, filepath)
        
        # Move models back to the original device
        self.policy_net.to(device)
        self.target_net.to(device)
        
        print(f"Model saved to {filepath}")
        
    # Class-level cache for model compatibility checks
    _model_compatibility_cache = {}
    
    def load_model(self, filepath, silent=False, for_gameplay=False):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to the model file
            silent: If True, suppress output messages
            for_gameplay: If True, set epsilon to 0 for pure exploitation (no random moves)
        """
        # Check cache first to avoid repeated messages and checks
        if filepath in AdvancedDQNAgent._model_compatibility_cache:
            cached_result = AdvancedDQNAgent._model_compatibility_cache[filepath]
            if cached_result['exists'] and not silent:
                print(f"Model loaded successfully. Training steps: {cached_result['steps']}")
            # Force epsilon to 0 for gameplay even with cached model
            if for_gameplay:
                self.epsilon = 0.0
            return cached_result['exists']
        
        if os.path.exists(filepath):
            # Load the checkpoint, mapping tensors to the appropriate device
            checkpoint = torch.load(filepath, map_location=device)
            
            # Print information about the saved model
            saved_device = checkpoint.get('device', 'unknown device')
            saved_gpu = checkpoint.get('is_gpu_available', False)
            current_gpu = is_gpu_available
            
            if not silent:
                print(f"Loading model trained on {saved_device}")
                if saved_gpu != current_gpu:
                    print(f"Warning: Model was trained with GPU={saved_gpu}, but current GPU={current_gpu}")
            
            # Load network parameters
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            
            # Ensure models are on the correct device
            self.policy_net.to(device)
            self.target_net.to(device)
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Fix optimizer device mapping if needed
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            
            # Load other parameters
            self.epsilon = checkpoint['epsilon']
            
            # Force epsilon to 0 for gameplay mode (pure exploitation, no random moves)
            if for_gameplay:
                self.epsilon = 0.0
                if not silent:
                    print(f"Gameplay mode: Epsilon forced to 0.0 (pure exploitation)")
            elif not silent:
                print(f"Training mode: Epsilon loaded as {self.epsilon:.4f}")
            
            self.batch_size = checkpoint.get('batch_size', self.batch_size)
            
            # Convert stats back to deques
            saved_stats = checkpoint['stats']
            max_stats_size = self.stats['losses'].maxlen
            for key in self.stats:
                if key in saved_stats:
                    self.stats[key] = deque(saved_stats[key], maxlen=max_stats_size)
            
            self.learn_step_counter = checkpoint['learn_step_counter']
            self.beta = checkpoint.get('beta', self.beta)  # Default to init value if not found
            self.tau = checkpoint.get('tau', self.tau)     # Default to init value if not found
            
            # Set target network to eval mode
            self.target_net.eval()
            
            # Cache the result
            AdvancedDQNAgent._model_compatibility_cache[filepath] = {
                'exists': True,
                'device': saved_device,
                'steps': self.learn_step_counter,
                'gpu_mismatch': saved_gpu != current_gpu
            }
            
            if not silent:
                print(f"Model loaded successfully. Training steps: {self.learn_step_counter}")
            return True
            
        # Cache negative result
        AdvancedDQNAgent._model_compatibility_cache[filepath] = {
            'exists': False,
            'steps': 0
        }
        
        if not silent:
            print(f"No model found at {filepath}")
        return False
    
    def get_action(self, state):
        """Get the next move based on the trained DQN for gameplay."""
        action_idx = self.select_action(state, training=False)
        return self.get_absolute_direction_from_relative(action_idx)