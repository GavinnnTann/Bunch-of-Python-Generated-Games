"""
Enhanced DQN module with curriculum learning and A* guidance.

Key Improvements:
1. Enhanced state representation with spatial awareness
2. Curriculum learning - progressive difficulty
3. A* algorithm integration for reward shaping
4. Additional safety features to avoid self-collision
5. Advanced reward structure with lookahead
"""

import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque
from constants import *
from advanced_dqn import AdvancedDQNAgent, DuelingDQN, device
from algorithms import SnakeAlgorithms


class EnhancedStateRepresentation:
    """
    Enhanced state representation with more spatial information.
    Increases state size from 11 to ~30 features for better decision making.
    """
    
    @staticmethod
    def get_enhanced_state(game_engine):
        """
        Create enhanced state with:
        - Danger detection in multiple directions
        - Distance to walls
        - Body segment proximity
        - Food location (multiple representations)
        - Path feasibility
        - Snake length context
        """
        snake = game_engine.snake
        head = snake[0]
        food = game_engine.food
        current_direction = game_engine.direction
        
        state_features = []
        
        # 1. IMMEDIATE DANGER (3 features) - straight, right, left
        danger_info = EnhancedStateRepresentation._get_danger_info(snake, head, current_direction)
        state_features.extend(danger_info)
        
        # 2. EXTENDED DANGER (3 features) - two steps ahead
        extended_danger = EnhancedStateRepresentation._get_extended_danger(snake, head, current_direction)
        state_features.extend(extended_danger)
        
        # 3. WALL DISTANCES (4 features) - normalized distances to each wall
        wall_distances = EnhancedStateRepresentation._get_wall_distances(head)
        state_features.extend(wall_distances)
        
        # 4. FOOD DIRECTION (4 features) - up, right, down, left
        food_direction = [
            int(food[0] < head[0]),  # food up
            int(food[1] > head[1]),  # food right
            int(food[0] > head[0]),  # food down
            int(food[1] < head[1])   # food left
        ]
        state_features.extend(food_direction)
        
        # 5. FOOD DISTANCE (2 features) - Manhattan and Euclidean
        manhattan_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        euclidean_dist = np.sqrt((head[0] - food[0])**2 + (head[1] - food[1])**2)
        state_features.extend([
            manhattan_dist / (GRID_WIDTH + GRID_HEIGHT),  # Normalized
            euclidean_dist / np.sqrt(GRID_WIDTH**2 + GRID_HEIGHT**2)  # Normalized
        ])
        
        # 6. CURRENT DIRECTION (4 features) - one-hot encoded
        dir_features = [
            int(current_direction == UP),
            int(current_direction == RIGHT),
            int(current_direction == DOWN),
            int(current_direction == LEFT)
        ]
        state_features.extend(dir_features)
        
        # 7. BODY PROXIMITY (4 features) - closest body segment in each direction
        body_proximity = EnhancedStateRepresentation._get_body_proximity(snake, head)
        state_features.extend(body_proximity)
        
        # 8. SNAKE LENGTH (1 feature) - normalized
        snake_length = len(snake) / (GRID_WIDTH * GRID_HEIGHT)
        state_features.append(snake_length)
        
        # 9. AVAILABLE SPACE (3 features) - space available in each direction
        available_space = EnhancedStateRepresentation._get_available_space(snake, head, current_direction)
        state_features.extend(available_space)
        
        # 10. TAIL DIRECTION (3 features) - where is tail relative to head (safety metric)
        tail = snake[-1]
        tail_dir = [
            float(tail[0] < head[0]),  # tail above
            float(tail[1] > head[1]),  # tail right
            float(tail[0] > head[0])   # tail below
        ]
        state_features.extend(tail_dir)
        
        # 11. A* PATH GUIDANCE (3 features) - NEW! What direction does optimal path suggest?
        # This provides A* information WITHOUT taking over decision making
        try:
            from algorithms import SnakeAlgorithms
            algorithms = SnakeAlgorithms(game_engine)
            path = algorithms._find_path_astar(head, food)
            
            if path and len(path) > 1:
                next_optimal = path[1]
                # Which relative direction is the A* suggested move?
                move_dir = (next_optimal[0] - head[0], next_optimal[1] - head[1])
                
                # Convert to relative direction (straight=0, right=1, left=2)
                relative_dirs = EnhancedStateRepresentation._get_relative_directions(current_direction)
                astar_direction = [0, 0, 0]  # One-hot: [straight, right, left]
                
                for idx, rel_dir in enumerate(relative_dirs):
                    if move_dir == rel_dir:
                        astar_direction[idx] = 1
                        break
                
                state_features.extend(astar_direction)
            else:
                # No path found - all zeros
                state_features.extend([0, 0, 0])
        except:
            # If A* fails, use zeros
            state_features.extend([0, 0, 0])
        
        # Convert to tensor
        # Total: 3 + 3 + 4 + 4 + 2 + 4 + 4 + 1 + 3 + 3 + 3 = 34 features (increased from 31)
        return torch.tensor(state_features, dtype=torch.float32, device=device)
    
    @staticmethod
    def _get_danger_info(snake, head, direction):
        """Get immediate danger in three directions."""
        danger = [0, 0, 0]  # straight, right, left
        
        # Get directional vectors
        directions = EnhancedStateRepresentation._get_relative_directions(direction)
        
        for i, dir_vec in enumerate(directions):
            next_pos = (head[0] + dir_vec[0], head[1] + dir_vec[1])
            if EnhancedStateRepresentation._is_collision(next_pos, snake):
                danger[i] = 1
        
        return danger
    
    @staticmethod
    def _get_extended_danger(snake, head, direction):
        """Look two steps ahead for danger."""
        danger = [0, 0, 0]  # straight, right, left
        
        directions = EnhancedStateRepresentation._get_relative_directions(direction)
        
        for i, dir_vec in enumerate(directions):
            # First step
            next_pos = (head[0] + dir_vec[0], head[1] + dir_vec[1])
            if not EnhancedStateRepresentation._is_collision(next_pos, snake):
                # Second step
                next_next_pos = (next_pos[0] + dir_vec[0], next_pos[1] + dir_vec[1])
                if EnhancedStateRepresentation._is_collision(next_next_pos, snake):
                    danger[i] = 1
        
        return danger
    
    @staticmethod
    def _get_wall_distances(head):
        """Get normalized distances to walls."""
        return [
            head[0] / GRID_HEIGHT,  # Distance to top
            (GRID_WIDTH - head[1] - 1) / GRID_WIDTH,  # Distance to right
            (GRID_HEIGHT - head[0] - 1) / GRID_HEIGHT,  # Distance to bottom
            head[1] / GRID_WIDTH  # Distance to left
        ]
    
    @staticmethod
    def _get_body_proximity(snake, head):
        """Get closest body segment distance in each direction."""
        body = list(snake)[1:]  # Exclude head
        proximity = [GRID_WIDTH + GRID_HEIGHT] * 4  # up, right, down, left - initialize with max
        
        for segment in body:
            # Same column
            if segment[1] == head[1]:
                if segment[0] < head[0]:  # Above
                    proximity[0] = min(proximity[0], head[0] - segment[0])
                else:  # Below
                    proximity[2] = min(proximity[2], segment[0] - head[0])
            # Same row
            if segment[0] == head[0]:
                if segment[1] > head[1]:  # Right
                    proximity[1] = min(proximity[1], segment[1] - head[1])
                else:  # Left
                    proximity[3] = min(proximity[3], head[1] - segment[1])
        
        # Normalize
        max_dist = GRID_WIDTH + GRID_HEIGHT
        return [p / max_dist for p in proximity]
    
    @staticmethod
    def _get_available_space(snake, head, direction):
        """
        Calculate available space using flood fill in each direction.
        This helps avoid trapping situations.
        """
        directions = EnhancedStateRepresentation._get_relative_directions(direction)
        space_counts = []
        
        for dir_vec in directions:
            next_pos = (head[0] + dir_vec[0], head[1] + dir_vec[1])
            if EnhancedStateRepresentation._is_collision(next_pos, snake):
                space_counts.append(0)
            else:
                # Simple flood fill to count available spaces
                space = EnhancedStateRepresentation._count_reachable_spaces(next_pos, snake)
                space_counts.append(space / (GRID_WIDTH * GRID_HEIGHT))
        
        return space_counts
    
    @staticmethod
    def _count_reachable_spaces(start_pos, snake):
        """Count reachable empty spaces from start position using BFS."""
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        count = 0
        max_iterations = 50  # Limit to prevent slowdown
        
        while queue and count < max_iterations:
            pos = queue.popleft()
            count += 1
            
            for direction in [UP, RIGHT, DOWN, LEFT]:
                next_pos = (pos[0] + direction[0], pos[1] + direction[1])
                if (next_pos not in visited and 
                    not EnhancedStateRepresentation._is_collision(next_pos, snake)):
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return min(count, max_iterations)
    
    @staticmethod
    def _get_relative_directions(current_direction):
        """Get straight, right, left directions relative to current."""
        if current_direction == UP:
            return [UP, RIGHT, LEFT]
        elif current_direction == RIGHT:
            return [RIGHT, DOWN, UP]
        elif current_direction == DOWN:
            return [DOWN, LEFT, RIGHT]
        else:  # LEFT
            return [LEFT, UP, DOWN]
    
    @staticmethod
    def _is_collision(pos, snake):
        """Check if position results in collision."""
        return (pos[0] < 0 or pos[0] >= GRID_HEIGHT or
                pos[1] < 0 or pos[1] >= GRID_WIDTH or
                pos in list(snake)[:-1])


class EnhancedDQNAgent(AdvancedDQNAgent):
    """
    Enhanced DQN Agent with:
    - Improved state representation (34 features including A* guidance)
    - A* guided rewards (not action override)
    - Curriculum learning
    - Better exploration strategy
    """
    
    def __init__(self, game_engine, state_size=34, action_size=3):
        """Initialize with enhanced state size (34 features)."""
        # Initialize parent with larger state size
        super().__init__(game_engine, state_size=state_size, action_size=action_size)
        
        # Create A* algorithms helper
        self.algorithms = SnakeAlgorithms(game_engine)
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.curriculum_thresholds = [20, 50, 100, 200]  # UPDATED: Further lowered for realistic progression
        self.curriculum_consistency_required = 3  # Must meet threshold this many times consecutively
        self.curriculum_success_count = 0  # Track consecutive successful evaluations
        
        # Enhanced exploration with INCREASED A* weight
        self.use_astar_guidance = True  # Use A* to guide early training
        self.astar_guidance_prob = 0.5  # INCREASED from 0.3 to 0.5 for stronger guidance
        
        # Track performance for stuck detection (increased window for better assessment)
        self.recent_scores = deque(maxlen=50)  # Increased from 10 to 50 for better trend analysis
        self.stuck_counter = 0
        self.last_avg_score = 0  # Track if we're improving
        self.last_epsilon_boost_episode = -200  # Track when we last boosted epsilon (prevent oscillation)
        
        # PERFORMANCE BOOST: Set initial learning rate based on curriculum stage
        self.update_learning_rate_for_stage()
        
    def get_state(self):
        """Override to use enhanced state representation."""
        return EnhancedStateRepresentation.get_enhanced_state(self.game_engine)
    
    def update_learning_rate_for_stage(self):
        """
        PERFORMANCE BOOST: Adjust learning rate based on curriculum stage.
        Fast learning early, fine-tuning later.
        """
        stage_learning_rates = {
            0: 0.005,   # Stage 0: FAST learning for basics
            1: 0.003,   # Stage 1: Medium learning
            2: 0.002,   # Stage 2: Standard learning
            3: 0.001,   # Stage 3: Conservative learning
            4: 0.0005   # Stage 4: Fine-tuning
        }
        
        new_lr = stage_learning_rates.get(self.curriculum_stage, 0.001)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.learning_rate = new_lr
        return new_lr
    
    def perform_action(self, action):
        """
        Convert relative action to absolute direction and perform it.
        Actions: 0=turn right, 1=straight, 2=turn left
        """
        current_dir = self.game_engine.direction
        
        # Map relative action to new direction
        if current_dir == UP:
            new_dirs = [RIGHT, UP, LEFT]  # right, straight, left
        elif current_dir == RIGHT:
            new_dirs = [DOWN, RIGHT, UP]
        elif current_dir == DOWN:
            new_dirs = [LEFT, DOWN, RIGHT]
        else:  # LEFT
            new_dirs = [UP, LEFT, DOWN]
        
        new_direction = new_dirs[action]
        self.game_engine.change_direction(new_direction)
        self.game_engine.move_snake()  # Use move_snake instead of update
    
    def select_action(self, state, training=True):
        """
        Action selection using standard epsilon-greedy.
        A* guidance is now provided through state features and reward shaping,
        NOT by overriding actions. This allows DQN to actually LEARN pathfinding.
        """
        # Always use standard epsilon-greedy (no A* override)
        return super().select_action(state, training)
    
    def _get_astar_guided_action(self):
        """Get action suggestion from A* algorithm."""
        try:
            # Use A* to find path to food
            head = self.game_engine.snake[0]
            food = self.game_engine.food
            path = self.algorithms._find_path_astar(head, food)
            
            if path and len(path) > 1:
                # Get the next position from A* path
                next_pos = path[1]
                current_head = self.game_engine.snake[0]
                current_dir = self.game_engine.direction
                
                # Convert position to relative action
                move_dir = (next_pos[0] - current_head[0], next_pos[1] - current_head[1])
                
                # Map to relative action
                if current_dir == UP:
                    if move_dir == UP: return 1  # Straight
                    elif move_dir == RIGHT: return 0  # Right (turn right from up)
                    elif move_dir == LEFT: return 2  # Left
                elif current_dir == RIGHT:
                    if move_dir == RIGHT: return 1  # Straight
                    elif move_dir == DOWN: return 0  # Right
                    elif move_dir == UP: return 2  # Left
                elif current_dir == DOWN:
                    if move_dir == DOWN: return 1  # Straight
                    elif move_dir == LEFT: return 0  # Right
                    elif move_dir == RIGHT: return 2  # Left
                else:  # LEFT
                    if move_dir == LEFT: return 1  # Straight
                    elif move_dir == UP: return 0  # Right
                    elif move_dir == DOWN: return 2  # Left
        except:
            pass
        
        # Fallback to random if A* fails
        return random.randrange(self.action_size)
    
    def calculate_reward(self, old_score, game_over, old_distance, new_distance, action_taken=None):
        """
        Enhanced reward calculation with:
        - A* path alignment bonus (reward for following optimal path)
        - Safety bonus
        - Progress tracking
        - Curriculum-based scaling
        
        This allows DQN to LEARN from A* rather than being overridden by it.
        """
        current_score = self.game_engine.score
        reward = 0
        
        # Base rewards
        if game_over:
            reward += REWARD_DEATH * (1 + self.curriculum_stage * 0.5)  # Harsher penalty as it improves
        elif current_score > old_score:  # Ate food
            # UPDATED: Curriculum-scaled food reward to encourage aggressive food-seeking
            # Scale food reward based on snake length (harder to get food when longer)
            length_bonus = 1 + (len(self.game_engine.snake) / 100)
            
            # OPTIMAL HYPERPARAMETERS: Keep Stage 1 rewards at Stage 2 for exponential growth
            stage_food_multiplier = {
                0: 3.0,  # Triple reward at Stage 0 (30-60 points!) - fastest learning
                1: 2.5,  # Stage 1 optimal (achieved 230 score!) 25-50 points
                2: 2.5,  # FIXED: Keep Stage 1 level (was 2.0, caused collapse)
                3: 2.0,  # UPDATED: Gradual reduction (was 1.5)
                4: 1.0   # 10-20 points (standard)
            }.get(self.curriculum_stage, 2.0)
            
            reward += (REWARD_FOOD * 2) * length_bonus * stage_food_multiplier
            
            # PERFORMANCE BOOST: Extra bonus for first food at stage 0
            if self.curriculum_stage == 0 and current_score <= 10:
                reward += 10  # Big encouragement for early success
        else:
            # Survival reward decreases as snake gets longer (should be hunting food)
            survival_penalty = len(self.game_engine.snake) / 1000
            reward += REWARD_SURVIVAL - survival_penalty
            
            # PERFORMANCE BOOST: Stage 0 survival bonus (encourage staying alive to learn)
            if self.curriculum_stage == 0:
                survival_bonus = min(len(self.game_engine.snake) * 0.5, 5)  # Up to +5
                reward += survival_bonus
        
        # NEW: A* Alignment Bonus - Reward for moving along optimal path
        # This teaches the DQN to pathfind WITHOUT overriding its decisions
        try:
            head = self.game_engine.snake[0]
            food = self.game_engine.food
            path = self.algorithms._find_path_astar(head, food)
            
            if path and len(path) > 1:
                # Check if we moved toward the A* suggested direction
                next_optimal = path[1]
                
                # Check if current head is on the A* path (means we followed it)
                if head in path[1:min(3, len(path))]:  # Within first 2-3 steps of path
                    # OPTIMAL HYPERPARAMETERS: Keep Stage 1 support at Stage 2 for exponential growth
                    stage_astar_weight = {
                        0: 1.0,   # Strong guidance at Stage 0 (4x original!)
                        1: 0.75,  # Stage 1 optimal (achieved 230 score!)
                        2: 0.75,  # FIXED: Keep Stage 1 level (was 0.50, caused collapse)
                        3: 0.60,  # UPDATED: Gradual reduction (was 0.25)
                        4: 0.0    # No guidance at Stage 4 (fully independent)
                    }.get(self.curriculum_stage, 0.5)
                    
                    astar_bonus = self.astar_guidance_prob * stage_astar_weight
                    reward += astar_bonus
        except:
            pass
        
        # PERFORMANCE BOOST: Progressive distance rewards - scale by actual improvement
        # This provides a gradient for learning "closer = better" much faster
        if old_distance > 0:  # Avoid division by zero
            distance_change = old_distance - new_distance
            distance_improvement_ratio = distance_change / old_distance
            
            if distance_improvement_ratio > 0:
                # Moving closer - scale reward by how much closer we got
                # e.g., 50% closer = much better than 5% closer
                reward += REWARD_MOVE_TOWARDS_FOOD * 10 * distance_improvement_ratio
            else:
                # Moving away - penalize based on how much farther
                reward += REWARD_MOVE_AWAY_FROM_FOOD * 5 * abs(distance_improvement_ratio)
        else:
            # Fallback to simple binary reward if distance is 0 (shouldn't happen)
            if new_distance < old_distance:
                reward += REWARD_MOVE_TOWARDS_FOOD * 3
            elif new_distance > old_distance:
                reward += REWARD_MOVE_AWAY_FROM_FOOD * 2
        
        # Safety bonus - reward for having escape routes
        head = self.game_engine.snake[0]
        current_dir = self.game_engine.direction
        safe_moves = 0
        
        for direction in EnhancedStateRepresentation._get_relative_directions(current_dir):
            next_pos = (head[0] + direction[0], head[1] + direction[1])
            if not EnhancedStateRepresentation._is_collision(next_pos, self.game_engine.snake):
                safe_moves += 1
        
        if safe_moves >= 2:
            reward += 0.05  # Small bonus for having options
        elif safe_moves == 0 and not game_over:
            reward -= 5  # Penalty for getting into tight spot
        
        # Anti-loop behavior - penalize if moving in circles
        if len(self.game_engine.snake) > 5:
            head = self.game_engine.snake[0]
            # Check if we've visited this general area recently
            recent_positions = list(self.game_engine.snake)[1:6]
            nearby_count = sum(1 for pos in recent_positions 
                             if abs(pos[0] - head[0]) <= 2 and abs(pos[1] - head[1]) <= 2)
            if nearby_count >= 3:
                reward -= 0.2  # Penalty for circling
        
        return reward
    
    def update_curriculum(self, score, current_episode=0):
        """
        Update curriculum stage based on SUSTAINED performance.
        Requires consistent achievement of threshold before advancing.
        
        Args:
            score: Score from the current episode
            current_episode: Current episode number (for cooldown tracking)
        """
        self.recent_scores.append(score)
        
        # Need at least 10 scores to evaluate performance
        if len(self.recent_scores) >= 10:
            avg_score = np.mean(self.recent_scores)
            min_score = np.min(self.recent_scores)
            
            # Check if we should advance curriculum
            if self.curriculum_stage < len(self.curriculum_thresholds):
                current_threshold = self.curriculum_thresholds[self.curriculum_stage]
                
                # OPTIMAL ADVANCEMENT: Stricter criteria based on 230-score analysis
                # Agent must MASTER current stage before advancing
                # Stage 0: Need to meet threshold (easy start)
                # Stage 1: Need avg 90 to advance (was 60) - ensures readiness for Stage 2
                # Stage 2+: Need 30% above threshold for stability
                
                if self.curriculum_stage == 0:
                    advancement_threshold = current_threshold  # 20 (lenient start)
                elif self.curriculum_stage == 1:
                    advancement_threshold = current_threshold * 1.8  # 90 (MUCH STRICTER - was 60)
                else:
                    advancement_threshold = current_threshold * 1.3  # 130, 260 (moderately strict)
                
                if avg_score >= advancement_threshold:
                    self.curriculum_success_count += 1
                    
                    # Require multiple consecutive successful evaluations
                    if self.curriculum_success_count >= self.curriculum_consistency_required:
                        old_stage = self.curriculum_stage
                        old_astar_prob = self.astar_guidance_prob
                        old_epsilon = self.epsilon
                        old_lr = self.learning_rate
                        
                        self.curriculum_stage += 1
                        self.curriculum_success_count = 0  # Reset for next stage
                        
                        # PERFORMANCE BOOST: Update learning rate for new stage
                        new_lr = self.update_learning_rate_for_stage()
                        
                        # OPTIMAL HYPERPARAMETERS: Maintain strong exploration at each stage
                        # Based on analysis: epsilon 0.05-0.10 was optimal for 230-score breakthrough
                        if self.curriculum_stage == 1:
                            self.astar_guidance_prob = 0.35  # Reduce A* from 0.5 to 0.35
                            # Allow lower epsilon for more exploitation
                            if self.epsilon < 0.1:  # Changed from 0.3
                                self.epsilon = 0.1
                        elif self.curriculum_stage == 2:
                            self.astar_guidance_prob = 0.20  # Further reduce A* to 0.20
                            if self.epsilon < 0.12:  # FIXED: Raise floor (was 0.05) for optimal exploration
                                self.epsilon = 0.12
                        elif self.curriculum_stage == 3:
                            self.astar_guidance_prob = 0.10  # Minimal A* guidance
                            if self.epsilon < 0.08:  # FIXED: Raise floor (was 0.04)
                                self.epsilon = 0.08
                        elif self.curriculum_stage >= 4:
                            self.astar_guidance_prob = 0.0  # No A* guidance at final stage
                            if self.epsilon < 0.05:  # UPDATED: Raise floor (was 0.03)
                                self.epsilon = 0.05
                        
                        # VISIBLE logging of curriculum advancement
                        print(f"\n{'='*70}")
                        print(f"[CURRICULUM] ADVANCED: Stage {old_stage} -> Stage {self.curriculum_stage}")
                        print(f"Average Score: {avg_score:.1f} >= Threshold: {advancement_threshold}")
                        print(f"Minimum Score: {min_score:.1f} (occasional low scores allowed)")
                        print(f"Consistency: {self.curriculum_consistency_required} consecutive successful evaluations")
                        print(f"STRATEGY CHANGES:")
                        print(f"  • A* Reward Weight: {old_astar_prob:.2f} -> {self.astar_guidance_prob:.2f} ({old_astar_prob - self.astar_guidance_prob:+.2f})")
                        print(f"  • Epsilon:          {old_epsilon:.4f} -> {self.epsilon:.4f}")
                        print(f"  • Learning Rate:    {old_lr:.5f} -> {new_lr:.5f} ({new_lr - old_lr:+.5f})")
                        print(f"  • Death Penalty:    {1 + old_stage * 0.5:.1f}x -> {1 + self.curriculum_stage * 0.5:.1f}x")
                        print(f"NOTE: A* now guides through REWARDS, not action override")
                        print(f"      DQN learns to pathfind by being rewarded for following A* hints")
                        print(f"{'='*70}\n")
                    else:
                        # Making progress but need more consistency
                        print(f"[CURRICULUM] Progress: {self.curriculum_success_count}/{self.curriculum_consistency_required} "
                              f"evaluations passed (Avg: {avg_score:.1f}, Min: {min_score:.1f})")
                else:
                    # Reset if criteria not met
                    if self.curriculum_success_count > 0:
                        print(f"[CURRICULUM] Criteria not met - resetting progress "
                              f"(Avg: {avg_score:.1f} < Required: {advancement_threshold:.1f})")
                    self.curriculum_success_count = 0
                
            # Detect if TRULY stuck (much more conservative than before)
            # Only trigger if we have enough data and scores are NOT improving
            if len(self.recent_scores) >= 50:
                avg_score = np.mean(self.recent_scores)
                score_variance = np.var(self.recent_scores)
                
                # UPDATED: Check if stuck at current curriculum level
                # At each stage, if we're not advancing for a long time, we're stuck
                stage_stuck_threshold = self.curriculum_thresholds[self.curriculum_stage] if self.curriculum_stage < len(self.curriculum_thresholds) else 300
                
                # PERFORMANCE FIX: Better plateau detection
                # If we're near the threshold but not advancing, we're in a local optimum
                near_threshold = abs(avg_score - stage_stuck_threshold) < stage_stuck_threshold * 0.3
                improvement = abs(avg_score - self.last_avg_score)
                
                # Stage-specific stuck detection criteria (more lenient at early stages)
                if self.curriculum_stage == 0:
                    # Stage 0: Stuck if near threshold (15-25) with no improvement OR very low scores
                    is_stuck = (
                        (avg_score < 10 and score_variance < 20 and improvement < 1) or  # Very low scores
                        (near_threshold and improvement < 2 and score_variance < 100)  # Plateau near threshold
                    )
                elif self.curriculum_stage == 1:
                    # Stage 1: Moderate stuck detection
                    is_stuck = (
                        (avg_score < 35 and score_variance < 40 and improvement < 2) or
                        (near_threshold and improvement < 3)
                    )
                else:
                    # Stage 2+: Original stuck detection logic
                    is_stuck = (
                        (score_variance < 50 and improvement < 2) or  # No improvement
                        (avg_score < stage_stuck_threshold + 50 and improvement < 5)  # Stuck near threshold
                    )
                
                if is_stuck:
                    self.stuck_counter += 1
                    if self.stuck_counter >= 3:  # Only after 3 consecutive stuck checks (150 episodes!)
                        # ANTI-OSCILLATION: Don't boost if we recently boosted (within 200 episodes)
                        episodes_since_boost = current_episode - self.last_epsilon_boost_episode
                        
                        if episodes_since_boost < 200:
                            print(f"\n[COOLDOWN] Stuck detected but skipping boost (last boost {episodes_since_boost} episodes ago)")
                            print(f"  • Need 200 episodes between boosts to prevent oscillation")
                            print(f"  • Current avg: {avg_score:.1f}, Variance: {score_variance:.1f}")
                            self.stuck_counter = 0  # Reset to avoid repeated messages
                        else:
                            print(f"\n[WARNING] Agent appears stuck at Stage {self.curriculum_stage}!")
                            print(f"  • Current avg: {avg_score:.1f}, Variance: {score_variance:.1f}")
                            print(f"  • Target threshold: {stage_stuck_threshold}")
                            print(f"  • Boosting epsilon from {self.epsilon:.4f} to {min(self.epsilon + 0.10, 0.4):.4f}")
                            self.epsilon = min(self.epsilon + 0.10, 0.4)  # Larger boost for exploration
                            self.last_epsilon_boost_episode = current_episode  # Track when we boosted
                            self.stuck_counter = 0
                else:
                    self.stuck_counter = 0
                
                self.last_avg_score = avg_score
