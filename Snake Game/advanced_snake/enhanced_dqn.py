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
        
    def get_state(self):
        """Override to use enhanced state representation."""
        return EnhancedStateRepresentation.get_enhanced_state(self.game_engine)
    
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
            # UPDATED: Doubled base food reward to encourage aggressive food-seeking
            # Scale food reward based on snake length (harder to get food when longer)
            length_bonus = 1 + (len(self.game_engine.snake) / 100)
            reward += (REWARD_FOOD * 2) * length_bonus  # Was REWARD_FOOD (10), now 20 base
        else:
            # Survival reward decreases as snake gets longer (should be hunting food)
            survival_penalty = len(self.game_engine.snake) / 1000
            reward += REWARD_SURVIVAL - survival_penalty
        
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
                    # Scale bonus by curriculum stage (higher bonus early, lower later)
                    astar_bonus = self.astar_guidance_prob * 0.5  # 0.25 at stage 0, 0 at stage 4
                    reward += astar_bonus
        except:
            pass
        
        # Movement rewards - UPDATED: More aggressive distance-based shaping
        if new_distance < old_distance:
            # Moving toward food - good!
            reward += REWARD_MOVE_TOWARDS_FOOD * 3  # Tripled from 2 to encourage seeking
        elif new_distance > old_distance:
            # Moving away from food - bad!
            reward += REWARD_MOVE_AWAY_FROM_FOOD * 2  # Doubled penalty
        
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
    
    def update_curriculum(self, score):
        """
        Update curriculum stage based on SUSTAINED performance.
        Requires consistent achievement of threshold before advancing.
        """
        self.recent_scores.append(score)
        
        # Need at least 10 scores to evaluate performance
        if len(self.recent_scores) >= 10:
            avg_score = np.mean(self.recent_scores)
            min_score = np.min(self.recent_scores)
            
            # Check if we should advance curriculum
            if self.curriculum_stage < len(self.curriculum_thresholds):
                current_threshold = self.curriculum_thresholds[self.curriculum_stage]
                
                # IMPROVED: Require both average AND minimum to meet reasonable criteria
                # Average must exceed threshold AND minimum score should be at least 50% of threshold
                # This ensures SUSTAINED performance, not just lucky peaks
                min_acceptable = current_threshold * 0.5
                
                if avg_score >= current_threshold and min_score >= min_acceptable:
                    self.curriculum_success_count += 1
                    
                    # Require multiple consecutive successful evaluations
                    if self.curriculum_success_count >= self.curriculum_consistency_required:
                        old_stage = self.curriculum_stage
                        old_astar_prob = self.astar_guidance_prob
                        old_epsilon = self.epsilon
                        
                        self.curriculum_stage += 1
                        self.curriculum_success_count = 0  # Reset for next stage
                        
                        # UPDATED: Lower epsilon requirements for better exploitation
                        # Only adjust epsilon if it's gotten too low (don't force it up unnecessarily)
                        if self.curriculum_stage == 1:
                            self.astar_guidance_prob = 0.35  # Reduce A* from 0.5 to 0.35
                            # Allow lower epsilon for more exploitation
                            if self.epsilon < 0.2:  # Changed from 0.3
                                self.epsilon = 0.2
                        elif self.curriculum_stage == 2:
                            self.astar_guidance_prob = 0.20  # Further reduce A* to 0.20
                            if self.epsilon < 0.15:  # Changed from 0.2
                                self.epsilon = 0.15
                        elif self.curriculum_stage == 3:
                            self.astar_guidance_prob = 0.10  # Minimal A* guidance
                            if self.epsilon < 0.1:  # Changed from 0.15
                                self.epsilon = 0.1
                        elif self.curriculum_stage >= 4:
                            self.astar_guidance_prob = 0.0  # No A* guidance at final stage
                            if self.epsilon < 0.05:  # Changed from 0.1 - allow very low epsilon
                                self.epsilon = 0.05
                        
                        # VISIBLE logging of curriculum advancement
                        print(f"\n{'='*70}")
                        print(f"[CURRICULUM] ADVANCED: Stage {old_stage} -> Stage {self.curriculum_stage}")
                        print(f"Average Score: {avg_score:.1f} >= Threshold: {current_threshold}")
                        print(f"Minimum Score: {min_score:.1f} >= Required: {min_acceptable:.1f}")
                        print(f"Consistency: {self.curriculum_consistency_required} consecutive successful evaluations")
                        print(f"STRATEGY CHANGES:")
                        print(f"  • A* Reward Weight: {old_astar_prob:.2f} -> {self.astar_guidance_prob:.2f} ({old_astar_prob - self.astar_guidance_prob:+.2f})")
                        print(f"  • Epsilon:          {old_epsilon:.4f} -> {self.epsilon:.4f}")
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
                              f"(Avg: {avg_score:.1f}/{current_threshold}, Min: {min_score:.1f}/{min_acceptable:.1f})")
                    self.curriculum_success_count = 0
                
            # Detect if TRULY stuck (much more conservative than before)
            # Only trigger if we have enough data and scores are NOT improving
            if len(self.recent_scores) >= 50:
                avg_score = np.mean(self.recent_scores)
                score_variance = np.var(self.recent_scores)
                
                # Check if stuck: low variance AND no improvement from last check
                is_stuck = (score_variance < 50 and  # Very low variance (scores almost identical)
                           avg_score < 100 and  # Still performing poorly
                           abs(avg_score - self.last_avg_score) < 2)  # No meaningful improvement
                
                if is_stuck:
                    self.stuck_counter += 1
                    if self.stuck_counter >= 3:  # Only after 3 consecutive stuck checks (150 episodes!)
                        print(f"\n[WARNING] Agent appears stuck! Very low variance and no improvement.")
                        print(f"  • Current avg: {avg_score:.1f}, Variance: {score_variance:.1f}")
                        print(f"  • Slightly boosting epsilon from {self.epsilon:.4f} to {min(self.epsilon + 0.05, 0.3):.4f}")
                        self.epsilon = min(self.epsilon + 0.05, 0.3)  # Small boost, cap at 0.3
                        self.stuck_counter = 0
                else:
                    self.stuck_counter = 0
                
                self.last_avg_score = avg_score
