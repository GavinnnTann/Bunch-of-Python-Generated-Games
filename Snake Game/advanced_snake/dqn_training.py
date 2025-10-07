"""
DQN Training module for the Snake Game.
Implements a training interface with real-time visualization of learning progress
for the Advanced DQN algorithm.
"""

import pygame
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for embedding in pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
import torch
import psutil
import gc

from constants import *
from game_engine import GameEngine
# Make sure we import these only after GPU checks are done
from gpu_utils import check_gpu_availability, get_batch_size
import torch
# Set up the device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
# Now import the agent that will use this device
from advanced_dqn import AdvancedDQNAgent

class DQNTrainer:
    def __init__(self):
        """Initialize the training environment."""
        pygame.init()
        
        # Set up resizable display with initial dimensions
        self.width = TRAINING_SCREEN_WIDTH
        self.height = TRAINING_SCREEN_HEIGHT
        self.screen = pygame.display.set_mode(
            (self.width, self.height), 
            pygame.RESIZABLE
        )
        pygame.display.set_caption("Snake Advanced DQN Training (Resizable)")
        
        # Check GPU/CPU information
        self.device_info = self.get_device_info()
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Create models directory if it doesn't exist
        os.makedirs(QMODEL_DIR, exist_ok=True)
        
        # Training variables
        self.episodes = DEFAULT_DQN_TRAINING_EPISODES
        self.current_episode = 0
        self.best_score = 0
        self.episode_scores = []
        self.episode_steps = []
        self.running_avg_scores = []
        
        # State for training screen
        self.is_training = False
        self.selected_episodes = 2  # Index in DQN_TRAINING_EPISODES_OPTIONS
        self.training_speed = 2     # 0=Slow, 1=Medium, 2=Fast, 3=Super Fast, 4=Max Speed
        self.speed_options = ["Slow", "Medium", "Fast", "Super Fast", "Max Speed"]
        self.speed_values = [5, 15, 30, 60, 0]  # 0 means no delay/rendering
        self.selected_option = 0   # 0=Episodes, 1=Speed, 2=Start, 3=Back
        self.menu_options = ["Episodes", "Training Speed", "Start Training", "Back to Main Menu"]
        
        # Graph surfaces
        self.score_graph_surface = None
        self.q_values_graph_surface = None
        self.loss_graph_surface = None
        
        # Model compatibility cache to avoid repeated checks
        self.model_compatibility = None
    
    def start_training(self):
        """Start the training process."""
        # Set the number of episodes from selected option
        self.episodes = DQN_TRAINING_EPISODES_OPTIONS[self.selected_episodes]
        self.current_episode = 0
        self.is_training = True
        self.best_score = 0
        self.episode_scores = []
        self.episode_steps = []
        self.running_avg_scores = []
        
        # Initialize game and agent
        self.game_engine = GameEngine()
        
        # Check if an old model with 4 actions exists - we need to handle the migration
        # from absolute directions (4 actions) to relative actions (3 actions)
        model_path = os.path.join(QMODEL_DIR, DQN_MODEL_FILE)
        old_model_exists = os.path.exists(model_path)
        
        if old_model_exists:
            # Check if it's an old model (4 actions) or new model (3 actions)
            try:
                # Create agent with default settings
                self.agent = AdvancedDQNAgent(self.game_engine)
                self.agent.load_model(model_path, silent=True)
                print(f"Loaded existing model from {model_path}")
            except RuntimeError as e:
                # If we get a size mismatch error, it's likely an old model with 4 actions
                if "size mismatch" in str(e):
                    print("Detected old model format (4 actions). Creating fresh model with 3 actions.")
                    # Rename old model file to preserve it
                    backup_path = os.path.join(QMODEL_DIR, "snake_dqn_model_4actions_backup.pth")
                    import shutil
                    shutil.copy(model_path, backup_path)
                    print(f"Old model backed up to {backup_path}")
                    
                    # Create fresh agent with 3 actions
                    self.agent = AdvancedDQNAgent(self.game_engine)
                else:
                    # For other errors, re-raise
                    raise e
        else:
            # No existing model, create fresh agent
            self.agent = AdvancedDQNAgent(self.game_engine)
        
        # Training loop
        self._run_training_loop()
    
    def _run_training_loop(self):
        """Main training loop for DQN."""
        training_start_time = time.time()
        training_speed = self.speed_values[self.training_speed]
        
        # Ensure window is expanded for training display
        if self.width < TRAINING_SCREEN_WIDTH or self.height < TRAINING_SCREEN_HEIGHT:
            self.width = TRAINING_SCREEN_WIDTH
            self.height = TRAINING_SCREEN_HEIGHT
            self.screen = pygame.display.set_mode(
                (self.width, self.height), 
                pygame.RESIZABLE
            )
        
        while self.is_training and self.current_episode < self.episodes:
            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_training = False
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_training = False
                        return
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize event
                    self.width = max(event.w, MIN_TRAINING_SCREEN_WIDTH)
                    self.height = max(event.h, MIN_TRAINING_SCREEN_HEIGHT)
                    self.screen = pygame.display.set_mode(
                        (self.width, self.height), 
                        pygame.RESIZABLE
                    )
            
            # New episode
            self.current_episode += 1
            self.game_engine.reset_game()
            episode_reward = 0
            steps = 0
            max_steps = GRID_WIDTH * GRID_HEIGHT * 2  # Prevent infinite loops
            
            # Episode loop
            while not self.game_engine.game_over and steps < max_steps:
                # Take one step in the environment
                reward, done, ate_food = self.agent.train_step()
                episode_reward += reward
                steps += 1
            
            # Episode complete - update statistics
            self.episode_scores.append(self.game_engine.score)
            self.episode_steps.append(steps)
            
            # Update best score
            if self.game_engine.score > self.best_score:
                self.best_score = self.game_engine.score
                
            # Calculate running average (over last 100 episodes)
            if len(self.episode_scores) > 100:
                avg = np.mean(self.episode_scores[-100:])
            else:
                avg = np.mean(self.episode_scores)
            self.running_avg_scores.append(avg)
            
            # Memory management - periodically check and clean
            if self.current_episode % 25 == 0 and torch.cuda.is_available():
                # Check GPU memory usage
                memory_percent = torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory * 100
                if memory_percent > 80:  # If using more than 80% of GPU memory
                    print(f"High GPU memory usage detected ({memory_percent:.1f}%). Cleaning up...")
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Save model periodically
            if self.current_episode % DQN_MODEL_SAVE_INTERVAL == 0:
                self.agent.save_model(os.path.join(QMODEL_DIR, DQN_MODEL_FILE))
            
            # Display progress in console
            if self.current_episode % 10 == 0:
                elapsed = time.time() - training_start_time
                print(f"DQN Episode: {self.current_episode}/{self.episodes}, Score: {self.game_engine.score}, " +
                      f"Steps: {steps}, Best: {self.best_score}, Avg: {avg:.2f}, Epsilon: {self.agent.epsilon:.4f}, " +
                      f"Time: {elapsed:.2f}s")
                
            # Update graphs
            self._update_graphs()
                
            # Show the training screen with the updated graphs
            if training_speed > 0:
                self._render_training_screen(steps, episode_reward)
                pygame.time.delay(1000 // training_speed)  # Control frame rate
        
        # Training complete - save final model
        self.agent.save_model(os.path.join(QMODEL_DIR, DQN_MODEL_FILE))
        print(f"DQN Training complete! Model saved to {os.path.join(QMODEL_DIR, DQN_MODEL_FILE)}")
        
        # Force garbage collection and memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Show final results
        self._update_graphs()
        self._render_final_results()
        
        # Wait for user to exit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
    
    def update_device_info(self):
        """Update GPU/CPU information during training."""
        # Update system memory info
        memory = psutil.virtual_memory()
        system_memory_used = memory.used / (1024**3)  # GB
        system_memory_total = memory.total / (1024**3)  # GB
        system_memory_percent = memory.percent
        
        # Find and update the system memory info
        for i, info in enumerate(self.device_info):
            if "System Memory:" in info:
                self.device_info[i] = f"System Memory: {system_memory_used:.2f}/{system_memory_total:.2f} GB ({system_memory_percent}%)"
        
        if torch.cuda.is_available():
            # Update GPU memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
            memory_percent = memory_reserved / torch.cuda.get_device_properties(0).total_memory * 100
            
            # Find and update the memory info lines
            for i, info in enumerate(self.device_info):
                if "Memory Allocated:" in info:
                    self.device_info[i] = f"Memory Allocated: {memory_allocated:.2f} MB"
                elif "Memory Reserved:" in info:
                    self.device_info[i] = f"Memory Reserved: {memory_reserved:.2f}/{memory_total:.0f} MB ({memory_percent:.1f}%)"
                    
        # Try to clear some memory if it's getting too high
        if system_memory_percent > 90 or (torch.cuda.is_available() and memory_percent > 80):
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _update_graphs(self):
        """Update the graph surfaces with current training data."""
        # Update device info
        self.update_device_info()
        # Create score graph
        if len(self.episode_scores) > 0:
            plt.figure(figsize=(7, 4), dpi=100)
            episodes = list(range(1, len(self.episode_scores) + 1))
            plt.plot(episodes, self.episode_scores, 'b-', linewidth=2, alpha=0.6, label='Score')
            
            if self.running_avg_scores:
                plt.plot(episodes, self.running_avg_scores, 'r-', linewidth=2, label='Avg (100)')
                
            plt.title('Score History', fontsize=14, fontweight='bold')
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Ensure tick labels are visible
            plt.tick_params(labelsize=10)
            
            # Add text to show current episode
            plt.figtext(0.02, 0.02, f"Episode: {self.current_episode}", fontsize=10)
            
            # Convert matplotlib figure to pygame surface
            canvas = FigureCanvasAgg(plt.gcf())
            canvas.draw()
            
            # Get the RGBA buffer and convert to a pygame surface
            buf = canvas.buffer_rgba()
            width, height = canvas.get_width_height()
            
            # Create a pygame surface directly from the RGBA buffer
            self.score_graph_surface = pygame.image.frombuffer(buf, (width, height), "RGBA")
            plt.close()
        
        # Create Q-values graph
        if hasattr(self, 'agent') and len(self.agent.stats['q_values']) > 0:
            plt.figure(figsize=(7, 4), dpi=100)
            steps = list(range(1, len(self.agent.stats['q_values']) + 1))
            plt.plot(steps, self.agent.stats['q_values'], 'g-', linewidth=2, alpha=0.7)
            plt.title('Q-Value Changes', fontsize=14, fontweight='bold')
            plt.xlabel('Update', fontsize=12)
            plt.ylabel('Q-Value', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Ensure tick labels are visible
            plt.tick_params(labelsize=10)
            
            # Convert matplotlib figure to pygame surface
            canvas = FigureCanvasAgg(plt.gcf())
            canvas.draw()
            
            # Get the RGBA buffer and convert to a pygame surface
            buf = canvas.buffer_rgba()
            width, height = canvas.get_width_height()
            
            # Create a pygame surface directly from the RGBA buffer
            self.q_values_graph_surface = pygame.image.frombuffer(buf, (width, height), "RGBA")
            plt.close()
            
        # Create Loss graph
        if hasattr(self, 'agent') and len(self.agent.stats['losses']) > 0:
            plt.figure(figsize=(7, 4), dpi=100)
            steps = list(range(1, len(self.agent.stats['losses']) + 1))
            plt.plot(steps, self.agent.stats['losses'], 'm-', linewidth=2, alpha=0.7)
            plt.title('Loss Changes', fontsize=14, fontweight='bold')
            plt.xlabel('Update', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Use log scale for loss to better visualize changes
            if any(l > 0 for l in self.agent.stats['losses']):
                plt.yscale('log')
                
            # Ensure tick labels are visible
            plt.tick_params(labelsize=10)
            
            # Convert matplotlib figure to pygame surface
            canvas = FigureCanvasAgg(plt.gcf())
            canvas.draw()
            
            # Get the RGBA buffer and convert to a pygame surface
            buf = canvas.buffer_rgba()
            width, height = canvas.get_width_height()
            
            # Create a pygame surface directly from the RGBA buffer
            self.loss_graph_surface = pygame.image.frombuffer(buf, (width, height), "RGBA")
            plt.close()
    
    def _render_training_screen(self, steps, episode_reward):
        """Render the training screen with game state, statistics, and graphs."""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Calculate game area size based on window size
        game_size = min(self.width // 3, self.height // 3)
        
        # Render game state in a smaller area
        game_surface = pygame.Surface((game_size, game_size))
        game_surface.fill(BLACK)
        
        # Draw boundaries on the game surface
        pygame.draw.rect(game_surface, BLUE, (0, 0, game_size, game_size), 2)
        
        # Calculate scale factor based on game size
        scale_factor = game_size / (GRID_SIZE * GRID_WIDTH)
        
        # Draw food
        food_row, food_col = self.game_engine.food
        food_rect = pygame.Rect(
            food_col * GRID_SIZE * scale_factor, 
            food_row * GRID_SIZE * scale_factor, 
            GRID_SIZE * scale_factor, 
            GRID_SIZE * scale_factor
        )
        pygame.draw.rect(game_surface, RED, food_rect)
        
        # Draw snake
        for i, (row, col) in enumerate(self.game_engine.snake):
            snake_segment = pygame.Rect(
                col * GRID_SIZE * scale_factor, 
                row * GRID_SIZE * scale_factor, 
                GRID_SIZE * scale_factor, 
                GRID_SIZE * scale_factor
            )
            if i == 0:  # Head
                pygame.draw.rect(game_surface, DARK_GREEN, snake_segment)
            else:  # Body
                pygame.draw.rect(game_surface, GREEN, snake_segment)
        
        # Place game surface on screen with padding
        padding = 20
        self.screen.blit(game_surface, (padding, padding))
        
        # Display training information
        info_texts = [
            f"Episode: {self.current_episode}/{self.episodes}",
            f"Score: {self.game_engine.score}",
            f"Best Score: {self.best_score}",
            f"Steps: {steps}",
            f"Epsilon: {self.agent.epsilon:.4f}",
            f"Memory Size: {len(self.agent.memory)}",
            f"Current Reward: {episode_reward:.2f}",
            f"Algorithm: DQN+Double+Dueling+PER",
            "",
            f"Device: {device} ({'GPU' if torch.cuda.is_available() else 'CPU'})",
            "",
            "Press ESC to stop training",
            "",
            f"Window Size: {self.width}x{self.height}"
        ]
        
        info_x = game_size + padding * 2
        y_offset = padding
        for text in info_texts:
            text_surface = self.font.render(text, True, WHITE)
            self.screen.blit(text_surface, (info_x, y_offset))
            y_offset += 25
        
        # Calculate graph positions based on window size
        graph_width = min(self.width // 2 - padding * 2, 400)
        graph_height = min(self.height // 3 - padding * 2, 240)
        
        # Display the score graph
        if self.score_graph_surface is not None:
            score_graph_rect = pygame.Rect(padding, self.height // 3 + padding, 
                                         graph_width, graph_height)
            pygame.draw.rect(self.screen, WHITE, score_graph_rect, 2)
            
            # Scale the graph surface to fit the rectangle
            scaled_surface = pygame.transform.scale(self.score_graph_surface, 
                                                  (graph_width - 10, graph_height - 10))
            self.screen.blit(scaled_surface, (padding + 5, self.height // 3 + padding + 5))
            
            # Title for score graph
            title_surf = self.font.render("SCORE HISTORY", True, YELLOW)
            self.screen.blit(title_surf, (padding, self.height // 3))
        
        # Display the Q-values graph
        if self.q_values_graph_surface is not None:
            q_graph_rect = pygame.Rect(self.width // 2 + padding, self.height // 3 + padding, 
                                      graph_width, graph_height)
            pygame.draw.rect(self.screen, WHITE, q_graph_rect, 2)
            
            # Scale the graph surface to fit the rectangle
            scaled_surface = pygame.transform.scale(self.q_values_graph_surface, 
                                                  (graph_width - 10, graph_height - 10))
            self.screen.blit(scaled_surface, (self.width // 2 + padding + 5, 
                                            self.height // 3 + padding + 5))
            
            # Title for Q-values graph
            title_surf = self.font.render("Q-VALUE CHANGES", True, YELLOW)
            self.screen.blit(title_surf, (self.width // 2 + padding, self.height // 3))
        
        # Display the Loss graph
        if self.loss_graph_surface is not None:
            loss_graph_rect = pygame.Rect(padding, 2 * self.height // 3 + padding, 
                                        graph_width, graph_height)
            pygame.draw.rect(self.screen, WHITE, loss_graph_rect, 2)
            
            # Scale the graph surface to fit the rectangle
            scaled_surface = pygame.transform.scale(self.loss_graph_surface, 
                                                  (graph_width - 10, graph_height - 10))
            self.screen.blit(scaled_surface, (padding + 5, 2 * self.height // 3 + padding + 5))
            
            # Title for Loss graph
            title_surf = self.font.render("LOSS FUNCTION", True, YELLOW)
            self.screen.blit(title_surf, (padding, 2 * self.height // 3))
        
        # Update the display
        pygame.display.flip()
    
    def _render_final_results(self):
        """Render the final training results screen."""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Display title
        title_font = pygame.font.Font(None, 48)
        title_text = title_font.render("DQN Training Complete!", True, GREEN)
        self.screen.blit(title_text, (self.width // 2 - title_text.get_width() // 2, 20))
        
        # Display statistics
        info_texts = [
            f"Total Episodes: {self.current_episode}",
            f"Best Score: {self.best_score}",
            f"Final Epsilon: {self.agent.epsilon:.4f}",
            f"Memory Size: {len(self.agent.memory)}",
            f"Model Saved: {os.path.join(QMODEL_DIR, DQN_MODEL_FILE)}",
            f"Algorithm: DQN+Double+Dueling+PER",
            "",
            "Press any key to continue",
            "",
            f"Window Size: {self.width}x{self.height}"
        ]
        
        y_offset = 100
        for text in info_texts:
            text_surface = self.font.render(text, True, WHITE)
            self.screen.blit(text_surface, (self.width // 2 - text_surface.get_width() // 2, y_offset))
            y_offset += 30
        
        # Calculate graph positions based on window size
        graph_width = min(self.width // 2 - 40, 400)
        graph_height = min(self.height // 3 - 40, 240)
        
        # Display the score graph
        if self.score_graph_surface is not None:
            score_graph_rect = pygame.Rect(20, self.height // 3 + 20, graph_width, graph_height)
            pygame.draw.rect(self.screen, WHITE, score_graph_rect, 2)
            
            # Scale the graph surface to fit the rectangle
            scaled_surface = pygame.transform.scale(self.score_graph_surface, 
                                                  (graph_width - 10, graph_height - 10))
            self.screen.blit(scaled_surface, (25, self.height // 3 + 25))
            
            # Title for score graph
            title_surf = self.font.render("SCORE HISTORY", True, YELLOW)
            self.screen.blit(title_surf, (20, self.height // 3))
        
        # Display the Q-values graph
        if self.q_values_graph_surface is not None:
            q_graph_rect = pygame.Rect(self.width // 2 + 20, self.height // 3 + 20, 
                                      graph_width, graph_height)
            pygame.draw.rect(self.screen, WHITE, q_graph_rect, 2)
            
            # Scale the graph surface to fit the rectangle
            scaled_surface = pygame.transform.scale(self.q_values_graph_surface, 
                                                  (graph_width - 10, graph_height - 10))
            self.screen.blit(scaled_surface, (self.width // 2 + 25, self.height // 3 + 25))
            
            # Title for Q-values graph
            title_surf = self.font.render("Q-VALUE CHANGES", True, YELLOW)
            self.screen.blit(title_surf, (self.width // 2 + 20, self.height // 3))
        
        # Display the Loss graph
        if self.loss_graph_surface is not None:
            loss_graph_rect = pygame.Rect(20, 2 * self.height // 3 + 20, graph_width, graph_height)
            pygame.draw.rect(self.screen, WHITE, loss_graph_rect, 2)
            
            # Scale the graph surface to fit the rectangle
            scaled_surface = pygame.transform.scale(self.loss_graph_surface, 
                                                  (graph_width - 10, graph_height - 10))
            self.screen.blit(scaled_surface, (25, 2 * self.height // 3 + 25))
            
            # Title for Loss graph
            title_surf = self.font.render("LOSS FUNCTION", True, YELLOW)
            self.screen.blit(title_surf, (20, 2 * self.height // 3))
        
        # Update the display
        pygame.display.flip()
    
    def render_training_menu(self):
        """Render the training configuration menu."""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw title
        title_font = pygame.font.Font(None, 48)
        title_text = title_font.render("Advanced DQN Training", True, GREEN)
        self.screen.blit(title_text, (self.width // 2 - title_text.get_width() // 2, 50))
        
        # Draw algorithm description
        description_font = pygame.font.Font(None, 28)
        description_text = description_font.render("Deep Q-Network + Double + Dueling + Prioritized Experience Replay", True, PURPLE)
        self.screen.blit(description_text, (self.width // 2 - description_text.get_width() // 2, 100))
        
        # Draw menu options
        option_font = pygame.font.Font(None, 36)
        y_offset = 180
        
        for i, option in enumerate(self.menu_options):
            color = YELLOW if i == self.selected_option else WHITE
            option_text = option_font.render(option, True, color)
            
            # Add current value for configurable options
            if i == 0:  # Episodes
                value_text = f": {DQN_TRAINING_EPISODES_OPTIONS[self.selected_episodes]}"
                option_text = option_font.render(option + value_text, True, color)
            elif i == 1:  # Speed
                value_text = f": {self.speed_options[self.training_speed]}"
                option_text = option_font.render(option + value_text, True, color)
            
            self.screen.blit(option_text, (self.width // 2 - option_text.get_width() // 2, y_offset))
            y_offset += 50
        
        # Draw instructions
        instruction_font = pygame.font.Font(None, 24)
        instructions = [
            "Use UP/DOWN to navigate",
            "Use LEFT/RIGHT to change values",
            "Press ENTER to select",
            "Press ESC to go back",
            "Resize window as needed for better graph visibility"
        ]
        
        y_offset = 400
        for instruction in instructions:
            instr_text = instruction_font.render(instruction, True, GRAY)
            self.screen.blit(instr_text, (self.width // 2 - instr_text.get_width() // 2, y_offset))
            y_offset += 30
        
        # Draw model info if available
        model_path = os.path.join(QMODEL_DIR, DQN_MODEL_FILE)
        backup_path = os.path.join(QMODEL_DIR, "snake_dqn_model_4actions_backup.pth")
        
        if os.path.exists(model_path):
            # Always use the verify_model_compatibility method which has caching built-in now
            is_compatible, msg = self.verify_model_compatibility()
            
            if is_compatible:
                model_info = f"Existing model found: {DQN_MODEL_FILE} (Compatible)"
                model_text = instruction_font.render(model_info, True, LIGHT_GRAY)
            else:
                model_info = f"Existing model found but incompatible: Will create new model"
                model_text = instruction_font.render(model_info, True, (255, 150, 150))
                
                if os.path.exists(backup_path):
                    backup_info = f"Old model backed up to: {os.path.basename(backup_path)}"
                    backup_text = instruction_font.render(backup_info, True, LIGHT_GRAY)
                    self.screen.blit(backup_text, (self.width // 2 - backup_text.get_width() // 2, 590))
                
            self.screen.blit(model_text, (self.width // 2 - model_text.get_width() // 2, 560))
        
        # Display device info (GPU/CPU)
        device_info_font = pygame.font.Font(None, 22)
        y_offset = self.height - 150
        
        # Title for hardware info section
        hw_title = device_info_font.render("HARDWARE INFORMATION:", True, YELLOW)
        self.screen.blit(hw_title, (self.width // 2 - hw_title.get_width() // 2, y_offset))
        y_offset += 25
        
        # Display each line of device info
        for info in self.device_info:
            info_text = device_info_font.render(info, True, LIGHT_GRAY)
            self.screen.blit(info_text, (self.width // 2 - info_text.get_width() // 2, y_offset))
            y_offset += 20
        
        # Display window size info
        size_text = instruction_font.render(f"Window Size: {self.width}x{self.height}", True, DARK_GRAY)
        self.screen.blit(size_text, (self.width // 2 - size_text.get_width() // 2, self.height - 40))
        
        # Update display
        pygame.display.flip()
    
    def handle_menu_input(self, key):
        """Handle input in the training menu."""
        if key == pygame.K_UP:
            self.selected_option = (self.selected_option - 1) % len(self.menu_options)
        elif key == pygame.K_DOWN:
            self.selected_option = (self.selected_option + 1) % len(self.menu_options)
        elif key == pygame.K_RETURN:
            if self.selected_option == 2:  # Start Training
                self.start_training()
            elif self.selected_option == 3:  # Back to Main Menu
                return False  # Signal to go back to main menu
        elif key == pygame.K_LEFT:
            if self.selected_option == 0:  # Episodes
                self.selected_episodes = (self.selected_episodes - 1) % len(DQN_TRAINING_EPISODES_OPTIONS)
            elif self.selected_option == 1:  # Speed
                self.training_speed = (self.training_speed - 1) % len(self.speed_options)
        elif key == pygame.K_RIGHT:
            if self.selected_option == 0:  # Episodes
                self.selected_episodes = (self.selected_episodes + 1) % len(DQN_TRAINING_EPISODES_OPTIONS)
            elif self.selected_option == 1:  # Speed
                self.training_speed = (self.training_speed + 1) % len(self.speed_options)
        elif key == pygame.K_ESCAPE:
            return False  # Signal to go back to main menu
            
        return True  # Continue in training menu
    
    def get_device_info(self):
        """Get detailed information about the device being used for training."""
        info = []
        
        # Get system info
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        system_memory_used = memory.used / (1024**3)  # GB
        system_memory_total = memory.total / (1024**3)  # GB
        system_memory_percent = memory.percent
        
        # System information
        info.append(f"System Memory: {system_memory_used:.2f}/{system_memory_total:.2f} GB ({system_memory_percent}%)")
        
        # CPU information
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        info.append(f"CPU: {cpu_count} cores (Usage: {cpu_percent}%)")
        
        # Check if CUDA is available
        info.append(f"PyTorch Device: {device}")
        
        # If GPU is available, get more information
        if torch.cuda.is_available():
            info.append(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Get detailed GPU info
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory = gpu_props.total_memory / 1024**3  # GB
            info.append(f"GPU Memory: {total_memory:.2f} GB")
            info.append(f"Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            info.append(f"CUDA Version: {torch.version.cuda}")
            
            # Get batch size information
            is_gpu, batch_size = check_gpu_availability()
            info.append(f"Batch Size: {batch_size} (GPU optimized)")
            
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
            memory_percent = memory_reserved / torch.cuda.get_device_properties(0).total_memory * 100
            info.append(f"Memory Allocated: {memory_allocated:.2f} MB")
            info.append(f"Memory Reserved: {memory_reserved:.2f}/{gpu_props.total_memory / (1024**2):.0f} MB ({memory_percent:.1f}%)")
            
            # Memory utilization
            if hasattr(torch.cuda, 'memory_stats'):
                try:
                    memory_stats = torch.cuda.memory_stats(0)
                    if 'active_bytes.all.peak' in memory_stats:
                        peak_memory = memory_stats['active_bytes.all.peak'] / 1024**2
                        info.append(f"Peak Memory: {peak_memory:.2f} MB")
                except:
                    pass
                
        else:
            # CPU-only information
            info.append(f"PyTorch Version: {torch.__version__}")
            info.append("CUDA: Not available (Using CPU only)")
            info.append(f"Batch Size: {DQN_BATCH_SIZE} (Default)")
            
            # Check if CUDA is installed but not being used
            if hasattr(torch, 'version') and hasattr(torch.version, 'cuda') and torch.version.cuda:
                info.append(f"Note: CUDA {torch.version.cuda} is installed but not being used")
                info.append("      Check USE_CUDA setting in constants.py")
            
        return info

    # Class-level cache to prevent repeated compatibility checks across instances
    _model_compatibility_cache = {}
    
    def verify_model_compatibility(self):
        """Check if the existing model is compatible with the current agent architecture."""
        model_path = os.path.join(QMODEL_DIR, DQN_MODEL_FILE)
        
        # Check class-level cache first
        if model_path in DQNTrainer._model_compatibility_cache:
            return DQNTrainer._model_compatibility_cache[model_path]
            
        if not os.path.exists(model_path):
            result = (False, "No model found")
            DQNTrainer._model_compatibility_cache[model_path] = result
            return result
            
        try:
            # Create temporary game engine and agent
            temp_game = GameEngine()
            temp_agent = AdvancedDQNAgent(temp_game)
            
            # Always use the silent flag
            result = temp_agent.load_model(model_path, silent=True)
            result = (True, "Model is compatible")
                
        except RuntimeError as e:
            result = (False, str(e))
        except Exception as e:
            result = (False, f"Unexpected error: {str(e)}")
            
        # Cache the result at class level
        DQNTrainer._model_compatibility_cache[model_path] = result
        return result

    def run_training_menu(self):
        """Run the training configuration menu loop."""
        running = True
        
        # Check model compatibility when menu opens - uses cached result if available
        is_compatible, msg = self.verify_model_compatibility()
        if not is_compatible and "size mismatch" in msg:
            print(f"Warning: Existing model is not compatible with current architecture: {msg}")
            print("You will need to start training from scratch.")
        
        # Store compatibility result to avoid checking repeatedly
        self.model_compatibility = {
            "is_compatible": is_compatible,
            "message": msg
        }
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # Signal to quit game
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_menu_input(event.key)
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize event
                    self.width = max(event.w, MIN_TRAINING_SCREEN_WIDTH)
                    self.height = max(event.h, MIN_TRAINING_SCREEN_HEIGHT)
                    self.screen = pygame.display.set_mode(
                        (self.width, self.height), 
                        pygame.RESIZABLE
                    )
            
            # Render menu
            self.render_training_menu()
            
            # Control frame rate
            self.clock.tick(30)
        
        return True  # Signal to continue game


if __name__ == "__main__":
    # Run the trainer standalone for testing
    trainer = DQNTrainer()
    trainer.run_training_menu()