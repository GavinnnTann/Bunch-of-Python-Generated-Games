"""
Training module for the Snake Game Q-learning agent.
Implements a training interface with real-time visualization of learning progress.
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

from constants import *
from game_engine import GameEngine
from q_learning import SnakeQLearningAgent

class SnakeTrainer:
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
        pygame.display.set_caption("Snake Q-Learning Training (Resizable)")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Create models directory if it doesn't exist
        os.makedirs(QMODEL_DIR, exist_ok=True)
        
        # Training variables
        self.episodes = DEFAULT_TRAINING_EPISODES
        self.current_episode = 0
        self.best_score = 0
        self.episode_scores = []
        self.episode_steps = []
        self.running_avg_scores = []
        
        # State for training screen
        self.is_training = False
        self.selected_episodes = 2  # Index in TRAINING_EPISODES_OPTIONS
        self.training_speed = 2     # 0=Slow, 1=Medium, 2=Fast, 3=Super Fast, 4=Max Speed
        self.speed_options = ["Slow", "Medium", "Fast", "Super Fast", "Max Speed"]
        self.speed_values = [5, 15, 30, 60, 0]  # 0 means no delay/rendering
        self.selected_option = 0   # 0=Episodes, 1=Speed, 2=Start, 3=Back
        self.menu_options = ["Episodes", "Training Speed", "Start Training", "Back to Main Menu"]
        
        # Graph surfaces
        self.score_graph_surface = None
        self.q_values_graph_surface = None
    
    def start_training(self):
        """Start the training process."""
        # Set the number of episodes from selected option
        self.episodes = TRAINING_EPISODES_OPTIONS[self.selected_episodes]
        self.current_episode = 0
        self.is_training = True
        self.best_score = 0
        self.episode_scores = []
        self.episode_steps = []
        self.running_avg_scores = []
        
        # Initialize game and agent
        self.game_engine = GameEngine()
        self.agent = SnakeQLearningAgent(self.game_engine)
        
        # Try to load existing model if available
        model_path = os.path.join(QMODEL_DIR, QMODEL_FILE)
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            print(f"Loaded existing model from {model_path}")
        
        # Training loop
        self._run_training_loop()
    
    def _run_training_loop(self):
        """Main training loop for Q-learning."""
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
            
            # Save model periodically
            if self.current_episode % MODEL_SAVE_INTERVAL == 0:
                self.agent.save_model(os.path.join(QMODEL_DIR, QMODEL_FILE))
            
            # Display progress in console
            if self.current_episode % 10 == 0:
                elapsed = time.time() - training_start_time
                print(f"Episode: {self.current_episode}/{self.episodes}, Score: {self.game_engine.score}, " +
                      f"Steps: {steps}, Best: {self.best_score}, Avg: {avg:.2f}, Epsilon: {self.agent.epsilon:.4f}, " +
                      f"Time: {elapsed:.2f}s")
                
            # Update graphs
            self._update_graphs()
                
            # Show the training screen with the updated graphs
            if training_speed > 0:
                self._render_training_screen(steps, episode_reward)
                pygame.time.delay(1000 // training_speed)  # Control frame rate
        
        # Training complete - save final model
        self.agent.save_model(os.path.join(QMODEL_DIR, QMODEL_FILE))
        print(f"Training complete! Model saved to {os.path.join(QMODEL_DIR, QMODEL_FILE)}")
        
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
    
    def _update_graphs(self):
        """Update the graph surfaces with current training data."""
        # Create score graph
        if len(self.episode_scores) > 0:  # Changed from > 1 to > 0 to show graphs earlier
            plt.figure(figsize=(7, 4), dpi=100)  # Increased size and DPI
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
        if hasattr(self, 'agent') and len(self.agent.stats['q_values']) > 0:  # Changed from > 1 to > 0
            plt.figure(figsize=(7, 4), dpi=100)  # Increased size and DPI
            steps = list(range(1, len(self.agent.stats['q_values']) + 1))
            plt.plot(steps, self.agent.stats['q_values'], 'g-', linewidth=2, alpha=0.7)
            plt.title('Q-Value Changes', fontsize=14, fontweight='bold')
            plt.xlabel('Update', fontsize=12)
            plt.ylabel('Avg. Q-Value Change', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Ensure tick labels are visible
            plt.tick_params(labelsize=10)
            
            # Only use log scale if there are values greater than 0
            if any(q > 0 for q in self.agent.stats['q_values']):
                plt.yscale('log')
            
            # Convert matplotlib figure to pygame surface
            canvas = FigureCanvasAgg(plt.gcf())
            canvas.draw()
            
            # Get the RGBA buffer and convert to a pygame surface
            buf = canvas.buffer_rgba()
            width, height = canvas.get_width_height()
            
            # Create a pygame surface directly from the RGBA buffer
            self.q_values_graph_surface = pygame.image.frombuffer(buf, (width, height), "RGBA")
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
            f"States Learned: {len(self.agent.q_table)}",
            f"Memory Size: {len(self.agent.memory)}",
            f"Current Reward: {episode_reward:.2f}",
            "",
            "Press ESC to stop training",
            "",
            "Window Size: {0}x{1}".format(self.width, self.height)
        ]
        
        info_x = game_size + padding * 2
        y_offset = padding
        for text in info_texts:
            text_surface = self.font.render(text, True, WHITE)
            self.screen.blit(text_surface, (info_x, y_offset))
            y_offset += 25
        
        # Calculate graph positions based on window size
        padding = 20
        graph_y = self.height // 2
        
        # Display graphs if available
        if self.score_graph_surface is not None:
            # Draw a border around the graph
            graph_rect = pygame.Rect(
                padding, 
                graph_y, 
                self.score_graph_surface.get_width() + 10,
                self.score_graph_surface.get_height() + 10
            )
            pygame.draw.rect(self.screen, WHITE, graph_rect, 2)
            
            # Draw the graph with a small padding inside the border
            self.screen.blit(self.score_graph_surface, (padding + 5, graph_y + 5))
            
            # Add a title above the graph
            title_surf = self.font.render("TRAINING PROGRESS", True, YELLOW)
            self.screen.blit(title_surf, (padding, graph_y - 30))
        
        if self.q_values_graph_surface is not None:
            # Calculate position for right side graph
            right_x = self.width - self.q_values_graph_surface.get_width() - padding - 5
            
            # Draw a border around the graph
            graph_rect = pygame.Rect(
                right_x - 5, 
                graph_y, 
                self.q_values_graph_surface.get_width() + 10,
                self.q_values_graph_surface.get_height() + 10
            )
            pygame.draw.rect(self.screen, WHITE, graph_rect, 2)
            
            # Draw the graph with a small padding inside the border
            self.screen.blit(self.q_values_graph_surface, (right_x, graph_y + 5))
            
            # Add a title above the graph
            title_surf = self.font.render("LEARNING RATE", True, YELLOW)
            self.screen.blit(title_surf, (right_x, graph_y - 30))
        
        # Update the display
        pygame.display.flip()
    
    def _render_final_results(self):
        """Render the final training results screen."""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Display title
        title_font = pygame.font.Font(None, 48)
        title_text = title_font.render("Training Complete!", True, GREEN)
        self.screen.blit(title_text, (self.width // 2 - title_text.get_width() // 2, 20))
        
        # Display statistics
        info_texts = [
            f"Total Episodes: {self.current_episode}",
            f"Best Score: {self.best_score}",
            f"Final Epsilon: {self.agent.epsilon:.4f}",
            f"States Learned: {len(self.agent.q_table)}",
            f"Model Saved: {os.path.join(QMODEL_DIR, QMODEL_FILE)}",
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
        padding = 20
        graph_y = self.height // 2
        
        # Display graphs
        if self.score_graph_surface is not None:
            # Draw a border around the graph
            graph_rect = pygame.Rect(
                padding, 
                graph_y, 
                self.score_graph_surface.get_width() + 10,
                self.score_graph_surface.get_height() + 10
            )
            pygame.draw.rect(self.screen, WHITE, graph_rect, 2)
            
            # Draw the graph with a small padding inside the border
            self.screen.blit(self.score_graph_surface, (padding + 5, graph_y + 5))
            
            # Add a title above the graph
            title_surf = self.font.render("SCORE HISTORY", True, YELLOW)
            self.screen.blit(title_surf, (padding, graph_y - 30))
        
        if self.q_values_graph_surface is not None:
            # Calculate position for right side graph
            right_x = self.width - self.q_values_graph_surface.get_width() - padding - 5
            
            # Draw a border around the graph
            graph_rect = pygame.Rect(
                right_x - 5, 
                graph_y, 
                self.q_values_graph_surface.get_width() + 10,
                self.q_values_graph_surface.get_height() + 10
            )
            pygame.draw.rect(self.screen, WHITE, graph_rect, 2)
            
            # Draw the graph with a small padding inside the border
            self.screen.blit(self.q_values_graph_surface, (right_x, graph_y + 5))
            
            # Add a title above the graph
            title_surf = self.font.render("Q-VALUE CHANGES", True, YELLOW)
            self.screen.blit(title_surf, (right_x, graph_y - 30))
        
        # Update the display
        pygame.display.flip()
    
    def render_training_menu(self):
        """Render the training configuration menu."""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw title
        title_font = pygame.font.Font(None, 48)
        title_text = title_font.render("Q-Learning Training", True, GREEN)
        self.screen.blit(title_text, (self.width // 2 - title_text.get_width() // 2, 50))
        
        # Draw menu options
        option_font = pygame.font.Font(None, 36)
        y_offset = 150
        
        for i, option in enumerate(self.menu_options):
            color = YELLOW if i == self.selected_option else WHITE
            option_text = option_font.render(option, True, color)
            
            # Add current value for configurable options
            if i == 0:  # Episodes
                value_text = f": {TRAINING_EPISODES_OPTIONS[self.selected_episodes]}"
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
        model_path = os.path.join(QMODEL_DIR, QMODEL_FILE)
        if os.path.exists(model_path):
            model_info = f"Existing model found: {QMODEL_FILE}"
            model_text = instruction_font.render(model_info, True, LIGHT_GRAY)
            self.screen.blit(model_text, (self.width // 2 - model_text.get_width() // 2, 500))
        
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
                self.selected_episodes = (self.selected_episodes - 1) % len(TRAINING_EPISODES_OPTIONS)
            elif self.selected_option == 1:  # Speed
                self.training_speed = (self.training_speed - 1) % len(self.speed_options)
        elif key == pygame.K_RIGHT:
            if self.selected_option == 0:  # Episodes
                self.selected_episodes = (self.selected_episodes + 1) % len(TRAINING_EPISODES_OPTIONS)
            elif self.selected_option == 1:  # Speed
                self.training_speed = (self.training_speed + 1) % len(self.speed_options)
        elif key == pygame.K_ESCAPE:
            return False  # Signal to go back to main menu
            
        return True  # Continue in training menu
    
    def run_training_menu(self):
        """Run the training configuration menu loop."""
        running = True
        
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
    trainer = SnakeTrainer()
    trainer.run_training_menu()