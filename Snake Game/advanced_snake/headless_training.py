"""
Headless DQN Training module for Snake Game.

This module provides high-speed training without pygame visualization,
with options for:
1. Terminal-only output (fastest)
2. Real-time matplotlib visualization in a separate window
3. Periodic model saving

Designed for maximum GPU utilization.
"""

import os
import time
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import psutil
import gc
import json

from constants import *
from game_engine import GameEngine
from gpu_utils import check_gpu_availability, get_batch_size
# Set up the device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
# Import the agent after device is configured
from advanced_dqn import AdvancedDQNAgent

class HeadlessDQNTrainer:
    def __init__(self, episodes=1000, show_graphs=False, save_interval=50):
        """
        Initialize the headless training environment.
        
        Args:
            episodes: Number of episodes to train for
            show_graphs: Whether to show real-time matplotlib graphs
            save_interval: How often to save the model (in episodes)
        """
        # Training parameters
        self.episodes = episodes
        self.current_episode = 0
        self.show_graphs = show_graphs
        self.save_interval = save_interval
        self.is_training = False
        self.running_avg_size = 100
        
        # Create models directory if it doesn't exist
        os.makedirs(QMODEL_DIR, exist_ok=True)
        
        # Training statistics
        self.episode_scores = []
        self.episode_lengths = []
        self.running_avg_scores = []
        self.episode_durations = []
        self.training_start_time = None
        
        # For plotting
        self.fig = None
        self.axs = None
        self.animation = None
        
        # Setup the game engine and agent
        self.game_engine = GameEngine()  # Using standard GameEngine
        self.setup_agent()

    def setup_agent(self):
        """Set up or load the DQN agent."""
        model_path = os.path.join(QMODEL_DIR, "snake_dqn_model.pth")
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self.game_engine = GameEngine()  # Create a fresh game engine
                self.agent = AdvancedDQNAgent(self.game_engine)
                self.agent.load_model(model_path)
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

    def setup_plots(self):
        """Set up the matplotlib plots for real-time visualization."""
        if not self.show_graphs:
            return
            
        plt.ion()  # Turn on interactive mode
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.canvas.manager.set_window_title("Snake DQN Training Metrics")
        
        # Score plot
        self.axs[0].set_title('Training Scores', fontsize=14, fontweight='bold')
        self.axs[0].set_xlabel('Episode')
        self.axs[0].set_ylabel('Score')
        self.axs[0].grid(True, alpha=0.3)
        
        # Q-values plot
        self.axs[1].set_title('Q-Values', fontsize=14, fontweight='bold')
        self.axs[1].set_xlabel('Update')
        self.axs[1].set_ylabel('Q-Value')
        self.axs[1].grid(True, alpha=0.3)
        
        # Loss plot
        self.axs[2].set_title('Loss', fontsize=14, fontweight='bold')
        self.axs[2].set_xlabel('Update')
        self.axs[2].set_ylabel('Loss')
        self.axs[2].grid(True, alpha=0.3)
        self.axs[2].set_yscale('log')
        
        plt.tight_layout()
        plt.show(block=False)

    def update_plots(self):
        """Update the matplotlib plots with current data."""
        if not self.show_graphs:
            return
            
        # Only update every few episodes to avoid slowing down training
        if self.current_episode % 5 != 0 and self.current_episode > 1:
            return
            
        # Clear previous plots
        for ax in self.axs:
            ax.clear()
        
        # Score plot
        episodes = list(range(1, len(self.episode_scores) + 1))
        self.axs[0].plot(episodes, self.episode_scores, 'b-', linewidth=2, alpha=0.6, label='Score')
        if self.running_avg_scores:
            self.axs[0].plot(episodes, self.running_avg_scores, 'r-', linewidth=2, label='Avg (100)')
        self.axs[0].set_title('Training Scores', fontsize=14, fontweight='bold')
        self.axs[0].set_xlabel('Episode')
        self.axs[0].set_ylabel('Score')
        self.axs[0].legend(loc='upper left', fontsize=10)
        self.axs[0].grid(True, alpha=0.3)
        
        # Q-values plot
        if hasattr(self, 'agent') and len(self.agent.stats['q_values']) > 0:
            steps = list(range(1, len(self.agent.stats['q_values']) + 1))
            self.axs[1].plot(steps, self.agent.stats['q_values'], 'g-', linewidth=2, alpha=0.7)
            self.axs[1].set_title('Q-Value Changes', fontsize=14, fontweight='bold')
            self.axs[1].set_xlabel('Update')
            self.axs[1].set_ylabel('Q-Value')
            self.axs[1].grid(True, alpha=0.3)
        
        # Loss plot
        if hasattr(self, 'agent') and len(self.agent.stats['losses']) > 0:
            steps = list(range(1, len(self.agent.stats['losses']) + 1))
            self.axs[2].plot(steps, self.agent.stats['losses'], 'm-', linewidth=2, alpha=0.7)
            self.axs[2].set_title('Loss Changes', fontsize=14, fontweight='bold')
            self.axs[2].set_xlabel('Update')
            self.axs[2].set_ylabel('Loss')
            self.axs[2].grid(True, alpha=0.3)
            # Use log scale for loss visualization
            if any(l > 0 for l in self.agent.stats['losses']):
                self.axs[2].set_yscale('log')
        
        # Update the display
        plt.tight_layout()
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def calculate_running_average(self):
        """Calculate running average of scores."""
        if len(self.episode_scores) > 0:
            window = min(self.running_avg_size, len(self.episode_scores))
            avg = sum(self.episode_scores[-window:]) / window
            self.running_avg_scores.append(avg)
        else:
            self.running_avg_scores = []

    def get_device_info(self):
        """Get information about the device being used for training."""
        device_info = {}
        
        # GPU info
        device_info["device"] = device.type
        device_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            device_info["gpu_name"] = torch.cuda.get_device_name(0)
            device_info["cuda_version"] = torch.version.cuda
            try:
                device_info["vram_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                device_info["vram_used"] = (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024**3)  # GB
            except:
                device_info["vram_total"] = "Unknown"
                device_info["vram_used"] = "Unknown"
        
        # CPU and RAM info
        device_info["cpu_percent"] = psutil.cpu_percent()
        device_info["ram_percent"] = psutil.virtual_memory().percent
        
        return device_info

    def print_training_info(self, episode_reward, steps, elapsed_time):
        """Print training information to the terminal."""
        # Calculate running average score
        avg_score = 0
        if len(self.episode_scores) > 0:
            window = min(self.running_avg_size, len(self.episode_scores))
            avg_score = sum(self.episode_scores[-window:]) / window
            
        # Calculate best score
        best_score = max(self.episode_scores) if self.episode_scores else 0
        
        # Print episode information
        print(f"DQN Episode: {self.current_episode}/{self.episodes}, Score: {episode_reward}, "
              f"Steps: {steps}, Best: {best_score}, Avg: {avg_score:.2f}, "
              f"Epsilon: {self.agent.epsilon:.4f}, Time: {elapsed_time:.2f}s", flush=True)
              
        # Periodically print device info
        if self.current_episode % 20 == 0:
            device_info = self.get_device_info()
            if device_info["device"] == "cuda":
                print(f"GPU: {device_info['gpu_name']} | "
                      f"VRAM: {device_info.get('vram_used', 'Unknown'):.2f}/{device_info.get('vram_total', 'Unknown'):.2f}GB | "
                      f"CPU: {device_info['cpu_percent']}% | RAM: {device_info['ram_percent']}%", flush=True)

    def save_model(self, is_best=False):
        """Save the trained model."""
        filename = "snake_dqn_model_best.pth" if is_best else "snake_dqn_model.pth"
        model_path = os.path.join(QMODEL_DIR, filename)
        self.agent.save_model(model_path)
        
        # Also save a checkpoint with episode number periodically
        if self.current_episode % 100 == 0:
            checkpoint_path = os.path.join(QMODEL_DIR, f"snake_dqn_checkpoint_ep{self.current_episode}.pth")
            self.agent.save_model(checkpoint_path)
            print(f"Checkpoint saved at episode {self.current_episode}", flush=True)

    def train(self):
        """Run the headless training process."""
        self.training_start_time = time.time()
        self.is_training = True
        
        # Set up plots if visualizations are enabled
        if self.show_graphs:
            self.setup_plots()
        
        print("Starting headless training...", flush=True)
        print(f"Using device: {device.type.upper()}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"CUDA Version: {torch.version.cuda}", flush=True)
            
        # Main training loop
        while self.is_training and self.current_episode < self.episodes:
            self.current_episode += 1
            episode_start_time = time.time()
            
            # Reset environment
            self.game_engine.reset_game()  # Using correct method name
            episode_reward = 0
            done = False
            steps = 0
            
            # Run episode
            while not done:
                reward, done, ate_food = self.agent.train_step()
                episode_reward += reward
                steps += 1
            
            # Store episode results
            self.episode_scores.append(episode_reward)
            self.episode_lengths.append(steps)
            
            # Calculate running average
            self.calculate_running_average()
            
            # Calculate elapsed time
            episode_time = time.time() - episode_start_time
            self.episode_durations.append(episode_time)
            total_elapsed = time.time() - self.training_start_time
            
            # Print info to terminal
            self.print_training_info(episode_reward, steps, total_elapsed)
            
            # Update visualization if enabled
            if self.show_graphs:
                self.update_plots()
            
            # Save model periodically
            if self.current_episode % self.save_interval == 0:
                self.save_model()
                
            # Save best model if this is the best score
            if episode_reward == max(self.episode_scores):
                self.save_model(is_best=True)
                
            # Save training history periodically
            if self.current_episode % (self.save_interval // 2) == 0:
                self.save_training_history()
                
            # Force garbage collection periodically to prevent memory leaks
            if self.current_episode % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final save when training completes
        self.save_model()
        print(f"Training completed. Final model saved.")
        
        # Save final training history
        self.save_training_history()
        
        # Keep plots open if visualizations were enabled
        if self.show_graphs:
            plt.ioff()
            plt.show()
            
    def save_training_history(self):
        """Save training history to a JSON file."""
        try:
            # Calculate best score and running average
            best_score = max(self.episode_scores) if self.episode_scores else 0
            
            if len(self.running_avg_scores) > 0:
                latest_avg = self.running_avg_scores[-1]
            else:
                latest_avg = 0.0
                
            # Prepare the history data
            # Safely get losses and q_values, ensuring they are lists
            losses = self.agent.stats.get('losses', []) if isinstance(self.agent.stats, dict) else []
            if isinstance(losses, list):
                losses = losses[:1000]  # Limit size for performance
            else:
                losses = []
                
            q_values = self.agent.stats.get('q_values', []) if isinstance(self.agent.stats, dict) else []
            if isinstance(q_values, list):
                q_values = q_values[:1000]  # Limit size for performance
            else:
                q_values = []
            
            history = {
                "scores": self.episode_scores,
                "running_avgs": self.running_avg_scores,
                "losses": losses,
                "q_values": q_values,
                "best_score": best_score,
                "latest_avg_score": latest_avg,
                "episodes_completed": self.current_episode,
                "training_time": time.time() - self.training_start_time if self.training_start_time else 0,
                "timestamp": time.time(),
                "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            
            # Save to file
            history_path = os.path.join(QMODEL_DIR, "snake_dqn_model_history.json")
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
            print(f"Training history saved to {history_path}")
            return True
        except Exception as e:
            print(f"Error saving training history: {e}")
            return False

def save_training_history(trainer):
    """Save training history to a JSON file."""
    try:
        # Safely get losses and q_values, ensuring they are lists
        losses = trainer.agent.stats.get('losses', []) if isinstance(trainer.agent.stats, dict) else []
        if isinstance(losses, list):
            losses = losses[:1000]  # Limit size for performance
        else:
            losses = []
            
        q_values = trainer.agent.stats.get('q_values', []) if isinstance(trainer.agent.stats, dict) else []
        if isinstance(q_values, list):
            q_values = q_values[:1000]  # Limit size for performance
        else:
            q_values = []
        
        # Prepare the history data
        history = {
            "scores": trainer.episode_scores,
            "running_avgs": trainer.running_avg_scores,
            "losses": losses,
            "q_values": q_values,
            "best_score": max(trainer.episode_scores) if trainer.episode_scores else 0,
            "episodes_completed": trainer.current_episode,
            "training_time": time.time() - trainer.training_start_time if trainer.training_start_time else 0,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # Save to file
        model_path = os.path.join(QMODEL_DIR, "snake_dqn_model.pth")
        history_path = os.path.join(QMODEL_DIR, "snake_dqn_model_history.json")
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
        print(f"Training history saved to {history_path}")
        return True
    except Exception as e:
        print(f"Error saving training history: {e}")
        return False

def main():
    """Parse command line arguments and start training."""
    parser = argparse.ArgumentParser(description='Headless DQN Training for Snake Game')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--save-interval', type=int, default=50, help='How often to save the model (episodes)')
    parser.add_argument('--show-graphs', action='store_true', help='Show real-time training graphs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--new-model', action='store_true', help='Start with a fresh model instead of loading checkpoint')
    
    args = parser.parse_args()
    
    # Check for GPU availability and print info
    check_gpu_availability()
    
    # Create and run the headless trainer
    trainer = HeadlessDQNTrainer(
        episodes=args.episodes,
        show_graphs=args.show_graphs,
        save_interval=args.save_interval
    )
    
    # If using a custom batch size or learning rate, update the agent
    if trainer.agent and args.batch_size != 64:
        print(f"Setting custom batch size: {args.batch_size}")
        trainer.agent.batch_size = args.batch_size
        
    if trainer.agent and args.learning_rate != 0.001:
        print(f"Setting custom learning rate: {args.learning_rate}")
        # Update optimizer with new learning rate
        for param_group in trainer.agent.optimizer.param_groups:
            param_group['lr'] = args.learning_rate
    
    # If new-model flag is set, reinitialize the agent
    if args.new_model:
        print("Starting with a fresh model (ignoring checkpoints)")
        trainer.game_engine = GameEngine()
        trainer.agent = AdvancedDQNAgent(trainer.game_engine)
    
    # Run training
    trainer.train()

if __name__ == "__main__":
    main()