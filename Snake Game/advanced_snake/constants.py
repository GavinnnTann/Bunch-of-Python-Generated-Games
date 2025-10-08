"""
Constants module for the Snake Game.
Contains game settings, colors, and other constants.
"""

import os

# Get the directory of this file (advanced_snake folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GAME_WIDTH = 600
GAME_HEIGHT = 600
INFO_WIDTH = 200
GRID_SIZE = 20
GRID_WIDTH = GAME_WIDTH // GRID_SIZE
GRID_HEIGHT = GAME_HEIGHT // GRID_SIZE

# Training window dimensions
TRAINING_SCREEN_WIDTH = 1300
TRAINING_SCREEN_HEIGHT = 800
MIN_TRAINING_SCREEN_WIDTH = 800
MIN_TRAINING_SCREEN_HEIGHT = 600

# Colors (R, G, B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 155, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Game settings
INITIAL_SNAKE_LENGTH = 3
FRAME_RATES = {"Slow": 5, "Medium": 10, "Fast": 15, "Very Fast": 50}
DEFAULT_SPEED = "Very Fast"
GAME_TITLE = "Advanced Snake Game"

# Game modes
MANUAL_MODE = "Manual"
ASTAR_MODE = "A* Algorithm"
DIJKSTRA_MODE = "Dijkstra Algorithm"
QLEARNING_MODE = "Q-Learning Algorithm"
GAME_MODES = [MANUAL_MODE, ASTAR_MODE, DIJKSTRA_MODE, QLEARNING_MODE]

# Direction vectors (row, col)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

# Key mappings
KEY_UP = "w"
KEY_DOWN = "s"
KEY_LEFT = "a"
KEY_RIGHT = "d"

# Game states
STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"
STATE_PAUSED = "paused"

# Score settings
POINTS_PER_FOOD = 10

# Q-learning parameters
QLEARNING_ALPHA = 0.1       # Learning rate
QLEARNING_GAMMA = 0.9       # Discount factor
QLEARNING_EPSILON = 1.0     # Initial exploration rate
QLEARNING_EPSILON_MIN = 0.01  # Minimum exploration rate
QLEARNING_EPSILON_DECAY = 0.995  # Decay rate for exploration
QLEARNING_BATCH_SIZE = 64   # Batch size for training

# Training settings
DEFAULT_TRAINING_EPISODES = 1000
TRAINING_EPISODES_OPTIONS = [100, 500, 1000, 5000, 10000]
MODEL_SAVE_INTERVAL = 100   # Save model every N episodes
TRAINING_DISPLAY_INTERVAL = 1  # Update display every N episodes

# Q-learning rewards
REWARD_FOOD = 10.0         # Reward for eating food
REWARD_DEATH = -10.0       # Penalty for dying
REWARD_MOVE_TOWARDS_FOOD = 0.1  # Small reward for moving towards food
REWARD_MOVE_AWAY_FROM_FOOD = -0.1  # Small penalty for moving away from food
REWARD_SURVIVAL = 0.01     # Tiny reward for surviving (encourages longer games)

# Model file paths
QMODEL_DIR = os.path.join(SCRIPT_DIR, "models")
QMODEL_FILE = "snake_qlearning_model.pkl"
DQN_MODEL_FILE = "snake_dqn_model.pth"

# DQN parameters
DQN_MODE = "Advanced DQN"  # Mode name for the menu
DQN_LEARNING_RATE = 0.003   # PERFORMANCE BOOST: Increased from 0.001 for faster learning (3x faster weight updates)
DQN_GAMMA = 0.99           # Discount factor
DQN_EPSILON = 1.0          # Initial exploration rate
DQN_EPSILON_MIN = 0.01     # Minimum exploration rate
DQN_EPSILON_DECAY = 0.997  # Decay rate for exploration (per episode, not per step)
DQN_BATCH_SIZE = 64        # Batch size for training (can be increased for GPU)
DQN_MEMORY_SIZE = 100000   # Size of replay buffer
DQN_TARGET_UPDATE = 25     # SPEED OPTIMIZATION: Increased from 10 for more stable Q-values (less oscillation)
DQN_PRIORITIZED_ALPHA = 0.6  # Alpha parameter for prioritized replay (0 = uniform sampling)
DQN_PRIORITIZED_BETA = 0.4   # Beta parameter for prioritized replay (importance sampling)
DQN_BETA_INCREMENT = 0.001   # How much to increase beta each sampling

# CUDA/GPU settings
USE_CUDA = True            # Whether to use CUDA when available
GPU_BATCH_SIZE = 512       # SPEED OPTIMIZATION: Increased from 256 for smoother gradients (faster convergence)
CPU_BATCH_SIZE = 128       # PERFORMANCE BOOST: Increased from 64 for better convergence

# Neural network architecture
DQN_HIDDEN_SIZE = 128      # Size of hidden layers
DQN_LEARNING_STARTS = 2000  # SPEED OPTIMIZATION: Increased from 1000 for more diverse initial experiences

# Training settings for DQN
DQN_TRAINING_EPISODES_OPTIONS = [100, 500, 1000, 5000, 10000]
DEFAULT_DQN_TRAINING_EPISODES = 1000
DQN_MODEL_SAVE_INTERVAL = 100  # Save model every N episodes