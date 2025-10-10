# Advanced Snake Game - Complete Reference Guide

**Version:** 2.0  
**Last Updated:** October 10, 2025  
**Author:** Gavin Tann

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Game Modes](#game-modes)
4. [Algorithms](#algorithms)
5. [Q-Learning Model](#q-learning-model)
6. [Enhanced DQN Model](#enhanced-dqn-model)
7. [Curriculum Learning System](#curriculum-learning-system)
8. [Training UI](#training-ui)
9. [Performance Features](#performance-features)
10. [Configuration & Tuning](#configuration--tuning)
11. [Command Reference](#command-reference)
12. [Troubleshooting](#troubleshooting)
13. [Tips & Best Practices](#tips--best-practices)

---

## Overview

The Advanced Snake Game is a comprehensive reinforcement learning project featuring multiple AI algorithms for playing Snake. It includes:

- **Manual play mode** for human players
- **Pathfinding algorithms** (A*, Dijkstra)
- **Q-Learning** tabular reinforcement learning
- **Enhanced DQN** with curriculum learning, prioritized replay, and A* guidance
- **Training UI** with real-time visualization and performance monitoring
- **GPU acceleration** for faster training

### Key Features

✅ **5 Curriculum Stages** - Progressive difficulty from beginner to expert  
✅ **Gradient Indicators** - Real-time learning momentum visualization  
✅ **Stuck Detection** - Automatic exploration boosts when plateauing  
✅ **Performance Optimizations** - 90% faster UI updates, GPU support  
✅ **Comprehensive Monitoring** - Score tracking, epsilon decay, learning rate visualization  

---

## Getting Started

### Installation

1. **Requirements:**
   ```
   Python 3.8+
   PyTorch (with CUDA support for GPU)
   NumPy
   Matplotlib
   Tkinter (usually included with Python)
   Pygame
   ```

2. **Install dependencies:**
   ```bash
   pip install torch numpy matplotlib pygame
   ```

3. **Verify GPU support (optional but recommended):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### File Structure

```
Snake Game/advanced_snake/
├── main.py                  # Main game entry point
├── game_engine.py           # Core game logic
├── ui.py                    # Game UI
├── algorithms.py            # A* and Dijkstra implementations
├── q_learning.py            # Q-Learning agent
├── enhanced_dqn.py          # Enhanced DQN agent
├── train_qlearning.py       # Q-Learning training script
├── train_enhanced.py        # Enhanced DQN training script
├── training_ui.py           # Training UI with visualizations
├── constants.py             # Configuration parameters
├── gpu_utils.py             # GPU utilities
└── models/                  # Saved models directory
    ├── snake_qlearning_model.pkl
    └── snake_dqn_model_epXXXX.pth
```

### Quick Start

**Play manually:**
```bash
python main.py
# Select "Manual" mode
# Use WASD keys to control the snake
```

**Watch A* algorithm:**
```bash
python main.py
# Select "A* Algorithm" mode
```

**Train Enhanced DQN with UI:**
```bash
python training_ui.py
# Configure settings in UI
# Click "Start Training"
```

**Train Enhanced DQN via command line:**
```bash
python train_enhanced.py --episodes 1000
```

---

## Game Modes

### 1. Manual Mode
- **Control:** WASD keys (W=up, A=left, S=down, D=right)
- **Goal:** Eat food to grow, avoid walls and yourself
- **Scoring:** +10 points per food

### 2. A* Algorithm
- **Type:** Pathfinding algorithm with heuristic
- **Behavior:** Always finds shortest path to food
- **Performance:** Deterministic, perfect pathfinding
- **Limitation:** No learning, can trap itself when snake gets long

### 3. Dijkstra Algorithm
- **Type:** Shortest path algorithm
- **Behavior:** Explores all possibilities to find optimal path
- **Performance:** Guaranteed shortest path, slower than A*
- **Limitation:** No learning, can trap itself

### 4. Q-Learning Algorithm
- **Type:** Tabular reinforcement learning
- **Behavior:** Learns from experience, builds Q-table
- **Performance:** Good for small state spaces
- **Limitation:** State space explosion in large grids

### 5. Advanced DQN (Enhanced DQN)
- **Type:** Deep reinforcement learning with neural networks
- **Behavior:** Learns complex patterns, uses curriculum learning
- **Performance:** Best overall, learns strategic play
- **Features:** Double DQN, Dueling architecture, Prioritized Experience Replay, A* guidance

---

## Algorithms

### A* Pathfinding

**How it works:**
1. Calculates Manhattan distance heuristic to food
2. Explores nodes with lowest f = g + h score
   - g = distance from start
   - h = heuristic distance to goal
3. Reconstructs optimal path when food is reached

**Advantages:**
- ✅ Fast and efficient
- ✅ Guaranteed shortest path
- ✅ Works well for simple cases

**Disadvantages:**
- ❌ No learning capability
- ❌ Can trap itself when snake body blocks paths
- ❌ No long-term strategy

**Use case:** Demonstrating optimal pathfinding, baseline comparison

### Dijkstra's Algorithm

**How it works:**
1. Explores all neighbors uniformly
2. Always picks unvisited node with smallest distance
3. Guarantees shortest path to food

**Advantages:**
- ✅ Guaranteed optimal path
- ✅ Simpler than A* (no heuristic needed)

**Disadvantages:**
- ❌ Slower than A* (explores more nodes)
- ❌ No learning capability
- ❌ Can trap itself

**Use case:** Educational purposes, comparison with A*

---

## Q-Learning Model

### Overview

Q-Learning is a tabular reinforcement learning algorithm that learns state-action values (Q-values) without needing a model of the environment.

### State Representation

**State features (11 dimensions):**
1-3. Danger straight, left, right (binary)
4-7. Food direction: up, down, left, right (binary)
8-11. Current direction: up, down, left, right (one-hot encoded)

### Q-Table

- **Size:** Dynamically grows as new states are encountered
- **Storage:** Dictionary mapping state tuples to action Q-values
- **Actions:** 3 possible (straight, left turn, right turn)

### Training Parameters

```python
QLEARNING_ALPHA = 0.1       # Learning rate
QLEARNING_GAMMA = 0.9       # Discount factor
QLEARNING_EPSILON = 1.0     # Initial exploration rate
QLEARNING_EPSILON_MIN = 0.01
QLEARNING_EPSILON_DECAY = 0.995
```

### Reward Structure

```python
REWARD_FOOD = 10.0              # Eating food
REWARD_DEATH = -10.0            # Dying (collision)
REWARD_MOVE_TOWARDS_FOOD = 0.1  # Moving closer to food
REWARD_MOVE_AWAY_FROM_FOOD = -0.1  # Moving away from food
REWARD_SURVIVAL = 0.01          # Staying alive
```

### Training Q-Learning

**Via command line:**
```bash
python train_qlearning.py --episodes 5000
```

**Via main menu:**
1. Run `python main.py`
2. Select "Q-Learning Algorithm"
3. Choose "Train New Model" or "Load Existing Model"

### Q-Learning Pros & Cons

**Advantages:**
- ✅ Simple and interpretable
- ✅ No neural network required
- ✅ Fast for small state spaces
- ✅ Guaranteed convergence (with proper parameters)

**Disadvantages:**
- ❌ State space explosion in complex environments
- ❌ Cannot generalize to unseen states
- ❌ Limited to discrete state representations
- ❌ Struggles with high-dimensional inputs

---

## Enhanced DQN Model

### Architecture Overview

The Enhanced DQN combines multiple advanced techniques:

1. **Double DQN** - Reduces overestimation bias
2. **Dueling DQN** - Separate value and advantage streams
3. **Prioritized Experience Replay (PER)** - Learn from important transitions
4. **A* Guidance** - Hybrid approach using pathfinding hints
5. **Curriculum Learning** - Progressive difficulty stages

### Neural Network Architecture

```
Input Layer (14 dimensions):
├── Danger detection (8): immediate dangers in all directions
├── Food direction (2): normalized x, y distance to food
├── Current direction (4): one-hot encoded direction
└── (Optional) A* guidance features

Hidden Layers:
├── Layer 1: 128 neurons + ReLU
├── Layer 2: 128 neurons + ReLU
└── Dropout (0.1) for regularization

Dueling Streams:
├── Value Stream: 128 → 1 (state value V(s))
└── Advantage Stream: 128 → 3 (action advantages A(s,a))

Output Layer:
└── Q-values: V(s) + (A(s,a) - mean(A(s,a)))
    3 actions: straight, turn left, turn right
```

### State Features (14 dimensions)

**Danger detection (8):**
- Danger straight ahead
- Danger to the left
- Danger to the right
- Danger moving up
- Danger moving down
- Danger moving left
- Danger moving right
- Danger at current position

**Food location (2):**
- Normalized x-distance to food
- Normalized y-distance to food

**Current direction (4):**
- Moving up (binary)
- Moving down (binary)
- Moving left (binary)
- Moving right (binary)

### Training Parameters

```python
# Learning
DQN_LEARNING_RATE = 0.003      # Initial LR (adjusted per stage)
DQN_GAMMA = 0.99               # Discount factor
DQN_BATCH_SIZE = 64/512        # CPU/GPU batch size
DQN_MEMORY_SIZE = 100000       # Replay buffer size

# Exploration
DQN_EPSILON = 1.0              # Initial exploration
DQN_EPSILON_MIN = 0.01         # Minimum exploration
DQN_EPSILON_DECAY = 0.997      # Decay rate per episode

# Neural Network
DQN_HIDDEN_SIZE = 128          # Hidden layer neurons
DQN_TARGET_UPDATE = 25         # Target network sync frequency

# Prioritized Replay
DQN_PRIORITIZED_ALPHA = 0.6    # Priority exponent
DQN_PRIORITIZED_BETA = 0.4     # Importance sampling
DQN_BETA_INCREMENT = 0.001     # Beta annealing rate
```

### Advanced Features

#### Double DQN
Reduces overestimation by:
1. Selecting action with online network
2. Evaluating Q-value with target network
3. Formula: `Q_target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))`

#### Dueling Architecture
Separates state value from action advantages:
- **Value stream V(s):** How good is this state?
- **Advantage stream A(s,a):** How much better is this action vs others?
- **Combined:** Q(s,a) = V(s) + (A(s,a) - mean(A))

#### Prioritized Experience Replay (PER)
Samples important transitions more frequently:
- **Priority:** Based on TD-error magnitude
- **Sampling:** Stochastic prioritization
- **Correction:** Importance sampling weights to prevent bias

#### A* Guidance
Hybrid approach combining learning with pathfinding:
- Uses A* to suggest safe moves when uncertain
- Helps bootstrap learning in early stages
- Gradually reduces reliance as agent learns

---

## Curriculum Learning System

### Overview

Curriculum learning progressively increases task difficulty, allowing the agent to master basics before tackling complex scenarios.

### 5-Stage Curriculum

| Stage | Threshold | Description | Learning Rate | Epsilon Min |
|-------|-----------|-------------|---------------|-------------|
| **0** | 0-20 avg  | **Beginner:** Learn survival, avoid immediate walls | 0.005 | 0.10 |
| **1** | 20-50 avg | **Novice:** Consistent food collection, basic paths | 0.003 | 0.05 |
| **2** | 50-100 avg | **Intermediate:** Longer survival, avoid traps | 0.002 | 0.04 |
| **3** | 100-200 avg | **Advanced:** Strategic play, long-term planning | 0.001 | 0.02 |
| **4** | 200+ avg | **Expert:** Master level, maximize score | 0.0005 | 0.01 |

### Stage Advancement Logic

**Criteria for advancing:**
1. Average score (last 100 episodes) must exceed threshold
2. Must maintain threshold for **3 consecutive checks** (consistency requirement)
3. Automatic parameter adjustment upon advancement

**What changes when advancing:**
- ✅ Learning rate resets and begins decay schedule
- ✅ Epsilon minimum adjusted (more exploitation)
- ✅ Target Q-value becomes more conservative
- ✅ Training becomes more stable

### Stage-Specific Decay Parameters

**Epsilon Decay (per episode):**
```python
STAGE_EPSILON_DECAY = {
    0: 0.996,   # Slower decay (more exploration initially)
    1: 0.9965,  # Medium decay
    2: 0.997,   # Standard decay
    3: 0.998,   # Conservative decay
    4: 0.999    # Very slow decay (fine-tuning)
}
```

**Learning Rate Decay (per episode):**
```python
STAGE_LR_DECAY = {
    0: 0.9990,  # Maintain strong learning
    1: 0.9993,  # Gradual reduction
    2: 0.9995,  # Standard decay
    3: 0.9997,  # Conservative
    4: 0.9998   # Very conservative (fine-tuning)
}
```

**Minimum Values:**
```python
# Epsilon minimums ensure continued exploration
STAGE_EPSILON_MINIMUMS = {0: 0.10, 1: 0.05, 2: 0.04, 3: 0.02, 4: 0.01}

# LR minimums prevent learning from stopping
STAGE_LR_MINIMUMS = {0: 0.002, 1: 0.0015, 2: 0.001, 3: 0.0005, 4: 0.0002}
```

### Why Curriculum Learning?

**Benefits:**
- ✅ **Faster learning:** Master fundamentals before complexity
- ✅ **More stable:** Gradual difficulty prevents confusion
- ✅ **Better exploration:** Appropriate epsilon for each stage
- ✅ **Adaptive learning rate:** Matches complexity to plasticity
- ✅ **Higher final performance:** Strong foundation enables mastery

**Without curriculum:**
- ❌ Agent overwhelmed by complexity early
- ❌ Random exploration ineffective
- ❌ Learning rate too high/low for current skill
- ❌ Unstable progress with frequent regressions

### Monitoring Curriculum Progress

**In Training UI:**
- Curriculum stage shown in status bar
- Stage advancements marked with cyan stars on graph
- Vertical lines indicate advancement episodes
- Annotations show stage transitions (S0→S1, S1→S2, etc.)

**Expected progression timeline:**
- **Stage 0→1:** ~50-200 episodes (learn basics)
- **Stage 1→2:** ~200-500 episodes (consistent performance)
- **Stage 2→3:** ~500-1000 episodes (strategic play)
- **Stage 3→4:** ~1000-2000 episodes (mastery)

---

## Training UI

### Overview

The Training UI (`training_ui.py`) provides a comprehensive interface for training, monitoring, and analyzing Enhanced DQN agents.

### UI Tabs

#### 1. Setup & Monitoring
- **Model directory:** Select where to save/load models
- **Training episodes:** Configure training length
- **Model browser:** View and load existing models
- **Stuck detection controls:** Enable/disable and tune stuck detection
- **Real-time log:** Training progress messages

#### 2. Training Performance
- **Score progression graph:** Episode scores and 100-episode running average
- **Gradient indicators:** Learning momentum visualization (3 time scales)
- **Curriculum markers:** Stage advancement annotations
- **Epsilon & LR tracking:** Exploration and learning rate over time
- **Max score highlighting:** Best episode marked with gold star

#### 3. Network Architecture
- **Layer visualization:** Neural network structure diagram
- **Parameter counts:** Neurons and connections per layer
- **Activation functions:** ReLU, Softmax details

#### 4. State Features
- **Feature importance:** Which inputs matter most
- **Input distribution:** Feature value histograms
- **Correlation matrix:** Feature relationships

#### 5. Live Gameplay
- **Real-time visualization:** Watch agent play during training
- **Performance metrics:** Current score, steps, episode
- **Decision visualization:** Q-values for each action

### Gradient Indicators (NEW!)

**Three gradient boxes in upper right of Score Progression graph:**

1. **Overall Gradient** - (Current avg - Initial avg) / Total episodes
   - Shows long-term learning effectiveness
   - Green = Good overall progress

2. **Mid-term Gradient** - (Current avg - Midpoint avg) / Half episodes
   - Shows if learning is accelerating or decelerating
   - Compare to Overall to detect trends

3. **Recent Gradient** - (Current avg - 100 eps ago) / 100
   - **MOST IMPORTANT:** Current learning momentum
   - Yellow = Stagnant, Green = Improving, Red = Regressing

**Color coding:**
- 🟢 **Bright Green** (>+0.05 pts/ep): Strong improvement
- 🟢 **Light Green** (+0.01 to +0.05): Slow improvement
- 🟡 **Yellow** (-0.01 to +0.01): Stagnant (plateau)
- 🟠 **Orange** (-0.05 to -0.01): Weak decline
- 🔴 **Red** (<-0.05): Strong decline (stop training!)

**Usage:**
- Check Recent gradient every 50 episodes
- If Yellow for 200+ episodes → Training likely complete
- If Red → Stop immediately, investigate issue
- Compare all three to understand learning trajectory

### Stuck Detection Controls

**Purpose:** Automatically boost exploration when agent plateaus

**Controls:**
- ☑ **Enable Stuck Detection:** Toggle on/off
- **Sensitivity (1-10 checks):** How many stuck checks before boost
  - Lower = More aggressive (boost sooner)
  - Higher = More conservative (boost later)
- **Cooldown (50-500 episodes):** Minimum gap between boosts
- **Boost Amount (0.05-0.30):** Epsilon increase when stuck
- **Min Improvement (2.0-15.0 points):** Required score gain to avoid "stuck" label

**When to disable:**
- Gradient indicators show Yellow despite boosts (boosts not helping)
- Training is unstable (scores oscillating wildly)
- You want full manual control over epsilon

**When to enable:**
- Agent genuinely stuck at plateaus for 300+ episodes
- Recent gradient improves to Green after boosts
- You want automated exploration management

### Performance Features

**Dynamic update intervals:**
- Episodes 1-50: 1 second updates
- Episodes 51-200: 2 second updates
- Episodes 201-500: 4 second updates
- Episodes 500+: 8 second updates

**Window state detection:**
- Zero CPU usage when minimized
- Instant recovery when restored
- Prevents lag when tabbing out

**Incremental graph updates:**
- 90% faster than full redraw
- Only redraws new data points
- Full redraw every 50 episodes for gradients

**Caching:**
- Epsilon values cached (no recalculation)
- Learning rate values cached
- Gradient boxes retained between updates

### Using the Training UI

**Step 1: Configure**
```
1. Set model directory (e.g., "models/")
2. Set training episodes (e.g., 1000)
3. Configure stuck detection (or disable)
4. (Optional) Adjust other parameters via constants.py
```

**Step 2: Start Training**
```
Click "Start Training" button
→ Training script launches in background
→ Real-time graphs update automatically
→ Log messages show progress
```

**Step 3: Monitor**
```
Watch:
• Recent gradient color (most important!)
• Curriculum stage advancements (cyan stars)
• Epsilon boosts (if stuck detection enabled)
• Max score progression (gold star)
```

**Step 4: Analyze**
```
• If Recent gradient Yellow for 200+ episodes → Consider stopping
• If boosts visible but gradient stays Yellow → Disable stuck detection
• If Recent gradient Red → Stop immediately, reduce learning rate
• If all gradients Green → Keep training!
```

**Step 5: Save**
```
Models automatically saved every 100 episodes to:
models/snake_dqn_model_epXXXX.pth
```

---

## Performance Features

### GPU Acceleration

**Automatic detection:**
- Training automatically uses CUDA if available
- Falls back to CPU if CUDA not found
- Batch size adjusted based on hardware

**GPU batch size:** 512 (vs 128 on CPU)
- Faster convergence
- Smoother gradients
- Better parallelization

**Enable/disable:**
```python
# In constants.py
USE_CUDA = True  # Set to False to force CPU
```

**Check GPU usage:**
```bash
python test_gpu_usage.py
```

### UI Performance Optimizations

**90% faster graph updates:**
- Incremental updates (append data vs full redraw)
- Selective recalculation (only when needed)
- Cached epsilon/LR values

**Window state detection:**
- Pauses updates when minimized
- Zero CPU usage in background
- Instant recovery on restore

**Simplified visualizations:**
- Last 5 curriculum advancements only
- Reduced annotation complexity
- Optimized matplotlib rendering

### Training Speed Optimizations

**Increased learning rate (Stage 0):**
- 0.005 vs 0.001 → 3x faster weight updates
- Faster bootstrap learning
- Decays to stable values

**Larger target update interval:**
- 25 episodes vs 10 → More stable Q-values
- Less oscillation
- Faster convergence

**Optimized batch sizes:**
- GPU: 512 samples → Smoother gradients
- CPU: 128 samples → Better than 64

**Learning starts at 2000:**
- More diverse initial experiences
- Better initialization
- Reduced early instability

---

## Configuration & Tuning

### Key Configuration Files

**constants.py:** Central configuration
- Game settings (grid size, colors, etc.)
- Training parameters (learning rate, epsilon, etc.)
- Curriculum thresholds and decay schedules
- Stuck detection parameters

**enhanced_dqn.py:** Agent implementation
- Neural network architecture
- Training loop logic
- Curriculum advancement rules

**training_ui.py:** UI settings
- Update intervals
- Graph configurations
- Performance optimizations

### Tuning Learning Rate

**Stage-specific starting values:**
```python
# In enhanced_dqn.py, update_learning_rate_for_stage()
stage_learning_rates = {
    0: 0.005,   # Beginner - fast learning
    1: 0.003,   # Novice - medium learning
    2: 0.002,   # Intermediate - standard
    3: 0.001,   # Advanced - conservative
    4: 0.0005   # Expert - fine-tuning
}
```

**Decay rates:**
```python
# In constants.py
STAGE_LR_DECAY = {
    0: 0.9990,  # Slower decay = maintain learning longer
    1: 0.9993,
    2: 0.9995,
    3: 0.9997,
    4: 0.9998   # Fastest decay = focus on fine-tuning
}
```

**When to adjust:**
- **LR too high:** Unstable training, scores oscillate wildly
- **LR too low:** Slow learning, gradual progress stalls
- **Solution:** Adjust starting values or decay rates

### Tuning Exploration (Epsilon)

**Epsilon decay rates:**
```python
# In constants.py
STAGE_EPSILON_DECAY = {
    0: 0.996,   # Slower decay = more exploration early
    1: 0.9965,
    2: 0.997,
    3: 0.998,
    4: 0.999    # Slowest decay = maintain exploration
}
```

**Minimum values:**
```python
STAGE_EPSILON_MINIMUMS = {
    0: 0.10,   # 10% random actions minimum
    1: 0.05,   # 5% random actions
    2: 0.04,
    3: 0.02,
    4: 0.01    # 1% random actions
}
```

**When to adjust:**
- **Too much exploration:** Agent doesn't exploit learned policy
- **Too little exploration:** Agent stuck in local optima
- **Solution:** Adjust minimums or decay rates per stage

### Tuning Stuck Detection

**Access via Training UI:**
1. Open Training UI
2. Find "Stuck Detection Controls" section
3. Adjust sliders:
   - **Sensitivity:** 1-10 (default: 3)
   - **Cooldown:** 50-500 episodes (default: 200)
   - **Boost amount:** 0.05-0.30 (default: 0.10)
   - **Min improvement:** 2.0-15.0 points (default: 5.0)

**Via constants.py:**
```python
ENABLE_STUCK_DETECTION = True
STUCK_COUNTER_THRESHOLD = 3    # Checks before boost
STUCK_BOOST_COOLDOWN = 200     # Episodes between boosts
STUCK_EPSILON_BOOST = 0.10     # Boost amount
STUCK_IMPROVEMENT_THRESHOLD = 5.0  # Points needed
```

**Recommendations:**
- **If boosts not helping:** Increase cooldown to 400+, reduce boost to 0.05-0.06
- **If agent genuinely stuck:** Decrease sensitivity to 2-3, moderate boost 0.08-0.10
- **If unstable:** Disable entirely or very conservative (sensitivity 10, cooldown 500)

### Tuning Curriculum Thresholds

**Default thresholds:**
```python
# In enhanced_dqn.py
self.curriculum_thresholds = [20, 50, 100, 200]
# Stage 0→1: 20 avg, Stage 1→2: 50 avg, etc.
```

**Consistency requirement:**
```python
self.curriculum_consistency_required = 3
# Must meet threshold 3 times consecutively
```

**When to adjust:**
- **Too easy:** Agent advances too quickly, doesn't master basics
  - Solution: Increase thresholds (e.g., [30, 75, 150, 300])
- **Too hard:** Agent stuck at low stages forever
  - Solution: Decrease thresholds (e.g., [15, 40, 80, 150])
- **Unstable:** Agent regresses after advancing
  - Solution: Increase consistency requirement to 4-5

### Batch Size Tuning

**GPU training:**
```python
GPU_BATCH_SIZE = 512  # Large batch for smoother gradients
```

**CPU training:**
```python
CPU_BATCH_SIZE = 128  # Balance speed vs memory
```

**When to adjust:**
- **GPU memory error:** Reduce GPU_BATCH_SIZE to 256 or 128
- **Too slow on CPU:** Reduce CPU_BATCH_SIZE to 64
- **Want smoother learning:** Increase batch size (if memory allows)

---

## Command Reference

### Training Commands

**Enhanced DQN (basic):**
```bash
python train_enhanced.py --episodes 1000
```

**Enhanced DQN (custom parameters):**
```bash
python train_enhanced.py \
  --episodes 5000 \
  --save-interval 200 \
  --resume models/snake_dqn_model_ep500.pth
```

**Enhanced DQN (stuck detection tuning):**
```bash
python train_enhanced.py \
  --episodes 1000 \
  --disable-stuck-detection
  
# Or with custom stuck detection settings
python train_enhanced.py \
  --episodes 1000 \
  --enable-stuck-detection \
  --stuck-sensitivity 8 \
  --stuck-cooldown 400 \
  --stuck-boost 0.06 \
  --stuck-improvement 9.0
```

**Q-Learning:**
```bash
python train_qlearning.py --episodes 5000
```

**Training UI:**
```bash
python training_ui.py
```

### Playing/Testing Commands

**Manual play:**
```bash
python main.py
# Select "Manual" from menu
```

**Watch trained agent:**
```bash
python main.py
# Select "Advanced DQN" from menu
# Choose "Load Existing Model"
```

**Test GPU:**
```bash
python test_gpu_usage.py
```

### Common Training Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--episodes` | Training episodes | 1000 | 100-10000 |
| `--save-interval` | Save frequency | 100 | 10-1000 |
| `--resume` | Continue from checkpoint | None | Path to .pth |
| `--stuck-sensitivity` | Stuck checks before boost | 3 | 1-10 |
| `--stuck-cooldown` | Episodes between boosts | 200 | 50-500 |
| `--stuck-boost` | Epsilon increase amount | 0.10 | 0.05-0.30 |
| `--enable-stuck-detection` | Enable boosts | True | - |
| `--disable-stuck-detection` | Disable boosts | False | - |

---

## Troubleshooting

### Training Issues

**Problem: Training too slow**
- **Check:** GPU availability (`nvidia-smi` or `python test_gpu_usage.py`)
- **Solution:** Install CUDA-enabled PyTorch, or reduce batch size
- **Workaround:** Use CPU with smaller batch size (128 → 64)

**Problem: Agent not improving**
- **Check:** Gradient indicators (Recent gradient Yellow?)
- **Solution:** 
  - If Yellow for 200+ episodes → Training likely complete
  - Try disabling stuck detection
  - Reduce learning rate or adjust curriculum thresholds

**Problem: Scores oscillating wildly**
- **Check:** Learning rate too high
- **Solution:** Reduce stage learning rates in `enhanced_dqn.py`
- **Temporary:** Lower batch size for more stable gradients

**Problem: Stuck detection not helping**
- **Check:** Recent gradient color before/after boosts
- **Solution:** If stays Yellow → Disable stuck detection
- **Alternative:** Increase cooldown to 400+, reduce boost to 0.05

### UI Issues

**Problem: UI freezing at 300+ episodes**
- **Solved:** Recent performance optimizations (90% faster updates)
- **Verify:** Check if running latest `training_ui.py` version
- **Workaround:** Restart UI, reduce update frequency

**Problem: Graphs not updating**
- **Check:** Training script still running
- **Solution:** Check log for errors, restart training
- **Verify:** Model files being saved to correct directory

**Problem: Gradient indicators not showing**
- **Check:** Need 100+ episodes for meaningful Recent gradient
- **Solution:** Train longer or load model with more history
- **Verify:** Running latest `training_ui.py` with gradient code

### GPU Issues

**Problem: CUDA out of memory**
- **Solution:** Reduce `GPU_BATCH_SIZE` from 512 to 256 or 128
- **Edit:** `constants.py` → `GPU_BATCH_SIZE = 256`

**Problem: GPU not detected**
- **Check:** `python -c "import torch; print(torch.cuda.is_available())"`
- **Solution:** Install CUDA toolkit and CUDA-enabled PyTorch
- **Workaround:** Set `USE_CUDA = False` in constants.py

### Model Loading Issues

**Problem: Model file not found**
- **Check:** Correct path in model directory
- **Solution:** Verify file exists, check for typos
- **Common:** Ensure `.pth` extension for DQN, `.pkl` for Q-Learning

**Problem: Incompatible model version**
- **Check:** Model architecture matches current code
- **Solution:** Retrain from scratch if major code changes
- **Workaround:** Keep backup of old code with old models

---

## Tips & Best Practices

### Training Strategy

**1. Start with defaults**
- Use default curriculum thresholds [20, 50, 100, 200]
- Keep stuck detection enabled initially
- Train for at least 500 episodes to see patterns

**2. Monitor gradients**
- **Most important:** Watch Recent gradient (bottom box)
- If Green → Keep training
- If Yellow for 200+ episodes → Likely complete
- If Red → Stop immediately, investigate

**3. Use stuck detection wisely**
- Enable during first training run
- Check if boosts change Recent gradient Yellow → Green
- If no improvement after boosts → Disable
- Conservative settings: Sensitivity 8, Cooldown 400, Boost 0.06

**4. Save checkpoints**
- Models auto-save every 100 episodes
- Keep best checkpoints (highest avg score)
- Don't delete intermediates until training complete

**5. Compare runs**
- Test with/without stuck detection
- Try different curriculum thresholds
- Document what works for your setup

### Curriculum Learning Tips

**Stage 0 (Beginner):**
- Expect slow initial progress (50-100 episodes)
- Agent learning basic survival
- High exploration (epsilon ~0.5-1.0)
- Fast learning rate (0.005)

**Stage 1-2 (Novice/Intermediate):**
- Should advance within 200-500 episodes total
- More consistent food collection
- Moderate exploration (epsilon ~0.2-0.5)
- Medium learning rate (0.002-0.003)

**Stage 3-4 (Advanced/Expert):**
- May take 1000-2000+ episodes
- Strategic long-term planning
- Low exploration (epsilon ~0.05-0.15)
- Fine-tuning learning rate (0.0005-0.001)

**If stuck at a stage for 500+ episodes:**
- Check Recent gradient (Yellow = not improving)
- Consider lowering threshold for that stage
- Or increase learning rate for faster adaptation

### Parameter Tuning Workflow

**When changing parameters:**

1. **Change ONE parameter at a time**
   - Easier to identify what helped/hurt
   - Document changes in training notes

2. **Train for sufficient episodes**
   - Minimum 200 episodes to see effect
   - Preferably 500+ for stable assessment

3. **Compare gradient indicators**
   - Did Recent gradient improve (Yellow → Green)?
   - Did Overall gradient increase?
   - Is learning accelerating (Mid-term > Overall)?

4. **Check final performance**
   - Max score achieved
   - Average score in last 100 episodes
   - Stability (low variance)

5. **Revert if worse**
   - Load previous checkpoint
   - Restore old parameters
   - Try different adjustment

### Optimal Training Conditions

**Hardware:**
- GPU with 4GB+ VRAM (recommended)
- 8GB+ system RAM
- Multi-core CPU

**Settings for fastest learning:**
- GPU enabled (`USE_CUDA = True`)
- Large batch size (512 on GPU)
- Default curriculum thresholds
- Stuck detection disabled (if agent learning steadily)

**Settings for most stable learning:**
- Conservative learning rates (reduce by 20-30%)
- High epsilon minimums (more exploration)
- Higher curriculum thresholds [30, 75, 150, 300]
- Stuck detection enabled with conservative settings

### When to Stop Training

**Stop if:**
1. ✅ Recent gradient Yellow for 300+ episodes
2. ✅ Max score hasn't improved in 500+ episodes
3. ✅ Average score plateaued (stable for 200+ episodes)
4. ✅ All gradient indicators Yellow
5. ✅ Stuck detection boosts not helping (gradient stays Yellow)

**Continue if:**
1. ✅ Recent gradient Green (currently improving)
2. ✅ Mid-term > Overall (learning accelerating)
3. ✅ Just advanced curriculum stage (give time to adapt)
4. ✅ Boosts changing gradient Yellow → Green (working)

**Investigate if:**
1. ⚠️ Recent gradient Red (recent regression)
2. ⚠️ Scores oscillating wildly (unstable)
3. ⚠️ Frequent curriculum stage regressions
4. ⚠️ GPU memory errors

---

## Appendix

### File Descriptions

**Core Game:**
- `main.py` - Entry point, menu system
- `game_engine.py` - Game logic, collision detection
- `ui.py` - Pygame rendering
- `constants.py` - Configuration parameters

**Algorithms:**
- `algorithms.py` - A* and Dijkstra pathfinding
- `q_learning.py` - Q-Learning agent
- `enhanced_dqn.py` - Enhanced DQN agent with curriculum

**Training:**
- `train_qlearning.py` - Q-Learning training script
- `train_enhanced.py` - Enhanced DQN training script
- `training_ui.py` - Training UI with visualizations

**Utilities:**
- `gpu_utils.py` - GPU detection and utilities
- `test_gpu_usage.py` - GPU test script
- `test_training.py` - Training test script

**Documentation:**
- `GRADIENT_INDICATORS.md` - Gradient indicators quick reference
- `GRADIENT_INDICATORS_GUIDE.md` - Comprehensive gradient guide
- `STUCK_DETECTION_TUNING_GUIDE.md` - Stuck detection tuning
- `STUCK_DETECTION_CONTROLS.md` - UI controls reference
- `PERFORMANCE_IMPROVEMENTS.md` - Performance optimizations

### Keyboard Shortcuts

**During manual play:**
- `W` - Move up
- `A` - Move left
- `S` - Move down
- `D` - Move right
- `ESC` - Pause/Resume
- `Q` - Quit to menu

**During training:**
- Window resize - Supported
- Minimize - Zero CPU usage
- Close - Graceful shutdown

### Common Error Messages

**"CUDA out of memory"**
- Reduce batch size in constants.py
- Close other GPU applications
- Use CPU mode (`USE_CUDA = False`)

**"Model file not found"**
- Check model directory path
- Verify .pth file exists
- Ensure correct filename

**"State space too large"** (Q-Learning)
- Use Enhanced DQN instead
- Reduce state representation
- Increase training episodes

**"Training not improving"**
- Check gradient indicators
- Adjust learning rate
- Disable stuck detection if not helping

### Performance Benchmarks

**Training speed (Enhanced DQN):**
- GPU (RTX 3060): ~200-300 episodes/hour
- GPU (GTX 1660): ~150-200 episodes/hour
- CPU (i7-9700K): ~50-100 episodes/hour
- CPU (i5-8400): ~30-60 episodes/hour

**Memory usage:**
- Q-Learning: ~100-500 MB
- Enhanced DQN (GPU): ~2-4 GB VRAM
- Enhanced DQN (CPU): ~500 MB-1 GB RAM
- Training UI: ~200-400 MB RAM

**Expected scores:**
- Manual player (skilled): 100-300
- A* algorithm: 50-200 (depends on luck)
- Q-Learning (trained): 50-150
- Enhanced DQN (Stage 2): 50-150
- Enhanced DQN (Stage 3): 100-250
- Enhanced DQN (Stage 4): 200-400+

### Version History

**v2.0 (October 2025)**
- ✨ Added gradient indicators (3 time scales)
- ✨ Added stuck detection tuning controls
- ✨ Performance optimizations (90% faster UI)
- ✨ Window state detection (zero CPU when minimized)
- ✨ Incremental graph updates
- ✨ Comprehensive documentation

**v1.5**
- Added curriculum learning (5 stages)
- Enhanced DQN with Double, Dueling, PER
- A* guidance integration
- Training UI with visualizations

**v1.0**
- Initial release
- Manual play, A*, Dijkstra
- Basic Q-Learning
- Simple DQN

---

## Quick Reference Card

### Essential Commands
```bash
# Train with UI (recommended)
python training_ui.py

# Train without UI
python train_enhanced.py --episodes 1000

# Play manually
python main.py → Select "Manual"

# Watch trained agent
python main.py → Select "Advanced DQN" → Load Model
```

### Key Files to Edit
```
constants.py          → All parameters
enhanced_dqn.py       → Curriculum thresholds, architecture
training_ui.py        → UI settings
```

### Critical Parameters
```python
# Learning
DQN_LEARNING_RATE = 0.003        # Adjust per stage
DQN_GAMMA = 0.99                 # Discount factor

# Exploration
DQN_EPSILON = 1.0                # Start exploration
STAGE_EPSILON_MINIMUMS = {...}   # Min per stage

# Curriculum
curriculum_thresholds = [20, 50, 100, 200]
curriculum_consistency_required = 3

# Stuck Detection
ENABLE_STUCK_DETECTION = True
STUCK_COUNTER_THRESHOLD = 3
STUCK_BOOST_COOLDOWN = 200
```

### Decision Guide
```
Recent Gradient:
  🟢 Green  → Keep training
  🟡 Yellow → Consider stopping (if 200+ episodes)
  🔴 Red    → Stop immediately

Stuck Detection:
  Boosts help (Yellow→Green)  → Keep enabled
  Boosts don't help (Yellow)  → Disable
  Training unstable           → Disable

Curriculum Stage:
  Stuck at stage 500+ episodes  → Lower threshold
  Advancing too fast            → Raise threshold
  Regressing after advance      → Increase consistency
```

---

**End of Reference Guide**

For detailed information on specific topics, refer to:
- Gradient indicators: `GRADIENT_INDICATORS_GUIDE.md`
- Stuck detection: `STUCK_DETECTION_TUNING_GUIDE.md`
- Performance: `PERFORMANCE_IMPROVEMENTS.md`
- Quick starts: This guide's "Getting Started" section

**Happy Training! 🐍🎮🚀**
