# Advanced Snake Game with Reinforcement Learning# Advanced Snake Game - Clean Project Structure



A comprehensive Snake game implementation featuring multiple AI algorithms, deep reinforcement learning with curriculum learning, and advanced training visualization.## 📁 Essential Files



---### Core Game Components

- **`constants.py`** - Game configuration and hyperparameters

## 🎮 Features- **`game_engine.py`** - Snake game logic and mechanics

- **`ui.py`** - User interface components

### Game Modes

- **Manual Play** - Classic Snake with WASD controls### Algorithm Implementations

- **A* Algorithm** - Intelligent pathfinding- **`algorithms.py`** - Traditional algorithms (A*, BFS, Hamiltonian)

- **Dijkstra Algorithm** - Guaranteed shortest path- **`q_learning.py`** - Q-Learning (Tabular) agent

- **Q-Learning** - Tabular reinforcement learning- **`advanced_dqn.py`** - Deep Q-Network agent

- **Enhanced DQN** - Deep Q-Network with curriculum learning- **`enhanced_dqn.py`** - Enhanced DQN with curriculum learning

- **`gpu_utils.py`** - GPU acceleration utilities

### Enhanced DQN Capabilities

- ✨ **Curriculum Learning** - 5 progressive difficulty stages### Main Entry Points

- ✨ **Advanced Architecture** - Double DQN + Dueling + Prioritized Replay- **`main.py`** - Main game interface (play/watch AI)

- ✨ **Real-time Visualization** - Comprehensive training UI- **`training_ui.py`** - **PRIMARY TRAINING TOOL** - Comprehensive training interface with:

- ✨ **Learning Momentum** - Gradient indicators across 3 time scales  - Model selection (Q-Learning, Original DQN, Enhanced DQN)

- ✨ **Stuck Detection** - Automatic exploration boosts (configurable)  - Real-time training graphs

- ✨ **GPU Acceleration** - CUDA support for faster training  - Hyperparameter controls

- ✨ **Performance Optimized** - 90% faster UI updates  - Model management



---### Training Scripts (Called by training_ui.py)

- **`train_qlearning.py`** - Q-Learning training backend

## 🚀 Quick Start- **`train_enhanced.py`** - Enhanced DQN training backend



### Installation### Model Storage

- **`models/`** - Saved models and training statistics

```bash  - `snake_qlearning_model.pkl` - Q-Learning model

# Install dependencies  - `snake_enhanced_dqn*.pth` - Enhanced DQN models

pip install torch numpy matplotlib pygame  - `*_history.json` - Training history files

  - `qlearning_training_stats.json` - Q-Learning stats

# Verify GPU support (optional)

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"### Documentation

```- **`QUICK_REFERENCE.md`** - Quick reference guide



### Train Your First Agent## 🗑️ Removed Non-Essential Files



**Option 1: Training UI (Recommended)**### Debug/Test Scripts (Deleted)

```bash- ❌ `check_cuda.py` - One-time CUDA setup check

python training_ui.py- ❌ `debug_model.py` - Model debugging tool

```- ❌ `test_batch_size.py` - GPU memory testing

Click "Start Training" and watch the visualization!- ❌ `test_enhanced.py` - Enhanced DQN testing

- ❌ `demo_new_graphs.py` - Graph visualization demo

**Option 2: Command Line**- ❌ `diagnose_stagnation.py` - Training analysis tool

```bash- ❌ `compare_training.py` - Model comparison tool

python train_enhanced.py --episodes 1000- ❌ `check_model_state.py` - Model state inspector

```

### Redundant Training Scripts (Deleted - Replaced by training_ui.py)

### Play with Trained Agent- ❌ `training.py` - Old training script

- ❌ `dqn_training.py` - Old DQN training

```bash- ❌ `headless_training.py` - Replaced by training UI

python main.py

# Select "Advanced DQN" → Load Existing Model### Batch Files (Deleted)

```- ❌ `run_headless_training.bat`

- ❌ `run_training_ui.bat`

---

### Analysis Documentation (Deleted)

## 📊 Training UI Features- ❌ `OPTIMAL_HYPERPARAMETERS_ANALYSIS.md`

- ❌ `STAGE_2_ANALYSIS.md`

### Real-time Monitoring

- **Score Progression Graph** - Episode scores + 100-episode average### Old Model Checkpoints (Deleted)

- **Gradient Indicators** - Learning momentum visualization- ❌ `snake_dqn_model_ep*_interrupted_*.json` - Interrupted training files

  - Overall gradient (long-term progress)

  - Mid-term gradient (acceleration/deceleration)## 🚀 How to Use

  - Recent gradient (current momentum) - **MOST IMPORTANT**

- **Curriculum Markers** - Stage advancement annotations### Play the Game

- **Epsilon & Learning Rate Tracking** - Exploration/exploitation balance```bash

python main.py

### Gradient Color Coding```

- 🟢 **Bright Green** (>+0.05 pts/ep): Strong improvement**Main Menu Options:**

- 🟢 **Light Green** (+0.01 to +0.05): Slow improvement- Select game mode (Manual, A*, BFS, Hamiltonian, Q-Learning, Enhanced DQN)

- 🟡 **Yellow** (-0.01 to +0.01): Stagnant (plateau)- Adjust speed

- 🟠 **Orange** (-0.05 to -0.01): Weak decline- Browse and load trained models

- 🔴 **Red** (<-0.05): Strong decline- Start playing!



### Stuck Detection Controls**Note:** Training options have been removed from main.py. Use `training_ui.py` for all training needs.

- ☑ Enable/Disable toggle

- Sensitivity slider (1-10 checks before boost)### Train Models (Recommended)

- Cooldown slider (50-500 episodes between boosts)```bash

- Boost amount slider (0.05-0.30 epsilon increase)python training_ui.py

- Min improvement threshold (2.0-15.0 points)```

The training UI provides:

---- Model type selection (Q-Learning, Original DQN, Enhanced DQN)

- Hyperparameter adjustment

## 🎓 Curriculum Learning System- Real-time performance graphs

- Model save/load management

Progressive 5-stage difficulty system that adapts learning parameters:- Training progress tracking



| Stage | Threshold | Description | Learning Rate | Epsilon Min |### Quick Training (Command Line)

|-------|-----------|-------------|---------------|-------------|```bash

| **0** | 0-20 avg | Beginner: Survival basics | 0.005 | 0.10 |# Q-Learning

| **1** | 20-50 avg | Novice: Consistent food collection | 0.003 | 0.05 |python train_qlearning.py --episodes 1000 --learning-rate 0.1 --batch-size 64

| **2** | 50-100 avg | Intermediate: Avoid traps | 0.002 | 0.04 |

| **3** | 100-200 avg | Advanced: Strategic planning | 0.001 | 0.02 |# Enhanced DQN

| **4** | 200+ avg | Expert: Score maximization | 0.0005 | 0.01 |python train_enhanced.py --episodes 2000 --batch-size 512 --save-interval 200

```

---

## 📊 Model Comparison

## 📚 Documentation

| Model | State Space | Action Space | Training Time | Performance |

### Quick References|-------|-------------|--------------|---------------|-------------|

- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Get started in 5 minutes| **Q-Learning** | 11 features (discrete) | 4 absolute (UP/DOWN/LEFT/RIGHT) | Fast (~50-100 eps/sec) | Excellent for Snake |

- **[COMPLETE_REFERENCE_GUIDE.md](COMPLETE_REFERENCE_GUIDE.md)** - Comprehensive documentation| **Original DQN** | 11 features | 3 relative (Turn Left/Straight/Right) | Medium (~10-30 eps/sec) | Moderate |

| **Enhanced DQN** | 34 features | 3 relative | Slower (~10-20 eps/sec) | Best with curriculum |

### Feature Guides

- **[GRADIENT_INDICATORS_GUIDE.md](GRADIENT_INDICATORS_GUIDE.md)** - Learning momentum explained## 💡 Key Insights

- **[STUCK_DETECTION_TUNING_GUIDE.md](STUCK_DETECTION_TUNING_GUIDE.md)** - When and how to use boosts

- **[PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md)** - Optimization details### Why Q-Learning Outperforms DQN for Snake

1. **Absolute Actions**: Q-Learning uses direct spatial actions (UP/DOWN/LEFT/RIGHT)

---2. **Perfect Memory**: Tabular approach stores exact Q-values for each state

3. **Small State Space**: ~500-800 states for typical Snake gameplay

## 🎯 Training Tips4. **Fast Training**: Reaches competence in ~1000 episodes



### When to Stop Training### When to Use Enhanced DQN

**Stop if:**- Larger, more complex environments

- ✅ Recent gradient Yellow for 300+ episodes- Need for generalization beyond seen states

- ✅ Max score hasn't improved in 500+ episodes- When state space is too large for tabular methods

- ✅ Average score plateaued (stable for 200+ episodes)- Continuous or high-dimensional state spaces



**Continue if:**## 🔧 Project Cleanup Summary

- ✅ Recent gradient Green (improving)

- ✅ Mid-term > Overall (accelerating)**Removed**: 15+ non-essential files (debug scripts, redundant trainers, old docs)  

- ✅ Just advanced curriculum stage**Result**: Clean, maintainable codebase focused on core functionality  

**Benefit**: Easier navigation, reduced confusion, faster development

---

---

## 🛠️ Command Reference

**Last Updated**: October 8, 2025  

```bash**Project**: Advanced Snake Game with RL Agents

# Train with UI (recommended)
python training_ui.py

# Train without UI
python train_enhanced.py --episodes 1000

# Play manually
python main.py → Select "Manual"

# Watch trained agent
python main.py → Select "Advanced DQN" → Load Model

# Custom stuck detection
python train_enhanced.py --stuck-sensitivity 8 --stuck-cooldown 400

# Disable stuck detection
python train_enhanced.py --disable-stuck-detection
```

---

## 📊 Expected Results

### Typical Training Timeline
```
Episodes 1-50:     Stage 0 - Learn survival (gradient: bright green)
Episodes 50-200:   Stage 0→1 - Consistent food (gradient: green)
Episodes 200-500:  Stage 1→2 - Better strategy (gradient: light green)
Episodes 500-1000: Stage 2→3 - Advanced tactics (gradient: may turn yellow)
Episodes 1000+:    Fine-tuning (gradient: yellow = done)
```

### Final Performance
- **Stage 2 (50-100 avg):** Decent
- **Stage 3 (100-200 avg):** Good
- **Stage 4 (200+ avg):** Excellent!

---

## 📁 Project Structure

```
advanced_snake/
├── main.py                    # Game entry point
├── training_ui.py             # Training UI (PRIMARY TOOL)
├── enhanced_dqn.py            # Enhanced DQN with curriculum
├── train_enhanced.py          # CLI training script
├── constants.py               # All configuration
│
├── Documentation/
│   ├── README.md              # This file
│   ├── QUICK_START_GUIDE.md
│   ├── COMPLETE_REFERENCE_GUIDE.md
│   └── [Feature guides...]
│
└── models/                    # Saved models
```

---

## 🚀 Quick Decision Guide

```
Want to play? 
  → python main.py

Want to train?
  → python training_ui.py (with UI)
  → python train_enhanced.py (without UI, faster)

Want to understand?
  → Read QUICK_START_GUIDE.md (5 minutes)
  → Read COMPLETE_REFERENCE_GUIDE.md (comprehensive)

Want to know when to stop training?
  → Watch Recent gradient (bottom colored box)
  → Yellow for 200+ episodes = done
```

---

**Happy Training! 🐍🎮🚀**

*Last Updated: October 10, 2025*
