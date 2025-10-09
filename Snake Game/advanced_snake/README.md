# Advanced Snake Game - Clean Project Structure

## 📁 Essential Files

### Core Game Components
- **`constants.py`** - Game configuration and hyperparameters
- **`game_engine.py`** - Snake game logic and mechanics
- **`ui.py`** - User interface components

### Algorithm Implementations
- **`algorithms.py`** - Traditional algorithms (A*, BFS, Hamiltonian)
- **`q_learning.py`** - Q-Learning (Tabular) agent
- **`advanced_dqn.py`** - Deep Q-Network agent
- **`enhanced_dqn.py`** - Enhanced DQN with curriculum learning
- **`gpu_utils.py`** - GPU acceleration utilities

### Main Entry Points
- **`main.py`** - Main game interface (play/watch AI)
- **`training_ui.py`** - **PRIMARY TRAINING TOOL** - Comprehensive training interface with:
  - Model selection (Q-Learning, Original DQN, Enhanced DQN)
  - Real-time training graphs
  - Hyperparameter controls
  - Model management

### Training Scripts (Called by training_ui.py)
- **`train_qlearning.py`** - Q-Learning training backend
- **`train_enhanced.py`** - Enhanced DQN training backend

### Model Storage
- **`models/`** - Saved models and training statistics
  - `snake_qlearning_model.pkl` - Q-Learning model
  - `snake_enhanced_dqn*.pth` - Enhanced DQN models
  - `*_history.json` - Training history files
  - `qlearning_training_stats.json` - Q-Learning stats

### Documentation
- **`QUICK_REFERENCE.md`** - Quick reference guide

## 🗑️ Removed Non-Essential Files

### Debug/Test Scripts (Deleted)
- ❌ `check_cuda.py` - One-time CUDA setup check
- ❌ `debug_model.py` - Model debugging tool
- ❌ `test_batch_size.py` - GPU memory testing
- ❌ `test_enhanced.py` - Enhanced DQN testing
- ❌ `demo_new_graphs.py` - Graph visualization demo
- ❌ `diagnose_stagnation.py` - Training analysis tool
- ❌ `compare_training.py` - Model comparison tool
- ❌ `check_model_state.py` - Model state inspector

### Redundant Training Scripts (Deleted - Replaced by training_ui.py)
- ❌ `training.py` - Old training script
- ❌ `dqn_training.py` - Old DQN training
- ❌ `headless_training.py` - Replaced by training UI

### Batch Files (Deleted)
- ❌ `run_headless_training.bat`
- ❌ `run_training_ui.bat`

### Analysis Documentation (Deleted)
- ❌ `OPTIMAL_HYPERPARAMETERS_ANALYSIS.md`
- ❌ `STAGE_2_ANALYSIS.md`

### Old Model Checkpoints (Deleted)
- ❌ `snake_dqn_model_ep*_interrupted_*.json` - Interrupted training files

## 🚀 How to Use

### Play the Game
```bash
python main.py
```
**Main Menu Options:**
- Select game mode (Manual, A*, BFS, Hamiltonian, Q-Learning, Enhanced DQN)
- Adjust speed
- Browse and load trained models
- Start playing!

**Note:** Training options have been removed from main.py. Use `training_ui.py` for all training needs.

### Train Models (Recommended)
```bash
python training_ui.py
```
The training UI provides:
- Model type selection (Q-Learning, Original DQN, Enhanced DQN)
- Hyperparameter adjustment
- Real-time performance graphs
- Model save/load management
- Training progress tracking

### Quick Training (Command Line)
```bash
# Q-Learning
python train_qlearning.py --episodes 1000 --learning-rate 0.1 --batch-size 64

# Enhanced DQN
python train_enhanced.py --episodes 2000 --batch-size 512 --save-interval 200
```

## 📊 Model Comparison

| Model | State Space | Action Space | Training Time | Performance |
|-------|-------------|--------------|---------------|-------------|
| **Q-Learning** | 11 features (discrete) | 4 absolute (UP/DOWN/LEFT/RIGHT) | Fast (~50-100 eps/sec) | Excellent for Snake |
| **Original DQN** | 11 features | 3 relative (Turn Left/Straight/Right) | Medium (~10-30 eps/sec) | Moderate |
| **Enhanced DQN** | 34 features | 3 relative | Slower (~10-20 eps/sec) | Best with curriculum |

## 💡 Key Insights

### Why Q-Learning Outperforms DQN for Snake
1. **Absolute Actions**: Q-Learning uses direct spatial actions (UP/DOWN/LEFT/RIGHT)
2. **Perfect Memory**: Tabular approach stores exact Q-values for each state
3. **Small State Space**: ~500-800 states for typical Snake gameplay
4. **Fast Training**: Reaches competence in ~1000 episodes

### When to Use Enhanced DQN
- Larger, more complex environments
- Need for generalization beyond seen states
- When state space is too large for tabular methods
- Continuous or high-dimensional state spaces

## 🔧 Project Cleanup Summary

**Removed**: 15+ non-essential files (debug scripts, redundant trainers, old docs)  
**Result**: Clean, maintainable codebase focused on core functionality  
**Benefit**: Easier navigation, reduced confusion, faster development

---

**Last Updated**: October 8, 2025  
**Project**: Advanced Snake Game with RL Agents
