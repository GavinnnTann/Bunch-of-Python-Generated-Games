# Stable DQN - Production-Ready Deep Q-Network

## Overview

**Stable DQN** is a carefully designed DQN agent built to **reliably match or outperform tabular Q-Learning** on the Snake game through conservative hyperparameters, proper data handling, and comprehensive monitoring.

**Goal:** Overcome DQN's typical instability and approximation errors to achieve consistent, reproducible performance.

---

## Key Differentiators from Enhanced DQN

| Feature | Enhanced DQN | Stable DQN |
|---------|--------------|------------|
| **Focus** | Max performance through curriculum & A* | Reliability & data efficiency |
| **Architecture** | 34 → 256 → 128 → 4 | 34 → 128 → 128 → Dueling(64) |
| **Training** | Episodes-based | **Steps-based** (proper budget) |
| **Input** | Raw features | ✅ **Z-score normalized** |
| **Replay** | PER (α=0.4) | ✅ **PER (α=0.6, β annealing)** |
| **Target Updates** | Hard (every 1000 steps) | ✅ **Soft (τ=0.005)** |
| **Warmup** | None | ✅ **20k random steps** |
| **Monitoring** | Basic | ✅ **Comprehensive diagnostics** |
| **Grad Clipping** | 10.0 | ✅ **10.0 (monitored pre/post)** |
| **Learning Rate** | 1e-3 | ✅ **3e-4 (conservative)** |
| **Epsilon Decay** | Fast (995 per episode) | ✅ **Linear (1.0 → 0.05 over 200k steps)** |

---

## Architecture

```
Input (34 normalized features)
    ↓
Dense(128, ReLU)  [Glorot init]
    ↓
Dense(128, ReLU)  [Glorot init]
    ↓
Split into Dueling Streams:
    ├─ Value Stream: Dense(64, ReLU) → Dense(1)
    └─ Advantage Stream: Dense(64, ReLU) → Dense(3)
    ↓
Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
    ↓
Output: Q-Values for [Turn Left, Straight, Turn Right]
```

### Why This Architecture?

1. **Smaller Hidden Layers (128 vs 256):**
   - Reduces overfitting risk
   - Faster training per step
   - Sufficient capacity for 34 features

2. **Dueling Network:**
   - Separates "how good is this state" from "how good is each action"
   - Faster learning for states where action choice doesn't matter much

3. **3 Actions (Relative):**
   - Turn Left, Straight, Turn Right
   - Relative to current direction
   - Matches Enhanced DQN (allows comparison)

---

## Input Normalization

**Critical Feature:** Fixed Z-score normalization prevents distribution shift.

### Process

1. **Warmup Phase (20k steps):**
   - Collect transitions using random policy
   - Compute mean (μ) and std (σ) for each of 34 features
   - Store these statistics permanently

2. **Training & Inference:**
   - Normalize every state: `z = (x - μ) / σ`
   - Use SAME statistics (no re-fitting)

3. **Why This Matters:**
   ```
   Without normalization:
   - Food distance: ~0-20
   - Danger flags: 0-1
   - Body proximity: 0-1
   → Network struggles with different scales
   
   With normalization:
   - All features: mean=0, std=1
   → Network learns faster, more stable
   ```

---

## Prioritized Experience Replay (PER)

**Enhanced** implementation with proper β annealing.

### Parameters

- **Buffer Size:** 500,000 transitions
- **Priority Exponent (α):** 0.6
  - How much prioritization (0 = uniform, 1 = full priority)
  - 0.6 balances diversity and important transitions
  
- **Importance Sampling (β):** 0.4 → 1.0
  - Starts at 0.4 (allows prioritization bias early)
  - Anneals to 1.0 (fully corrects bias later)
  - Increment: +0.001 per batch

### How It Works

1. **Initial Priority:** New experiences get max priority
2. **TD-Error Priority:** `priority = (|TD error| + ε)^α`
3. **Sampling:** Higher priority = higher selection probability
4. **Importance Weights:** Correct bias: `w = (N × P(i))^(-β)`

### Benefits

- Learns more from surprising transitions
- Faster convergence on critical states
- Better sample efficiency

---

## Double DQN

Reduces overestimation bias in Q-value targets.

### Standard DQN (Overestimates)
```python
target = r + γ × max_a Q_target(s', a)
# Problem: max operation picks highest value (includes noise)
```

### Double DQN (Stable)
```python
# Use online network to SELECT action
best_action = argmax_a Q_online(s', a)

# Use target network to EVALUATE action
target = r + γ × Q_target(s', best_action)

# Benefit: Separates selection from evaluation → less overestimation
```

---

## Soft Target Updates

**Polyak averaging:** Slowly blend online weights into target network.

### Hard Updates (Enhanced DQN)
```python
Every 1000 steps:
    target_net ← online_net  # Abrupt change
```

### Soft Updates (Stable DQN)
```python
Every gradient step:
    target_params ← τ × online_params + (1-τ) × target_params
    # τ = 0.005 → smooth, continuous updates
```

### Why Soft is Better

- **Smoother targets:** Less volatility in TD errors
- **Faster updates:** No waiting 1000 steps
- **More stable:** Gradual tracking prevents sudden target shifts

---

## Training Protocol

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 3e-4 | Conservative (vs Enhanced's 1e-3) |
| **Batch Size** | 128 | Balance speed/stability |
| **Gamma (γ)** | 0.99 | Standard discount |
| **Gradient Clip** | 10.0 | Prevent exploding gradients |
| **Loss Function** | Smooth L1 (Huber, δ=1.0) | Robust to outliers |
| **Optimizer** | Adam (β1=0.9, β2=0.999) | Standard |

### Exploration Schedule

```
Epsilon-Greedy:
- Start: ε = 1.0 (100% random)
- End: ε = 0.05 (5% random)
- Decay: Linear over 200,000 steps
- Rationale: Slow decay ensures thorough exploration
```

### Training Budget

```
Total Steps: 5,000,000 - 10,000,000
- Warmup: 20,000 (random policy, normalization)
- Training: Remaining steps (1 gradient update per step)

Evaluation: Every 50,000 steps
- Run 20 episodes at ε=0.05
- Track mean, median, max score
```

---

## Monitoring & Diagnostics

**Comprehensive logging** to detect issues early.

### Per-Batch Metrics

1. **TD Error:**
   - Mean, 95th percentile
   - Watch for: Sudden spikes (target network desync)

2. **Loss (Huber):**
   - Should decrease over time
   - Watch for: Divergence, NaNs

3. **Q-Values:**
   - Mean, std, max
   - Watch for: Exploding Qs (overestimation)

4. **Gradient Norms:**
   - Pre-clip & post-clip
   - Watch for: Consistently hitting clip limit

5. **Target Lag:**
   - `mean |Q_online - Q_target|`
   - Watch for: Large lag (targets too slow)

### Per-Evaluation (50k steps)

1. **Score Distribution:**
   - Mean, median, max, std
   - Track improvement over time

2. **Buffer Health:**
   - Fill percentage
   - PER sampling diversity

3. **Epsilon Value:**
   - Confirm decay schedule

---

## Success Criteria

### Early (2-3M steps)

- ✅ **Median eval score ≥ 20**
- ✅ **Stable loss** (not diverging)
- ✅ **Q-values in reasonable range** (not exploding)

### Mid (5M steps)

- ✅ **Median eval score ≥ 25**
- ✅ **Surpassing 20-point plateau**
- ✅ **Consistent improvement**

### Final (10M steps)

- ✅ **Median eval ≥ Q-Learning median**
- ✅ **Low variance** (IQR stable)
- ✅ **Reproducible** (same performance across runs)

---

## Common Issues & Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **Plateau at ~20** | Epsilon decayed too fast | Extend to 400k steps |
| **Diverging loss** | LR too high | Reduce to 1e-4 |
| **Exploding Q-values** | Overestimation | Confirm Double DQN active, reduce τ |
| **Slow learning** | Not enough exploration | Increase batch to 256, try n-step=3 |
| **High variance** | PER bias too strong | Ensure β annealing to 1.0 |

---

## Usage

### Training (Command Line)

```bash
# Default (5M steps)
python train_stable_dqn.py

# Custom budget
python train_stable_dqn.py --steps 10000000

# Faster evaluation cycles
python train_stable_dqn.py --eval-interval 25000

# More frequent checkpoints
python train_stable_dqn.py --save-interval 50000
```

### Training (UI)

1. Launch training UI: `python training_ui.py`
2. Select **"Stable DQN (34 features)"**
3. Set episodes (will convert to ~50 steps/episode)
4. Click **Start Training**

**Note:** Episodes in UI are converted to steps (Episodes × 50)

### Model Files

- **Main checkpoint:** `models/snake_stable_dqn.pth`
- **Best model:** `models/snake_stable_dqn_best.pth`
- **Normalizer stats:** `models/snake_stable_dqn_normalizer.npz`
- **Training history:** `models/snake_stable_dqn_history.json`
- **Stats log:** `models/snake_stable_dqn_stats.json`

---

## Comparison: Q-Learning vs Stable DQN

| Metric | Q-Learning | Stable DQN |
|--------|------------|------------|
| **State Space** | 11 binary → 2K states | 34 continuous → infinite |
| **Action Space** | 4 absolute (UP/DOWN/LEFT/RIGHT) | 3 relative (L/S/R) |
| **Memory** | Perfect (tabular) | Approximate (neural net) |
| **Training Speed** | ⚡ ~100 eps/sec (CPU) | 🐌 ~10 eps/sec (GPU) |
| **Sample Efficiency** | 1000 episodes to competence | 2-3M steps (~40-60k episodes) |
| **Generalization** | None (only seen states) | Good (unseen states) |
| **Scalability** | ❌ Limited to small state space | ✅ Scales to large/continuous |
| **Best for Snake?** | ✅ Absolute actions easier | ⚠️ Relative actions harder |

### When to Use Each

**Use Q-Learning when:**
- State space is small (<50k states)
- Features are discrete/binary
- Absolute actions available
- Want fast training
- Need interpretability (inspect Q-table)

**Use Stable DQN when:**
- State space is large or continuous
- Need generalization to unseen states
- Testing DQN implementation
- Want to scale to harder tasks
- Have GPU available

---

## Implementation Details

### State Representation (34 Features)

Uses same features as Enhanced DQN:

1. **Immediate Danger (3):** Straight, Right, Left
2. **Food Direction (4):** Up, Right, Down, Left
3. **Current Direction (4):** One-hot encoding
4. **Wall Proximity (4):** Distance in each direction
5. **Food Distance (2):** Manhattan, Euclidean
6. **Body Proximity (4):** Closest body segment per direction
7. **Snake Length (1):** Normalized
8. **Available Space (3):** Flood-fill in each direction
9. **Tail Direction (3):** Relative to head
10. **Grid Occupancy (6):** Quadrant densities

### Reward Function

**Identical to Q-Learning:**

```python
if game_over:
    reward = -10  # Death penalty
elif ate_food:
    reward = +10  # Food reward
else:
    reward = 0    # Neutral survival
```

**Why simple rewards?**
- Easier to compare with Q-Learning
- Stable DQN should learn from sparse signals
- No hand-crafted shaping needed

---

## Ablation Studies (To Run)

Test these variations to understand what matters:

1. **Learning Rate:** {1e-4, 3e-4, 1e-3}
2. **Epsilon Decay:** {200k, 400k, 800k steps}
3. **Target Update:** τ {0.002, 0.005, 0.01}
4. **PER vs Uniform:** Compare PER on/off
5. **Network Size:** {64, 128, 256} hidden units
6. **Double DQN:** Compare on/off (expect worse without)
7. **Gradient Clip:** Test removal (expect instability)
8. **N-step:** Try n={1, 3, 5}

---

## Expected Results

### Realistic Expectations

**After 2M steps:**
- Median score: ~15-25
- Max score: ~30-50
- Stable loss, reasonable Q-values

**After 5M steps:**
- Median score: ~25-35
- Max score: ~50-80
- Consistently above plateau

**After 10M steps:**
- Median score: ~30-40
- Max score: ~80-150
- Approaching Q-Learning performance

### Why DQN is Harder

1. **Relative Actions:** Harder to learn than absolute
2. **Approximation Error:** Neural net vs perfect table
3. **Moving Targets:** Target network changes over time
4. **Exploration:** Must balance exploration/exploitation

---

## Future Improvements

If Stable DQN still struggles:

1. **Rainbow DQN:** Add n-step, noisy nets, distributional RL
2. **Absolute Actions:** Switch to UP/DOWN/LEFT/RIGHT
3. **Auxiliary Tasks:** Add value prediction, reward prediction
4. **Curriculum:** Start with small grid, scale up
5. **Behavioral Cloning:** Pre-train on A* trajectories

---

## Files

- `stable_dqn.py` - Agent implementation
- `train_stable_dqn.py` - Training script
- `training_ui.py` - Integrated into GUI
- `STABLE_DQN_GUIDE.md` - This file

---

**Last Updated:** October 8, 2025  
**Status:** Ready for training  
**Goal:** Reliable DQN that meets/exceeds Q-Learning performance
