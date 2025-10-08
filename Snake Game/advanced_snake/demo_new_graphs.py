"""
Quick script to test the new training graphs with sample data
"""
import numpy as np
import matplotlib.pyplot as plt

# Generate sample training data
episodes = 1500
scores = []
epsilon_start = 1.0
epsilon = epsilon_start
epsilon_decay = 0.997
curriculum_advancements = []  # Track curriculum stage changes: [(episode, old_stage, new_stage), ...]

# Simulate realistic training progression
for ep in range(episodes):
    # Score increases with experience but has variance
    base_score = min(250, ep / 5)  # Slow improvement
    
    # Add stage-based jumps and record advancements
    if ep > 200 and ep < 210:
        if ep == 205:  # Advancement to stage 1
            curriculum_advancements.append((ep, 0, 1))
        base_score += 20
    if ep > 600 and ep < 610:
        if ep == 605:  # Advancement to stage 2
            curriculum_advancements.append((ep, 1, 2))
        base_score += 50
    if ep > 1000 and ep < 1010:
        if ep == 1005:  # Advancement to stage 3
            curriculum_advancements.append((ep, 2, 3))
        base_score += 30
    
    # Add variance (decreases as epsilon decreases)
    variance = 50 * epsilon
    score = max(0, base_score + np.random.normal(0, variance))
    scores.append(score)
    
    # Epsilon decay with stage minimums
    avg = np.mean(scores[max(0, ep-100):ep+1])
    if avg >= 200:
        stage_min = 0.01
    elif avg >= 100:
        stage_min = 0.02
    elif avg >= 50:
        stage_min = 0.04
    elif avg >= 20:
        stage_min = 0.05
    else:
        stage_min = 0.10
    
    if epsilon > stage_min:
        epsilon *= epsilon_decay
    else:
        epsilon = stage_min

# Calculate running averages
window = 100
running_avgs = [sum(scores[max(0, i-window):i+1]) / min(window, i+1) for i in range(len(scores))]

print("Sample data generated!")
print(f"Episodes: {len(scores)}")
print(f"Final score: {scores[-1]:.1f}")
print(f"Final avg: {running_avgs[-1]:.1f}")
print(f"Best score: {max(scores):.1f}")
print(f"Final epsilon: {epsilon:.4f}")

# Create the 2x2 graph layout (matching new training_ui.py)
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Training Performance Analysis - Sample Data', fontsize=18, fontweight='bold')

episodes_list = list(range(1, len(scores) + 1))

# ===== GRAPH 1: Score Progression =====
axs[0, 0].set_title('Score Progression', fontsize=14, fontweight='bold')
axs[0, 0].set_xlabel('Episode', fontsize=12)
axs[0, 0].set_ylabel('Score', fontsize=12)
axs[0, 0].plot(episodes_list, scores, 'b-', linewidth=0.8, alpha=0.3, label='Score')
axs[0, 0].plot(episodes_list, running_avgs, 'r-', linewidth=2.5, label='Avg (100)')

# Add curriculum thresholds
thresholds = [20, 50, 100, 200]
colors = ['green', 'orange', 'purple', 'red']
labels = ['Stage 0→1', 'Stage 1→2', 'Stage 2→3', 'Stage 3→4']
for threshold, color, label in zip(thresholds, colors, labels):
    axs[0, 0].axhline(y=threshold, color=color, linestyle='--', alpha=0.6, linewidth=1.5, label=label)

# ===== NEW: Highlight curriculum advancements =====
for episode, old_stage, new_stage in curriculum_advancements:
    score_at_advancement = scores[episode - 1]
    
    # Add vertical line at advancement
    axs[0, 0].axvline(x=episode, color='cyan', linestyle=':', linewidth=2, alpha=0.8, zorder=5)
    
    # Add star marker
    axs[0, 0].scatter([episode], [score_at_advancement], 
                      marker='*', s=400, color='cyan', 
                      edgecolors='darkblue', linewidths=2, zorder=15)
    
    # Add annotation
    axs[0, 0].annotate(f'Stage {new_stage}', 
                       xy=(episode, score_at_advancement),
                       xytext=(episode + 50, score_at_advancement + 20),
                       arrowprops=dict(facecolor='cyan', shrink=0.05, width=1.5),
                       fontsize=10, fontweight='bold', color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.7))

max_score = max(scores)
max_idx = scores.index(max_score)
axs[0, 0].scatter([episodes_list[max_idx]], [max_score], color='gold', s=100, zorder=15, edgecolors='black', linewidths=2)
axs[0, 0].annotate(f'Best: {max_score:.0f}', xy=(episodes_list[max_idx], max_score),
                   xytext=(episodes_list[max_idx] + 100, max_score),
                   arrowprops=dict(facecolor='gold', shrink=0.05, width=1.5),
                   fontsize=11, fontweight='bold')
axs[0, 0].legend(loc='upper left', fontsize=9, ncol=2)
axs[0, 0].grid(True, alpha=0.3)

# ===== GRAPH 2: Epsilon Decay =====
axs[0, 1].set_title('Exploration Rate (Epsilon)', fontsize=14, fontweight='bold')
axs[0, 1].set_xlabel('Episode', fontsize=12)
axs[0, 1].set_ylabel('Epsilon', fontsize=12)

# Calculate actual epsilon values
epsilon_values = []
current_eps = epsilon_start
for ep in range(len(scores)):
    avg = running_avgs[ep]
    if avg >= 200:
        stage_min = 0.01
    elif avg >= 100:
        stage_min = 0.02
    elif avg >= 50:
        stage_min = 0.04
    elif avg >= 20:
        stage_min = 0.05
    else:
        stage_min = 0.10
    
    if current_eps > stage_min:
        current_eps *= epsilon_decay
    else:
        current_eps = stage_min
    epsilon_values.append(current_eps)

axs[0, 1].plot(episodes_list, epsilon_values, 'purple', linewidth=2.5, label='Epsilon')

# Add stage minimums
stage_minimums = {0: 0.10, 1: 0.05, 2: 0.04, 3: 0.02, 4: 0.01}
for stage, min_eps in stage_minimums.items():
    axs[0, 1].axhline(y=min_eps, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    axs[0, 1].text(len(episodes_list) * 0.02, min_eps + 0.01, f'Stage {stage} min', fontsize=8, color='gray')

current_eps_val = epsilon_values[-1]
axs[0, 1].scatter([len(epsilon_values)], [current_eps_val], color='red', s=100, zorder=10, edgecolors='black', linewidths=2)
axs[0, 1].annotate(f'{current_eps_val:.4f}', xy=(len(epsilon_values), current_eps_val),
                   xytext=(len(epsilon_values) * 0.85, current_eps_val + 0.05),
                   fontsize=11, fontweight='bold', color='red')
axs[0, 1].set_ylim(-0.05, 1.05)
axs[0, 1].legend(loc='upper right', fontsize=10)
axs[0, 1].grid(True, alpha=0.3)

# ===== GRAPH 3: Episode Duration =====
axs[1, 0].set_title('Episode Duration (Steps)', fontsize=14, fontweight='bold')
axs[1, 0].set_xlabel('Episode', fontsize=12)
axs[1, 0].set_ylabel('Steps Survived', fontsize=12)

estimated_steps = [max(20, int(s * 2.5 + np.random.randint(-10, 10))) for s in scores]

window = 50
steps_avg = [sum(estimated_steps[max(0, i-window):i+1]) / min(window, i+1) for i in range(len(estimated_steps))]
axs[1, 0].plot(episodes_list, steps_avg, 'orange', linewidth=2.5, label=f'Avg ({window} eps)', zorder=10)
axs[1, 0].scatter(episodes_list, estimated_steps, c='blue', alpha=0.2, s=10, label='Steps')

# Trend line
z = np.polyfit(episodes_list, estimated_steps, 2)
p = np.poly1d(z)
axs[1, 0].plot(episodes_list, p(episodes_list), "g--", alpha=0.7, linewidth=2, label='Trend')
axs[1, 0].legend(loc='upper left', fontsize=10)
axs[1, 0].grid(True, alpha=0.3)

# ===== GRAPH 4: Score Distribution =====
axs[1, 1].set_title('Score Distribution (Last 100)', fontsize=14, fontweight='bold')
axs[1, 1].set_xlabel('Score', fontsize=12)
axs[1, 1].set_ylabel('Frequency', fontsize=12)

recent_scores = scores[-100:]
n, bins, patches = axs[1, 1].hist(recent_scores, bins=20, edgecolor='black', alpha=0.7)

# Color code bins
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center < 20:
        patch.set_facecolor('lightcoral')
    elif bin_center < 50:
        patch.set_facecolor('lightyellow')
    elif bin_center < 100:
        patch.set_facecolor('lightgreen')
    elif bin_center < 200:
        patch.set_facecolor('lightblue')
    else:
        patch.set_facecolor('mediumpurple')

mean_score = np.mean(recent_scores)
median_score = np.median(recent_scores)
std_score = np.std(recent_scores)

axs[1, 1].axvline(mean_score, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_score:.1f}')
axs[1, 1].axvline(median_score, color='green', linestyle='--', linewidth=2.5, label=f'Median: {median_score:.1f}')

stats_text = f'μ={mean_score:.1f}\nσ={std_score:.1f}\nMin={min(recent_scores):.0f}\nMax={max(recent_scores):.0f}'
axs[1, 1].text(0.98, 0.97, stats_text, transform=axs[1, 1].transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axs[1, 1].legend(loc='upper left', fontsize=10)
axs[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.93)

# Save the figure
plt.savefig('models/sample_training_graphs.png', dpi=150, bbox_inches='tight')
print("\n✅ Sample graphs saved to: models/sample_training_graphs.png")
print("\nThis is what you'll see in the Training UI!")
print("\nKey observations from sample data:")
print(f"  • Started at stage 0, progressed through stages 1-3")
print(f"  • Curriculum advancements marked with cyan stars:")
for ep, old, new in curriculum_advancements:
    print(f"    - Episode {ep}: Stage {old} → {new}")
print(f"  • Epsilon decayed from 1.0 → {epsilon_values[-1]:.4f}")
print(f"  • Score variance decreased as training progressed")
print(f"  • Steps increased proportionally with scores")

plt.show()
