"""
Diagnostic script to analyze training stagnation
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Find the most recent training history
history_files = list(Path("models").glob("*_history.json"))
if not history_files:
    print("No training history files found!")
    exit()

latest_history = max(history_files, key=lambda p: p.stat().st_mtime)
print(f"Analyzing: {latest_history}")

with open(latest_history, 'r') as f:
    data = json.load(f)

scores = data.get('scores', [])
running_avgs = data.get('running_avgs', [])
curriculum_stage = data.get('curriculum_stage', 0)
epsilon = data.get('epsilon', 0)

if len(scores) < 100:
    print("Not enough training data yet!")
    exit()

print(f"\n{'='*70}")
print("TRAINING DIAGNOSTICS")
print(f"{'='*70}")

# Recent performance (last 100 episodes)
recent_scores = scores[-100:]
recent_avg = np.mean(recent_scores)
recent_std = np.std(recent_scores)
recent_min = np.min(recent_scores)
recent_max = np.max(recent_scores)
recent_median = np.median(recent_scores)

print(f"\n📊 LAST 100 EPISODES:")
print(f"  • Average:  {recent_avg:.1f}")
print(f"  • Median:   {recent_median:.1f}")
print(f"  • Std Dev:  {recent_std:.1f}")
print(f"  • Min:      {recent_min:.1f}")
print(f"  • Max:      {recent_max:.1f}")
print(f"  • Variance: {recent_std**2:.1f}")

# Curriculum info
print(f"\n🎓 CURRICULUM STATUS:")
print(f"  • Current Stage: {curriculum_stage}")
print(f"  • Epsilon: {epsilon:.4f}")

thresholds = [20, 50, 100, 200]
if curriculum_stage < len(thresholds):
    next_threshold = thresholds[curriculum_stage]
    min_required_old = next_threshold * 0.5  # Old requirement
    min_required_new = next_threshold * 0.3  # New requirement
    
    print(f"  • Next Threshold: {next_threshold}")
    print(f"  • Min Required (OLD): {min_required_old:.1f} (50% of threshold)")
    print(f"  • Min Required (NEW): {min_required_new:.1f} (30% of threshold)")
    print(f"  • Recent Min Score: {recent_min:.1f}")
    
    if recent_avg >= next_threshold:
        print(f"  • Average Requirement: ✅ MET ({recent_avg:.1f} >= {next_threshold})")
    else:
        print(f"  • Average Requirement: ❌ NOT MET ({recent_avg:.1f} < {next_threshold})")
    
    if recent_min >= min_required_old:
        print(f"  • Min Requirement (OLD): ✅ MET ({recent_min:.1f} >= {min_required_old:.1f})")
    else:
        print(f"  • Min Requirement (OLD): ❌ NOT MET ({recent_min:.1f} < {min_required_old:.1f})")
    
    if recent_min >= min_required_new:
        print(f"  • Min Requirement (NEW): ✅ MET ({recent_min:.1f} >= {min_required_new:.1f})")
    else:
        print(f"  • Min Requirement (NEW): ❌ NOT MET ({recent_min:.1f} < {min_required_new:.1f})")

# Check for stagnation
print(f"\n⚠️  STAGNATION DETECTION:")

# Last 500 episodes
if len(scores) >= 500:
    last_500_avg = np.mean(scores[-500:])
    last_250_avg = np.mean(scores[-250:])
    last_100_avg = np.mean(scores[-100:])
    
    print(f"  • Last 500 eps avg: {last_500_avg:.1f}")
    print(f"  • Last 250 eps avg: {last_250_avg:.1f}")
    print(f"  • Last 100 eps avg: {last_100_avg:.1f}")
    
    improvement_500_to_250 = last_250_avg - last_500_avg
    improvement_250_to_100 = last_100_avg - last_250_avg
    
    print(f"  • Improvement (500→250): {improvement_500_to_250:+.1f}")
    print(f"  • Improvement (250→100): {improvement_250_to_100:+.1f}")
    
    if abs(improvement_250_to_100) < 5:
        print(f"  • Status: ⚠️ STAGNATING (< 5 point improvement)")
    else:
        print(f"  • Status: ✅ Improving")

# Score distribution
print(f"\n📈 SCORE DISTRIBUTION (Last 100):")
bins = [0, 50, 100, 150, 200, 300, 500, 1000]
for i in range(len(bins)-1):
    count = sum(1 for s in recent_scores if bins[i] <= s < bins[i+1])
    percentage = (count / len(recent_scores)) * 100
    bar = '█' * int(percentage / 2)
    print(f"  • {bins[i]:4d}-{bins[i+1]:4d}: {count:3d} episodes ({percentage:5.1f}%) {bar}")

# Recommendations
print(f"\n💡 RECOMMENDATIONS:")

if recent_avg >= 100 and recent_min < 30 and curriculum_stage == 2:
    print("  1. ✅ Curriculum fix applied - minimum requirement lowered to 30%")
    print("     You should advance to Stage 3 soon!")

if epsilon < 0.05:
    print(f"  2. ⚠️ Epsilon very low ({epsilon:.4f}) - limited exploration")
    print("     Stuck detection will boost it automatically after 150 episodes")

if recent_std > 100:
    print(f"  3. ⚠️ High variance ({recent_std:.1f}) - inconsistent performance")
    print("     Model needs more training to stabilize")

print(f"\n{'='*70}\n")

# Optional: Plot if matplotlib available
try:
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Scores over time
    plt.subplot(2, 1, 1)
    plt.plot(scores, alpha=0.3, label='Scores')
    plt.plot(running_avgs, label='Running Average', linewidth=2)
    plt.axhline(y=100, color='r', linestyle='--', label='Stage 2→3 Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Training Progress - Stage {curriculum_stage}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Recent score distribution
    plt.subplot(2, 1, 2)
    plt.hist(recent_scores, bins=20, edgecolor='black')
    plt.axvline(x=recent_avg, color='r', linestyle='--', linewidth=2, label=f'Avg: {recent_avg:.1f}')
    plt.axvline(x=100, color='g', linestyle='--', linewidth=2, label='Threshold: 100')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution (Last 100 Episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_diagnosis.png', dpi=150)
    print("📊 Graph saved to: models/training_diagnosis.png")
    
except Exception as e:
    print(f"Could not generate graph: {e}")
