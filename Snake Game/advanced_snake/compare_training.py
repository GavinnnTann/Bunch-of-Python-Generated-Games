"""
Compare training results between original and enhanced DQN models.

Usage:
    python compare_training.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from constants import QMODEL_DIR


def load_history(filename):
    """Load training history from JSON file."""
    path = os.path.join(QMODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def plot_comparison():
    """Create comprehensive comparison plots."""
    # Load histories
    original = load_history("snake_dqn_model_history.json")
    enhanced = load_history("snake_enhanced_dqn_history.json")
    
    if original is None and enhanced is None:
        print("❌ No training histories found!")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('DQN Training Comparison: Original vs Enhanced', fontsize=16, fontweight='bold')
    
    # 1. Running Average Scores
    ax1 = plt.subplot(2, 3, 1)
    if original:
        ax1.plot(original['running_avgs'], label='Original DQN', linewidth=2, alpha=0.8)
    if enhanced:
        ax1.plot(enhanced['running_avgs'], label='Enhanced DQN', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Running Average Score (100 ep)')
    ax1.set_title('Learning Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Raw Scores (scatter with trend)
    ax2 = plt.subplot(2, 3, 2)
    if original:
        ax2.scatter(range(len(original['scores'])), original['scores'], 
                   alpha=0.3, s=10, label='Original DQN')
    if enhanced:
        ax2.scatter(range(len(enhanced['scores'])), enhanced['scores'],
                   alpha=0.3, s=10, label='Enhanced DQN')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Score')
    ax2.set_title('Raw Scores (with variance)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Score Distribution (histogram)
    ax3 = plt.subplot(2, 3, 3)
    if original:
        ax3.hist(original['scores'], bins=30, alpha=0.5, label='Original DQN', edgecolor='black')
    if enhanced:
        ax3.hist(enhanced['scores'], bins=30, alpha=0.5, label='Enhanced DQN', edgecolor='black')
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Table
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    metrics = []
    headers = ['Metric', 'Original DQN', 'Enhanced DQN', 'Improvement']
    
    if original and enhanced:
        orig_avg = original['latest_avg_score']
        enh_avg = enhanced['latest_avg_score']
        avg_improvement = ((enh_avg - orig_avg) / orig_avg * 100) if orig_avg > 0 else 0
        
        orig_best = original['best_score']
        enh_best = enhanced['best_score']
        best_improvement = ((enh_best - orig_best) / orig_best * 100) if orig_best > 0 else 0
        
        orig_var = np.var(original['running_avgs'][-100:])
        enh_var = np.var(enhanced['running_avgs'][-100:]) if len(enhanced['running_avgs']) >= 100 else 0
        
        metrics = [
            ['Avg Score (final)', f'{orig_avg:.1f}', f'{enh_avg:.1f}', f'{avg_improvement:+.1f}%'],
            ['Best Score', f'{orig_best:.1f}', f'{enh_best:.1f}', f'{best_improvement:+.1f}%'],
            ['Episodes Trained', str(original['episodes_completed']), 
             str(enhanced.get('episodes_completed', 0)), '-'],
            ['Variance (last 100)', f'{orig_var:.1f}', f'{enh_var:.1f}', 
             f'{((orig_var - enh_var)/orig_var*100):+.1f}%' if orig_var > 0 else '-'],
            ['State Features', '11', enhanced.get('state_features', 31), '+181%']
        ]
    elif original:
        metrics = [
            ['Avg Score', f"{original['latest_avg_score']:.1f}", '-', '-'],
            ['Best Score', f"{original['best_score']:.1f}", '-', '-'],
            ['Episodes', str(original['episodes_completed']), '-', '-']
        ]
    elif enhanced:
        metrics = [
            ['Avg Score', '-', f"{enhanced['latest_avg_score']:.1f}", '-'],
            ['Best Score', '-', f"{enhanced['best_score']:.1f}", '-'],
            ['Episodes', '-', str(enhanced.get('episodes_completed', 0)), '-']
        ]
    
    table = ax4.table(cellText=metrics, colLabels=headers, loc='center',
                      cellLoc='center', colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 5. Learning Rate (improvement per episode)
    ax5 = plt.subplot(2, 3, 5)
    if original:
        orig_improvement = np.diff(original['running_avgs'])
        ax5.plot(orig_improvement, label='Original DQN', alpha=0.7, linewidth=1.5)
    if enhanced:
        enh_improvement = np.diff(enhanced['running_avgs'])
        ax5.plot(enh_improvement, label='Enhanced DQN', alpha=0.7, linewidth=1.5)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Change in Running Average')
    ax5.set_title('Learning Rate (episode-to-episode improvement)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Milestone Achievement
    ax6 = plt.subplot(2, 3, 6)
    milestones = [50, 100, 150, 200, 250, 300, 350, 400]
    
    def episodes_to_milestone(running_avgs, milestone):
        """Find first episode where running avg >= milestone."""
        for i, avg in enumerate(running_avgs):
            if avg >= milestone:
                return i + 1
        return None
    
    orig_milestones = []
    enh_milestones = []
    
    if original:
        orig_milestones = [episodes_to_milestone(original['running_avgs'], m) for m in milestones]
    if enhanced:
        enh_milestones = [episodes_to_milestone(enhanced['running_avgs'], m) for m in milestones]
    
    x = np.arange(len(milestones))
    width = 0.35
    
    if orig_milestones and any(orig_milestones):
        orig_y = [m if m else 0 for m in orig_milestones]
        ax6.bar(x - width/2, orig_y, width, label='Original DQN', alpha=0.8)
    
    if enh_milestones and any(enh_milestones):
        enh_y = [m if m else 0 for m in enh_milestones]
        ax6.bar(x + width/2, enh_y, width, label='Enhanced DQN', alpha=0.8)
    
    ax6.set_xlabel('Score Milestone')
    ax6.set_ylabel('Episodes to Achieve')
    ax6.set_title('Milestone Achievement Speed')
    ax6.set_xticks(x)
    ax6.set_xticklabels(milestones)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(QMODEL_DIR, "training_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {save_path}")
    
    plt.show()


def print_summary():
    """Print text summary of comparison."""
    original = load_history("snake_dqn_model_history.json")
    enhanced = load_history("snake_enhanced_dqn_history.json")
    
    print("\n" + "="*70)
    print("📊 TRAINING COMPARISON SUMMARY")
    print("="*70)
    
    if original:
        print("\n🔵 ORIGINAL DQN:")
        print(f"   Episodes: {original['episodes_completed']}")
        print(f"   Best Score: {original['best_score']:.1f}")
        print(f"   Final Avg: {original['latest_avg_score']:.1f}")
        print(f"   Training Time: {original.get('training_time', 0)/60:.1f} minutes")
        print(f"   State Features: 11")
    
    if enhanced:
        print("\n🟢 ENHANCED DQN:")
        print(f"   Episodes: {enhanced.get('episodes_completed', 0)}")
        print(f"   Best Score: {enhanced['best_score']:.1f}")
        print(f"   Final Avg: {enhanced['latest_avg_score']:.1f}")
        print(f"   Training Time: {enhanced.get('training_time', 0)/60:.1f} minutes")
        print(f"   State Features: {enhanced.get('state_features', 31)}")
        print(f"   Curriculum Stage: {enhanced.get('curriculum_stage', 0)}/4")
        print(f"   Model Type: {enhanced.get('model_type', 'Enhanced DQN')}")
    
    if original and enhanced:
        print("\n📈 IMPROVEMENTS:")
        avg_diff = enhanced['latest_avg_score'] - original['latest_avg_score']
        avg_pct = (avg_diff / original['latest_avg_score'] * 100) if original['latest_avg_score'] > 0 else 0
        print(f"   Average Score: {avg_diff:+.1f} ({avg_pct:+.1f}%)")
        
        best_diff = enhanced['best_score'] - original['best_score']
        best_pct = (best_diff / original['best_score'] * 100) if original['best_score'] > 0 else 0
        print(f"   Best Score: {best_diff:+.1f} ({best_pct:+.1f}%)")
        
        orig_var = np.var(original['scores'])
        enh_var = np.var(enhanced['scores'])
        var_diff = orig_var - enh_var
        print(f"   Variance Reduction: {var_diff:+.1f} (lower is better)")
        
        # Calculate stability
        orig_stability = 1 - (np.std(original['running_avgs'][-50:]) / np.mean(original['running_avgs'][-50:]))
        enh_stability = 1 - (np.std(enhanced['running_avgs'][-50:]) / np.mean(enhanced['running_avgs'][-50:]))
        print(f"   Stability Score: {enh_stability:.3f} vs {orig_stability:.3f} (closer to 1 is better)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_summary()
    plot_comparison()
