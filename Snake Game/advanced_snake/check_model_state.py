"""
Model State Diagnostic Tool
============================
Check the current state of your trained Enhanced DQN model.

This script inspects:
- Epsilon value (exploration rate)
- Curriculum stage
- A* guidance probability
- Training episodes completed
- Best score achieved
- Model performance metrics

Usage:
    python check_model_state.py

Or to check a specific model file:
    python check_model_state.py models/snake_enhanced_dqn_ep500.pth
"""

import sys
import os
import torch
import json
import numpy as np
from datetime import datetime

def check_model_state(model_path='models/snake_enhanced_dqn.pth', history_path='models/snake_enhanced_dqn_history.json'):
    """Check and display model state information."""
    
    print("=" * 70)
    print("ENHANCED DQN MODEL STATE DIAGNOSTIC")
    print("=" * 70)
    
    # Fix paths if running from parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(model_path):
        # Try multiple locations
        if not os.path.exists(model_path):
            # Try in script directory
            alt_path = os.path.join(script_dir, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            # Try in Snake Game/advanced_snake/models
            elif 'Snake Game' not in os.getcwd():
                alt_path = os.path.join(os.getcwd(), 'Snake Game', 'advanced_snake', model_path)
                if os.path.exists(alt_path):
                    model_path = alt_path
    
    if not os.path.isabs(history_path):
        if not os.path.exists(history_path):
            alt_path = os.path.join(script_dir, history_path)
            if os.path.exists(alt_path):
                history_path = alt_path
            elif 'Snake Game' not in os.getcwd():
                alt_path = os.path.join(os.getcwd(), 'Snake Game', 'advanced_snake', history_path)
                if os.path.exists(alt_path):
                    history_path = alt_path
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model file not found: {model_path}")
        print("\nAvailable models:")
        # Check multiple locations
        search_dirs = [
            'models',
            os.path.join(script_dir, 'models'),
            os.path.join(os.getcwd(), 'Snake Game', 'advanced_snake', 'models')
        ]
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print(f"\nIn {search_dir}:")
                for file in os.listdir(search_dir):
                    if file.endswith('.pth'):
                        print(f"  • {file}")
        return
    
    try:
        # Load model checkpoint
        print(f"\n📂 Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract information
        if isinstance(checkpoint, dict):
            epsilon = checkpoint.get('epsilon', 'Unknown')
            episodes = checkpoint.get('episodes', checkpoint.get('episodes_completed', 'Unknown'))
            best_score = checkpoint.get('best_score', 'Unknown')
            curriculum_stage = checkpoint.get('curriculum_stage', 'Unknown')
            
            # Try to infer A* probability from curriculum stage
            if curriculum_stage != 'Unknown':
                astar_probs = {0: 0.5, 1: 0.35, 2: 0.20, 3: 0.10, 4: 0.0}
                astar_prob = astar_probs.get(curriculum_stage, 'Unknown')
            else:
                astar_prob = 'Unknown'
            
            print("\n" + "=" * 70)
            print("MODEL CHECKPOINT INFORMATION")
            print("=" * 70)
            print(f"Episodes Completed:    {episodes}")
            print(f"Best Score:            {best_score}")
            print(f"Current Epsilon:       {epsilon if epsilon != 'Unknown' else 'Unknown'} ({'Low' if isinstance(epsilon, float) and epsilon < 0.1 else 'High' if isinstance(epsilon, float) and epsilon > 0.3 else 'Medium' if epsilon != 'Unknown' else '?'} exploration)")
            print(f"Curriculum Stage:      {curriculum_stage} / 4")
            print(f"A* Guidance Prob:      {astar_prob if astar_prob != 'Unknown' else 'Unknown'} ({'High' if isinstance(astar_prob, float) and astar_prob > 0.3 else 'Medium' if isinstance(astar_prob, float) and astar_prob > 0.1 else 'Low' if isinstance(astar_prob, float) else '?'} reward weight)")
            
            # Epsilon analysis
            if epsilon != 'Unknown':
                print(f"\n📊 EPSILON ANALYSIS:")
                if epsilon > 0.3:
                    print(f"  ⚠️  Epsilon is HIGH ({epsilon:.4f}) - Model is exploring randomly {epsilon*100:.1f}% of the time")
                    print(f"      This causes poor scores. Should decay to <0.1 by episode 1000.")
                elif epsilon > 0.1:
                    print(f"  ⚡ Epsilon is MEDIUM ({epsilon:.4f}) - Reasonable balance of exploration/exploitation")
                else:
                    print(f"  ✅ Epsilon is LOW ({epsilon:.4f}) - Model mostly exploiting learned knowledge")
            
            # Curriculum analysis
            print(f"\n📚 CURRICULUM ANALYSIS:")
            thresholds = {0: 20, 1: 50, 2: 100, 3: 200}
            if curriculum_stage != 'Unknown' and curriculum_stage in thresholds:
                next_threshold = thresholds[curriculum_stage]
                print(f"  Current Stage: {curriculum_stage}")
                print(f"  Next Threshold: {next_threshold} points (need avg ≥{next_threshold} for 3 consecutive evaluations)")
                print(f"  A* Reward Weight: {astar_prob if astar_prob != 'Unknown' else '?'}")
                
                if curriculum_stage == 0:
                    print(f"  ⚠️  Still in Stage 0 - Model heavily relies on A* guidance")
                    print(f"      Need to reach avg score of {next_threshold}+ to advance")
                elif curriculum_stage == 1:
                    print(f"  ⚡ Stage 1 - Learning basic strategies with reduced A* help")
                elif curriculum_stage == 2:
                    print(f"  ✅ Stage 2 - Developing independence from A*")
                elif curriculum_stage == 3:
                    print(f"  🎯 Stage 3 - Advanced strategies, minimal A* guidance")
                elif curriculum_stage >= 4:
                    print(f"  🏆 Stage 4 - Expert level, no A* guidance!")
            
        else:
            print("\n⚠️  Model checkpoint format not recognized (old format)")
            epsilon = 'Unknown'
            episodes = 'Unknown'
            best_score = 'Unknown'
            curriculum_stage = 'Unknown'
        
        # Load training history if available
        if os.path.exists(history_path):
            print(f"\n📈 Loading training history: {history_path}")
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            scores = history.get('scores', [])
            if scores:
                print("\n" + "=" * 70)
                print("TRAINING PERFORMANCE")
                print("=" * 70)
                
                # Recent performance
                recent_100 = scores[-100:] if len(scores) >= 100 else scores
                recent_50 = scores[-50:] if len(scores) >= 50 else scores
                recent_10 = scores[-10:] if len(scores) >= 10 else scores
                
                print(f"Total Episodes:        {len(scores)}")
                print(f"Best Score Ever:       {max(scores):.1f}")
                print(f"Average (last 100):    {np.mean(recent_100):.2f}")
                print(f"Average (last 50):     {np.mean(recent_50):.2f}")
                print(f"Average (last 10):     {np.mean(recent_10):.2f}")
                print(f"Std Dev (last 100):    {np.std(recent_100):.2f}")
                
                # Performance trend
                if len(scores) >= 200:
                    first_100 = np.mean(scores[100:200])
                    last_100 = np.mean(recent_100)
                    trend = last_100 - first_100
                    
                    print(f"\n📊 PERFORMANCE TREND:")
                    print(f"  Episodes 100-200 avg:  {first_100:.2f}")
                    print(f"  Recent 100 avg:        {last_100:.2f}")
                    print(f"  Change:                {trend:+.2f} ({trend/first_100*100:+.1f}%)")
                    
                    if trend > 5:
                        print(f"  ✅ IMPROVING - Model is learning!")
                    elif trend > -2:
                        print(f"  ⚡ STABLE - Performance plateaued")
                    else:
                        print(f"  ⚠️  DECLINING - Performance dropping (possible overfitting or high epsilon)")
                
                # Score distribution
                print(f"\n📊 SCORE DISTRIBUTION (last 100 episodes):")
                score_0 = sum(1 for s in recent_100 if s == 0)
                score_10 = sum(1 for s in recent_100 if 0 < s <= 10)
                score_20 = sum(1 for s in recent_100 if 10 < s <= 30)
                score_50 = sum(1 for s in recent_100 if 30 < s <= 60)
                score_100 = sum(1 for s in recent_100 if 60 < s <= 100)
                score_high = sum(1 for s in recent_100 if s > 100)
                
                print(f"  0 points:      {score_0:3d} episodes ({score_0/len(recent_100)*100:5.1f}%)")
                print(f"  1-10 points:   {score_10:3d} episodes ({score_10/len(recent_100)*100:5.1f}%)")
                print(f"  11-30 points:  {score_20:3d} episodes ({score_20/len(recent_100)*100:5.1f}%)")
                print(f"  31-60 points:  {score_50:3d} episodes ({score_50/len(recent_100)*100:5.1f}%)")
                print(f"  61-100 points: {score_100:3d} episodes ({score_100/len(recent_100)*100:5.1f}%)")
                print(f"  100+ points:   {score_high:3d} episodes ({score_high/len(recent_100)*100:5.1f}%)")
                
                # Diagnosis
                print(f"\n🔍 DIAGNOSIS:")
                if curriculum_stage == 0 and np.mean(recent_100) < 20:
                    print(f"  ⚠️  STUCK IN STAGE 0:")
                    print(f"     • Average score ({np.mean(recent_100):.1f}) below threshold (20)")
                    print(f"     • Model can't advance curriculum")
                    print(f"     • Heavily dependent on A* guidance")
                    print(f"  💡 SOLUTIONS:")
                    print(f"     1. Continue training (need sustained avg ≥20)")
                    print(f"     2. Check if epsilon is too high (should be <0.2)")
                    print(f"     3. Increase food rewards to encourage aggressive play")
                
                if epsilon != 'Unknown' and epsilon > 0.2 and len(scores) > 500:
                    print(f"  ⚠️  EPSILON TOO HIGH:")
                    print(f"     • Epsilon {epsilon:.4f} at episode {len(scores)}")
                    print(f"     • Should be <0.1 by episode 500")
                    print(f"     • Random exploration hurting performance")
                    print(f"  💡 SOLUTION: Let epsilon decay naturally or manually set lower")
                
                if abs(np.mean(recent_50) - np.mean(scores[max(0, len(scores)-150):max(50, len(scores)-100)])) > 3:
                    print(f"  ⚠️  HIGH VARIANCE:")
                    print(f"     • Performance fluctuating significantly")
                    print(f"     • May indicate learning rate too high or unstable training")
                    print(f"  💡 SOLUTION: Reduce learning rate for fine-tuning")
                
                if np.mean(recent_100) > 20 and curriculum_stage == 0:
                    print(f"  ⚡ READY TO ADVANCE:")
                    print(f"     • Average score ({np.mean(recent_100):.1f}) above threshold")
                    print(f"     • Should advance to Stage 1 soon")
                    print(f"     • Need 3 consecutive evaluations ≥20")
        
        else:
            print(f"\n⚠️  Training history not found: {history_path}")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        # Provide actionable recommendations
        if epsilon != 'Unknown' and epsilon > 0.15 and isinstance(episodes, int) and episodes > 500:
            print("1. ⚠️  REDUCE EPSILON - It's too high for this stage of training")
            print("   • Current decay may be too slow")
            print("   • Consider manually setting epsilon = 0.05 for better exploitation")
        
        if curriculum_stage == 0 and isinstance(episodes, int) and episodes > 300:
            print("2. 📚 FOCUS ON CURRICULUM ADVANCEMENT")
            print("   • Model stuck in Stage 0 for too long")
            print("   • Increase food rewards to boost scores")
            print("   • Thresholds have been lowered: [20, 50, 100, 200]")
        
        if isinstance(best_score, (int, float)) and best_score < 50:
            print("3. 🎯 IMPROVE BASE PERFORMANCE")
            print("   • Best score is low - model needs more learning")
            print("   • Check if A* pathfinding is working correctly")
            print("   • Verify reward shaping is encouraging food collection")
        
        print("\n💡 Next steps:")
        print("   • Continue training with updated parameters")
        print("   • Monitor curriculum advancement messages")
        print("   • Use Model Visualization tools to analyze feature importance")
        print("   • Check if DQN is learning from A* hints")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        # Infer history path from model path
        history_path = model_path.replace('.pth', '_history.json')
        check_model_state(model_path, history_path)
    else:
        check_model_state()
