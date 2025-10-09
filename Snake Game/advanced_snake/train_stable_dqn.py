"""
Stable DQN Training Script
==========================
Conservative, monitored training following best practices.

Training budget: 5-10M steps
Evaluation: Every 50k steps, 20 episodes at ε=0.05
"""

import sys
import os
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import GameEngine
from stable_dqn import StableDQNAgent
from constants import *


def train_stable_dqn(
    total_steps=5000000,
    eval_interval=50000,
    save_interval=100000,
    warmup_steps=20000
):
    """
    Train Stable DQN agent.
    
    Args:
        total_steps: Total environment steps (default 5M)
        eval_interval: Steps between evaluations (default 50k)
        save_interval: Steps between checkpoints (default 100k)
        warmup_steps: Random policy collection (default 20k)
    """
    print("=" * 70)
    print("STABLE DQN TRAINING - SNAKE GAME")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  • Total Steps: {total_steps:,}")
    print(f"  • Warmup Steps: {warmup_steps:,}")
    print(f"  • Eval Interval: {eval_interval:,}")
    print(f"  • Save Interval: {save_interval:,}")
    print(f"  • Batch Size: 128")
    print(f"  • Learning Rate: 3e-4")
    print(f"  • Gamma: 0.99")
    print(f"  • Epsilon: 1.0 -> 0.05 over 200k steps")
    print(f"  • PER: alpha=0.6, beta=0.4->1.0")
    print(f"  • Target Update: tau=0.005 (soft)")
    print("=" * 70)
    
    # Initialize
    game_engine = GameEngine()
    agent = StableDQNAgent(game_engine)
    
    # Check for existing checkpoint
    model_path = os.path.join(QMODEL_DIR, "snake_stable_dqn.pth")
    if os.path.exists(model_path):
        print(f"\n[INFO] Found existing checkpoint: {model_path}")
        response = input("Continue training from checkpoint? (y/n): ")
        if response.lower() == 'y':
            agent.load_model(model_path)
            print(f"[OK] Resumed from step {agent.total_steps:,}")
        else:
            print("[INFO] Starting fresh training")
    
    # Warmup phase (if needed)
    if agent.total_steps == 0:
        agent.collect_warmup_data(warmup_steps)
    
    # Training loop
    print(f"\n[INFO] Starting training...\n")
    
    episode = 0
    last_eval_step = 0
    last_save_step = 0
    best_eval_score = 0
    
    training_history = {
        'episodes': [],
        'scores': [],
        'steps_per_episode': [],
        'total_steps': [],
        'epsilon': [],
        'eval_mean': [],
        'eval_median': [],
        'eval_max': [],
        'loss': [],
        'td_error': [],
        'q_values': []
    }
    
    try:
        while agent.total_steps < total_steps:
            # Train one episode
            result = agent.train_episode()
            episode += 1
            
            # Log training progress
            training_history['episodes'].append(episode)
            training_history['scores'].append(result['score'])
            training_history['steps_per_episode'].append(result['steps'])
            training_history['total_steps'].append(agent.total_steps)
            training_history['epsilon'].append(agent.epsilon)
            
            # Aggregate recent stats
            if len(agent.stats['losses']) > 0:
                training_history['loss'].append(agent.stats['losses'][-1])
                training_history['td_error'].append(agent.stats['td_errors'][-1])
                training_history['q_values'].append(agent.stats['q_values'][-1])
            
            # Progress output every 10 episodes
            if episode % 10 == 0:
                recent_scores = training_history['scores'][-100:]
                avg_score = sum(recent_scores) / len(recent_scores)
                
                loss_str = f"{training_history['loss'][-1]:.4f}" if training_history['loss'] else "N/A"
                td_str = f"{training_history['td_error'][-1]:.4f}" if training_history['td_error'] else "N/A"
                q_str = f"{training_history['q_values'][-1]:.2f}" if training_history['q_values'] else "N/A"
                
                print(f"Ep {episode:5d} | "
                      f"Steps: {agent.total_steps:7,d} | "
                      f"Score: {result['score']:3d} | "
                      f"Avg100: {avg_score:6.2f} | "
                      f"ε: {agent.epsilon:.4f} | "
                      f"Loss: {loss_str} | "
                      f"TD: {td_str} | "
                      f"Q: {q_str}")
            
            # Evaluation
            if agent.total_steps - last_eval_step >= eval_interval:
                print(f"\n{'='*70}")
                print(f"EVALUATION @ {agent.total_steps:,} steps")
                print(f"{'='*70}")
                
                eval_results = agent.evaluate(num_episodes=20, epsilon=0.05)
                
                print(f"  Mean:   {eval_results['mean']:.2f}")
                print(f"  Median: {eval_results['median']:.2f}")
                print(f"  Max:    {eval_results['max']}")
                print(f"  Min:    {eval_results['min']}")
                print(f"  Std:    {eval_results['std']:.2f}")
                
                training_history['eval_mean'].append(eval_results['mean'])
                training_history['eval_median'].append(eval_results['median'])
                training_history['eval_max'].append(eval_results['max'])
                
                # Save best model
                if eval_results['median'] > best_eval_score:
                    best_eval_score = eval_results['median']
                    best_path = model_path.replace('.pth', '_best.pth')
                    agent.save_model(best_path)
                    print(f"  🏆 New best! Saved to {best_path}")
                
                print(f"{'='*70}\n")
                last_eval_step = agent.total_steps
            
            # Checkpoint save
            if agent.total_steps - last_save_step >= save_interval:
                agent.save_model(model_path)
                
                # Save training history
                history_path = model_path.replace('.pth', '_history.json')
                with open(history_path, 'w') as f:
                    json.dump(training_history, f, indent=2)
                
                print(f"\n[SAVE] Checkpoint @ {agent.total_steps:,} steps\n")
                last_save_step = agent.total_steps
        
        # Final save
        agent.save_model(model_path)
        history_path = model_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Final evaluation
        print(f"\n{'='*70}")
        print(f"FINAL EVALUATION")
        print(f"{'='*70}")
        final_eval = agent.evaluate(num_episodes=50, epsilon=0.05)
        print(f"  Mean:   {final_eval['mean']:.2f}")
        print(f"  Median: {final_eval['median']:.2f}")
        print(f"  Max:    {final_eval['max']}")
        print(f"  Std:    {final_eval['std']:.2f}")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Training interrupted by user.")
        agent.save_model(model_path)
        history_path = model_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print("[SAVE] Progress saved.")
    
    except Exception as e:
        print(f"\n[ERROR] Training error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save progress
        try:
            agent.save_model(model_path)
            print("[SAVE] Emergency save successful.")
        except:
            print("[ERROR] Could not save model.")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Training Summary:")
    print(f"  • Total Episodes: {episode}")
    print(f"  • Total Steps: {agent.total_steps:,}")
    print(f"  • Final Epsilon: {agent.epsilon:.4f}")
    print(f"  • Best Eval Median: {best_eval_score:.2f}")
    print("="*70)


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train Stable DQN agent for Snake')
    
    parser.add_argument('--steps', type=int, default=5000000,
                       help='Total training steps (default: 5M)')
    parser.add_argument('--eval-interval', type=int, default=50000,
                       help='Steps between evaluations (default: 50k)')
    parser.add_argument('--save-interval', type=int, default=100000,
                       help='Steps between checkpoints (default: 100k)')
    parser.add_argument('--warmup', type=int, default=20000,
                       help='Warmup steps (default: 20k)')
    
    args = parser.parse_args()
    
    train_stable_dqn(
        total_steps=args.steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        warmup_steps=args.warmup
    )


if __name__ == "__main__":
    main()
