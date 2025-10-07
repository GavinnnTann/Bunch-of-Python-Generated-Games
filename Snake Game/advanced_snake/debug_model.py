"""
Debug DQN Model - Check model state and test inference
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from game_engine import GameEngine
from enhanced_dqn import EnhancedDQNAgent

def main():
    print("=" * 70)
    print("DQN MODEL DEBUG TOOL")
    print("=" * 70)
    
    # Load checkpoint
    checkpoint_path = "models/snake_enhanced_dqn.pth"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n📊 MODEL CHECKPOINT INFO:")
    print(f"  Epsilon: {checkpoint.get('epsilon', 'Not found')}")
    print(f"  Curriculum Stage: {checkpoint.get('curriculum_stage', 'Not found')}")
    print(f"  Learn Steps: {checkpoint.get('learn_step_counter', 'Not found')}")
    print(f"  Model Type: {checkpoint.get('model_type', 'Not found')}")
    
    # Check network architecture
    if 'policy_net_state_dict' in checkpoint:
        policy_dict = checkpoint['policy_net_state_dict']
        if 'feature_layer.0.weight' in policy_dict:
            state_size = policy_dict['feature_layer.0.weight'].shape[1]
            print(f"  State Size: {state_size}")
        if 'action_layer.2.weight' in policy_dict:
            action_size = policy_dict['action_layer.2.weight'].shape[0]
            print(f"  Action Size: {action_size}")
    
    # Create game and agent
    print("\n🎮 TESTING MODEL INFERENCE:")
    game = GameEngine()
    agent = EnhancedDQNAgent(game)
    
    # Load model
    print(f"  Loading model...")
    if agent.load_model(checkpoint_path, silent=True, for_gameplay=True):
        print(f"  ✅ Model loaded successfully")
        print(f"  Epsilon after loading: {agent.epsilon}")
    else:
        print(f"  ❌ Failed to load model")
        return
    
    # Test state generation
    print("\n🧪 TESTING STATE GENERATION:")
    state = agent.get_state()
    print(f"  State shape: {state.shape}")
    print(f"  State dtype: {state.dtype}")
    print(f"  State device: {state.device}")
    print(f"  State values (first 10): {state[:10].tolist()}")
    
    # Test action selection
    print("\n🎯 TESTING ACTION SELECTION:")
    for i in range(5):
        game.reset_game()
        state = agent.get_state()
        action_idx = agent.select_action(state, training=False)
        
        with torch.no_grad():
            q_values = agent.policy_net(state.unsqueeze(0))
        
        print(f"\n  Test {i+1}:")
        print(f"    State: Snake at {game.snake[0]}, Food at {game.food}")
        print(f"    Direction: {game.direction}")
        print(f"    Q-values: {q_values.squeeze().tolist()}")
        print(f"    Selected action: {action_idx} (0=straight, 1=right, 2=left)")
        print(f"    Max Q-value: {q_values.max().item():.4f}")
    
    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
