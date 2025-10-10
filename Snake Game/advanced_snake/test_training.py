"""
Test script to verify DQN training is actually happening.
This will show when gradients are being computed and weights are updating.
"""

import torch
import numpy as np
from game_engine import GameEngine
from enhanced_dqn import EnhancedDQNAgent

def test_training():
    """Test if DQN is actually training and updating weights."""
    
    print("=" * 70)
    print("DQN TRAINING VERIFICATION TEST")
    print("=" * 70)
    
    # Create game and agent
    game = GameEngine()
    agent = EnhancedDQNAgent(game)
    
    print(f"\n[SETUP]")
    print(f"  Device: {agent.policy_net.feature_layer[0].weight.device}")
    print(f"  Train start threshold: {agent.train_start} experiences")
    print(f"  Batch size: {agent.batch_size}")
    print(f"  Memory size: {len(agent.memory)}")
    
    # Get initial weights
    initial_weights = agent.policy_net.feature_layer[0].weight.data.clone()
    print(f"\n[INITIAL WEIGHTS] First layer sample: {initial_weights[0, :5]}")
    
    # Collect some experiences
    print(f"\n[COLLECTING EXPERIENCES]")
    total_steps = 0
    for episode in range(5):
        game.reset_game()
        state = agent.get_state()
        steps = 0
        
        while not game.game_over and steps < 200:
            action = agent.select_action(state, training=True)
            agent.perform_action(action)
            new_state = agent.get_state()
            
            reward = 1.0 if game.score > 0 else -0.1
            agent.memory.add(state, action, reward, new_state, game.game_over)
            
            state = new_state
            steps += 1
            total_steps += 1
        
        print(f"  Episode {episode + 1}: {steps} steps, Memory: {len(agent.memory)}/{agent.train_start}")
    
    print(f"\n[TRAINING TEST]")
    if len(agent.memory) < agent.train_start:
        print(f"  ⚠️  Not enough experiences yet ({len(agent.memory)}/{agent.train_start})")
        print(f"  Collecting {agent.train_start - len(agent.memory)} more experiences...")
        
        # Collect more
        while len(agent.memory) < agent.train_start:
            game.reset_game()
            state = agent.get_state()
            for _ in range(100):
                if game.game_over:
                    break
                action = agent.select_action(state, training=True)
                agent.perform_action(action)
                new_state = agent.get_state()
                reward = 1.0 if game.score > 0 else -0.1
                agent.memory.add(state, action, reward, new_state, game.game_over)
                state = new_state
        
        print(f"  ✓ Now have {len(agent.memory)} experiences")
    
    # Try training
    print(f"\n[PERFORMING TRAINING]")
    losses = []
    for i in range(10):
        loss = agent.optimize_model()
        losses.append(loss)
        if i == 0:
            print(f"  Training iteration {i + 1}: Loss = {loss:.4f}")
    
    print(f"  Completed 10 training iterations")
    print(f"  Average loss: {np.mean(losses):.4f}")
    print(f"  Loss range: {min(losses):.4f} - {max(losses):.4f}")
    
    # Check if weights changed
    final_weights = agent.policy_net.feature_layer[0].weight.data.clone()
    weight_diff = torch.abs(final_weights - initial_weights).mean().item()
    
    print(f"\n[WEIGHT CHANGE VERIFICATION]")
    print(f"  Initial weights (sample): {initial_weights[0, :5]}")
    print(f"  Final weights (sample):   {final_weights[0, :5]}")
    print(f"  Average absolute change:  {weight_diff:.6f}")
    
    if weight_diff > 1e-6:
        print(f"\n  ✅ SUCCESS! Weights are being updated!")
        print(f"  ✅ DQN training is working correctly!")
    else:
        print(f"\n  ❌ PROBLEM! Weights are NOT changing!")
        print(f"  ❌ Training may not be working!")
    
    # Check gradient flow
    print(f"\n[GRADIENT CHECK]")
    agent.policy_net.train()
    
    # Forward pass
    sample_state = agent.get_state()
    if isinstance(sample_state, torch.Tensor):
        sample_state = sample_state.unsqueeze(0)
    else:
        sample_state = torch.FloatTensor(sample_state).unsqueeze(0).to(agent.policy_net.feature_layer[0].weight.device)
    
    output = agent.policy_net(sample_state)
    loss = output.sum()
    
    # Backward pass
    agent.optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in agent.policy_net.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            print(f"  ✓ {name}: gradient norm = {param.grad.norm().item():.6f}")
            break
    
    if has_gradients:
        print(f"\n  ✅ Gradients are flowing through the network!")
    else:
        print(f"\n  ❌ No gradients detected!")
    
    print(f"\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_training()
