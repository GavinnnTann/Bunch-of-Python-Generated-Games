"""
Test GPU utilization during training.
This will show how much GPU is actually being used.
"""

import torch
import time
import numpy as np
from game_engine import GameEngine
from enhanced_dqn import EnhancedDQNAgent

def test_gpu_usage():
    """Test GPU usage during training."""
    
    print("=" * 70)
    print("GPU UTILIZATION TEST")
    print("=" * 70)
    
    # Check CUDA availability
    print(f"\n[CUDA INFO]")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Current device: {torch.cuda.current_device()}")
    
    # Create agent
    game = GameEngine()
    agent = EnhancedDQNAgent(game)
    
    print(f"\n[AGENT INFO]")
    print(f"  State size: {agent.state_size}")
    print(f"  Action size: {agent.action_size}")
    print(f"  Network on device: {next(agent.policy_net.parameters()).device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    trainable_params = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Collect experiences
    print(f"\n[COLLECTING 500 EXPERIENCES]")
    while len(agent.memory) < 500:
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
    print(f"  Collected {len(agent.memory)} experiences")
    
    # Test training speed
    print(f"\n[TRAINING SPEED TEST]")
    print(f"  Performing 100 training iterations...")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_time = time.time()
    losses = []
    
    for i in range(100):
        loss = agent.optimize_model()
        losses.append(loss)
        
        if i % 20 == 0 and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            mem_cached = torch.cuda.memory_reserved() / 1024**2  # MB
            print(f"    Iteration {i}: Loss={loss:.4f}, GPU Mem: {mem_allocated:.1f}MB / {mem_cached:.1f}MB")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n[RESULTS]")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Time per iteration: {total_time / 100 * 1000:.2f}ms")
    print(f"  Iterations per second: {100 / total_time:.1f}")
    print(f"  Average loss: {np.mean(losses):.4f}")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"\n[GPU MEMORY]")
        print(f"  Peak memory used: {peak_mem:.1f}MB")
        print(f"  Current allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"  Current reserved: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
    
    # Analysis
    print(f"\n[ANALYSIS]")
    if total_time / 100 < 0.01:  # Less than 10ms per iteration
        print(f"  ⚠️  Training is VERY fast ({total_time / 100 * 1000:.2f}ms/iteration)")
        print(f"  This is typical for small networks and may not fully utilize GPU")
    
    if torch.cuda.is_available():
        if peak_mem < 100:
            print(f"  ⚠️  Low GPU memory usage ({peak_mem:.1f}MB)")
            print(f"  Reasons:")
            print(f"    • Small batch size (64 samples)")
            print(f"    • Small network (34 input features, 256 hidden)")
            print(f"    • Fast forward/backward pass")
        
        print(f"\n  💡 To increase GPU utilization:")
        print(f"    1. Increase batch size (e.g., 256 or 512)")
        print(f"    2. Increase network size (e.g., 512 hidden units)")
        print(f"    3. Train more frequently (more iterations per episode)")
        print(f"    4. Use multiple training updates per step")
    
    print(f"\n" + "=" * 70)

if __name__ == "__main__":
    test_gpu_usage()
