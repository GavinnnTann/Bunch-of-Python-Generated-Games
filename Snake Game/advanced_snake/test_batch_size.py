"""
Quick test to see if batch size 512 works with your GPU
"""
import torch
from constants import *

print(f"Testing batch size: 512")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
    print(f"Cached VRAM: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    
    try:
        # Simulate a batch of 512 with 34 features (Enhanced DQN)
        device = torch.device("cuda")
        
        # Create dummy tensors similar to training
        batch_size = 512
        state_size = 34  # Enhanced DQN features
        
        states = torch.randn(batch_size, state_size, device=device)
        next_states = torch.randn(batch_size, state_size, device=device)
        
        # Simulate a neural network forward pass
        from advanced_dqn import DuelingDQN
        network = DuelingDQN(state_size, 3).to(device)
        
        with torch.no_grad():
            q_values = network(states)
            next_q_values = network(next_states)
        
        print(f"\n✅ SUCCESS! Batch size 512 works on your GPU")
        print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"VRAM reserved: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
        
        # Test even larger batch
        try:
            states_large = torch.randn(1024, state_size, device=device)
            q_large = network(states_large)
            print(f"\n✅ Could even handle batch size 1024!")
            print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        except RuntimeError as e:
            print(f"\n⚠️ Batch size 1024 is too large: {e}")
            
    except RuntimeError as e:
        print(f"\n❌ ERROR with batch size 512: {e}")
        print(f"\nRecommendation: Use smaller batch size (256 or 128)")
        
else:
    print("\n❌ No CUDA GPU available - batch size 512 not recommended on CPU")
    print("Recommendation: Use CPU_BATCH_SIZE = 64")
