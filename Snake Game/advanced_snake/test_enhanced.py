"""
Test the enhanced DQN implementation to verify it works correctly.

This script runs a quick test to ensure:
1. Enhanced state representation works (31 features)
2. A* guidance integration functions
3. Curriculum system initializes
4. Training loop executes without errors

Usage:
    python test_enhanced.py
"""

import sys
import torch
from game_engine import GameEngine
from enhanced_dqn import EnhancedDQNAgent, EnhancedStateRepresentation


def test_state_representation():
    """Test that enhanced state has correct size and values."""
    print("\n" + "="*70)
    print("TEST 1: Enhanced State Representation")
    print("="*70)
    
    game = GameEngine()
    state = EnhancedStateRepresentation.get_enhanced_state(game)
    
    print(f"✓ State shape: {state.shape}")
    print(f"✓ Expected: torch.Size([31])")
    
    if state.shape[0] == 31:
        print("✅ PASS: State has correct 31 features")
    else:
        print(f"❌ FAIL: State has {state.shape[0]} features, expected 31")
        return False
    
    # Check that values are reasonable
    if torch.all(state >= -1) and torch.all(state <= 1):
        print("✅ PASS: All state values in reasonable range")
    else:
        print("⚠️ WARNING: Some state values outside [-1, 1] range")
    
    print(f"\nSample state vector (first 10 features):")
    print(f"  {state[:10].tolist()}")
    
    return True


def test_agent_initialization():
    """Test that enhanced agent initializes correctly."""
    print("\n" + "="*70)
    print("TEST 2: Enhanced Agent Initialization")
    print("="*70)
    
    game = GameEngine()
    agent = EnhancedDQNAgent(game)
    
    print(f"✓ Agent created")
    print(f"✓ State size: {agent.state_size}")
    print(f"✓ Action size: {agent.action_size}")
    print(f"✓ Curriculum stage: {agent.curriculum_stage}")
    print(f"✓ A* guidance probability: {agent.astar_guidance_prob}")
    
    if agent.state_size == 31:
        print("✅ PASS: Agent has correct state size (31)")
    else:
        print(f"❌ FAIL: Agent state size is {agent.state_size}, expected 31")
        return False
    
    if agent.action_size == 3:
        print("✅ PASS: Agent has correct action size (3)")
    else:
        print(f"❌ FAIL: Agent action size is {agent.action_size}, expected 3")
        return False
    
    return True


def test_astar_guidance():
    """Test A* guidance integration."""
    print("\n" + "="*70)
    print("TEST 3: A* Guidance Integration")
    print("="*70)
    
    game = GameEngine()
    agent = EnhancedDQNAgent(game)
    
    # Try to get A* guided action
    try:
        action = agent._get_astar_guided_action()
        print(f"✓ A* guided action: {action}")
        
        if 0 <= action <= 2:
            print("✅ PASS: A* guidance returns valid action")
        else:
            print(f"❌ FAIL: Invalid action {action}, expected 0-2")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: A* guidance failed with error: {e}")
        return False
    
    return True


def test_action_selection():
    """Test action selection with different modes."""
    print("\n" + "="*70)
    print("TEST 4: Action Selection")
    print("="*70)
    
    game = GameEngine()
    agent = EnhancedDQNAgent(game)
    state = agent.get_state()
    
    # Test training mode
    try:
        action_train = agent.select_action(state, training=True)
        print(f"✓ Training mode action: {action_train}")
        
        if 0 <= action_train <= 2:
            print("✅ PASS: Training mode returns valid action")
        else:
            print(f"❌ FAIL: Invalid action {action_train}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Training action selection failed: {e}")
        return False
    
    # Test evaluation mode
    try:
        action_eval = agent.select_action(state, training=False)
        print(f"✓ Evaluation mode action: {action_eval}")
        
        if 0 <= action_eval <= 2:
            print("✅ PASS: Evaluation mode returns valid action")
        else:
            print(f"❌ FAIL: Invalid action {action_eval}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Evaluation action selection failed: {e}")
        return False
    
    return True


def test_reward_calculation():
    """Test enhanced reward calculation."""
    print("\n" + "="*70)
    print("TEST 5: Enhanced Reward Calculation")
    print("="*70)
    
    game = GameEngine()
    agent = EnhancedDQNAgent(game)
    
    # Test different scenarios
    scenarios = [
        ("Food eaten", 0, False, 5, 0, "Should be positive"),
        ("Death", 10, True, 5, 10, "Should be very negative"),
        ("Move toward food", 10, False, 5, 4, "Should be slightly positive"),
        ("Move away from food", 10, False, 5, 6, "Should be slightly negative")
    ]
    
    all_passed = True
    for name, old_score, game_over, old_dist, new_dist, expected in scenarios:
        game.game_over = game_over
        game.score = old_score + (10 if name == "Food eaten" else 0)
        
        reward = agent.calculate_reward(old_score, game_over, old_dist, new_dist)
        print(f"✓ {name}: reward = {reward:.2f} ({expected})")
        
        # Basic sanity checks
        if name == "Death" and reward >= 0:
            print(f"  ❌ FAIL: Death should give negative reward")
            all_passed = False
        elif name == "Food eaten" and reward <= 0:
            print(f"  ❌ FAIL: Food should give positive reward")
            all_passed = False
    
    if all_passed:
        print("✅ PASS: Reward calculations seem reasonable")
    
    return all_passed


def test_curriculum_system():
    """Test curriculum advancement."""
    print("\n" + "="*70)
    print("TEST 6: Curriculum Learning System")
    print("="*70)
    
    game = GameEngine()
    agent = EnhancedDQNAgent(game)
    
    print(f"✓ Initial curriculum stage: {agent.curriculum_stage}")
    print(f"✓ Curriculum thresholds: {agent.curriculum_thresholds}")
    
    # Simulate high scores to test advancement
    print("\nSimulating 10 episodes with score 60...")
    for i in range(10):
        agent.update_curriculum(60)
    
    print(f"✓ Curriculum stage after high scores: {agent.curriculum_stage}")
    
    if agent.curriculum_stage > 0:
        print("✅ PASS: Curriculum advanced as expected")
    else:
        print("❌ FAIL: Curriculum did not advance")
        return False
    
    return True


def test_training_step():
    """Test a single training step end-to-end."""
    print("\n" + "="*70)
    print("TEST 7: Single Training Step")
    print("="*70)
    
    try:
        game = GameEngine()
        agent = EnhancedDQNAgent(game)
        
        # Get initial state
        state = agent.get_state()
        print("✓ Got initial state")
        
        # Select action
        action = agent.select_action(state, training=True)
        print(f"✓ Selected action: {action}")
        
        # Perform action
        agent.perform_action(action)
        print("✓ Performed action")
        
        # Get new state
        new_state = agent.get_state()
        print("✓ Got new state")
        
        # Calculate reward
        head = game.snake[0]
        food = game.food
        distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
        reward = agent.calculate_reward(0, game.game_over, distance, distance)
        print(f"✓ Calculated reward: {reward:.2f}")
        
        # Store experience
        agent.memory.add(state, action, reward, new_state, game.game_over)
        print("✓ Stored experience in memory")
        
        print("✅ PASS: Complete training step executed successfully")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Training step failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("🧪 ENHANCED DQN TESTING SUITE")
    print("="*70)
    print("Testing enhanced DQN implementation...")
    
    tests = [
        ("State Representation", test_state_representation),
        ("Agent Initialization", test_agent_initialization),
        ("A* Guidance", test_astar_guidance),
        ("Action Selection", test_action_selection),
        ("Reward Calculation", test_reward_calculation),
        ("Curriculum System", test_curriculum_system),
        ("Training Step", test_training_step)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("="*70)
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Enhanced DQN is ready to use.")
        print("\nNext step: Run training with:")
        print("  python train_enhanced.py --episodes 1000 --new-model")
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")
        print("Please check the error messages above.")
    
    print("="*70 + "\n")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
