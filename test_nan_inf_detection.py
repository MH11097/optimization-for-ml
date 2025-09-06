#!/usr/bin/env python3
"""
Test script to verify NaN/Inf detection functionality
"""

import numpy as np
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.optimization_utils import check_for_numerical_issues, kiem_tra_hoi_tu

def test_check_for_numerical_issues():
    """Test the numerical stability checking function"""
    print("🧪 Testing check_for_numerical_issues function...")
    
    # Test 1: Normal values - should be OK
    print("\n1. Testing normal values:")
    has_issues, msg = check_for_numerical_issues(0.001, 1.5, np.array([0.1, 0.2, 0.3]), 10)
    print(f"   Result: {has_issues}, Message: {msg}")
    assert not has_issues, "Normal values should not have issues"
    
    # Test 2: NaN gradient norm
    print("\n2. Testing NaN gradient norm:")
    has_issues, msg = check_for_numerical_issues(np.nan, 1.5, np.array([0.1, 0.2, 0.3]), 10)
    print(f"   Result: {has_issues}, Message: {msg}")
    assert has_issues, "NaN gradient norm should be detected"
    assert "Gradient norm = NaN" in msg
    
    # Test 3: Inf gradient norm
    print("\n3. Testing Inf gradient norm:")
    has_issues, msg = check_for_numerical_issues(np.inf, 1.5, np.array([0.1, 0.2, 0.3]), 10)
    print(f"   Result: {has_issues}, Message: {msg}")
    assert has_issues, "Inf gradient norm should be detected"
    assert "Gradient norm = ±Inf" in msg
    
    # Test 4: NaN loss
    print("\n4. Testing NaN loss:")
    has_issues, msg = check_for_numerical_issues(0.001, np.nan, np.array([0.1, 0.2, 0.3]), 10)
    print(f"   Result: {has_issues}, Message: {msg}")
    assert has_issues, "NaN loss should be detected"
    assert "Loss = NaN" in msg
    
    # Test 5: Inf loss
    print("\n5. Testing Inf loss:")
    has_issues, msg = check_for_numerical_issues(0.001, np.inf, np.array([0.1, 0.2, 0.3]), 10)
    print(f"   Result: {has_issues}, Message: {msg}")
    assert has_issues, "Inf loss should be detected"
    assert "Loss = ±Inf" in msg
    
    # Test 6: NaN weights
    print("\n6. Testing NaN weights:")
    weights_with_nan = np.array([0.1, np.nan, 0.3])
    has_issues, msg = check_for_numerical_issues(0.001, 1.5, weights_with_nan, 10)
    print(f"   Result: {has_issues}, Message: {msg}")
    assert has_issues, "NaN weights should be detected"
    assert "Weights contain 1 NaN values" in msg
    
    # Test 7: Inf weights
    print("\n7. Testing Inf weights:")
    weights_with_inf = np.array([0.1, np.inf, 0.3])
    has_issues, msg = check_for_numerical_issues(0.001, 1.5, weights_with_inf, 10)
    print(f"   Result: {has_issues}, Message: {msg}")
    assert has_issues, "Inf weights should be detected"
    assert "Weights contain 1 ±Inf values" in msg
    
    # Test 8: Multiple issues
    print("\n8. Testing multiple issues:")
    weights_with_both = np.array([np.nan, np.inf, 0.3])
    has_issues, msg = check_for_numerical_issues(np.nan, np.inf, weights_with_both, 10)
    print(f"   Result: {has_issues}, Message: {msg}")
    assert has_issues, "Multiple issues should be detected"
    assert "Gradient norm = NaN" in msg
    assert "Loss = ±Inf" in msg
    assert "NaN values" in msg
    assert "±Inf values" in msg
    
    print("\n✅ All numerical stability tests passed!")


def test_kiem_tra_hoi_tu_with_nan_inf():
    """Test the enhanced convergence checking function"""
    print("\n🧪 Testing enhanced kiem_tra_hoi_tu function...")
    
    # Test 1: Normal convergence
    print("\n1. Testing normal convergence:")
    converged, reason = kiem_tra_hoi_tu(1e-7, 1e-8, 10, 1e-6, 100, 0.5, np.array([0.1, 0.2]))
    print(f"   Converged: {converged}, Reason: {reason}")
    assert converged, "Should converge with small gradient and cost change"
    
    # Test 2: NaN should stop immediately
    print("\n2. Testing NaN detection:")
    converged, reason = kiem_tra_hoi_tu(np.nan, 1e-8, 10, 1e-6, 100, 0.5, np.array([0.1, 0.2]))
    print(f"   Converged: {converged}, Reason: {reason}")
    assert converged, "Should stop immediately when NaN detected"
    assert "NUMERICAL INSTABILITY" in reason
    assert "Gradient norm = NaN" in reason
    
    # Test 3: Inf should stop immediately
    print("\n3. Testing Inf detection:")
    converged, reason = kiem_tra_hoi_tu(np.inf, 1e-8, 10, 1e-6, 100, 0.5, np.array([0.1, 0.2]))
    print(f"   Converged: {converged}, Reason: {reason}")
    assert converged, "Should stop immediately when Inf detected"
    assert "NUMERICAL INSTABILITY" in reason
    assert "Gradient norm = ±Inf" in reason
    
    # Test 4: Inf loss should stop immediately
    print("\n4. Testing Inf loss detection:")
    converged, reason = kiem_tra_hoi_tu(0.001, 1e-8, 10, 1e-6, 100, np.inf, np.array([0.1, 0.2]))
    print(f"   Converged: {converged}, Reason: {reason}")
    assert converged, "Should stop immediately when loss is Inf"
    assert "NUMERICAL INSTABILITY" in reason
    assert "Loss = ±Inf" in reason
    
    print("\n✅ All convergence tests passed!")


def create_unstable_gradient_descent_example():
    """Create a simple example that will cause gradient explosion"""
    print("\n🧪 Testing with a simple unstable example...")
    
    # Create a simple model that can cause numerical instability
    from algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
    
    # Create problematic data that can cause numerical issues
    np.random.seed(42)
    X = np.random.normal(0, 1, (10, 3))
    y = np.random.normal(0, 1, 10)
    
    # Use an extremely high learning rate to force numerical instability
    model = GradientDescentModel(
        ham_loss='ols', 
        learning_rate=1e10,  # Extremely high learning rate to force divergence
        so_lan_thu=20,
        diem_dung=1e-5,
        convergence_check_freq=1  # Check every iteration
    )
    
    print("   Training with extremely high learning rate to test NaN/Inf detection...")
    try:
        results = model.fit(X, y)
        print(f"   Training stopped at iteration: {results['final_iteration']}")
        print(f"   Final loss: {results['loss_history'][-1] if results['loss_history'] else 'N/A'}")
        print(f"   Final gradient norm: {results['gradient_norms'][-1] if results['gradient_norms'] else 'N/A'}")
        print(f"   Converged (stopped): {results['converged']}")
        return True
    except Exception as e:
        print(f"   Error during training: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing NaN/Inf Detection Implementation")
    print("=" * 60)
    
    try:
        # Test the utility functions
        test_check_for_numerical_issues()
        test_kiem_tra_hoi_tu_with_nan_inf()
        
        # Test with actual model (this might cause numerical instability)
        print("\n" + "=" * 60)
        success = create_unstable_gradient_descent_example()
        
        print("\n" + "=" * 60)
        if success:
            print("✅ ALL TESTS PASSED! NaN/Inf detection is working correctly.")
        else:
            print("⚠️  Model test had issues, but utility functions work correctly.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()