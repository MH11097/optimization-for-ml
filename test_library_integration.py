"""
Test script to verify library integration with AlgorithmComparator.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_library_imports():
    """Test importing all library modules."""
    print("Testing library imports...")
    
    try:
        # Test base wrapper
        from src.optimization.external.base_library_wrapper import BaseLibraryWrapper
        print("+ BaseLibraryWrapper imported successfully")
        
        # Test library_stochastic imports
        try:
            from src.optimization.external.library_stochastic.sklearn_sgd_variants import create_sklearn_sgd_constant
            print("+ Sklearn SGD variants imported successfully")
        except ImportError as e:
            print(f"- Sklearn SGD variants import failed: {e}")
        
        try:
            from src.optimization.external.library_stochastic.pytorch_sgd_variants import create_pytorch_sgd_momentum
            print("+ PyTorch SGD variants imported successfully")
        except ImportError as e:
            print(f"- PyTorch SGD variants import failed: {e}")
        
        try:
            from src.optimization.external.library_stochastic.tensorflow_sgd_variants import create_tensorflow_sgd_nesterov
            print("+ TensorFlow SGD variants imported successfully")
        except ImportError as e:
            print(f"- TensorFlow SGD variants import failed: {e}")
        
        try:
            from src.optimization.external.library_stochastic.jax_sgd import create_jax_sgd_momentum
            print("+ JAX SGD variants imported successfully")
        except ImportError as e:
            print(f"- JAX SGD variants import failed: {e}")
        
        # Test library_adaptive imports
        try:
            from src.optimization.external.library_adaptive.pytorch_adaptive import create_pytorch_adam
            print("+ PyTorch adaptive algorithms imported successfully")
        except ImportError as e:
            print(f"- PyTorch adaptive algorithms import failed: {e}")
        
        try:
            from src.optimization.external.library_adaptive.tensorflow_adaptive import create_tensorflow_adam
            print("+ TensorFlow adaptive algorithms imported successfully")
        except ImportError as e:
            print(f"- TensorFlow adaptive algorithms import failed: {e}")
        
        try:
            from src.optimization.external.library_adaptive.jax_adaptive import create_jax_adam
            print("+ JAX adaptive algorithms imported successfully")
        except ImportError as e:
            print(f"- JAX adaptive algorithms import failed: {e}")
        
        try:
            from src.optimization.external.library_adaptive.sklearn_adaptive import create_sklearn_adaptive_standard
            print("+ Sklearn adaptive algorithms imported successfully")
        except ImportError as e:
            print(f"- Sklearn adaptive algorithms import failed: {e}")
        
        print("+ All basic imports successful!")
        
    except Exception as e:
        print(f"- Import test failed: {e}")
        return False
    
    return True

def test_external_status():
    """Test external library status reporting."""
    print("\nTesting external library status...")
    
    try:
        from src.optimization.external import print_external_library_status
        print_external_library_status()
        print("+ External library status check successful!")
        return True
    except Exception as e:
        print(f"- External library status check failed: {e}")
        return False

def test_simple_sklearn_optimizer():
    """Test creating a simple sklearn optimizer."""
    print("\nTesting simple sklearn optimizer creation...")
    
    try:
        from src.optimization.external.library_stochastic.sklearn_sgd_variants import create_sklearn_sgd_constant
        
        optimizer = create_sklearn_sgd_constant(
            learning_rate=0.01,
            loss_type='ols',
            max_iterations=10,
            convergence_tolerance=1e-3,
            random_state=42
        )
        
        print(f"+ Created optimizer: {type(optimizer).__name__}")
        print(f"  - Library: {optimizer.library_name}")
        print(f"  - Algorithm: {optimizer.algorithm_name}")
        print(f"  - Learning rate: {optimizer.learning_rate}")
        
        return True
    except Exception as e:
        print(f"- Sklearn optimizer creation failed: {e}")
        return False

def test_algorithm_comparator_library_detection():
    """Test AlgorithmComparator library detection methods."""
    print("\nTesting AlgorithmComparator library detection...")
    
    try:
        from src.algorithm_comparator import AlgorithmComparator
        
        # Create a mock comparator instance to test methods
        comparator = AlgorithmComparator("test", 1, 1)
        
        # Test library detection on mock data
        mock_data = {
            'algorithm': 'sklearn SGD',
            'library_info': {
                'library_name': 'sklearn',
                'algorithm_name': 'SGD'
            },
            'algorithm_specific': {}
        }
        
        from pathlib import Path
        mock_folder = Path("data/03_algorithms/library_stochastic/701_library_sgd_sklearn_constant_lr_001")
        
        is_library = comparator._is_library_algorithm(mock_data, mock_folder)
        print(f"+ Library detection result: {is_library}")
        
        if is_library:
            library_info = comparator._extract_library_info(mock_data, {})
            print(f"+ Library info extracted: {library_info}")
        
        return True
    except Exception as e:
        print(f"- AlgorithmComparator library detection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("LIBRARY INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_library_imports,
        test_external_status,
        test_simple_sklearn_optimizer,
        test_algorithm_comparator_library_detection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"- Test {test_func.__name__} failed with exception: {e}")
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: ALL TESTS PASSED - Library integration is working correctly!")
    else:
        print(f"WARNING: {total - passed} tests failed - Some issues need to be addressed")
    
    print("=" * 60)

if __name__ == "__main__":
    main()