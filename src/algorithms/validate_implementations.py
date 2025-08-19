"""
Validation script để đảm bảo tất cả implementations hoạt động đúng
Includes unit tests, integration tests, và mathematical correctness checks
"""

import numpy as np
import sys
import os
from pathlib import Path
import time

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all modules to validate
from newton_method.pure_newton import PureNewtonOptimizer, newton_standard_setup, analytical_solution
from newton_method.damped_newton import DampedNewtonOptimizer, damped_newton_standard_setup
from quasi_newton.bfgs import BFGSOptimizer, bfgs_standard_setup
from quasi_newton.lbfgs import LBFGSOptimizer, lbfgs_standard_setup
from quasi_newton.sr1 import SR1Optimizer, sr1_standard_setup

from utils.optimization_utils import (
    tinh_gradient_hoi_quy_tuyen_tinh,
    tinh_ma_tran_hessian_hoi_quy_tuyen_tinh,
    xac_minh_gradient,
    xac_minh_hessian,
    kiem_tra_positive_definite,
    safe_matrix_inverse,
    giai_he_phuong_trinh_tuyen_tinh,
    tinh_mse,
    compute_r2_score,
    predict
)


class ValidationSuite:
    """
    Comprehensive validation suite cho tất cả optimization implementations
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0
    
    def log(self, message: str):
        """Log message nếu verbose mode"""
        if self.verbose:
            print(message)
    
    def assert_test(self, condition: bool, test_name: str, error_msg: str = ""):
        """Assert test condition and log result"""
        self.total_tests += 1
        
        if condition:
            self.passed_tests += 1
            self.log(f"✅ PASS: {test_name}")
            return True
        else:
            self.log(f"❌ FAIL: {test_name} - {error_msg}")
            return False
    
    def test_utils_functions(self):
        """Test utility functions correctness"""
        self.log("\n=== Testing Utility Functions ===")
        
        # Generate test data
        np.random.seed(42)
        n_samples, n_features = 50, 4
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        weights = np.random.randn(n_features)
        bias = np.random.randn()
        
        # Test gradient computation
        try:
            grad_w, grad_b = tinh_gradient_hoi_quy_tuyen_tinh(X, y, weights, bias, 0.0)
            self.assert_test(
                grad_w.shape == (n_features,) and isinstance(grad_b, (int, float, np.floating)),
                "Gradient computation shapes",
                f"Expected shapes ({n_features},) and scalar, got {grad_w.shape} and {type(grad_b)}"
            )
        except Exception as e:
            self.assert_test(False, "Gradient computation", f"Exception: {e}")
        
        # Test Hessian computation
        try:
            hessian = tinh_ma_tran_hessian_hoi_quy_tuyen_tinh(X, 0.0)
            self.assert_test(
                hessian.shape == (n_features, n_features),
                "Hessian computation shape",
                f"Expected shape ({n_features}, {n_features}), got {hessian.shape}"
            )
            
            # Test Hessian is symmetric
            is_symmetric = np.allclose(hessian, hessian.T)
            self.assert_test(is_symmetric, "Hessian symmetry")
            
        except Exception as e:
            self.assert_test(False, "Hessian computation", f"Exception: {e}")
        
        # Test positive definiteness check
        try:
            pd_matrix = np.array([[2, 1], [1, 2]])  # Positive definite
            non_pd_matrix = np.array([[1, 2], [2, 1]])  # Not positive definite
            
            self.assert_test(kiem_tra_positive_definite(pd_matrix), "Positive definite detection (positive case)")
            self.assert_test(not kiem_tra_positive_definite(non_pd_matrix), "Positive definite detection (negative case)")
        except Exception as e:
            self.assert_test(False, "Positive definite check", f"Exception: {e}")
        
        # Test safe matrix inverse
        try:
            regular_matrix = np.array([[2, 1], [1, 2]])
            singular_matrix = np.array([[1, 1], [1, 1]])
            
            inv_regular = safe_matrix_inverse(regular_matrix)
            inv_singular = safe_matrix_inverse(singular_matrix)
            
            self.assert_test(inv_regular.shape == regular_matrix.shape, "Safe inverse (regular matrix)")
            self.assert_test(inv_singular.shape == singular_matrix.shape, "Safe inverse (singular matrix)")
        except Exception as e:
            self.assert_test(False, "Safe matrix inverse", f"Exception: {e}")
    
    def test_mathematical_correctness(self):
        """Test mathematical correctness của implementations"""
        self.log("\n=== Testing Mathematical Correctness ===")
        
        # Create well-conditioned test problem
        np.random.seed(123)
        n_samples, n_features = 100, 3
        X = np.random.randn(n_samples, n_features)
        true_weights = np.array([1.5, -2.0, 0.5])
        true_bias = 1.0
        noise = 0.01 * np.random.randn(n_samples)
        y = X @ true_weights + true_bias + noise
        
        # Test Pure Newton vs Analytical Solution
        try:
            newton_result = newton_standard_setup(X, y, verbose=False)
            analytical_result = analytical_solution(X, y, regularization=1e-8)
            
            weights_diff = np.linalg.norm(newton_result['weights'] - analytical_result['weights'])
            bias_diff = abs(newton_result['bias'] - analytical_result['bias'])
            
            self.assert_test(
                weights_diff < 1e-6,
                "Newton vs Analytical (weights)",
                f"Weight difference: {weights_diff:.2e}"
            )
            
            self.assert_test(
                bias_diff < 1e-6,
                "Newton vs Analytical (bias)",
                f"Bias difference: {bias_diff:.2e}"
            )
        except Exception as e:
            self.assert_test(False, "Newton vs Analytical comparison", f"Exception: {e}")
        
        # Test convergence to same solution (all methods should converge to similar solution)
        methods_to_test = {
            'Pure Newton': newton_standard_setup,
            'BFGS': bfgs_standard_setup,
            'L-BFGS': lbfgs_standard_setup
        }
        
        method_results = {}
        for method_name, method_func in methods_to_test.items():
            try:
                result = method_func(X, y, verbose=False)
                method_results[method_name] = result
            except Exception as e:
                self.assert_test(False, f"{method_name} execution", f"Exception: {e}")
                continue
        
        # Compare solutions between methods
        if len(method_results) >= 2:
            method_names = list(method_results.keys())
            reference_result = method_results[method_names[0]]
            
            for i in range(1, len(method_names)):
                compare_method = method_names[i]
                compare_result = method_results[compare_method]
                
                weights_diff = np.linalg.norm(
                    reference_result['weights'] - compare_result['weights']
                )
                bias_diff = abs(reference_result['bias'] - compare_result['bias'])
                
                self.assert_test(
                    weights_diff < 1e-3,
                    f"Solution consistency: {method_names[0]} vs {compare_method} (weights)",
                    f"Weight difference: {weights_diff:.2e}"
                )
                
                self.assert_test(
                    bias_diff < 1e-3,
                    f"Solution consistency: {method_names[0]} vs {compare_method} (bias)",
                    f"Bias difference: {bias_diff:.2e}"
                )
    
    def test_optimizer_classes(self):
        """Test optimizer classes directly"""
        self.log("\n=== Testing Optimizer Classes ===")
        
        # Generate test data
        np.random.seed(456)
        n_samples, n_features = 80, 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Test Pure Newton Optimizer
        try:
            optimizer = PureNewtonOptimizer(max_iterations=20, tolerance=1e-6, verbose=False)
            result = optimizer.optimize(X, y)
            
            self.assert_test(
                'weights' in result and 'bias' in result,
                "PureNewtonOptimizer basic functionality"
            )
            
            self.assert_test(
                result['weights'].shape == (n_features,),
                "PureNewtonOptimizer weight shape"
            )
        except Exception as e:
            self.assert_test(False, "PureNewtonOptimizer", f"Exception: {e}")
        
        # Test Damped Newton Optimizer
        try:
            optimizer = DampedNewtonOptimizer(max_iterations=20, tolerance=1e-6, verbose=False)
            result = optimizer.optimize(X, y)
            
            self.assert_test(
                'weights' in result and 'bias' in result and 'step_sizes' in result,
                "DampedNewtonOptimizer basic functionality"
            )
            
            self.assert_test(
                len(result['step_sizes']) > 0,
                "DampedNewtonOptimizer step sizes tracking"
            )
        except Exception as e:
            self.assert_test(False, "DampedNewtonOptimizer", f"Exception: {e}")
        
        # Test BFGS Optimizer
        try:
            optimizer = BFGSOptimizer(max_iterations=20, tolerance=1e-6, verbose=False)
            result = optimizer.optimize(X, y)
            
            self.assert_test(
                'weights' in result and 'final_hessian_approximation' in result,
                "BFGSOptimizer basic functionality"
            )
            
            hessian_approx = result['final_hessian_approximation']
            expected_shape = (n_features + 1, n_features + 1)  # weights + bias
            self.assert_test(
                hessian_approx.shape == expected_shape,
                "BFGSOptimizer Hessian approximation shape",
                f"Expected {expected_shape}, got {hessian_approx.shape}"
            )
        except Exception as e:
            self.assert_test(False, "BFGSOptimizer", f"Exception: {e}")
        
        # Test L-BFGS Optimizer
        try:
            optimizer = LBFGSOptimizer(memory_size=5, max_iterations=20, tolerance=1e-6, verbose=False)
            result = optimizer.optimize(X, y)
            
            self.assert_test(
                'weights' in result and 'memory_usage' in result,
                "LBFGSOptimizer basic functionality"
            )
            
            self.assert_test(
                max(result['memory_usage']) <= 5,
                "LBFGSOptimizer memory limit",
                f"Max memory usage: {max(result['memory_usage'])}"
            )
        except Exception as e:
            self.assert_test(False, "LBFGSOptimizer", f"Exception: {e}")
        
        # Test SR1 Optimizer
        try:
            optimizer = SR1Optimizer(max_iterations=20, tolerance=1e-6, verbose=False)
            result = optimizer.optimize(X, y)
            
            self.assert_test(
                'weights' in result and 'sr1_updates_applied' in result,
                "SR1Optimizer basic functionality"
            )
            
            self.assert_test(
                'positive_definite_history' in result,
                "SR1Optimizer positive definite tracking"
            )
        except Exception as e:
            self.assert_test(False, "SR1Optimizer", f"Exception: {e}")
    
    def test_edge_cases(self):
        """Test edge cases và error handling"""
        self.log("\n=== Testing Edge Cases ===")
        
        # Test với very small dataset
        try:
            X_small = np.array([[1, 2], [3, 4]])  # 2x2
            y_small = np.array([1, 2])
            
            result = newton_standard_setup(X_small, y_small, verbose=False)
            self.assert_test(
                'weights' in result,
                "Small dataset handling"
            )
        except Exception as e:
            self.assert_test(False, "Small dataset handling", f"Exception: {e}")
        
        # Test với ill-conditioned problem
        try:
            # Create ill-conditioned matrix
            np.random.seed(789)
            X_ill = np.random.randn(50, 3)
            X_ill[:, 2] = X_ill[:, 0] + X_ill[:, 1] + 1e-10 * np.random.randn(50)  # Nearly dependent
            y_ill = np.random.randn(50)
            
            # Should handle gracefully with regularization
            result = newton_standard_setup(X_ill, y_ill, verbose=False)
            self.assert_test(
                'weights' in result,
                "Ill-conditioned problem handling"
            )
        except Exception as e:
            self.assert_test(False, "Ill-conditioned problem", f"Exception: {e}")
        
        # Test với zero target
        try:
            X_zero = np.random.randn(30, 3)
            y_zero = np.zeros(30)
            
            result = bfgs_standard_setup(X_zero, y_zero, verbose=False)
            self.assert_test(
                result['final_mse'] < 1e-6,
                "Zero target handling",
                f"Final MSE: {result.get('final_mse', np.inf)}"
            )
        except Exception as e:
            self.assert_test(False, "Zero target handling", f"Exception: {e}")
    
    def test_convergence_behavior(self):
        """Test convergence behavior và stopping criteria"""
        self.log("\n=== Testing Convergence Behavior ===")
        
        # Create easy problem that should converge quickly
        np.random.seed(999)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.01
        
        methods = {
            'Pure Newton': newton_standard_setup,
            'Damped Newton': damped_newton_standard_setup,
            'BFGS': bfgs_standard_setup,
            'L-BFGS': lbfgs_standard_setup,
            'SR1': sr1_standard_setup
        }
        
        for method_name, method_func in methods.items():
            try:
                result = method_func(X, y, verbose=False)
                
                # Check convergence
                converged = result.get('convergence_info', {}).get('converged', False)
                final_mse = result.get('final_mse', np.inf)
                
                self.assert_test(
                    converged,
                    f"{method_name} convergence"
                )
                
                self.assert_test(
                    final_mse < 1e-3,
                    f"{method_name} final accuracy",
                    f"Final MSE: {final_mse:.2e}"
                )
                
            except Exception as e:
                self.assert_test(False, f"{method_name} convergence test", f"Exception: {e}")
    
    def run_all_tests(self):
        """Run all validation tests"""
        start_time = time.time()
        
        self.log("🧪 STARTING COMPREHENSIVE VALIDATION SUITE")
        self.log("=" * 60)
        
        # Run all test categories
        self.test_utils_functions()
        self.test_mathematical_correctness()
        self.test_optimizer_classes()
        self.test_edge_cases()
        self.test_convergence_behavior()
        
        # Summary
        end_time = time.time()
        
        self.log("\n" + "=" * 60)
        self.log("📊 VALIDATION SUMMARY")
        self.log("=" * 60)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        self.log(f"Total Tests: {self.total_tests}")
        self.log(f"Passed: {self.passed_tests}")
        self.log(f"Failed: {self.total_tests - self.passed_tests}")
        self.log(f"Success Rate: {success_rate:.1f}%")
        self.log(f"Validation Time: {end_time - start_time:.2f} seconds")
        
        if success_rate == 100:
            self.log("\n🎉 ALL TESTS PASSED! Implementations are validated.")
        elif success_rate >= 90:
            self.log("\n✅ Most tests passed. Minor issues detected.")
        elif success_rate >= 70:
            self.log("\n⚠️  Some tests failed. Review implementations.")
        else:
            self.log("\n❌ Many tests failed. Major issues detected.")
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'success_rate': success_rate,
            'validation_time': end_time - start_time
        }


def main():
    """Main function để run validation"""
    validator = ValidationSuite(verbose=True)
    results = validator.run_all_tests()
    
    # Save validation report
    output_dir = Path("data/03_algorithms/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "validation_report.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Validation report saved to {output_dir / 'validation_report.json'}")
    
    return results


if __name__ == "__main__":
    main()