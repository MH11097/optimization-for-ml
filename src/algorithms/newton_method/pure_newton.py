"""Implementation of Pure Newton Method cho Linear Regression"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.calculus_utils import (
    compute_gradient_linear_regression,
    compute_hessian_linear_regression,
    safe_matrix_inverse,
    solve_linear_system,
    check_positive_definite,
    compute_condition_number,
    print_matrix_info,
    print_gradient_info
)
from utils.optimization_utils import compute_mse, predict


class PureNewtonOptimizer:
    """
    Pure Newton Method implementation cho Linear Regression
    
    Sử dụng công thức: x_{k+1} = x_k - H^{-1} * gradient
    """
    
    def __init__(self, 
                 regularization: float = 1e-8,
                 max_iterations: int = 50,
                 tolerance: float = 1e-10,
                 verbose: bool = False):
        """
        Khởi tạo Pure Newton Optimizer
        
        Args:
            regularization: hệ số regularization cho Hessian (λ)
            max_iterations: số iteration tối đa
            tolerance: tolerance cho convergence
            verbose: có in thông tin debug không
        """
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Tracking results
        self.cost_history: List[float] = []
        self.gradient_norms: List[float] = []
        self.condition_numbers: List[float] = []
        self.convergence_info: Dict = {}
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, 
                     weights: np.ndarray, bias: float) -> float:
        """Tính cost function (MSE với regularization)"""
        predictions = predict(X, weights, bias)
        mse = compute_mse(y, predictions)
        
        # Thêm regularization term
        regularization_term = 0.5 * self.regularization * np.sum(weights**2)
        
        return mse + regularization_term
    
    def _check_convergence(self, gradient_norm: float, 
                          cost_change: float, iteration: int) -> Tuple[bool, str]:
        """Kiểm tra điều kiện convergence"""
        
        # Gradient norm convergence
        if gradient_norm < self.tolerance:
            return True, f"Gradient norm convergence: {gradient_norm:.2e} < {self.tolerance:.2e}"
        
        # Cost change convergence (sau iteration đầu tiên)
        if iteration > 0 and abs(cost_change) < self.tolerance:
            return True, f"Cost change convergence: {abs(cost_change):.2e} < {self.tolerance:.2e}"
        
        # Max iterations
        if iteration >= self.max_iterations:
            return True, f"Max iterations reached: {iteration}"
        
        return False, ""
    
    def optimize(self, X: np.ndarray, y: np.ndarray,
                initial_weights: Optional[np.ndarray] = None,
                initial_bias: float = 0.0) -> Dict:
        """
        Thực hiện Pure Newton optimization
        
        Args:
            X: ma trận đặc trưng (n_samples, n_features)
            y: vector target (n_samples,)
            initial_weights: trọng số khởi tạo (nếu None sẽ dùng zeros)
            initial_bias: bias khởi tạo
        
        Returns:
            results: dictionary chứa kết quả optimization
        """
        # Khởi tạo
        n_samples, n_features = X.shape
        
        if initial_weights is None:
            weights = np.zeros(n_features)
        else:
            weights = initial_weights.copy()
        
        bias = initial_bias
        
        # Reset tracking
        self.cost_history = []
        self.gradient_norms = []
        self.condition_numbers = []
        
        start_time = time.time()
        
        if self.verbose:
            print("=== Pure Newton Method Optimization ===")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Regularization: {self.regularization}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tolerance}")
            print()
        
        # Pre-compute Hessian (constant cho linear regression)
        hessian = compute_hessian_linear_regression(X, self.regularization)
        condition_number = compute_condition_number(hessian)
        is_positive_definite = check_positive_definite(hessian)
        
        if self.verbose:
            print_matrix_info(hessian, "Hessian Matrix")
            print(f"Is positive definite: {is_positive_definite}")
            print()
        
        if not is_positive_definite:
            print("WARNING: Hessian is not positive definite!")
            print("Consider increasing regularization parameter.")
        
        # Main optimization loop
        for iteration in range(self.max_iterations + 1):
            # Tính cost và gradient
            current_cost = self._compute_cost(X, y, weights, bias)
            gradient_w, gradient_b = compute_gradient_linear_regression(
                X, y, weights, bias, self.regularization
            )
            
            # Tính gradient norm (kết hợp weights và bias)
            gradient_norm = np.sqrt(np.sum(gradient_w**2) + gradient_b**2)
            
            # Lưu tracking info
            self.cost_history.append(current_cost)
            self.gradient_norms.append(gradient_norm)
            self.condition_numbers.append(condition_number)
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: Cost = {current_cost:.8f}, "
                      f"||grad|| = {gradient_norm:.2e}")
            
            # Kiểm tra convergence
            cost_change = 0.0 if iteration == 0 else (self.cost_history[-2] - current_cost)
            converged, reason = self._check_convergence(gradient_norm, cost_change, iteration)
            
            if converged:
                self.convergence_info = {
                    'converged': True,
                    'reason': reason,
                    'iterations': iteration,
                    'final_cost': current_cost,
                    'final_gradient_norm': gradient_norm,
                    'condition_number': condition_number
                }
                
                if self.verbose:
                    print(f"\nConverged after {iteration} iterations")
                    print(f"Reason: {reason}")
                    print(f"Final cost: {current_cost:.8f}")
                    print(f"Final gradient norm: {gradient_norm:.2e}")
                
                break
            
            # Newton step: giải H * step = gradient
            # Thay vì tính H^{-1}, ta giải hệ phương trình
            try:
                # Giải cho weights step
                weights_step = solve_linear_system(hessian, gradient_w)
                
                # Bias step (không có cross terms với weights trong Hessian)
                bias_step = gradient_b
                
                # Update parameters
                weights = weights - weights_step
                bias = bias - bias_step
                
            except Exception as e:
                print(f"Error in Newton step at iteration {iteration}: {e}")
                self.convergence_info = {
                    'converged': False,
                    'reason': f'Newton step failed: {e}',
                    'iterations': iteration,
                    'final_cost': current_cost,
                    'final_gradient_norm': gradient_norm,
                    'condition_number': condition_number
                }
                break
        
        end_time = time.time()
        
        # Tính final metrics
        final_predictions = predict(X, weights, bias)
        final_mse = compute_mse(y, final_predictions)
        
        # Prepare results
        results = {
            'weights': weights,
            'bias': bias,
            'cost_history': self.cost_history,
            'gradient_norms': self.gradient_norms,
            'condition_numbers': self.condition_numbers,
            'convergence_info': self.convergence_info,
            'final_mse': final_mse,
            'predictions': final_predictions,
            'optimization_time': end_time - start_time,
            'hessian_condition_number': condition_number,
            'method': 'Pure Newton'
        }
        
        if self.verbose:
            print(f"\nOptimization completed in {end_time - start_time:.4f} seconds")
            print(f"Final MSE: {final_mse:.8f}")
            print(f"Hessian condition number: {condition_number:.2e}")
        
        return results


def newton_standard_setup(X: np.ndarray, y: np.ndarray, 
                         verbose: bool = False) -> Dict:
    """
    Standard setup cho Pure Newton Method
    
    Parameters được chọn cho most common use cases
    """
    optimizer = PureNewtonOptimizer(
        regularization=1e-8,    # Minimal regularization
        max_iterations=50,      # Usually enough for Newton
        tolerance=1e-10,        # Very strict tolerance
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def newton_robust_setup(X: np.ndarray, y: np.ndarray, 
                       verbose: bool = False) -> Dict:
    """
    Robust setup cho Pure Newton Method
    
    Parameters được chọn cho ill-conditioned problems
    """
    optimizer = PureNewtonOptimizer(
        regularization=1e-6,    # Higher regularization for stability
        max_iterations=100,     # More iterations allowed
        tolerance=1e-8,         # Slightly relaxed tolerance
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def newton_fast_setup(X: np.ndarray, y: np.ndarray, 
                     verbose: bool = False) -> Dict:
    """
    Fast setup cho Pure Newton Method
    
    Parameters được chọn cho quick convergence
    """
    optimizer = PureNewtonOptimizer(
        regularization=1e-12,   # Minimal regularization
        max_iterations=20,      # Few iterations
        tolerance=1e-6,         # Relaxed tolerance
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


# Convenience function để compare với analytical solution
def analytical_solution(X: np.ndarray, y: np.ndarray, 
                       regularization: float = 1e-8) -> Dict:
    """
    Tính analytical solution cho linear regression
    
    Solution: w* = (X^T X + λI)^{-1} X^T y
    """
    n_features = X.shape[1]
    
    # Compute closed-form solution
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Add regularization
    regularized_XTX = XTX + regularization * np.eye(n_features)
    
    # Solve: (X^T X + λI) w = X^T y
    weights = solve_linear_system(regularized_XTX, XTy)
    
    # Bias computation (assuming we don't regularize bias)
    predictions_no_bias = X @ weights
    bias = np.mean(y - predictions_no_bias)
    
    # Final predictions and metrics
    predictions = predict(X, weights, bias)
    mse = compute_mse(y, predictions)
    
    return {
        'weights': weights,
        'bias': bias,
        'final_mse': mse,
        'predictions': predictions,
        'method': 'Analytical Solution'
    }


if __name__ == "__main__":
    # Test với synthetic data
    np.random.seed(42)
    
    # Generate test data
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    noise = 0.1 * np.random.randn(n_samples)
    y = X @ true_weights + true_bias + noise
    
    print("Testing Pure Newton Method...")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    print()
    
    # Test different setups
    print("=== Standard Setup ===")
    result_standard = newton_standard_setup(X, y, verbose=True)
    print(f"Learned weights: {result_standard['weights']}")
    print(f"Learned bias: {result_standard['bias']:.6f}")
    print()
    
    print("=== Analytical Solution ===")
    result_analytical = analytical_solution(X, y)
    print(f"Analytical weights: {result_analytical['weights']}")
    print(f"Analytical bias: {result_analytical['bias']:.6f}")
    print(f"Analytical MSE: {result_analytical['final_mse']:.8f}")
    
    # Compare results
    weights_diff = np.linalg.norm(result_standard['weights'] - result_analytical['weights'])
    bias_diff = abs(result_standard['bias'] - result_analytical['bias'])
    
    print(f"\nDifference from analytical solution:")
    print(f"Weights difference: {weights_diff:.2e}")
    print(f"Bias difference: {bias_diff:.2e}")