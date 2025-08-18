"""Implementation of BFGS Quasi-Newton Method cho Linear Regression"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.calculus_utils import (
    compute_gradient_linear_regression,
    check_positive_definite,
    compute_condition_number,
    print_matrix_info,
    print_gradient_info
)
from utils.optimization_utils import compute_mse, predict


class BFGSOptimizer:
    """
    BFGS Quasi-Newton Method implementation cho Linear Regression
    
    BFGS builds approximation to inverse Hessian using gradient information
    từ previous iterations. Secant condition: B_{k+1} s_k = y_k
    """
    
    def __init__(self, 
                 regularization: float = 1e-8,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 armijo_c1: float = 1e-4,
                 wolfe_c2: float = 0.9,
                 backtrack_rho: float = 0.8,
                 max_line_search_iter: int = 50,
                 restart_frequency: Optional[int] = None,
                 verbose: bool = False):
        """
        Khởi tạo BFGS Optimizer
        
        Args:
            regularization: hệ số regularization cho cost function
            max_iterations: số iteration tối đa
            tolerance: tolerance cho convergence
            armijo_c1: constant cho Armijo condition (sufficient decrease)
            wolfe_c2: constant cho curvature condition (Wolfe conditions)
            backtrack_rho: factor để giảm step size trong line search
            max_line_search_iter: số iterations tối đa cho line search
            restart_frequency: restart BFGS approximation every N iterations (None = no restart)
            verbose: có in thông tin debug không
        """
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.armijo_c1 = armijo_c1
        self.wolfe_c2 = wolfe_c2
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.restart_frequency = restart_frequency
        self.verbose = verbose
        
        # Tracking results
        self.cost_history: List[float] = []
        self.gradient_norms: List[float] = []
        self.step_sizes: List[float] = []
        self.line_search_iterations: List[int] = []
        self.hessian_condition_numbers: List[float] = []
        self.convergence_info: Dict = {}
        self.curvature_conditions: List[bool] = []
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, 
                     weights: np.ndarray, bias: float) -> float:
        """Tính cost function (MSE với regularization)"""
        predictions = predict(X, weights, bias)
        mse = compute_mse(y, predictions)
        
        # Thêm regularization term
        regularization_term = 0.5 * self.regularization * np.sum(weights**2)
        
        return mse + regularization_term
    
    def _compute_combined_gradient(self, X: np.ndarray, y: np.ndarray,
                                 weights: np.ndarray, bias: float) -> np.ndarray:
        """Tính combined gradient cho weights và bias"""
        gradient_w, gradient_b = compute_gradient_linear_regression(
            X, y, weights, bias, self.regularization
        )
        # Combine weights và bias gradients
        return np.concatenate([gradient_w, [gradient_b]])
    
    def _split_combined_params(self, combined_params: np.ndarray) -> Tuple[np.ndarray, float]:
        """Split combined parameters thành weights và bias"""
        weights = combined_params[:-1]
        bias = combined_params[-1]
        return weights, bias
    
    def _bfgs_update(self, H: np.ndarray, s: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        BFGS update cho inverse Hessian approximation
        
        H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T
        
        Args:
            H: current inverse Hessian approximation
            s: step vector (x_{k+1} - x_k)
            y: gradient difference (∇f_{k+1} - ∇f_k)
        
        Returns:
            H_new: updated inverse Hessian approximation
            curvature_ok: whether curvature condition was satisfied
        """
        # Check curvature condition: s^T y > 0
        sy = np.dot(s, y)
        curvature_ok = sy > 1e-8
        
        if not curvature_ok:
            if self.verbose:
                print(f"Warning: Curvature condition violated: s^T y = {sy:.2e}")
            # Skip update, return current H
            return H, False
        
        # BFGS update using Sherman-Morrison-Woodbury formula
        rho = 1.0 / sy
        I = np.eye(len(s))
        
        # A = I - ρ s y^T
        A = I - rho * np.outer(s, y)
        
        # H_{k+1} = A H_k A^T + ρ s s^T
        H_new = A @ H @ A.T + rho * np.outer(s, s)
        
        return H_new, True
    
    def _backtracking_line_search(self, X: np.ndarray, y: np.ndarray,
                                 weights: np.ndarray, bias: float,
                                 direction: np.ndarray, gradient: np.ndarray) -> Tuple[float, int]:
        """
        Backtracking line search với Armijo condition
        
        Returns:
            step_size: α tìm được
            iterations: số iterations của line search
        """
        current_cost = self._compute_cost(X, y, weights, bias)
        
        # Directional derivative: ∇f^T * d
        directional_derivative = np.dot(gradient, direction)
        
        # Initial step size
        alpha = 1.0
        
        for i in range(self.max_line_search_iter):
            # Thử parameters mới
            combined_params = np.concatenate([weights, [bias]])
            new_combined_params = combined_params + alpha * direction
            new_weights, new_bias = self._split_combined_params(new_combined_params)
            
            # Tính cost mới
            new_cost = self._compute_cost(X, y, new_weights, new_bias)
            
            # Kiểm tra Armijo condition
            armijo_condition = current_cost + self.armijo_c1 * alpha * directional_derivative
            
            if new_cost <= armijo_condition:
                return alpha, i + 1
            
            # Giảm step size
            alpha *= self.backtrack_rho
        
        # Nếu line search fail, return very small step
        return alpha, self.max_line_search_iter
    
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
        Thực hiện BFGS optimization
        
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
        
        # Khởi tạo inverse Hessian approximation với identity matrix
        n_params = n_features + 1  # weights + bias
        H = np.eye(n_params)
        
        # Reset tracking
        self.cost_history = []
        self.gradient_norms = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.hessian_condition_numbers = []
        self.curvature_conditions = []
        
        start_time = time.time()
        
        if self.verbose:
            print("=== BFGS Quasi-Newton Method Optimization ===")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Regularization: {self.regularization}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tolerance}")
            print(f"Armijo constant: {self.armijo_c1}")
            print(f"Restart frequency: {self.restart_frequency}")
            print()
        
        # Previous gradient cho BFGS update
        prev_gradient = None
        prev_combined_params = None
        
        # Main optimization loop
        for iteration in range(self.max_iterations + 1):
            # Tính cost và gradient
            current_cost = self._compute_cost(X, y, weights, bias)
            current_gradient = self._compute_combined_gradient(X, y, weights, bias)
            gradient_norm = np.linalg.norm(current_gradient)
            
            # Current combined parameters
            current_combined_params = np.concatenate([weights, [bias]])
            
            # Lưu tracking info
            self.cost_history.append(current_cost)
            self.gradient_norms.append(gradient_norm)
            
            # Condition number của inverse Hessian approximation
            condition_number = compute_condition_number(H)
            self.hessian_condition_numbers.append(condition_number)
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: Cost = {current_cost:.8f}, "
                      f"||grad|| = {gradient_norm:.2e}, κ(H) = {condition_number:.2e}")
            
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
                    'final_condition_number': condition_number
                }
                
                if self.verbose:
                    print(f"\nConverged after {iteration} iterations")
                    print(f"Reason: {reason}")
                    print(f"Final cost: {current_cost:.8f}")
                    print(f"Final gradient norm: {gradient_norm:.2e}")
                
                break
            
            # BFGS update (từ iteration thứ 2)
            if iteration > 0 and prev_gradient is not None:
                # s = x_k - x_{k-1}
                s = current_combined_params - prev_combined_params
                # y = ∇f_k - ∇f_{k-1}
                y = current_gradient - prev_gradient
                
                # BFGS update
                H, curvature_ok = self._bfgs_update(H, s, y)
                self.curvature_conditions.append(curvature_ok)
                
                if not curvature_ok and self.verbose:
                    print(f"         Skipped BFGS update due to curvature condition")
            
            # Restart BFGS approximation nếu cần
            if (self.restart_frequency is not None and 
                iteration > 0 and iteration % self.restart_frequency == 0):
                H = np.eye(n_params)
                if self.verbose:
                    print(f"         Restarted BFGS approximation at iteration {iteration}")
            
            # Compute search direction: d = -H * ∇f
            try:
                direction = -H @ current_gradient
                
                # Line search để tìm step size
                step_size, ls_iterations = self._backtracking_line_search(
                    X, y, weights, bias, direction, current_gradient
                )
                
                # Lưu line search info
                self.step_sizes.append(step_size)
                self.line_search_iterations.append(ls_iterations)
                
                # Update parameters
                new_combined_params = current_combined_params + step_size * direction
                weights, bias = self._split_combined_params(new_combined_params)
                
                if self.verbose and iteration % 10 == 0:
                    print(f"         Step size = {step_size:.6f}, "
                          f"Line search iterations = {ls_iterations}")
                
                # Save for next BFGS update
                prev_gradient = current_gradient.copy()
                prev_combined_params = current_combined_params.copy()
                
            except Exception as e:
                print(f"Error in BFGS step at iteration {iteration}: {e}")
                self.convergence_info = {
                    'converged': False,
                    'reason': f'BFGS step failed: {e}',
                    'iterations': iteration,
                    'final_cost': current_cost,
                    'final_gradient_norm': gradient_norm,
                    'final_condition_number': condition_number
                }
                break
        
        end_time = time.time()
        
        # Tính final metrics
        final_predictions = predict(X, weights, bias)
        final_mse = compute_mse(y, final_predictions)
        
        # Statistics
        curvature_success_rate = (np.sum(self.curvature_conditions) / 
                                len(self.curvature_conditions) if self.curvature_conditions else 0.0)
        
        # Prepare results
        results = {
            'weights': weights,
            'bias': bias,
            'cost_history': self.cost_history,
            'gradient_norms': self.gradient_norms,
            'step_sizes': self.step_sizes,
            'line_search_iterations': self.line_search_iterations,
            'hessian_condition_numbers': self.hessian_condition_numbers,
            'curvature_conditions': self.curvature_conditions,
            'convergence_info': self.convergence_info,
            'final_mse': final_mse,
            'predictions': final_predictions,
            'optimization_time': end_time - start_time,
            'final_hessian_approximation': H,
            'method': 'BFGS',
            'average_step_size': np.mean(self.step_sizes) if self.step_sizes else 0.0,
            'average_line_search_iterations': np.mean(self.line_search_iterations) if self.line_search_iterations else 0.0,
            'curvature_success_rate': curvature_success_rate
        }
        
        if self.verbose:
            print(f"\nOptimization completed in {end_time - start_time:.4f} seconds")
            print(f"Final MSE: {final_mse:.8f}")
            print(f"Average step size: {results['average_step_size']:.6f}")
            print(f"Average line search iterations: {results['average_line_search_iterations']:.1f}")
            print(f"Curvature condition success rate: {curvature_success_rate:.1%}")
            print(f"Final Hessian approximation condition number: {condition_number:.2e}")
        
        return results


def bfgs_standard_setup(X: np.ndarray, y: np.ndarray, 
                       verbose: bool = False) -> Dict:
    """
    Standard setup cho BFGS Method
    
    Parameters được chọn cho robust performance
    """
    optimizer = BFGSOptimizer(
        regularization=1e-8,        # Minimal regularization
        max_iterations=100,         # Sufficient for most problems
        tolerance=1e-6,             # Standard tolerance
        armijo_c1=1e-4,            # Standard Armijo constant
        wolfe_c2=0.9,              # Standard curvature constant
        backtrack_rho=0.8,         # Conservative backtracking
        max_line_search_iter=50,   # Sufficient line search attempts
        restart_frequency=None,     # No restart for standard setup
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def bfgs_robust_setup(X: np.ndarray, y: np.ndarray, 
                     verbose: bool = False) -> Dict:
    """
    Robust setup cho BFGS Method
    
    Parameters được chọn cho ill-conditioned problems
    """
    optimizer = BFGSOptimizer(
        regularization=1e-6,        # Higher regularization
        max_iterations=200,         # More iterations allowed
        tolerance=1e-6,             # Same tolerance
        armijo_c1=1e-3,            # Less strict Armijo condition
        wolfe_c2=0.9,              # Standard curvature constant
        backtrack_rho=0.5,         # More aggressive backtracking
        max_line_search_iter=100,  # More line search attempts
        restart_frequency=50,       # Restart every 50 iterations
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def bfgs_fast_setup(X: np.ndarray, y: np.ndarray, 
                   verbose: bool = False) -> Dict:
    """
    Fast setup cho BFGS Method
    
    Parameters được chọn cho quick convergence
    """
    optimizer = BFGSOptimizer(
        regularization=1e-10,       # Minimal regularization
        max_iterations=50,          # Fewer iterations
        tolerance=1e-5,             # Relaxed tolerance
        armijo_c1=1e-4,            # Standard Armijo
        wolfe_c2=0.9,              # Standard curvature constant
        backtrack_rho=0.9,         # Less aggressive backtracking
        max_line_search_iter=20,   # Fewer line search attempts
        restart_frequency=None,     # No restart for fast setup
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


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
    
    print("Testing BFGS Quasi-Newton Method...")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    print()
    
    # Test different setups
    print("=== Standard Setup ===")
    result_standard = bfgs_standard_setup(X, y, verbose=True)
    print(f"Learned weights: {result_standard['weights']}")
    print(f"Learned bias: {result_standard['bias']:.6f}")
    print(f"Average step size: {result_standard['average_step_size']:.6f}")
    print(f"Curvature success rate: {result_standard['curvature_success_rate']:.1%}")
    print()
    
    print("=== Robust Setup ===")
    result_robust = bfgs_robust_setup(X, y, verbose=True)
    print(f"Learned weights: {result_robust['weights']}")
    print(f"Learned bias: {result_robust['bias']:.6f}")
    print(f"Average step size: {result_robust['average_step_size']:.6f}")
    print(f"Curvature success rate: {result_robust['curvature_success_rate']:.1%}")
    
    # Compare convergence
    print(f"\nConvergence Comparison:")
    print(f"Standard: {result_standard['convergence_info']['iterations']} iterations")
    print(f"Robust: {result_robust['convergence_info']['iterations']} iterations")