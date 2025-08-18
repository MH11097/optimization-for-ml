"""Implementation of SR1 (Symmetric Rank-1) Quasi-Newton Method cho Linear Regression"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import sys
import os

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.calculus_utils import (
    compute_gradient_linear_regression,
    check_positive_definite,
    compute_condition_number,
    print_gradient_info
)
from utils.optimization_utils import compute_mse, predict


class SR1Optimizer:
    """
    SR1 (Symmetric Rank-1) Quasi-Newton Method implementation cho Linear Regression
    
    SR1 update: B_{k+1} = B_k + [(y_k - B_k s_k)(y_k - B_k s_k)^T] / [(y_k - B_k s_k)^T s_k]
    
    Đặc điểm của SR1:
    - Không đảm bảo positive definiteness
    - Có thể approximate indefinite Hessians
    - Update có thể bị skip nếu denominator quá nhỏ
    - Thường dùng với trust region methods
    """
    
    def __init__(self, 
                 regularization: float = 1e-8,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 sr1_skip_threshold: float = 1e-8,
                 armijo_c1: float = 1e-4,
                 backtrack_rho: float = 0.8,
                 max_line_search_iter: int = 50,
                 restart_frequency: Optional[int] = None,
                 damping_factor: float = 1e-6,
                 verbose: bool = False):
        """
        Khởi tạo SR1 Optimizer
        
        Args:
            regularization: hệ số regularization cho cost function
            max_iterations: số iteration tối đa
            tolerance: tolerance cho convergence
            sr1_skip_threshold: threshold để skip SR1 update khi denominator quá nhỏ
            armijo_c1: constant cho Armijo condition (sufficient decrease)
            backtrack_rho: factor để giảm step size trong line search
            max_line_search_iter: số iterations tối đa cho line search
            restart_frequency: restart approximation every N iterations (None = no restart)
            damping_factor: factor để regularize Hessian approximation nếu cần
            verbose: có in thông tin debug không
        """
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.sr1_skip_threshold = sr1_skip_threshold
        self.armijo_c1 = armijo_c1
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.restart_frequency = restart_frequency
        self.damping_factor = damping_factor
        self.verbose = verbose
        
        # Tracking results
        self.cost_history: List[float] = []
        self.gradient_norms: List[float] = []
        self.step_sizes: List[float] = []
        self.line_search_iterations: List[int] = []
        self.hessian_condition_numbers: List[float] = []
        self.sr1_updates_applied: List[bool] = []
        self.positive_definite_history: List[bool] = []
        self.convergence_info: Dict = {}
    
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
    
    def _sr1_update(self, B: np.ndarray, s: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        SR1 update cho Hessian approximation
        
        B_{k+1} = B_k + [(y - B s)(y - B s)^T] / [(y - B s)^T s]
        
        Args:
            B: current Hessian approximation
            s: step vector (x_{k+1} - x_k)
            y: gradient difference (∇f_{k+1} - ∇f_k)
        
        Returns:
            B_new: updated Hessian approximation
            update_applied: whether update was applied
        """
        # Compute y - B*s
        Bs = B @ s
        v = y - Bs
        
        # Check denominator: v^T s
        vs = np.dot(v, s)
        
        # Skip update if denominator is too small
        if abs(vs) < self.sr1_skip_threshold:
            if self.verbose:
                print(f"Warning: SR1 update skipped: |v^T s| = {abs(vs):.2e} < {self.sr1_skip_threshold:.2e}")
            return B, False
        
        # SR1 update: B + (v v^T) / (v^T s)
        B_new = B + np.outer(v, v) / vs
        
        return B_new, True
    
    def _regularize_hessian_if_needed(self, B: np.ndarray) -> np.ndarray:
        """
        Regularize Hessian approximation nếu không positive definite
        
        Args:
            B: Hessian approximation
        
        Returns:
            B_regularized: regularized Hessian approximation
        """
        is_pos_def = check_positive_definite(B)
        
        if not is_pos_def:
            # Add regularization to make it positive definite
            eigenvals = np.linalg.eigvals(B)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval <= 0:
                regularization = abs(min_eigenval) + self.damping_factor
                B_regularized = B + regularization * np.eye(B.shape[0])
                
                if self.verbose:
                    print(f"         Regularized Hessian with λ = {regularization:.2e}")
                
                return B_regularized
        
        return B
    
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
        
        # Nếu direction không phải descent direction, dùng steepest descent
        if directional_derivative >= 0:
            if self.verbose:
                print(f"Warning: Non-descent direction detected, using steepest descent")
            direction = -gradient
            directional_derivative = -np.dot(gradient, gradient)
        
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
        Thực hiện SR1 optimization
        
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
        
        # Khởi tạo Hessian approximation với identity matrix
        n_params = n_features + 1  # weights + bias
        B = np.eye(n_params)
        
        # Reset tracking
        self.cost_history = []
        self.gradient_norms = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.hessian_condition_numbers = []
        self.sr1_updates_applied = []
        self.positive_definite_history = []
        
        start_time = time.time()
        
        if self.verbose:
            print("=== SR1 (Symmetric Rank-1) Quasi-Newton Method ===")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Regularization: {self.regularization}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tolerance}")
            print(f"SR1 skip threshold: {self.sr1_skip_threshold}")
            print(f"Damping factor: {self.damping_factor}")
            print()
        
        # Previous gradient và params cho SR1 update
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
            
            # Analyze current Hessian approximation
            condition_number = compute_condition_number(B)
            is_positive_definite = check_positive_definite(B)
            
            # Lưu tracking info
            self.cost_history.append(current_cost)
            self.gradient_norms.append(gradient_norm)
            self.hessian_condition_numbers.append(condition_number)
            self.positive_definite_history.append(is_positive_definite)
            
            if self.verbose and iteration % 10 == 0:
                pos_def_status = "PD" if is_positive_definite else "Non-PD"
                print(f"Iteration {iteration:3d}: Cost = {current_cost:.8f}, "
                      f"||grad|| = {gradient_norm:.2e}, κ(B) = {condition_number:.2e} ({pos_def_status})")
            
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
                    'final_condition_number': condition_number,
                    'final_positive_definite': is_positive_definite
                }
                
                if self.verbose:
                    print(f"\nConverged after {iteration} iterations")
                    print(f"Reason: {reason}")
                    print(f"Final cost: {current_cost:.8f}")
                    print(f"Final gradient norm: {gradient_norm:.2e}")
                    print(f"Final Hessian positive definite: {is_positive_definite}")
                
                break
            
            # SR1 update (từ iteration thứ 2)
            update_applied = False
            if iteration > 0 and prev_gradient is not None:
                # s = x_k - x_{k-1}
                s = current_combined_params - prev_combined_params
                # y = ∇f_k - ∇f_{k-1}
                y = current_gradient - prev_gradient
                
                # SR1 update
                B, update_applied = self._sr1_update(B, s, y)
                self.sr1_updates_applied.append(update_applied)
                
                if not update_applied and self.verbose:
                    print(f"         SR1 update skipped at iteration {iteration}")
            
            # Restart SR1 approximation nếu cần
            if (self.restart_frequency is not None and 
                iteration > 0 and iteration % self.restart_frequency == 0):
                B = np.eye(n_params)
                if self.verbose:
                    print(f"         Restarted SR1 approximation at iteration {iteration}")
            
            # Regularize Hessian nếu cần để ensure descent direction
            B_regularized = self._regularize_hessian_if_needed(B)
            
            # Compute search direction: solve B * d = -∇f
            try:
                # Giải hệ phương trình thay vì invert matrix
                direction = np.linalg.solve(B_regularized, -current_gradient)
                
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
                    update_status = "Updated" if update_applied else "Skipped"
                    print(f"         Step size = {step_size:.6f}, "
                          f"Line search = {ls_iterations}, SR1 = {update_status}")
                
                # Save for next SR1 update
                prev_gradient = current_gradient.copy()
                prev_combined_params = current_combined_params.copy()
                
            except Exception as e:
                print(f"Error in SR1 step at iteration {iteration}: {e}")
                self.convergence_info = {
                    'converged': False,
                    'reason': f'SR1 step failed: {e}',
                    'iterations': iteration,
                    'final_cost': current_cost,
                    'final_gradient_norm': gradient_norm,
                    'final_condition_number': condition_number,
                    'final_positive_definite': is_positive_definite
                }
                break
        
        end_time = time.time()
        
        # Tính final metrics
        final_predictions = predict(X, weights, bias)
        final_mse = compute_mse(y, final_predictions)
        
        # Statistics
        sr1_success_rate = (np.sum(self.sr1_updates_applied) / 
                           len(self.sr1_updates_applied) if self.sr1_updates_applied else 0.0)
        positive_definite_rate = (np.sum(self.positive_definite_history) / 
                                len(self.positive_definite_history) if self.positive_definite_history else 0.0)
        
        # Prepare results
        results = {
            'weights': weights,
            'bias': bias,
            'cost_history': self.cost_history,
            'gradient_norms': self.gradient_norms,
            'step_sizes': self.step_sizes,
            'line_search_iterations': self.line_search_iterations,
            'hessian_condition_numbers': self.hessian_condition_numbers,
            'sr1_updates_applied': self.sr1_updates_applied,
            'positive_definite_history': self.positive_definite_history,
            'convergence_info': self.convergence_info,
            'final_mse': final_mse,
            'predictions': final_predictions,
            'optimization_time': end_time - start_time,
            'final_hessian_approximation': B,
            'method': 'SR1',
            'average_step_size': np.mean(self.step_sizes) if self.step_sizes else 0.0,
            'average_line_search_iterations': np.mean(self.line_search_iterations) if self.line_search_iterations else 0.0,
            'sr1_success_rate': sr1_success_rate,
            'positive_definite_rate': positive_definite_rate
        }
        
        if self.verbose:
            print(f"\nOptimization completed in {end_time - start_time:.4f} seconds")
            print(f"Final MSE: {final_mse:.8f}")
            print(f"Average step size: {results['average_step_size']:.6f}")
            print(f"Average line search iterations: {results['average_line_search_iterations']:.1f}")
            print(f"SR1 update success rate: {sr1_success_rate:.1%}")
            print(f"Positive definite rate: {positive_definite_rate:.1%}")
        
        return results


def sr1_standard_setup(X: np.ndarray, y: np.ndarray, 
                      verbose: bool = False) -> Dict:
    """
    Standard setup cho SR1 Method
    
    Parameters được chọn cho balance giữa stability và performance
    """
    optimizer = SR1Optimizer(
        regularization=1e-8,            # Minimal regularization
        max_iterations=100,             # Standard iterations
        tolerance=1e-6,                 # Standard tolerance
        sr1_skip_threshold=1e-8,        # Standard skip threshold
        armijo_c1=1e-4,                # Standard Armijo constant
        backtrack_rho=0.8,             # Conservative backtracking
        max_line_search_iter=50,       # Sufficient line search attempts
        restart_frequency=None,         # No restart for standard setup
        damping_factor=1e-6,           # Small damping for regularization
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def sr1_robust_setup(X: np.ndarray, y: np.ndarray, 
                    verbose: bool = False) -> Dict:
    """
    Robust setup cho SR1 Method
    
    Parameters được chọn cho challenging optimization problems
    """
    optimizer = SR1Optimizer(
        regularization=1e-6,            # Higher regularization
        max_iterations=200,             # More iterations allowed
        tolerance=1e-6,                 # Same tolerance
        sr1_skip_threshold=1e-6,        # More conservative skip threshold
        armijo_c1=1e-3,                # Less strict Armijo condition
        backtrack_rho=0.5,             # More aggressive backtracking
        max_line_search_iter=100,      # More line search attempts
        restart_frequency=50,           # Restart every 50 iterations
        damping_factor=1e-4,           # Higher damping for stability
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def sr1_aggressive_setup(X: np.ndarray, y: np.ndarray, 
                        verbose: bool = False) -> Dict:
    """
    Aggressive setup cho SR1 Method
    
    Parameters cho cases where indefinite Hessian approximation is acceptable
    """
    optimizer = SR1Optimizer(
        regularization=1e-10,           # Minimal regularization
        max_iterations=150,             # Moderate iterations
        tolerance=1e-6,                 # Standard tolerance
        sr1_skip_threshold=1e-10,       # Very permissive skip threshold
        armijo_c1=1e-4,                # Standard Armijo constant
        backtrack_rho=0.9,             # Less aggressive backtracking
        max_line_search_iter=30,       # Fewer line search attempts
        restart_frequency=None,         # No restart
        damping_factor=1e-8,           # Minimal damping
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


if __name__ == "__main__":
    # Test với synthetic data
    np.random.seed(42)
    
    # Generate test data
    n_samples, n_features = 100, 8
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    noise = 0.1 * np.random.randn(n_samples)
    y = X @ true_weights + true_bias + noise
    
    print("Testing SR1 (Symmetric Rank-1) Method...")
    print(f"Problem size: {n_samples} samples, {n_features} features")
    print(f"True bias: {true_bias}")
    print()
    
    # Test different setups
    print("=== Standard Setup ===")
    result_standard = sr1_standard_setup(X, y, verbose=True)
    print(f"Learned bias: {result_standard['bias']:.6f}")
    print(f"SR1 success rate: {result_standard['sr1_success_rate']:.1%}")
    print(f"Positive definite rate: {result_standard['positive_definite_rate']:.1%}")
    print()
    
    print("=== Robust Setup ===")
    result_robust = sr1_robust_setup(X, y, verbose=True)
    print(f"Learned bias: {result_robust['bias']:.6f}")
    print(f"SR1 success rate: {result_robust['sr1_success_rate']:.1%}")
    print(f"Positive definite rate: {result_robust['positive_definite_rate']:.1%}")
    print()
    
    print("=== Aggressive Setup ===")
    result_aggressive = sr1_aggressive_setup(X, y, verbose=True)
    print(f"Learned bias: {result_aggressive['bias']:.6f}")
    print(f"SR1 success rate: {result_aggressive['sr1_success_rate']:.1%}")
    print(f"Positive definite rate: {result_aggressive['positive_definite_rate']:.1%}")
    
    # Compare convergence
    print(f"\nConvergence Comparison:")
    print(f"Standard: {result_standard['convergence_info']['iterations']} iterations")
    print(f"Robust: {result_robust['convergence_info']['iterations']} iterations") 
    print(f"Aggressive: {result_aggressive['convergence_info']['iterations']} iterations")