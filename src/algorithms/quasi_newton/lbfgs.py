"""Implementation of L-BFGS (Limited Memory BFGS) cho Linear Regression"""

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
    print_gradient_info
)
from utils.optimization_utils import compute_mse, predict


class LBFGSOptimizer:
    """
    L-BFGS (Limited Memory BFGS) implementation cho Linear Regression
    
    L-BFGS maintains limited history của (s, y) pairs thay vì full inverse Hessian
    Memory usage: O(mn) thay vì O(n²), với m là memory size
    """
    
    def __init__(self, 
                 memory_size: int = 10,
                 regularization: float = 1e-8,
                 max_iterations: int = 200,
                 tolerance: float = 1e-6,
                 armijo_c1: float = 1e-4,
                 wolfe_c2: float = 0.9,
                 backtrack_rho: float = 0.8,
                 max_line_search_iter: int = 50,
                 verbose: bool = False):
        """
        Khởi tạo L-BFGS Optimizer
        
        Args:
            memory_size: số lượng (s, y) pairs được lưu trữ (thường 5-20)
            regularization: hệ số regularization cho cost function
            max_iterations: số iteration tối đa
            tolerance: tolerance cho convergence
            armijo_c1: constant cho Armijo condition (sufficient decrease)
            wolfe_c2: constant cho curvature condition (Wolfe conditions)
            backtrack_rho: factor để giảm step size trong line search
            max_line_search_iter: số iterations tối đa cho line search
            verbose: có in thông tin debug không
        """
        self.memory_size = memory_size
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.armijo_c1 = armijo_c1
        self.wolfe_c2 = wolfe_c2
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.verbose = verbose
        
        # L-BFGS memory
        self.s_history: Deque[np.ndarray] = deque(maxlen=memory_size)  # step vectors
        self.y_history: Deque[np.ndarray] = deque(maxlen=memory_size)  # gradient differences
        self.rho_history: Deque[float] = deque(maxlen=memory_size)     # 1/(s^T y)
        
        # Tracking results
        self.cost_history: List[float] = []
        self.gradient_norms: List[float] = []
        self.step_sizes: List[float] = []
        self.line_search_iterations: List[int] = []
        self.memory_usage: List[int] = []
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
    
    def _update_memory(self, s: np.ndarray, y: np.ndarray) -> bool:
        """
        Update L-BFGS memory với new (s, y) pair
        
        Args:
            s: step vector (x_{k+1} - x_k)
            y: gradient difference (∇f_{k+1} - ∇f_k)
        
        Returns:
            success: whether update was successful (curvature condition satisfied)
        """
        # Check curvature condition: s^T y > 0
        sy = np.dot(s, y)
        
        if sy <= 1e-8:
            if self.verbose:
                print(f"Warning: Curvature condition violated: s^T y = {sy:.2e}")
            return False
        
        # Add to memory
        rho = 1.0 / sy
        
        self.s_history.append(s.copy())
        self.y_history.append(y.copy())
        self.rho_history.append(rho)
        
        return True
    
    def _two_loop_recursion(self, gradient: np.ndarray) -> np.ndarray:
        """
        L-BFGS two-loop recursion để compute search direction
        
        Algorithm:
        1. First loop (backward): compute α_i values
        2. Apply initial Hessian scaling
        3. Second loop (forward): compute final direction
        
        Args:
            gradient: current gradient
        
        Returns:
            direction: search direction
        """
        if len(self.s_history) == 0:
            # No history, use steepest descent
            return -gradient
        
        q = gradient.copy()
        alpha = np.zeros(len(self.s_history))
        
        # First loop (backward through history)
        for i in reversed(range(len(self.s_history))):
            alpha[i] = self.rho_history[i] * np.dot(self.s_history[i], q)
            q = q - alpha[i] * self.y_history[i]
        
        # Apply initial Hessian approximation H_0
        if len(self.s_history) > 0:
            # Use scaling: γ_k = (s^T y) / (y^T y) from most recent update
            s_k = self.s_history[-1]
            y_k = self.y_history[-1]
            gamma = np.dot(s_k, y_k) / np.dot(y_k, y_k)
            z = gamma * q
        else:
            z = q
        
        # Second loop (forward through history)
        for i in range(len(self.s_history)):
            beta = self.rho_history[i] * np.dot(self.y_history[i], z)
            z = z + (alpha[i] - beta) * self.s_history[i]
        
        return -z
    
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
        Thực hiện L-BFGS optimization
        
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
        
        # Reset memory và tracking
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()
        
        self.cost_history = []
        self.gradient_norms = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.memory_usage = []
        self.curvature_conditions = []
        
        start_time = time.time()
        
        if self.verbose:
            print("=== L-BFGS (Limited Memory BFGS) Optimization ===")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Memory size: {self.memory_size}")
            print(f"Regularization: {self.regularization}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tolerance}")
            print(f"Armijo constant: {self.armijo_c1}")
            print()
        
        # Previous gradient và params cho L-BFGS update
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
            self.memory_usage.append(len(self.s_history))
            
            if self.verbose and iteration % 20 == 0:
                print(f"Iteration {iteration:3d}: Cost = {current_cost:.8f}, "
                      f"||grad|| = {gradient_norm:.2e}, Memory = {len(self.s_history)}/{self.memory_size}")
            
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
                    'final_memory_usage': len(self.s_history)
                }
                
                if self.verbose:
                    print(f"\nConverged after {iteration} iterations")
                    print(f"Reason: {reason}")
                    print(f"Final cost: {current_cost:.8f}")
                    print(f"Final gradient norm: {gradient_norm:.2e}")
                    print(f"Final memory usage: {len(self.s_history)}/{self.memory_size}")
                
                break
            
            # Update L-BFGS memory (từ iteration thứ 2)
            if iteration > 0 and prev_gradient is not None:
                # s = x_k - x_{k-1}
                s = current_combined_params - prev_combined_params
                # y = ∇f_k - ∇f_{k-1}
                y = current_gradient - prev_gradient
                
                # Update memory
                curvature_ok = self._update_memory(s, y)
                self.curvature_conditions.append(curvature_ok)
                
                if not curvature_ok and self.verbose:
                    print(f"         Skipped memory update due to curvature condition")
            
            # Compute search direction using two-loop recursion
            try:
                direction = self._two_loop_recursion(current_gradient)
                
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
                
                if self.verbose and iteration % 20 == 0:
                    print(f"         Step size = {step_size:.6f}, "
                          f"Line search iterations = {ls_iterations}")
                
                # Save for next L-BFGS update
                prev_gradient = current_gradient.copy()
                prev_combined_params = current_combined_params.copy()
                
            except Exception as e:
                print(f"Error in L-BFGS step at iteration {iteration}: {e}")
                self.convergence_info = {
                    'converged': False,
                    'reason': f'L-BFGS step failed: {e}',
                    'iterations': iteration,
                    'final_cost': current_cost,
                    'final_gradient_norm': gradient_norm,
                    'final_memory_usage': len(self.s_history)
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
            'memory_usage': self.memory_usage,
            'curvature_conditions': self.curvature_conditions,
            'convergence_info': self.convergence_info,
            'final_mse': final_mse,
            'predictions': final_predictions,
            'optimization_time': end_time - start_time,
            'method': 'L-BFGS',
            'memory_size': self.memory_size,
            'average_step_size': np.mean(self.step_sizes) if self.step_sizes else 0.0,
            'average_line_search_iterations': np.mean(self.line_search_iterations) if self.line_search_iterations else 0.0,
            'curvature_success_rate': curvature_success_rate,
            'average_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0.0
        }
        
        if self.verbose:
            print(f"\nOptimization completed in {end_time - start_time:.4f} seconds")
            print(f"Final MSE: {final_mse:.8f}")
            print(f"Average step size: {results['average_step_size']:.6f}")
            print(f"Average line search iterations: {results['average_line_search_iterations']:.1f}")
            print(f"Curvature condition success rate: {curvature_success_rate:.1%}")
            print(f"Average memory usage: {results['average_memory_usage']:.1f}/{self.memory_size}")
        
        return results


def lbfgs_standard_setup(X: np.ndarray, y: np.ndarray, 
                        verbose: bool = False) -> Dict:
    """
    Standard setup cho L-BFGS Method
    
    Parameters được chọn cho balance giữa memory và performance
    """
    optimizer = LBFGSOptimizer(
        memory_size=10,             # Standard memory size
        regularization=1e-8,        # Minimal regularization
        max_iterations=200,         # More iterations than full BFGS
        tolerance=1e-6,             # Standard tolerance
        armijo_c1=1e-4,            # Standard Armijo constant
        wolfe_c2=0.9,              # Standard curvature constant
        backtrack_rho=0.8,         # Conservative backtracking
        max_line_search_iter=50,   # Sufficient line search attempts
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def lbfgs_memory_efficient_setup(X: np.ndarray, y: np.ndarray, 
                                verbose: bool = False) -> Dict:
    """
    Memory efficient setup cho L-BFGS Method
    
    Parameters được chọn cho large-scale problems
    """
    optimizer = LBFGSOptimizer(
        memory_size=5,              # Smaller memory footprint
        regularization=1e-8,        # Minimal regularization
        max_iterations=500,         # More iterations allowed
        tolerance=1e-6,             # Standard tolerance
        armijo_c1=1e-4,            # Standard Armijo constant
        wolfe_c2=0.9,              # Standard curvature constant
        backtrack_rho=0.8,         # Conservative backtracking
        max_line_search_iter=50,   # Sufficient line search attempts
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def lbfgs_high_memory_setup(X: np.ndarray, y: np.ndarray, 
                           verbose: bool = False) -> Dict:
    """
    High memory setup cho L-BFGS Method
    
    Parameters được chọn cho fast convergence với more memory
    """
    optimizer = LBFGSOptimizer(
        memory_size=20,             # Larger memory for better approximation
        regularization=1e-8,        # Minimal regularization
        max_iterations=100,         # Fewer iterations expected
        tolerance=1e-6,             # Standard tolerance
        armijo_c1=1e-4,            # Standard Armijo constant
        wolfe_c2=0.9,              # Standard curvature constant
        backtrack_rho=0.8,         # Conservative backtracking
        max_line_search_iter=50,   # Sufficient line search attempts
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


if __name__ == "__main__":
    # Test với synthetic data
    np.random.seed(42)
    
    # Generate test data
    n_samples, n_features = 1000, 50  # Larger problem to show L-BFGS advantage
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    noise = 0.1 * np.random.randn(n_samples)
    y = X @ true_weights + true_bias + noise
    
    print("Testing L-BFGS Method...")
    print(f"Problem size: {n_samples} samples, {n_features} features")
    print(f"True bias: {true_bias}")
    print()
    
    # Test different setups
    print("=== Standard Setup (memory=10) ===")
    result_standard = lbfgs_standard_setup(X, y, verbose=True)
    print(f"Learned bias: {result_standard['bias']:.6f}")
    print(f"Average step size: {result_standard['average_step_size']:.6f}")
    print(f"Average memory usage: {result_standard['average_memory_usage']:.1f}/10")
    print()
    
    print("=== Memory Efficient Setup (memory=5) ===")
    result_efficient = lbfgs_memory_efficient_setup(X, y, verbose=True)
    print(f"Learned bias: {result_efficient['bias']:.6f}")
    print(f"Average step size: {result_efficient['average_step_size']:.6f}")
    print(f"Average memory usage: {result_efficient['average_memory_usage']:.1f}/5")
    print()
    
    print("=== High Memory Setup (memory=20) ===")
    result_high_mem = lbfgs_high_memory_setup(X, y, verbose=True)
    print(f"Learned bias: {result_high_mem['bias']:.6f}")
    print(f"Average step size: {result_high_mem['average_step_size']:.6f}")
    print(f"Average memory usage: {result_high_mem['average_memory_usage']:.1f}/20")
    
    # Compare convergence
    print(f"\nConvergence Comparison:")
    print(f"Standard (mem=10): {result_standard['convergence_info']['iterations']} iterations")
    print(f"Efficient (mem=5): {result_efficient['convergence_info']['iterations']} iterations")
    print(f"High memory (mem=20): {result_high_mem['convergence_info']['iterations']} iterations")