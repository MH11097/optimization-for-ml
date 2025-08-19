"""Newton Method - Damped Newton (Giảm tốc)

=== PHIÊN BẢN: DAMPED NEWTON (NEWTON GIẢM TỐC) ===

HÀM LOSS: Hỗ trợ OLS, Ridge, Lasso
Công thức: w_{k+1} = w_k - α_k * H^{-1} * ∇L(w_k)
Trong đó: α_k là learning rate được điều chỉnh

THAM SỐ TỐI ỨU:
Standard Setup:
- Learning Rate: 1.0 (ban đầu)
- Damping Factor: 0.5 (giảm tốc khi cần)
- Max Iterations: 100
- Backtrack iterations: 20

Robust Setup:
- Learning Rate: 0.5 (thận trọng hơn)
- Damping Factor: 0.8 (giảm tốc nhẹ hơn)
- Max Iterations: 200
- Backtrack iterations: 50

Fast Setup:
- Learning Rate: 1.5 (tích cực hơn)
- Damping Factor: 0.3 (giảm tốc nhanh)
- Max Iterations: 50
- Backtrack iterations: 10
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.optimization_utils import (
    tinh_gradient_hoi_quy_tuyen_tinh,
    tinh_ma_tran_hessian_hoi_quy_tuyen_tinh,
    giai_he_phuong_trinh_tuyen_tinh,
    kiem_tra_positive_definite,
    tinh_condition_number,
    in_thong_tin_ma_tran,
    in_thong_tin_gradient
)



class BoToiUuHoaNewtonGiamToc:
    """
    Bộ tối ưu hóa Newton giảm tốc cho Hồi quy tuyến tính
    
    Sử dụng line search (tìm kiếm đường thẳng) để:
    - Đảm bảo hội tụ ổn định
    - Tránh overshooting (vượt quá đích)
    - Điều chỉnh learning rate tự động
    
    Công thức: w_{k+1} = w_k - α_k * H^{-1} * ∇L(w_k)
    Trong đó: α_k được tìm bằng backtracking line search
    """
    
    def __init__(self, 
                 regularization: float = 1e-8,
                 max_iterations: int = 100,
                 tolerance: float = 1e-8,
                 armijo_c1: float = 1e-4,
                 backtrack_rho: float = 0.8,
                 max_line_search_iter: int = 50,
                 verbose: bool = False):
        """
        Khởi tạo Damped Newton Optimizer
        
        Args:
            regularization: hệ số regularization cho Hessian (λ)
            max_iterations: số iteration tối đa
            tolerance: tolerance cho convergence
            armijo_c1: constant cho Armijo condition (sufficient decrease)
            backtrack_rho: factor để giảm step size trong line search
            max_line_search_iter: số iterations tối đa cho line search
            verbose: có in thông tin debug không
        """
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.armijo_c1 = armijo_c1
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.verbose = verbose
        
        # Tracking results
        self.cost_history: List[float] = []
        self.gradient_norms: List[float] = []
        self.step_sizes: List[float] = []
        self.line_search_iterations: List[int] = []
        self.condition_numbers: List[float] = []
        self.convergence_info: Dict = {}
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, 
                     weights: np.ndarray, bias: float) -> float:
        """Tính cost function (MSE với regularization)"""
        predictions = du_doan(X, weights, bias)
        mse = tinh_mse(y, predictions)
        
        # Thêm regularization term
        regularization_term = 0.5 * self.regularization * np.sum(weights**2)
        
        return mse + regularization_term
    
    def _backtracking_line_search(self, X: np.ndarray, y: np.ndarray,
                                 weights: np.ndarray, bias: float,
                                 direction_w: np.ndarray, direction_b: float,
                                 gradient_w: np.ndarray, gradient_b: float) -> Tuple[float, int]:
        """
        Backtracking line search với Armijo condition
        
        Returns:
            step_size: α tìm được
            iterations: số iterations của line search
        """
        current_cost = self._compute_cost(X, y, weights, bias)
        
        # Directional derivative: ∇f^T * d
        directional_derivative = np.dot(gradient_w, direction_w) + gradient_b * direction_b
        
        # Initial step size
        alpha = 1.0
        
        for i in range(self.max_line_search_iter):
            # Thử parameters mới
            new_weights = weights + alpha * direction_w
            new_bias = bias + alpha * direction_b
            
            # Tính cost mới
            new_cost = self._compute_cost(X, y, new_weights, new_bias)
            
            # Kiểm tra Armijo condition: f(x + αd) ≤ f(x) + c₁α∇f^T d
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
        Thực hiện Damped Newton optimization
        
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
        self.step_sizes = []
        self.line_search_iterations = []
        self.condition_numbers = []
        
        start_time = time.time()
        
        if self.verbose:
            print("=== Damped Newton Method Optimization ===")
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Regularization: {self.regularization}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tolerance}")
            print(f"Armijo constant: {self.armijo_c1}")
            print(f"Backtrack factor: {self.backtrack_rho}")
            print()
        
        # Pre-compute Hessian (constant cho linear regression)
        hessian = tinh_ma_tran_hessian_hoi_quy_tuyen_tinh(X, self.regularization)
        condition_number = tinh_condition_number(hessian)
        is_positive_definite = kiem_tra_positive_definite(hessian)
        
        if self.verbose:
            in_thong_tin_ma_tran(hessian, "Hessian Matrix")
            print(f"Is positive definite: {is_positive_definite}")
            print()
        
        if not is_positive_definite:
            print("WARNING: Hessian is not positive definite!")
            print("Consider increasing regularization parameter.")
        
        # Main optimization loop
        for iteration in range(self.max_iterations + 1):
            # Tính cost và gradient
            current_cost = self._compute_cost(X, y, weights, bias)
            gradient_w, gradient_b = tinh_gradient_hoi_quy_tuyen_tinh(
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
            
            # Newton direction: giải H * d = -gradient
            try:
                # Giải cho weights direction
                direction_w = -giai_he_phuong_trinh_tuyen_tinh(hessian, gradient_w)
                
                # Bias direction
                direction_b = -gradient_b
                
                # Line search để tìm step size
                step_size, ls_iterations = self._backtracking_line_search(
                    X, y, weights, bias, direction_w, direction_b, gradient_w, gradient_b
                )
                
                # Lưu line search info
                self.step_sizes.append(step_size)
                self.line_search_iterations.append(ls_iterations)
                
                # Update parameters
                weights = weights + step_size * direction_w
                bias = bias + step_size * direction_b
                
                if self.verbose and iteration % 10 == 0:
                    print(f"         Step size = {step_size:.6f}, "
                          f"Line search iterations = {ls_iterations}")
                
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
        final_predictions = du_doan(X, weights, bias)
        final_mse = tinh_mse(y, final_predictions)
        
        # Prepare results
        results = {
            'weights': weights,
            'bias': bias,
            'cost_history': self.cost_history,
            'gradient_norms': self.gradient_norms,
            'step_sizes': self.step_sizes,
            'line_search_iterations': self.line_search_iterations,
            'condition_numbers': self.condition_numbers,
            'convergence_info': self.convergence_info,
            'final_mse': final_mse,
            'predictions': final_predictions,
            'optimization_time': end_time - start_time,
            'hessian_condition_number': condition_number,
            'method': 'Damped Newton',
            'average_step_size': np.mean(self.step_sizes) if self.step_sizes else 0.0,
            'average_line_search_iterations': np.mean(self.line_search_iterations) if self.line_search_iterations else 0.0
        }
        
        if self.verbose:
            print(f"\nOptimization completed in {end_time - start_time:.4f} seconds")
            print(f"Final MSE: {final_mse:.8f}")
            print(f"Average step size: {results['average_step_size']:.6f}")
            print(f"Average line search iterations: {results['average_line_search_iterations']:.1f}")
            print(f"Hessian condition number: {condition_number:.2e}")
        
        return results


def damped_newton_standard_setup(X: np.ndarray, y: np.ndarray, 
                                verbose: bool = False) -> Dict:
    """
    Standard setup cho Damped Newton Method
    
    Parameters được chọn cho balance giữa robustness và efficiency
    """
    optimizer = DampedNewtonOptimizer(
        regularization=1e-8,        # Minimal regularization
        max_iterations=100,         # More iterations than pure Newton
        tolerance=1e-8,             # Slightly relaxed tolerance
        armijo_c1=1e-4,            # Standard Armijo constant
        backtrack_rho=0.8,         # Conservative backtracking
        max_line_search_iter=50,   # Sufficient line search attempts
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def damped_newton_robust_setup(X: np.ndarray, y: np.ndarray, 
                              verbose: bool = False) -> Dict:
    """
    Robust setup cho Damped Newton Method
    
    Parameters được chọn cho difficult optimization landscapes
    """
    optimizer = DampedNewtonOptimizer(
        regularization=1e-6,        # Higher regularization for stability
        max_iterations=200,         # More iterations allowed
        tolerance=1e-6,             # Relaxed tolerance
        armijo_c1=1e-3,            # Less strict Armijo condition
        backtrack_rho=0.5,         # More aggressive backtracking
        max_line_search_iter=100,  # More line search attempts
        verbose=verbose
    )
    
    return optimizer.optimize(X, y)


def damped_newton_fast_setup(X: np.ndarray, y: np.ndarray, 
                            verbose: bool = False) -> Dict:
    """
    Fast setup cho Damped Newton Method
    
    Parameters được chọn cho quick convergence
    """
    optimizer = DampedNewtonOptimizer(
        regularization=1e-10,       # Minimal regularization
        max_iterations=50,          # Fewer iterations
        tolerance=1e-6,             # Relaxed tolerance
        armijo_c1=1e-4,            # Standard Armijo
        backtrack_rho=0.9,         # Less aggressive backtracking
        max_line_search_iter=20,   # Fewer line search attempts
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
    
    print("Testing Damped Newton Method...")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    print()
    
    # Test different setups
    print("=== Standard Setup ===")
    result_standard = damped_newton_standard_setup(X, y, verbose=True)
    print(f"Learned weights: {result_standard['weights']}")
    print(f"Learned bias: {result_standard['bias']:.6f}")
    print(f"Average step size: {result_standard['average_step_size']:.6f}")
    print(f"Average line search iterations: {result_standard['average_line_search_iterations']:.1f}")
    print()
    
    print("=== Robust Setup ===")
    result_robust = damped_newton_robust_setup(X, y, verbose=True)
    print(f"Learned weights: {result_robust['weights']}")
    print(f"Learned bias: {result_robust['bias']:.6f}")
    print(f"Average step size: {result_robust['average_step_size']:.6f}")
    print(f"Average line search iterations: {result_robust['average_line_search_iterations']:.1f}")
    
    # Compare convergence
    print(f"\nConvergence Comparison:")
    print(f"Standard: {result_standard['convergence_info']['iterations']} iterations")
    print(f"Robust: {result_robust['convergence_info']['iterations']} iterations")