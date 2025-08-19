"""Newton Method v3 - Lasso Regression (Approximate)

=== PHIÊN BẢN: LASSO REGRESSION (L1 REGULARIZATION) ===

HÀM LOSS: Lasso Regression (Xấp xỉ với Smooth L1)
Công thức: L(w) = (1/2n) * Σ(y_i - ŷ_i)² + λ * Σ|w_i|
Lưu ý: Newton không thể áp dụng trực tiếp cho L1 (không differentiable)
Sử dụng smooth approximation: |w| ≈ sqrt(w² + ε)

THAM SỐ TỐI ỨU:
Standard Setup (OLS/Ridge):
- Regularization: 1e-8 (minimal)
- Max Iterations: 50
- Tolerance: 1e-10 (rất nghiêm ngặt)
- Sử dụng cho: bài toán well-conditioned

Robust Setup (Ridge):
- Regularization: 1e-6 (cao hơn cho stability)
- Max Iterations: 100
- Tolerance: 1e-8 (relaxed hơn)
- Sử dụng cho: bài toán ill-conditioned

Fast Setup (OLS):
- Regularization: 1e-12 (tối thiểu)
- Max Iterations: 20
- Tolerance: 1e-6 (nhanh chóng)
- Sử dụng cho: convergence nhanh
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
    safe_matrix_inverse,
    giai_he_phuong_trinh_tuyen_tinh,
    kiem_tra_positive_definite,
    tinh_condition_number,
    in_thong_tin_ma_tran,
    in_thong_tin_gradient
)



class BoToiUuHoaNewtonLasso:
    """
    Bộ tối ưu hóa Newton cho Lasso Regression (L1 approximation)
    
    Sử dụng công thức: w_{k+1} = w_k - H^{-1} * ∇L(w_k)
    Trong đó:
    - L(w): Hàm loss Lasso ≈ (1/2n) * ||y - Xw||² + λ Σ sqrt(w_i² + ε)
    - H: Ma trận Hessian approximate
    - ∇L(w): Gradient của smooth L1 approximation
    
    Lưu ý: Đây là xấp xỉ vì L1 norm không differentiable tại 0
    """
    
    def __init__(self, 
                 regularization: float = 1e-3,
                 max_iterations: int = 50,
                 tolerance: float = 1e-10,
                 verbose: bool = False):
        """
        Khởi tạo Bộ tối ưu hóa Newton cơ bản
        
        Tham số:
            regularization: hệ số điều chỉnh cho ma trận Hessian (λ)
            max_iterations: số vòng lặp tối đa
            tolerance: ngưỡng sai số cho hội tụ
            verbose: có hiển thị thông tin chi tiết không
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
    
    def _tinh_chi_phi(self, X: np.ndarray, y: np.ndarray, 
                     trong_so: np.ndarray, he_so_tu_do: float) -> float:
        """Tính hàm chi phí (MSE với điều chỉnh)"""
        du_doan = du_doan(X, trong_so, he_so_tu_do)
        mse = tinh_mse(y, du_doan)
        
        # Lasso: sử dụng smooth L1 approximation
        epsilon = 1e-8  # để tránh chia cho 0
        smooth_l1 = self.regularization * np.sum(np.sqrt(trong_so**2 + epsilon))
        
        return mse + smooth_l1
    
    def _kiem_tra_hoi_tu(self, chuan_gradient: float, 
                        thay_doi_chi_phi: float, vong_lap: int) -> Tuple[bool, str]:
        """Kiểm tra điều kiện hội tụ"""
        
        # Hội tụ theo chuẩn gradient
        if chuan_gradient < self.tolerance:
            return True, f"Hội tụ theo chuẩn gradient: {chuan_gradient:.2e} < {self.tolerance:.2e}"
        
        # Hội tụ theo thay đổi chi phí (sau vòng lặp đầu tiên)
        if vong_lap > 0 and abs(thay_doi_chi_phi) < self.tolerance:
            return True, f"Hội tụ theo thay đổi chi phí: {abs(thay_doi_chi_phi):.2e} < {self.tolerance:.2e}"
        
        # Đạt giới hạn vòng lặp
        if vong_lap >= self.max_iterations:
            return True, f"Đạt giới hạn vòng lặp: {vong_lap}"
        
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
            
            # Newton step: giải H * step = gradient
            # Thay vì tính H^{-1}, ta giải hệ phương trình
            try:
                # Giải cho weights step
                weights_step = giai_he_phuong_trinh_tuyen_tinh(hessian, gradient_w)
                
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
        final_predictions = du_doan(X, weights, bias)
        final_mse = tinh_mse(y, final_predictions)
        
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
    weights = giai_he_phuong_trinh_tuyen_tinh(regularized_XTX, XTy)
    
    # Bias computation (assuming we don't regularize bias)
    predictions_no_bias = X @ weights
    bias = np.mean(y - predictions_no_bias)
    
    # Final predictions and metrics
    predictions = du_doan(X, weights, bias)
    mse = tinh_mse(y, predictions)
    
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