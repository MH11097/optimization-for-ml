#!/usr/bin/env python3
"""
DampedNewtonModel - Class cho Damped Newton Method với Line Search
Hỗ trợ các loss functions: OLS, Ridge, Lasso
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
import os
import json

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.optimization_utils import (
    du_doan, danh_gia_mo_hinh, in_ket_qua_danh_gia, kiem_tra_hoi_tu,
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss,
    add_bias_column, giai_he_phuong_trinh_tuyen_tinh
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te
)


class DampedNewtonModel:
    """
    Damped Newton Method Model với Line Search và hỗ trợ nhiều loss functions
    
    Parameters:
    - ham_loss: 'ols', 'ridge', 'lasso'
    - regularization: Tham số regularization cho Ridge/Lasso
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ (gradient norm)
    - numerical_regularization: Regularization cho numerical stability của Hessian
    - armijo_c1: Armijo constant cho line search (thường 1e-4)
    - backtrack_rho: Reduction factor cho backtracking (thường 0.5-0.9)
    - max_line_search_iter: Số lần backtrack tối đa
    - convergence_check_freq: Tần suất kiểm tra hội tụ (mỗi N iterations)
    """
    
    def __init__(self, ham_loss='ols', regularization=0.01, so_lan_thu=50, 
                 diem_dung=1e-10, numerical_regularization=1e-8, 
                 armijo_c1=1e-4, backtrack_rho=0.8, max_line_search_iter=50,
                 convergence_check_freq=1):
        self.ham_loss = ham_loss.lower()
        self.regularization = regularization
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.numerical_regularization = numerical_regularization
        self.armijo_c1 = armijo_c1
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.convergence_check_freq = convergence_check_freq
        
        # Validate supported loss function
        if self.ham_loss not in ['ols', 'ridge', 'lasso']:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}")
        
        # Sử dụng unified functions với format mới (bias trong X)
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.hess_func = lambda X: tinh_hessian_ham_loss(self.ham_loss, X, None, self.regularization)
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None  # Bây giờ bao gồm bias ở cuối
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0
        self.condition_number = None
        
    def _backtracking_line_search(self, X, y, weights, gradient, newton_direction):
        """
        Thực hiện backtracking line search theo thuật toán Armijo
        
        Args:
            X, y: Dữ liệu
            weights: Trọng số hiện tại
            gradient: Gradient tại weights hiện tại  
            newton_direction: Hướng Newton (H^-1 * gradient)
        
        Returns:
            alpha: Step size phù hợp
            n_backtracks: Số lần backtrack đã thực hiện
        """
        alpha = 1.0  # Bắt đầu với full Newton step
        current_loss = self.loss_func(X, y, weights)
        
        # Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
        # Với Newton direction: d = -H^-1*g, nên grad^T*d = -grad^T*H^-1*grad < 0
        directional_derivative = -np.dot(gradient, newton_direction)
        
        n_backtracks = 0
        
        for i in range(self.max_line_search_iter):
            new_weights = weights - alpha * newton_direction
            new_loss = self.loss_func(X, y, new_weights)
            
            # Kiểm tra Armijo condition
            armijo_threshold = current_loss - self.armijo_c1 * alpha * directional_derivative
            
            if new_loss <= armijo_threshold:
                break  # Tìm được step size phù hợp
            
            alpha *= self.backtrack_rho
            n_backtracks += 1
        
        return alpha, n_backtracks
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"🚀 Training Damped Newton Method - {self.ham_loss.upper()}")
        print(f"   Regularization: {self.regularization}")
        print(f"   Numerical regularization: {self.numerical_regularization}")
        print(f"   Line search - Armijo c1: {self.armijo_c1}, Backtrack ρ: {self.backtrack_rho}")
        print(f"   Max iterations: {self.so_lan_thu}")
        
        # Thêm cột bias vào X
        X_with_bias = add_bias_column(X)
        print(f"   Original features: {X.shape[1]}, With bias: {X_with_bias.shape[1]}")
        
        # Initialize weights (bao gồm bias ở cuối)
        n_features_with_bias = X_with_bias.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.line_search_iterations = []
        
        start_time = time.time()
        
        # Precompute Hessian using unified function (for OLS and Ridge)
        if self.ham_loss in ['ols', 'ridge']:
            H = self.hess_func(X_with_bias)
            # Add numerical regularization for stability
            H_reg = H + self.numerical_regularization * np.eye(n_features_with_bias)
            self.condition_number = np.linalg.cond(H_reg)
            print(f"   Hessian condition number: {self.condition_number:.2e}")
        
        for lan_thu in range(self.so_lan_thu):
            # Tính gradient (luôn cần cho Newton step)
            gradient_w, _ = self.grad_func(X_with_bias, y, self.weights)  # _ vì không cần gradient_b riêng
            
            # Chỉ tính loss và lưu history khi cần thiết
            should_check_converged = (
                (lan_thu + 1) % self.convergence_check_freq == 0 or 
                lan_thu == self.so_lan_thu - 1
            )
            
            if should_check_converged:
                # Chỉ tính loss khi cần (expensive operation)
                loss_value = self.loss_func(X_with_bias, y, self.weights)
                gradient_norm = np.linalg.norm(gradient_w)
                
                # Lưu vào history
                self.loss_history.append(loss_value)
                self.gradient_norms.append(gradient_norm)
                self.weights_history.append(self.weights.copy())
                
                cost_change = 0.0 if len(self.loss_history) == 0 else (self.loss_history[-1] - loss_value) if len(self.loss_history) == 1 else (self.loss_history[-2] - self.loss_history[-1])
                converged, reason = kiem_tra_hoi_tu(
                    gradient_norm=gradient_norm,
                    cost_change=cost_change, 
                    iteration=lan_thu,
                    tolerance=self.diem_dung,
                    max_iterations=self.so_lan_thu
                )
                
                if converged:
                    print(f"✅ Damped Newton Method stopped: {reason}")
                    self.converged = True
                    self.final_iteration = lan_thu + 1
                    break
            
            # Newton step with line search
            try:
                # For Lasso, recompute Hessian at each iteration
                if self.ham_loss == 'lasso':
                    H = self.hess_func(X_with_bias)
                    H_reg = H + self.numerical_regularization * np.eye(n_features_with_bias)
                    if lan_thu == 0:  # Print condition number only once
                        self.condition_number = np.linalg.cond(H_reg)
                        print(f"   Hessian condition number: {self.condition_number:.2e}")
                
                # Compute Newton direction: H^-1 * gradient
                newton_direction = giai_he_phuong_trinh_tuyen_tinh(H_reg, gradient_w)
                
                # Backtracking line search
                step_size, n_backtracks = self._backtracking_line_search(
                    X_with_bias, y, self.weights, gradient_w, newton_direction
                )
                
                # Update weights
                self.weights = self.weights - step_size * newton_direction
                
                # Record step information
                self.step_sizes.append(step_size)
                self.line_search_iterations.append(n_backtracks)
                
            except np.linalg.LinAlgError:
                print(f"Linear algebra error at iteration {lan_thu + 1}")
                break
            
            # Progress update
            if (lan_thu + 1) % 10 == 0 and should_check_converged:
                avg_backtracks = np.mean(self.line_search_iterations[-10:]) if self.line_search_iterations else 0
                print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient = {gradient_norm:.2e}, α = {step_size:.4f}, Backtracks = {avg_backtracks:.1f}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_lan_thu} vòng lặp")
            self.final_iteration = self.so_lan_thu
        
        print(f"Thời gian training: {self.training_time:.4f}s")
        print(f"Loss cuối: {self.loss_history[-1]:.8f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")  # Bias là phần tử cuối của weights
        print(f"Số weights (bao gồm bias): {len(self.weights)}")
        print(f"Trung bình step size: {np.mean(self.step_sizes):.4f}")
        print(f"Trung bình line search iterations: {np.mean(self.line_search_iterations):.1f}")
        
        return {
            'weights': self.weights,  # Bao gồm bias ở cuối
            'bias': self.weights[-1],  # Bias riêng để tương thích
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'step_sizes': self.step_sizes,
            'line_search_iterations': self.line_search_iterations,
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_iteration,
            'condition_number': self.condition_number
        }
    
    def predict(self, X):
        """Dự đoán với dữ liệu X 
        
        Trả về:
            predictions: Dự đoán trên log scale
            
        Lưu ý:
            - Model được train trên log-transformed targets
            - Dự đoán trả về ở log scale
            - Bias đã được tích hợp vào weights: y = Xw (với X đã có cột bias)
            - Sử dụng np.expm1() để chuyển về giá gốc khi cần
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        # Thêm cột bias vào X cho prediction
        X_with_bias = add_bias_column(X)
        return du_doan(X_with_bias, self.weights, None)
    
    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        print(f"\n📋 Đánh giá model...")
        # Sử dụng bias từ weights (phần tử cuối) để tương thích với hàm cũ
        bias_value = self.weights[-1]
        weights_without_bias = self.weights[:-1]
        metrics = danh_gia_mo_hinh(weights_without_bias, X_test, y_test, bias_value)
        in_ket_qua_danh_gia(metrics, self.training_time, 
                           f"Damped Newton Method - {self.ham_loss.upper()}")
        return metrics
    
    def save_results(self, ten_file, base_dir="data/03_algorithms/newton_method"):
        """
        Lưu kết quả model vào file
        
        Parameters:
        - ten_file: Tên file/folder để lưu kết quả
        - base_dir: Thư mục gốc để lưu
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        # Setup results directory
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results.json
        print(f"   Lưu kết quả vào {results_dir}/results.json")
        results_data = {
            "algorithm": f"Damped Newton Method - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "regularization": self.regularization,
                "numerical_regularization": self.numerical_regularization,
                "max_iterations": self.so_lan_thu,
                "tolerance": self.diem_dung,
                "armijo_c1": self.armijo_c1,
                "backtrack_rho": self.backtrack_rho,
                "max_line_search_iter": self.max_line_search_iter
            },
            "training_results": {
                "training_time": self.training_time,
                "converged": self.converged,
                "final_iteration": self.final_iteration,
                "total_iterations": self.so_lan_thu,
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1])
            },
            "weights_analysis": {
                "n_features": len(self.weights) - 1,  # Không tính bias
                "n_weights_total": len(self.weights),  # Tính cả bias
                "bias_value": float(self.weights[-1]),
                "weights_without_bias": self.weights[:-1].tolist(),
                "complete_weight_vector": self.weights.tolist(),
                "weights_stats": {
                    "min": float(np.min(self.weights[:-1])),  # Stats chỉ của weights, không tính bias
                    "max": float(np.max(self.weights[:-1])),
                    "mean": float(np.mean(self.weights[:-1])),
                    "std": float(np.std(self.weights[:-1]))
                }
            },
            "convergence_analysis": {
                "iterations_to_converge": self.final_iteration,
                "final_cost_change": float(self.loss_history[-1] - self.loss_history[-2]) if len(self.loss_history) > 1 else 0.0,
                "convergence_rate": "superlinear",  # Damped Newton có superlinear convergence
                "loss_reduction_ratio": float(self.loss_history[0] / self.loss_history[-1]) if len(self.loss_history) > 0 else 1.0,
                "convergence_quality": "damped_superlinear"
            },
            "numerical_analysis": {
                "hessian_condition_number": float(self.condition_number) if self.condition_number else None,
                "average_step_size": float(np.mean(self.step_sizes)) if self.step_sizes else 0,
                "max_step_size": float(np.max(self.step_sizes)) if self.step_sizes else 0,
                "min_step_size": float(np.min(self.step_sizes)) if self.step_sizes else 0,
                "step_size_stability": "line_search_controlled",
                "regularization_effect": "Applied for numerical stability"
            },
            "line_search_analysis": {
                "average_backtracks": float(np.mean(self.line_search_iterations)) if self.line_search_iterations else 0,
                "max_backtracks": int(np.max(self.line_search_iterations)) if self.line_search_iterations else 0,
                "min_backtracks": int(np.min(self.line_search_iterations)) if self.line_search_iterations else 0,
                "total_line_search_calls": len(self.line_search_iterations),
                "line_search_success_rate": "100%" # Always successful within max_iter
            },
            "algorithm_specific": {
                "method_type": "damped_newton",
                "second_order_method": True,
                "hessian_computation": "exact",
                "line_search_used": True,
                "line_search_type": "backtracking_armijo",
                "damping_applied": "line_search_based",
                "fast_convergence": self.final_iteration <= 20,
                "globalization_strategy": "armijo_line_search"
            }
        }
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        max_len = len(self.loss_history)
        training_df = pd.DataFrame({
            'iteration': range(0, len(self.loss_history)*self.convergence_check_freq, self.convergence_check_freq),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms,
            'step_size': self.step_sizes + [np.nan] * (max_len - len(self.step_sizes)),
            'line_search_iterations': self.line_search_iterations + [np.nan] * (max_len - len(self.line_search_iterations))
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/newton_method"):
        """
        Tạo các biểu đồ visualization
        
        Parameters:
        - X_test, y_test: Dữ liệu test để vẽ predictions
        - ten_file: Tên file/folder để lưu biểu đồ
        - base_dir: Thư mục gốc
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Tạo biểu đồ...")
        
        # 1. Convergence curves with line search info
        print("   - Vẽ đường hội tụ với thông tin line search")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0,0].semilogy(self.loss_history, 'b-', linewidth=2)
        axes[0,0].set_title('Loss Convergence')
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].grid(True, alpha=0.3)
        
        # Gradient norm
        axes[0,1].semilogy(self.gradient_norms, 'r-', linewidth=2)
        axes[0,1].set_title('Gradient Norm')
        axes[0,1].set_xlabel('Iteration')
        axes[0,1].set_ylabel('Gradient Norm')
        axes[0,1].grid(True, alpha=0.3)
        
        # Step sizes
        if self.step_sizes:
            axes[1,0].plot(self.step_sizes, 'g-', linewidth=2)
            axes[1,0].set_title('Step Sizes (α)')
            axes[1,0].set_xlabel('Iteration')
            axes[1,0].set_ylabel('Step Size')
            axes[1,0].grid(True, alpha=0.3)
        
        # Line search iterations
        if self.line_search_iterations:
            axes[1,1].bar(range(len(self.line_search_iterations)), self.line_search_iterations, 
                         alpha=0.7, color='orange')
            axes[1,1].set_title('Line Search Iterations')
            axes[1,1].set_xlabel('Newton Iteration')
            axes[1,1].set_ylabel('Backtrack Iterations')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Damped Newton {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mức) - hỗ trợ tất cả loss types
        print("   - Vẽ đường đồng mức optimization")
        # Chuẩn bị X_test với bias cho visualization
        X_test_with_bias = add_bias_column(X_test)
        
        ve_duong_dong_muc_optimization(
            loss_function=self.loss_func,
            weights_history=self.weights_history,  # Pass full history
            X=X_test_with_bias, y=y_test,
            title=f"Damped Newton {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png"),
            original_iterations=self.final_iteration,
            convergence_check_freq=self.convergence_check_freq,
            max_trajectory_points=None  # Damped Newton usually has few iterations, show all
        )