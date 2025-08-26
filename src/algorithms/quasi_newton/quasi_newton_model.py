#!/usr/bin/env python3
"""
QuasiNewtonModel - Class cho BFGS Quasi-Newton Method
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
    add_bias_column
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te
)


class QuasiNewtonModel:
    """
    BFGS Quasi-Newton Method Model với hỗ trợ nhiều loss functions
    
    Parameters:
    - ham_loss: 'ols', 'ridge', 'lasso'
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ (gradient norm)
    - regularization: Tham số regularization cho Ridge/Lasso
    - armijo_c1: Armijo constant cho line search
    - wolfe_c2: Wolfe curvature constant
    - backtrack_rho: Backtrack factor cho line search
    - max_line_search_iter: Số lần line search tối đa
    - damping: Damping factor cho BFGS update
    - convergence_check_freq: Tần suất kiểm tra hội tụ (mỗi N iterations)
    """
    
    def __init__(self, ham_loss='ols', so_lan_thu=100, diem_dung=1e-6, 
                 regularization=0.01, armijo_c1=1e-4, wolfe_c2=0.9,
                 backtrack_rho=0.8, max_line_search_iter=50, damping=1e-8, convergence_check_freq=10):
        self.ham_loss = ham_loss.lower()
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.regularization = regularization
        self.convergence_check_freq = convergence_check_freq
        self.armijo_c1 = armijo_c1
        self.wolfe_c2 = wolfe_c2
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.damping = damping
        
        # Validate supported loss function
        if self.ham_loss not in ['ols', 'ridge', 'lasso']:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}")
        
        # Sử dụng unified functions với format mới (bias trong X)
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None  # Bây giờ bao gồm bias ở cuối
        self.H_inv = None  # Inverse Hessian approximation
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.condition_numbers = []
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0
    
    def _wolfe_line_search(self, X, y, weights, direction, gradient):
        """
        Wolfe line search để satisfy both Armijo và curvature conditions
        """
        current_loss = self.loss_func(X, y, weights)
        
        # Directional derivative: ∇f^T * d  
        directional_derivative = np.dot(gradient, direction)
        
        # Initial step size
        alpha = 1.0
        
        for i in range(self.max_line_search_iter):
            # Thử weights mới
            new_weights = weights + alpha * direction
            
            # Tính loss và gradient mới
            new_loss = self.loss_func(X, y, new_weights)
            new_gradient = self.grad_func(X, y, new_weights)
            
            # Kiểm tra Armijo condition (sufficient decrease)
            armijo_condition = current_loss + self.armijo_c1 * alpha * directional_derivative
            
            if new_loss <= armijo_condition:
                # Kiểm tra curvature condition (Wolfe)
                curvature_condition = np.dot(new_gradient, direction)
                if curvature_condition >= self.wolfe_c2 * directional_derivative:
                    return alpha, i + 1, new_gradient
            
            # Giảm step size
            alpha *= self.backtrack_rho
        
        # Nếu line search fail, return step cuối và gradient mới
        new_weights = weights + alpha * direction
        new_gradient = self.grad_func(X, y, new_weights)
        return alpha, self.max_line_search_iter, new_gradient
    
    def _cap_nhat_bfgs(self, H_inv, s, y):
        """
        Cập nhật inverse Hessian approximation theo BFGS formula
        
        H_{k+1}^{-1} = (I - ρ s y^T) H_k^{-1} (I - ρ y s^T) + ρ s s^T
        where ρ = 1 / (y^T s)
        """
        sy = np.dot(s, y)
        
        # Kiểm tra curvature condition để đảm bảo positive definiteness
        if sy < self.damping:
            print(f"   Warning: Curvature condition violated (sy = {sy:.2e}), skipping BFGS update")
            return H_inv
        
        rho = 1.0 / sy
        n = len(s)
        I = np.eye(n)
        
        # BFGS update formula
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H_inv_new = np.dot(A, np.dot(H_inv, B)) + rho * np.outer(s, s)
        
        return H_inv_new
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"🚀 Training BFGS Quasi-Newton Method - {self.ham_loss.upper()}")
        print(f"   Max iterations: {self.so_lan_thu}")
        if self.ham_loss in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")
        print(f"   Armijo c1: {self.armijo_c1}, Wolfe c2: {self.wolfe_c2}")
        
        # Thêm cột bias vào X
        X_with_bias = add_bias_column(X)
        print(f"   Original features: {X.shape[1]}, With bias: {X_with_bias.shape[1]}")
        
        # Initialize weights và inverse Hessian approximation (bao gồm bias ở cuối)
        n_features_with_bias = X_with_bias.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        self.H_inv = np.eye(n_features_with_bias)  # Initial approximation
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.condition_numbers = []
        
        start_time = time.time()
        
        # Initial gradient
        gradient_prev, _ = self.grad_func(X_with_bias, y, self.weights)  # _ vì không cần gradient_b riêng
        
        for lan_thu in range(self.so_lan_thu):
            # Tính gradient (luôn cần cho BFGS)
            gradient_curr, _ = self.grad_func(X_with_bias, y, self.weights)  # _ vì không cần gradient_b riêng
            
            # Chỉ tính loss và lưu history khi cần thiết
            should_check_converged = (
                (lan_thu + 1) % self.convergence_check_freq == 0 or 
                lan_thu == self.so_lan_thu - 1
            )
            
            if should_check_converged:
                # Chỉ tính loss khi cần (expensive operation)
                loss_value = self.loss_func(X_with_bias, y, self.weights)
                gradient_norm = np.linalg.norm(gradient_curr)
                
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
                    print(f"✅ BFGS Quasi-Newton stopped: {reason}")
                    self.converged = True
                    self.final_iteration = lan_thu + 1
                    break
            
            # BFGS direction: d = -H_inv * gradient
            direction = -np.dot(self.H_inv, gradient_curr)
            
            # Line search để tìm step size
            step_size, ls_iter, gradient_new = self._wolfe_line_search(
                X_with_bias, y, self.weights, direction, gradient_curr
            )
            
            self.step_sizes.append(step_size)
            self.line_search_iterations.append(ls_iter)
            
            # Update weights
            weights_new = self.weights + step_size * direction
            
            # BFGS update
            s = weights_new - self.weights  # step
            y = gradient_new - gradient_curr  # gradient change
            
            self.H_inv = self._cap_nhat_bfgs(self.H_inv, s, y)
            
            # Store condition number
            cond_num = np.linalg.cond(self.H_inv)
            self.condition_numbers.append(cond_num)
            
            # Update for next iteration
            self.weights = weights_new
            gradient_prev = gradient_curr
            
            # Progress update
            if (lan_thu + 1) % 10 == 0 and should_check_converged:
                print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient = {gradient_norm:.2e}, Step = {step_size:.6f}, Cond = {cond_num:.2e}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_lan_thu} vòng lặp")
            self.final_iteration = self.so_lan_thu
        
        print(f"Thời gian training: {self.training_time:.4f}s")
        print(f"Loss cuối: {self.loss_history[-1]:.8f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")  # Bias là phần tử cuối của weights
        print(f"Số weights (bao gồm bias): {len(self.weights)}")
        if self.step_sizes:
            print(f"Average step size: {np.mean(self.step_sizes):.6f}")
            print(f"Average line search iterations: {np.mean(self.line_search_iterations):.1f}")
        
        return {
            'weights': self.weights,  # Bao gồm bias ở cuối
            'bias': self.weights[-1],  # Bias riêng để tương thích
            'H_inv': self.H_inv,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'step_sizes': self.step_sizes,
            'line_search_iterations': self.line_search_iterations,
            'condition_numbers': self.condition_numbers,
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_iteration
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
                           f"BFGS Quasi-Newton - {self.ham_loss.upper()}")
        return metrics
    
    def save_results(self, ten_file, base_dir="data/03_algorithms/quasi_newton"):
        """
        Lưu kết quả model vào file
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        # Setup results directory
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results.json
        print(f"   Lưu kết quả vào {results_dir}/results.json")
        results_data = {
            "algorithm": f"BFGS Quasi-Newton - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "max_iterations": self.so_lan_thu,
                "tolerance": self.diem_dung,
                "armijo_c1": self.armijo_c1,
                "wolfe_c2": self.wolfe_c2,
                "backtrack_rho": self.backtrack_rho,
                "damping": self.damping
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
                "convergence_rate": "superlinear",  # BFGS có superlinear convergence
                "loss_reduction_ratio": float(self.loss_history[0] / self.loss_history[-1]) if len(self.loss_history) > 0 else 1.0
            },
            "numerical_analysis": {
                "average_step_size": float(np.mean(self.step_sizes)) if self.step_sizes else 0,
                "average_line_search_iterations": float(np.mean(self.line_search_iterations)) if self.line_search_iterations else 0,
                "final_condition_number": float(self.condition_numbers[-1]) if self.condition_numbers else 0,
                "max_condition_number": float(np.max(self.condition_numbers)) if self.condition_numbers else 0,
                "min_condition_number": float(np.min(self.condition_numbers)) if self.condition_numbers else 0,
                "hessian_approximation_quality": "BFGS_secant_approximation",
                "line_search_efficiency": "Wolfe_conditions_satisfied"
            },
            "algorithm_specific": {
                "method_type": "quasi_newton_bfgs",
                "second_order_approximation": True,
                "hessian_computation": "secant_approximation",
                "line_search_used": True,
                "line_search_type": "wolfe_conditions",
                "superlinear_convergence": "BFGS provides superlinear convergence rate",
                "memory_efficient": "inverse_hessian_approximation"
            }
        }
        
        if self.ham_loss in ['ridge', 'lasso']:
            results_data["parameters"]["regularization"] = self.regularization
        
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
            'line_search_iter': self.line_search_iterations + [np.nan] * (max_len - len(self.line_search_iterations)),
            'condition_number': self.condition_numbers + [np.nan] * (max_len - len(self.condition_numbers))
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/quasi_newton"):
        """
        Tạo các biểu đồ visualization
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Tạo biểu đồ...")
        
        # 1. Convergence curves với condition number
        print("   - Vẽ đường hội tụ")
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
        
        # Condition number
        if self.condition_numbers:
            axes[1,0].semilogy(self.condition_numbers, 'g-', linewidth=2)
            axes[1,0].set_title('Condition Number of H_inv')
            axes[1,0].set_xlabel('Iteration')
            axes[1,0].set_ylabel('Condition Number')
            axes[1,0].grid(True, alpha=0.3)
        
        # Step sizes
        if self.step_sizes:
            axes[1,1].plot(self.step_sizes, 'm-', linewidth=2)
            axes[1,1].set_title('Step Sizes')
            axes[1,1].set_xlabel('Iteration')
            axes[1,1].set_ylabel('Step Size')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"BFGS Quasi-Newton {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mức)
        print("   - Vẽ đường đồng mức optimization")
        sample_frequency = max(1, len(self.weights_history) // 50)
        sampled_weights = self.weights_history[::sample_frequency]
        
        # Chuẩn bị X_test với bias cho visualization
        X_test_with_bias = add_bias_column(X_test)
        
        ve_duong_dong_muc_optimization(
            loss_function=self.loss_func,
            weights_history=sampled_weights,
            X=X_test_with_bias, y=y_test,
            title=f"BFGS Quasi-Newton {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png")
        )
        
