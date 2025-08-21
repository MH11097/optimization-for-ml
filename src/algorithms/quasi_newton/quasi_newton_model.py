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
    tinh_mse, du_doan, 
    tinh_gia_tri_ham_OLS, tinh_gradient_OLS,
    tinh_gia_tri_ham_Ridge, tinh_gradient_Ridge,
    tinh_gia_tri_ham_Lasso_smooth, tinh_gradient_Lasso_smooth,
    danh_gia_mo_hinh, in_ket_qua_danh_gia
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
    """
    
    def __init__(self, ham_loss='ols', so_lan_thu=100, diem_dung=1e-6, 
                 regularization=0.01, armijo_c1=1e-4, wolfe_c2=0.9,
                 backtrack_rho=0.8, max_line_search_iter=50, damping=1e-8):
        self.ham_loss = ham_loss.lower()
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.regularization = regularization
        self.armijo_c1 = armijo_c1
        self.wolfe_c2 = wolfe_c2
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.damping = damping
        
        # Chọn loss function và gradient function
        if self.ham_loss == 'ols':
            self.loss_func = tinh_gia_tri_ham_OLS
            self.grad_func = tinh_gradient_OLS
        elif self.ham_loss == 'ridge':
            self.loss_func = lambda X, y, w: tinh_gia_tri_ham_Ridge(X, y, w, self.regularization)
            self.grad_func = lambda X, y, w: tinh_gradient_Ridge(X, y, w, self.regularization)
        elif self.ham_loss == 'lasso':
            self.loss_func = lambda X, y, w: tinh_gia_tri_ham_Lasso_smooth(X, y, w, self.regularization)
            self.grad_func = lambda X, y, w: tinh_gradient_Lasso_smooth(X, y, w, self.regularization)
        else:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}")
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None
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
        print(f"Training BFGS Quasi-Newton Method - {self.ham_loss.upper()}")
        print(f"   Max iterations: {self.so_lan_thu}")
        print(f"   Tolerance: {self.diem_dung}")
        if self.ham_loss in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")
        print(f"   Armijo c1: {self.armijo_c1}")
        print(f"   Wolfe c2: {self.wolfe_c2}")
        
        # Initialize weights và inverse Hessian approximation
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        self.H_inv = np.eye(n_features)  # Initial approximation
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.condition_numbers = []
        
        start_time = time.time()
        
        # Initial gradient
        gradient_prev = self.grad_func(X, y, self.weights)
        
        for lan_thu in range(self.so_lan_thu):
            # Compute loss and gradient
            loss_value = self.loss_func(X, y, self.weights)
            gradient_curr = self.grad_func(X, y, self.weights)
            
            # Store history
            self.loss_history.append(loss_value)
            gradient_norm = np.linalg.norm(gradient_curr)
            self.gradient_norms.append(gradient_norm)
            self.weights_history.append(self.weights.copy())
            
            # Check convergence
            if gradient_norm < self.diem_dung:
                print(f"Converged after {lan_thu + 1} iterations (gradient norm: {gradient_norm:.2e})")
                self.converged = True
                self.final_iteration = lan_thu + 1
                break
            
            # BFGS direction: d = -H_inv * gradient
            direction = -np.dot(self.H_inv, gradient_curr)
            
            # Line search để tìm step size
            step_size, ls_iter, gradient_new = self._wolfe_line_search(
                X, y, self.weights, direction, gradient_curr
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
            if (lan_thu + 1) % 10 == 0:
                print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient norm = {gradient_norm:.2e}, Step size = {step_size:.6f}, Cond = {cond_num:.2e}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"Reached maximum iterations ({self.so_lan_thu})")
            self.final_iteration = self.so_lan_thu
        
        print(f"Training time: {self.training_time:.4f} seconds")
        print(f"Final loss: {self.loss_history[-1]:.8f}")
        print(f"Final gradient norm: {self.gradient_norms[-1]:.2e}")
        if self.step_sizes:
            print(f"Average step size: {np.mean(self.step_sizes):.6f}")
            print(f"Average line search iterations: {np.mean(self.line_search_iterations):.1f}")
        
        return {
            'weights': self.weights,
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
        """Dự đoán với dữ liệu X"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        return du_doan(X, self.weights, 0)
    
    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        print(f"\\nĐánh giá model trên test set")
        metrics = danh_gia_mo_hinh(self.weights, X_test, y_test)
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
        
        # Save results.json
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
            "training_time": self.training_time,
            "convergence": {
                "converged": self.converged,
                "iterations": self.final_iteration,
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1])
            },
            "numerical_analysis": {
                "average_step_size": float(np.mean(self.step_sizes)) if self.step_sizes else 0,
                "average_line_search_iterations": float(np.mean(self.line_search_iterations)) if self.line_search_iterations else 0,
                "final_condition_number": float(self.condition_numbers[-1]) if self.condition_numbers else 0,
                "superlinear_convergence": "BFGS provides superlinear convergence rate"
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
            'iteration': range(max_len),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms,
            'step_size': self.step_sizes + [np.nan] * (max_len - len(self.step_sizes)),
            'line_search_iter': self.line_search_iterations + [np.nan] * (max_len - len(self.line_search_iterations)),
            'condition_number': self.condition_numbers + [np.nan] * (max_len - len(self.condition_numbers))
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/quasi_newton"):
        """
        Tạo các biểu đồ visualization
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\\n Tạo các biểu đồ visualization")
        
        # 1. Convergence curves với condition number
        print("   Vẽ đường hội tụ với condition number")
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
        print("   Vẽ so sánh dự đoán với thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"BFGS Quasi-Newton {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        print(f"   Biểu đồ đã được lưu vào: {results_dir.absolute()}")