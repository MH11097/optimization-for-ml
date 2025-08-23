#!/usr/bin/env python3
"""
NewtonModel - Class cho Pure Newton Method
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
    tinh_gia_tri_ham_OLS, tinh_gradient_OLS, tinh_hessian_OLS,
    tinh_gia_tri_ham_Ridge, tinh_gradient_Ridge, tinh_hessian_Ridge,
    tinh_gia_tri_ham_Lasso_smooth, tinh_gradient_Lasso_smooth, tinh_hessian_Lasso_smooth,
    giai_he_phuong_trinh_tuyen_tinh,
    danh_gia_mo_hinh, in_ket_qua_danh_gia, kiem_tra_hoi_tu,
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te
)


class NewtonModel:
    """
    Pure Newton Method Model với hỗ trợ nhiều loss functions
    
    Parameters:
    - ham_loss: 'ols', 'ridge', 'lasso'
    - regularization: Tham số regularization cho Ridge/Lasso và numerical stability
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ (gradient norm)
    - numerical_regularization: Regularization cho numerical stability của Hessian
    """
    
    def __init__(self, ham_loss='ols', regularization=0.01, so_lan_thu=50, 
                 diem_dung=1e-10, numerical_regularization=1e-8):
        self.ham_loss = ham_loss.lower()
        self.regularization = regularization
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.numerical_regularization = numerical_regularization
        
        # Validate supported loss function
        if self.ham_loss not in ['ols', 'ridge', 'lasso']:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}")
        
        # Sử dụng unified functions thay vì if-else logic
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, 0.0, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, 0.0, self.regularization)[0]  # chỉ lấy gradient_w
        self.hess_func = lambda X: tinh_hessian_ham_loss(self.ham_loss, X, None, self.regularization)
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0
        self.condition_number = None
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"Training Newton Method - {self.ham_loss.upper()}")
        print(f"   Regularization: {self.regularization}")
        print(f"   Numerical regularization: {self.numerical_regularization}")
        print(f"   Max iterations: {self.so_lan_thu}")
        print(f"   Tolerance: {self.diem_dung}")
        
        # Initialize weights
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        
        start_time = time.time()
        
        # Precompute Hessian using unified function
        H = self.hess_func(X)
        
        # Add numerical regularization for stability
        H_reg = H + self.numerical_regularization * np.eye(n_features)
        self.condition_number = np.linalg.cond(H_reg)
        
        print(f"   Hessian condition number: {self.condition_number:.2e}")
        
        for lan_thu in range(self.so_lan_thu):
            # Compute loss and gradient
            loss_value = self.loss_func(X, y, self.weights)
            gradient_w = self.grad_func(X, y, self.weights)
            
            # Store history
            self.loss_history.append(loss_value)
            gradient_norm = np.linalg.norm(gradient_w)
            self.gradient_norms.append(gradient_norm)
            self.weights_history.append(self.weights.copy())
            
            # Check convergence using updated function (requires both conditions)
            cost_change = 0.0 if lan_thu == 0 else (self.loss_history[-2] - self.loss_history[-1])
            converged, reason = kiem_tra_hoi_tu(
                gradient_norm=gradient_norm,
                cost_change=cost_change, 
                iteration=lan_thu,
                tolerance=self.diem_dung,
                max_iterations=self.so_lan_thu
            )
            
            if converged:
                print(f"Newton Method stopped: {reason}")
                self.converged = True
                self.final_iteration = lan_thu + 1
                break
            
            # Newton step
            try:
                # For Lasso, might need to recompute Hessian
                if self.ham_loss == 'lasso':
                    H = self.hess_func(X)
                    H_reg = H + self.numerical_regularization * np.eye(n_features)
                
                delta_w = giai_he_phuong_trinh_tuyen_tinh(H_reg, gradient_w)
                step_size = np.linalg.norm(delta_w)
                self.step_sizes.append(step_size)
                
                self.weights = self.weights - delta_w
                
            except np.linalg.LinAlgError:
                print(f"Linear algebra error at iteration {lan_thu + 1}")
                break
            
            # Progress update
            if (lan_thu + 1) % 10 == 0:
                print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient norm = {gradient_norm:.2e}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"Reached maximum iterations ({self.so_lan_thu})")
            self.final_iteration = self.so_lan_thu
        
        print(f"Training time: {self.training_time:.4f} seconds")
        print(f"Final loss: {self.loss_history[-1]:.8f}")
        print(f"Final gradient norm: {self.gradient_norms[-1]:.2e}")
        
        return {
            'weights': self.weights,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'step_sizes': self.step_sizes,
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_iteration,
            'condition_number': self.condition_number
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
                           f"Newton Method - {self.ham_loss.upper()}")
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
        
        # Save results.json
        print(f"   Lưu kết quả vào {results_dir}/results.json")
        results_data = {
            "algorithm": f"Newton Method - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "regularization": self.regularization,
                "numerical_regularization": self.numerical_regularization,
                "max_iterations": self.so_lan_thu,
                "tolerance": self.diem_dung
            },
            "training_time": self.training_time,
            "convergence": {
                "converged": self.converged,
                "iterations": self.final_iteration,
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1])
            },
            "numerical_analysis": {
                "hessian_condition_number": float(self.condition_number),
                "average_step_size": float(np.mean(self.step_sizes)) if self.step_sizes else 0,
                "quadratic_convergence": self.final_iteration <= 20,
                "regularization_effect": "Applied for numerical stability"
            }
        }
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        max_len = len(self.loss_history)
        training_df = pd.DataFrame({
            'iteration': range(max_len),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms,
            'step_size': self.step_sizes + [np.nan] * (max_len - len(self.step_sizes))
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
        
        print(f"\\n Tạo các biểu đồ visualization")
        
        # 1. Convergence curves
        print("   Vẽ đường hội tụ")
        ve_duong_hoi_tu(self.loss_history, self.gradient_norms, 
                        title=f"Newton Method {self.ham_loss.upper()} - Convergence Analysis",
                        save_path=str(results_dir / "convergence_analysis.png"))
        
        # 2. Predictions vs Actual
        print("   Vẽ so sánh dự đoán với thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Newton Method {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mực) - hỗ trợ tất cả loss types
        print("   Vẽ đường đẳng mực optimization")
        sample_frequency = max(1, len(self.weights_history) // 50)
        sampled_weights = self.weights_history[::sample_frequency]
        
        ve_duong_dong_muc_optimization(
            loss_function=self.loss_func,
            weights_history=sampled_weights,
            X=X_test, y=y_test,
            bias_history=None,  # Newton Method doesn't use bias
            title=f"Newton Method {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png")
        )
        
        print(f"   Biểu đồ đã được lưu vào: {results_dir.absolute()}")