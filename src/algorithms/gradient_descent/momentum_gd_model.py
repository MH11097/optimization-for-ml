#!/usr/bin/env python3
"""
MomentumGDModel - Class cho Gradient Descent with Momentum
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
    danh_gia_mo_hinh, in_ket_qua_danh_gia, kiem_tra_hoi_tu,
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te
)


class MomentumGDModel:
    """
    Gradient Descent with Momentum Model với hỗ trợ nhiều loss functions
    
    Parameters:
    - ham_loss: 'ols', 'ridge', 'lasso'
    - learning_rate: Tỷ lệ học
    - momentum: Momentum coefficient β (thường 0.9)
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ
    - regularization: Tham số regularization cho Ridge/Lasso
    """
    
    def __init__(self, ham_loss='ols', learning_rate=0.01, momentum=0.9, 
                 so_lan_thu=1000, diem_dung=1e-6, regularization=0.01):
        self.ham_loss = ham_loss.lower()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.regularization = regularization
        
        # Validate supported loss function
        if self.ham_loss not in ['ols', 'ridge', 'lasso']:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}")
        
        # Sử dụng unified functions thay vì if-else logic
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, 0.0, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, 0.0, self.regularization)[0]  # chỉ lấy gradient_w
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None
        self.velocity = None
        self.loss_history = []
        self.gradient_norms = []
        self.velocity_norms = []
        self.weights_history = []
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"Training Momentum Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Momentum β: {self.momentum}")
        print(f"   Max iterations: {self.so_lan_thu}")
        print(f"   Tolerance: {self.diem_dung}")
        if self.ham_loss in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")
        
        # Initialize weights and velocity
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        self.velocity = np.zeros(n_features)
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.velocity_norms = []
        self.weights_history = []
        
        start_time = time.time()
        
        for lan_thu in range(self.so_lan_thu):
            # Compute loss and gradient
            loss_value = self.loss_func(X, y, self.weights)
            gradient_w = self.grad_func(X, y, self.weights)
            
            # Momentum update
            self.velocity = self.momentum * self.velocity + gradient_w
            self.weights = self.weights - self.learning_rate * self.velocity
            
            # Store history
            self.loss_history.append(loss_value)
            gradient_norm = np.linalg.norm(gradient_w)
            self.gradient_norms.append(gradient_norm)
            velocity_norm = np.linalg.norm(self.velocity)
            self.velocity_norms.append(velocity_norm)
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
                print(f"Momentum GD stopped: {reason}")
                self.converged = True
                self.final_iteration = lan_thu + 1
                break
            
            # Progress update
            if (lan_thu + 1) % 100 == 0:
                print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.6f}, Gradient norm = {gradient_norm:.6f}, Velocity norm = {velocity_norm:.6f}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"Reached maximum iterations ({self.so_lan_thu})")
            self.final_iteration = self.so_lan_thu
        
        print(f"Training time: {self.training_time:.2f} seconds")
        print(f"Final loss: {self.loss_history[-1]:.6f}")
        print(f"Final gradient norm: {self.gradient_norms[-1]:.6f}")
        print(f"Final velocity norm: {self.velocity_norms[-1]:.6f}")
        
        return {
            'weights': self.weights,
            'velocity': self.velocity,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'velocity_norms': self.velocity_norms,
            'weights_history': self.weights_history,
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
                           f"Momentum Gradient Descent - {self.ham_loss.upper()}")
        return metrics
    
    def save_results(self, ten_file, base_dir="data/03_algorithms/gradient_descent"):
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
            "algorithm": f"Momentum Gradient Descent - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "learning_rate": self.learning_rate,
                "momentum": self.momentum,
                "max_iterations": self.so_lan_thu,
                "tolerance": self.diem_dung
            },
            "training_time": self.training_time,
            "convergence": {
                "converged": self.converged,
                "iterations": self.final_iteration,
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1]),
                "final_velocity_norm": float(self.velocity_norms[-1])
            }
        }
        
        if self.ham_loss in ['ridge', 'lasso']:
            results_data["parameters"]["regularization"] = self.regularization
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        training_df = pd.DataFrame({
            'iteration': range(len(self.loss_history)),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms,
            'velocity_norm': self.velocity_norms
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/gradient_descent"):
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
        
        # 1. Convergence curves với velocity
        print("   Vẽ đường hội tụ với velocity")
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
        
        # Velocity norm
        axes[1,0].plot(self.velocity_norms, 'g-', linewidth=2)
        axes[1,0].set_title('Velocity Norm')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Velocity Norm')
        axes[1,0].grid(True, alpha=0.3)
        
        # Combined plot
        ax2 = axes[1,1].twinx()
        axes[1,1].semilogy(self.loss_history, 'b-', linewidth=2, label='Loss')
        ax2.plot(self.velocity_norms, 'g-', linewidth=2, label='Velocity Norm')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Loss', color='b')
        ax2.set_ylabel('Velocity Norm', color='g')
        axes[1,1].set_title('Loss vs Velocity')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        print("   Vẽ so sánh dự đoán với thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Momentum GD {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mực)
        print("   Vẽ đường đồng mực optimization")
        sample_frequency = max(1, len(self.weights_history) // 50)
        sampled_weights = self.weights_history[::sample_frequency]
        
        ve_duong_dong_muc_optimization(
            loss_function=self.loss_func,
            weights_history=sampled_weights,
            X=X_test, y=y_test,
            bias_history=None,  # Momentum GD doesn't use bias
            title=f"Momentum GD {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png")
        )
        
        print(f"   Biểu đồ đã được lưu vào: {results_dir.absolute()}")