#!/usr/bin/env python3
"""
ProximalGDModel - Class cho Proximal Gradient Descent
Hỗ trợ Lasso và Elastic Net regularization
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
    danh_gia_mo_hinh, in_ket_qua_danh_gia
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_du_doan_vs_thuc_te
)


class ProximalGDModel:
    """
    Proximal Gradient Descent Model cho sparse learning
    
    Parameters:
    - ham_loss: 'lasso', 'elastic_net'
    - learning_rate: Tỷ lệ học
    - lambda_l1: L1 regularization strength
    - lambda_l2: L2 regularization strength (cho Elastic Net)
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ
    """
    
    def __init__(self, ham_loss='lasso', learning_rate=0.01, lambda_l1=0.01, 
                 lambda_l2=0.0, so_lan_thu=1000, diem_dung=1e-6):
        self.ham_loss = ham_loss.lower()
        self.learning_rate = learning_rate
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        
        if self.ham_loss not in ['lasso', 'elastic_net']:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}. Chỉ hỗ trợ 'lasso' và 'elastic_net'.")
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None
        self.loss_history = []
        self.gradient_norms = []
        self.sparsity_history = []  # Số lượng weights = 0
        self.weights_history = []
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0
        
    def _soft_threshold(self, x, threshold):
        """
        Soft thresholding operator: sign(x) * max(|x| - threshold, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _proximal_operator(self, z):
        """
        Proximal operator for L1 (và L2 nếu có)
        """
        if self.ham_loss == 'lasso':
            # Pure L1: soft thresholding
            return self._soft_threshold(z, self.learning_rate * self.lambda_l1)
        
        elif self.ham_loss == 'elastic_net':
            # Elastic Net: L1 + L2
            # Proximal operator: soft_threshold(z, λ1*α) / (1 + λ2*α)
            soft_thresh = self._soft_threshold(z, self.learning_rate * self.lambda_l1)
            return soft_thresh / (1 + self.learning_rate * self.lambda_l2)
    
    def _compute_loss(self, X, y, weights):
        """Tính loss function"""
        # MSE loss
        mse_loss = tinh_gia_tri_ham_OLS(X, y, weights)
        
        # L1 regularization
        l1_penalty = self.lambda_l1 * np.sum(np.abs(weights))
        
        # L2 regularization (cho Elastic Net)
        l2_penalty = 0
        if self.ham_loss == 'elastic_net':
            l2_penalty = 0.5 * self.lambda_l2 * np.sum(weights ** 2)
        
        return mse_loss + l1_penalty + l2_penalty
    
    def _compute_sparsity(self, weights, threshold=1e-8):
        """Tính số lượng weights gần bằng 0 (sparsity)"""
        return np.sum(np.abs(weights) < threshold)
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"Training Proximal Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Lambda L1: {self.lambda_l1}")
        if self.ham_loss == 'elastic_net':
            print(f"   Lambda L2: {self.lambda_l2}")
        print(f"   Max iterations: {self.so_lan_thu}")
        print(f"   Tolerance: {self.diem_dung}")
        
        # Initialize weights
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.sparsity_history = []
        self.weights_history = []
        
        start_time = time.time()
        
        for lan_thu in range(self.so_lan_thu):
            # Forward step: z = w - α∇f(w)
            gradient = tinh_gradient_OLS(X, y, self.weights)
            z = self.weights - self.learning_rate * gradient
            
            # Proximal step: w = prox_λ(z)
            self.weights = self._proximal_operator(z)
            
            # Compute loss (with regularization)
            loss_value = self._compute_loss(X, y, self.weights)
            
            # Store history
            self.loss_history.append(loss_value)
            gradient_norm = np.linalg.norm(gradient)
            self.gradient_norms.append(gradient_norm)
            
            sparsity = self._compute_sparsity(self.weights)
            self.sparsity_history.append(sparsity)
            self.weights_history.append(self.weights.copy())
            
            # Check convergence
            if lan_thu > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.diem_dung:
                print(f"Converged after {lan_thu + 1} iterations")
                self.converged = True
                self.final_iteration = lan_thu + 1
                break
            
            # Progress update
            if (lan_thu + 1) % 100 == 0:
                print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.6f}, Gradient norm = {gradient_norm:.6f}, Sparsity = {sparsity}/{n_features}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"Reached maximum iterations ({self.so_lan_thu})")
            self.final_iteration = self.so_lan_thu
        
        final_sparsity = self._compute_sparsity(self.weights)
        print(f"Training time: {self.training_time:.2f} seconds")
        print(f"Final loss: {self.loss_history[-1]:.6f}")
        print(f"Final gradient norm: {self.gradient_norms[-1]:.6f}")
        print(f"Final sparsity: {final_sparsity}/{n_features} ({final_sparsity/n_features*100:.1f}%)")
        
        return {
            'weights': self.weights,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'sparsity_history': self.sparsity_history,
            'weights_history': self.weights_history,
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_iteration,
            'final_sparsity': final_sparsity
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
                           f"Proximal Gradient Descent - {self.ham_loss.upper()}")
        return metrics
    
    def save_results(self, ten_file, base_dir="data/03_algorithms/proximal_gd"):
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
            "algorithm": f"Proximal Gradient Descent - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "learning_rate": self.learning_rate,
                "lambda_l1": self.lambda_l1,
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
            "sparsity_analysis": {
                "final_sparsity": int(self.sparsity_history[-1]) if self.sparsity_history else 0,
                "total_features": len(self.weights),
                "sparsity_ratio": float(self.sparsity_history[-1] / len(self.weights)) if self.sparsity_history else 0,
                "non_zero_weights": int(len(self.weights) - self.sparsity_history[-1]) if self.sparsity_history else len(self.weights)
            }
        }
        
        if self.ham_loss == 'elastic_net':
            results_data["parameters"]["lambda_l2"] = self.lambda_l2
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        training_df = pd.DataFrame({
            'iteration': range(len(self.loss_history)),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms,
            'sparsity': self.sparsity_history
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/proximal_gd"):
        """
        Tạo các biểu đồ visualization
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\\n Tạo các biểu đồ visualization")
        
        # 1. Convergence curves với sparsity
        print("   Vẽ đường hội tụ với sparsity analysis")
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
        
        # Sparsity evolution
        axes[1,0].plot(self.sparsity_history, 'g-', linewidth=2)
        axes[1,0].set_title('Sparsity Evolution')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Number of Zero Weights')
        axes[1,0].grid(True, alpha=0.3)
        
        # Weight magnitudes
        if len(self.weights) <= 20:  # Chỉ plot nếu không quá nhiều features
            axes[1,1].bar(range(len(self.weights)), np.abs(self.weights))
            axes[1,1].set_title('Final Weight Magnitudes')
            axes[1,1].set_xlabel('Feature Index')
            axes[1,1].set_ylabel('|Weight|')
            axes[1,1].grid(True, alpha=0.3)
        else:
            # Histogram of weight magnitudes
            axes[1,1].hist(np.abs(self.weights), bins=30, alpha=0.7, color='purple')
            axes[1,1].set_title('Weight Magnitude Distribution')
            axes[1,1].set_xlabel('|Weight|')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        print("   Vẽ so sánh dự đoán với thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Proximal GD {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        print(f"   Biểu đồ đã được lưu vào: {results_dir.absolute()}")