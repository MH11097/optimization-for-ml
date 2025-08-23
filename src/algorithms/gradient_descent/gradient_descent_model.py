#!/usr/bin/env python3
"""
GradientDescentModel - Class cho Gradient Descent Algorithm
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


class GradientDescentModel:
    """
    Gradient Descent Model với hỗ trợ nhiều loss functions
    
    Parameters:
    - ham_loss: 'ols', 'ridge', 'lasso'
    - learning_rate: Tỷ lệ học
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ
    - regularization: Tham số regularization cho Ridge/Lasso
    """
    
    def __init__(self, ham_loss='ols', learning_rate=0.1, so_lan_thu=500, 
                 diem_dung=1e-5, regularization=0.01):
        self.ham_loss = ham_loss.lower()
        self.learning_rate = learning_rate
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.regularization = regularization
        
        # Chọn loss function và gradient function
        if self.ham_loss == 'ols':
            self.loss_func = tinh_gia_tri_ham_OLS
            self.grad_func = tinh_gradient_OLS
        elif self.ham_loss == 'ridge':
            self.loss_func = lambda X, y, w, b: tinh_gia_tri_ham_Ridge(X, y, w, self.regularization, b)
            self.grad_func = lambda X, y, w, b: tinh_gradient_Ridge(X, y, w, self.regularization, b)
        elif self.ham_loss == 'lasso':
            self.loss_func = lambda X, y, w, b: tinh_gia_tri_ham_Lasso_smooth(X, y, w, self.regularization, b)
            self.grad_func = lambda X, y, w, b: tinh_gradient_Lasso_smooth(X, y, w, self.regularization, b)
        else:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}")
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None
        self.bias = None  # Thêm bias term
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.bias_history = []  # Thêm bias history
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y (bao gồm cả bias term)
        
        Returns:
        - dict: Kết quả training bao gồm weights, bias, loss_history, etc.
        """
        print(f"🚀 Training Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate: {self.learning_rate}, Max iterations: {self.so_lan_thu}")
        if self.ham_loss in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")
        
        # Initialize weights and bias
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = np.random.normal(0, 0.01)  # Khởi tạo bias
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.bias_history = []
        
        start_time = time.time()
        
        for lan_thu in range(self.so_lan_thu):
            # Compute loss and gradients với bias
            if self.ham_loss == 'ols':
                loss_value = tinh_gia_tri_ham_OLS(X, y, self.weights, self.bias)
                gradient_w, gradient_b = tinh_gradient_OLS(X, y, self.weights, self.bias)
            elif self.ham_loss == 'ridge':
                loss_value = tinh_gia_tri_ham_Ridge(X, y, self.weights, self.bias, self.regularization)
                gradient_w, gradient_b = tinh_gradient_Ridge(X, y, self.weights, self.bias, self.regularization)
            elif self.ham_loss == 'lasso':
                loss_value = tinh_gia_tri_ham_Lasso_smooth(X, y, self.weights, self.bias, self.regularization)
                gradient_w, gradient_b = tinh_gradient_Lasso_smooth(X, y, self.weights, self.bias, self.regularization)
            
            # Update weights and bias
            self.weights = self.weights - self.learning_rate * gradient_w
            self.bias = self.bias - self.learning_rate * gradient_b
            
            # Store history
            self.loss_history.append(loss_value)
            gradient_norm = np.sqrt(np.linalg.norm(gradient_w)**2 + gradient_b**2)  # Combined gradient norm
            self.gradient_norms.append(gradient_norm)
            self.weights_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            
            # Check convergence
            if lan_thu > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.diem_dung:
                print(f"✅ Hội tụ sau {lan_thu + 1} vòng lặp")
                self.converged = True
                self.final_iteration = lan_thu + 1
                break
            
            # Progress update
            if (lan_thu + 1) % 100 == 0:
                print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.6f}, Gradient = {gradient_norm:.6f}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_lan_thu} vòng lặp")
            self.final_iteration = self.so_lan_thu
        
        print(f"⏱️ Thời gian training: {self.training_time:.2f}s")
        print(f"📊 Loss cuối: {self.loss_history[-1]:.6f}")
        print(f"📈 Bias cuối: {self.bias:.6f}")
        
        return {
            'weights': self.weights,
            'bias': self.bias,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'bias_history': self.bias_history,
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_iteration
        }
    
    def predict(self, X):
        """Dự đoán với dữ liệu X (bao gồm bias term)
        
        Trả về:
            predictions: Dự đoán trên log scale
            
        Lưu ý:
            - Model được train trên log-transformed targets
            - Dự đoán trả về ở log scale
            - Bao gồm bias term: y = Xw + b
            - Sử dụng np.expm1() để chuyển về giá gốc khi cần
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        return du_doan(X, self.weights, self.bias)
    
    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set (với bias term)"""
        if self.weights is None or self.bias is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        print(f"\n📋 Đánh giá model...")
        metrics = danh_gia_mo_hinh(self.weights, X_test, y_test, self.bias)
        in_ket_qua_danh_gia(metrics, self.training_time, 
                           f"Gradient Descent - {self.ham_loss.upper()}")
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
            "algorithm": f"Gradient Descent - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "learning_rate": self.learning_rate,
                "max_iterations": self.so_lan_thu,
                "tolerance": self.diem_dung
            },
            "training_time": self.training_time,
            "convergence": {
                "converged": self.converged,
                "iterations": self.final_iteration,
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1])
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
            'gradient_norm': self.gradient_norms
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\\n Kết quả đã được lưu vào: {results_dir.absolute()}")
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
        
        print(f"\\n📊 Tạo biểu đồ...")
        
        # 1. Convergence curves
        print("   - Vẽ đường hội tụ")
        ve_duong_hoi_tu(self.loss_history, self.gradient_norms, 
                        title=f"Gradient Descent {self.ham_loss.upper()} - Hội tụ",
                        save_path=str(results_dir / "convergence_analysis.png"))
        
        # 2. Predictions vs Actual
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Gradient Descent {self.ham_loss.upper()} - Dự đoán vs Thực tế",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        print(f"✅ Biểu đồ đã lưu vào: {results_dir.absolute()}")