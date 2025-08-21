#!/usr/bin/env python3
"""
SGDModel - Class cho Stochastic Gradient Descent
Hỗ trợ các loss functions: MSE (Mean Squared Error)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
import os
import json
import pickle

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.optimization_utils import (
    tinh_mse, du_doan, danh_gia_mo_hinh, in_ket_qua_danh_gia
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_du_doan_vs_thuc_te
)


class SGDModel:
    """
    Stochastic Gradient Descent Model
    
    Parameters:
    - learning_rate: Tỷ lệ học (step size)
    - so_epochs: Số epochs (số lần duyệt qua toàn bộ dataset)
    - random_state: Random seed để tái tạo kết quả
    - batch_size: Kích thước batch (1 cho pure SGD, >1 cho mini-batch)
    - ham_loss: Loss function (hiện tại chỉ hỗ trợ 'mse')
    """
    
    def __init__(self, learning_rate=0.01, so_epochs=100, random_state=42, 
                 batch_size=1, ham_loss='mse'):
        self.learning_rate = learning_rate
        self.so_epochs = so_epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.ham_loss = ham_loss.lower()
        
        if self.ham_loss != 'mse':
            raise ValueError(f"Chưa hỗ trợ loss function: {ham_loss}. Chỉ hỗ trợ 'mse'.")
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None
        self.cost_history = []
        self.training_time = 0
        self.converged = False
        self.final_cost = None
        
    def _tinh_chi_phi(self, X, y, weights):
        """Tính chi phí Mean Squared Error"""
        predictions = X.dot(weights)
        errors = predictions - y
        cost = np.mean(errors ** 2)
        return cost
    
    def _tinh_gradient_sample(self, xi, yi, weights):
        """Tính gradient cho một sample"""
        # Prediction cho sample xi
        prediction = xi.dot(weights)
        error = prediction - yi
        
        # Gradient for single sample: 2 * xi.T * error
        gradient = 2 * xi * error
        return gradient
    
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, cost_history, etc.
        """
        print(f"Training Stochastic Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.so_epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Random state: {self.random_state}")
        
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Reset history
        self.cost_history = []
        
        start_time = time.time()
        
        for epoch in range(self.so_epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process in batches
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Tính gradient cho batch
                batch_gradient = np.zeros(n_features)
                
                for j in range(len(X_batch)):
                    xi = X_batch[j]
                    yi = y_batch[j]
                    sample_gradient = self._tinh_gradient_sample(xi, yi, self.weights)
                    batch_gradient += sample_gradient
                
                # Average gradient over batch
                batch_gradient /= len(X_batch)
                
                # Update weights
                self.weights -= self.learning_rate * batch_gradient
            
            # Calculate cost for entire dataset at end of epoch
            epoch_cost = self._tinh_chi_phi(X, y, self.weights)
            self.cost_history.append(epoch_cost)
            
            # Progress update
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch + 1}: Cost = {epoch_cost:.6f}")
        
        self.training_time = time.time() - start_time
        self.final_cost = self.cost_history[-1]
        
        print(f"Training time: {self.training_time:.2f} seconds")
        print(f"Final cost: {self.final_cost:.6f}")
        
        return {
            'weights': self.weights,
            'cost_history': self.cost_history,
            'training_time': self.training_time,
            'final_cost': self.final_cost,
            'epochs': self.so_epochs
        }
    
    def predict(self, X):
        """Dự đoán với dữ liệu X"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        return X.dot(self.weights)
    
    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        print(f"\\nĐánh giá model trên test set")
        
        # Tính các metrics thủ công
        y_pred = self.predict(X_test)
        
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        print(f"   MSE:  {metrics['mse']:.6f}")
        print(f"   RMSE: {metrics['rmse']:.6f}")
        print(f"   MAE:  {metrics['mae']:.6f}")
        print(f"   R²:   {metrics['r2']:.6f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def save_results(self, ten_file, base_dir="data/03_algorithms/stochastic_gd"):
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
        
        # Save model weights và training info
        print(f"   Lưu model vào {results_dir}/model.pkl")
        model_data = {
            'algorithm': f'Stochastic Gradient Descent',
            'weights': self.weights.tolist(),
            'cost_history': self.cost_history,
            'training_time': self.training_time,
            'epochs': self.so_epochs,
            'final_cost': self.final_cost,
            'parameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'random_state': self.random_state
            }
        }
        
        with open(results_dir / "model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        # Save results.json
        print(f"   Lưu kết quả vào {results_dir}/results.json")
        results_data = {
            "algorithm": f"Stochastic Gradient Descent",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "learning_rate": self.learning_rate,
                "epochs": self.so_epochs,
                "batch_size": self.batch_size,
                "random_state": self.random_state
            },
            "training_time": self.training_time,
            "convergence": {
                "epochs": self.so_epochs,
                "final_cost": float(self.final_cost)
            }
        }
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        training_df = pd.DataFrame({
            'epoch': range(len(self.cost_history)),
            'cost': self.cost_history
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/stochastic_gd"):
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
        
        # 1. Training curve (cost over epochs)
        print("   Vẽ đường training cost")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, 'orange', linewidth=2)
        plt.title('Stochastic Gradient Descent - Training Curve', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Cost (MSE)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "training_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        print("   Vẽ so sánh dự đoán với thực tế")
        y_pred_test = self.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_test, y_pred_test, alpha=0.6, s=20, color='orange')
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = 1 - np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Stochastic GD - Predictions vs Actual\\nR² = {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig(results_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Biểu đồ đã được lưu vào: {results_dir.absolute()}")