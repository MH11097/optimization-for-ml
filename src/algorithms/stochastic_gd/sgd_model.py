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
    du_doan, danh_gia_mo_hinh, in_ket_qua_danh_gia, kiem_tra_hoi_tu,
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss,
    add_bias_column
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_du_doan_vs_thuc_te, ve_duong_dong_muc_optimization
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
                 batch_size=1, ham_loss='ols', tolerance=1e-6, regularization=0.01):
        self.learning_rate = learning_rate
        self.so_epochs = so_epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.ham_loss = ham_loss.lower()
        self.tolerance = tolerance
        self.regularization = regularization
        
        # Validate supported loss function và mở rộng hỗ trợ
        if self.ham_loss not in ['ols', 'ridge', 'lasso', 'mse']:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}. Hỗ trợ: 'ols', 'ridge', 'lasso', 'mse'")
        
        # Map mse to ols cho compatibility
        if self.ham_loss == 'mse':
            self.ham_loss = 'ols'
        
        # Sử dụng unified functions với format mới (bias trong X)
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None  # Bây giờ bao gồm bias ở cuối
        self.cost_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.training_time = 0
        self.converged = False
        self.final_cost = None
        self.final_epoch = 0
        
    def _tinh_chi_phi(self, X, y, weights):
        """Tính chi phí sử dụng unified function"""
        return self.loss_func(X, y, weights)
    
    def _tinh_gradient_sample(self, xi, yi, weights):
        """Tính gradient cho một sample sử dụng unified function"""
        # Reshape để tương thích với unified function
        X_sample = xi.reshape(1, -1)  # (1, n_features) - đã bao gồm bias
        y_sample = np.array([yi])     # (1,)
        
        # Sử dụng unified function và lấy gradient weights
        gradient_w, _ = self.grad_func(X_sample, y_sample, weights)  # _ vì không cần gradient_b riêng
        return gradient_w
    
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, cost_history, etc.
        """
        print(f"🚀 Training Stochastic Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.so_epochs}, Batch size: {self.batch_size}")
        print(f"   Random state: {self.random_state}")
        
        np.random.seed(self.random_state)
        
        # Thêm cột bias vào X
        X_with_bias = add_bias_column(X)
        print(f"   Original features: {X.shape[1]}, With bias: {X_with_bias.shape[1]}")
        
        n_samples, n_features_with_bias = X_with_bias.shape
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        
        # Reset histories
        self.cost_history = []
        self.gradient_norms = []
        self.weights_history = []
        
        # Add convergence tracking
        self.converged = False
        self.final_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(self.so_epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
            
            epoch_gradients = []
            
            # Process in batches
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Tính gradient cho batch
                batch_gradient = np.zeros(n_features_with_bias)
                
                for j in range(len(X_batch)):
                    xi = X_batch[j]
                    yi = y_batch[j]
                    sample_gradient = self._tinh_gradient_sample(xi, yi, self.weights)
                    batch_gradient += sample_gradient
                
                # Average gradient over batch
                batch_gradient /= len(X_batch)
                epoch_gradients.append(batch_gradient)
                
                # Update weights
                self.weights -= self.learning_rate * batch_gradient
            
            # Calculate cost for entire dataset at end of epoch
            epoch_cost = self._tinh_chi_phi(X_with_bias, y, self.weights)
            self.cost_history.append(epoch_cost)
            
            # Store weights history
            self.weights_history.append(self.weights.copy())
            
            # Calculate average gradient norm for the epoch
            epoch_gradient_avg = np.mean(epoch_gradients, axis=0)
            gradient_norm = np.linalg.norm(epoch_gradient_avg)
            self.gradient_norms.append(gradient_norm)
            
            # Check convergence using updated function (requires both conditions)
            cost_change = 0.0 if epoch == 0 else (self.cost_history[-2] - self.cost_history[-1])
            converged, reason = kiem_tra_hoi_tu(
                gradient_norm=gradient_norm,
                cost_change=cost_change,
                iteration=epoch,
                tolerance=self.tolerance,
                max_iterations=self.so_epochs
            )
            
            if converged:
                print(f"✅ SGD stopped: {reason}")
                self.converged = True
                self.final_epoch = epoch + 1
                break
            
            # Progress update
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch + 1}: Cost = {epoch_cost:.6f}, Gradient = {gradient_norm:.6f}")
        
        self.training_time = time.time() - start_time
        self.final_cost = self.cost_history[-1]
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_epochs} epochs")
            self.final_epoch = self.so_epochs
            
        print(f"Thời gian training: {self.training_time:.2f}s")
        print(f"Loss cuối: {self.final_cost:.6f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")  # Bias là phần tử cuối của weights
        print(f"Số weights (bao gồm bias): {len(self.weights)}")
        
        return {
            'weights': self.weights,  # Bao gồm bias ở cuối
            'bias': self.weights[-1],  # Bias riêng để tương thích
            'cost_history': self.cost_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'training_time': self.training_time,
            'final_cost': self.final_cost,
            'converged': self.converged,
            'final_epoch': self.final_epoch
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
                           f"Stochastic Gradient Descent - {self.ham_loss.upper()}")
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
        
        # Save comprehensive results.json
        print(f"   Lưu kết quả vào {results_dir}/results.json")
        results_data = {
            "algorithm": f"Stochastic Gradient Descent - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "learning_rate": self.learning_rate,
                "epochs": self.so_epochs,
                "batch_size": self.batch_size,
                "random_state": self.random_state,
                "tolerance": self.tolerance
            },
            "training_results": {
                "training_time": self.training_time,
                "converged": self.converged,
                "final_epoch": self.final_epoch,
                "total_epochs": self.so_epochs,
                "final_cost": float(self.final_cost),
                "final_gradient_norm": float(self.gradient_norms[-1]) if self.gradient_norms else 0
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
                "epochs_to_converge": self.final_epoch,
                "final_cost_change": float(self.cost_history[-1] - self.cost_history[-2]) if len(self.cost_history) > 1 else 0.0,
                "convergence_rate": "sublinear",  # SGD có sublinear convergence
                "cost_reduction_ratio": float(self.cost_history[0] / self.cost_history[-1]) if len(self.cost_history) > 0 else 1.0
            },
            "algorithm_specific": {
                "method_type": "stochastic_gradient_descent",
                "batch_processing": True,
                "batch_size": self.batch_size,
                "epoch_based_training": True,
                "data_shuffling": True,
                "noisy_gradients": "inherent_in_SGD",
                "convergence_type": "probabilistic"
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
        
        print(f"\n Kết quả đã được lưu vào: {results_dir.absolute()}")
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
        
        print(f"\n📊 Tạo biểu đồ...")
        
        # 1. Training curve (cost over epochs)
        print("   - Vẽ đường training cost")
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
        print("   - So sánh dự đoán vs thực tế")
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
        
        # 3. Optimization trajectory (contour plot)
        print("   - Vẽ đường đồng mực optimization")
        if hasattr(self, 'weights_history') and len(self.weights_history) > 0:
            # Sample weights history for performance (every 10th point)
            step = max(1, len(self.weights_history) // 100)
            sampled_weights = np.array(self.weights_history[::step])
            
            # Chuẩn bị X_test với bias cho visualization
            X_test_with_bias = add_bias_column(X_test)
            
            ve_duong_dong_muc_optimization(
                loss_function=self.loss_func,
                weights_history=sampled_weights,
                X=X_test_with_bias, y=y_test,
                bias_history=None,  # Không cần bias riêng nữa
                title=f"Stochastic GD {self.ham_loss.upper()} - Optimization Path",
                save_path=str(results_dir / "optimization_trajectory.png")
            )
        else:
            print("     Không có weights history để vẽ contour plot")
        
        print(f"✅ Biểu đồ đã lưu vào: {results_dir.absolute()}")