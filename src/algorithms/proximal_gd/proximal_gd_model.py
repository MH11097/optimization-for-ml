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
    du_doan, danh_gia_mo_hinh, in_ket_qua_danh_gia, kiem_tra_hoi_tu,
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss,
    add_bias_column
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te
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
        
        # Sử dụng unified function cho gradient của phần smooth (OLS) với format mới (bias trong X)
        self.grad_smooth_func = lambda X, y, w: tinh_gradient_ham_loss('ols', X, y, w, None)
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None  # Bây giờ bao gồm bias ở cuối
        self.loss_history = []
        self.gradient_norms = []
        self.sparsity_history = []  # Số lượng weights = 0 (không tính bias)
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
        Lưu ý: Không áp dụng regularization cho bias (phần tử cuối)
        """
        result = z.copy()
        
        if self.ham_loss == 'lasso':
            # Pure L1: soft thresholding cho tất cả trừ bias
            result[:-1] = self._soft_threshold(z[:-1], self.learning_rate * self.lambda_l1)
            # Giữ nguyên bias (không regularize)
            result[-1] = z[-1]
        
        elif self.ham_loss == 'elastic_net':
            # Elastic Net: L1 + L2 cho tất cả trừ bias
            soft_thresh = self._soft_threshold(z[:-1], self.learning_rate * self.lambda_l1)
            result[:-1] = soft_thresh / (1 + self.learning_rate * self.lambda_l2)
            # Giữ nguyên bias (không regularize)
            result[-1] = z[-1]
            
        return result
    
    def _compute_loss(self, X, y, weights):
        """Tính loss function - không regularize bias (phần tử cuối)"""
        # MSE loss
        mse_loss = tinh_gia_tri_ham_loss('ols', X, y, weights, None)
        
        # L1 regularization (không áp dụng cho bias)
        l1_penalty = self.lambda_l1 * np.sum(np.abs(weights[:-1]))
        
        # L2 regularization (cho Elastic Net, không áp dụng cho bias)
        l2_penalty = 0
        if self.ham_loss == 'elastic_net':
            l2_penalty = 0.5 * self.lambda_l2 * np.sum(weights[:-1] ** 2)
        
        return mse_loss + l1_penalty + l2_penalty
    
    def _compute_sparsity(self, weights, threshold=1e-8):
        """Tính số lượng weights gần bằng 0 (sparsity) - không tính bias"""
        return np.sum(np.abs(weights[:-1]) < threshold)
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"🚀 Training Proximal Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Lambda L1: {self.lambda_l1}")
        if self.ham_loss == 'elastic_net':
            print(f"   Lambda L2: {self.lambda_l2}")
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
        self.sparsity_history = []
        self.weights_history = []
        
        start_time = time.time()
        
        for lan_thu in range(self.so_lan_thu):
            # Forward step: z = w - α∇f(w)
            gradient, _ = self.grad_smooth_func(X_with_bias, y, self.weights)  # _ vì không cần gradient_b riêng
            z = self.weights - self.learning_rate * gradient
            
            # Proximal step: w = prox_λ(z)
            self.weights = self._proximal_operator(z)
            
            # Compute loss (with regularization)
            loss_value = self._compute_loss(X_with_bias, y, self.weights)
            
            # Store history
            self.loss_history.append(loss_value)
            gradient_norm = np.linalg.norm(gradient)
            self.gradient_norms.append(gradient_norm)
            
            sparsity = self._compute_sparsity(self.weights)
            self.sparsity_history.append(sparsity)
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
                print(f"✅ Proximal GD stopped: {reason}")
                self.converged = True
                self.final_iteration = lan_thu + 1
                break
            
            # Progress update
            if (lan_thu + 1) % 100 == 0:
                n_weights_without_bias = n_features_with_bias - 1
                print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.6f}, Gradient = {gradient_norm:.6f}, Sparsity = {sparsity}/{n_weights_without_bias}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_lan_thu} vòng lặp")
            self.final_iteration = self.so_lan_thu
        
        final_sparsity = self._compute_sparsity(self.weights)
        n_weights_without_bias = n_features_with_bias - 1
        print(f"Thời gian training: {self.training_time:.2f}s")
        print(f"Loss cuối: {self.loss_history[-1]:.6f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")  # Bias là phần tử cuối của weights
        print(f"Final sparsity: {final_sparsity}/{n_weights_without_bias} ({final_sparsity/n_weights_without_bias*100:.1f}%)")
        print(f"Số weights (bao gồm bias): {len(self.weights)}")
        
        return {
            'weights': self.weights,  # Bao gồm bias ở cuối
            'bias': self.weights[-1],  # Bias riêng để tương thích
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
        
        # Save comprehensive results.json
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
                "convergence_rate": "linear",  # Proximal GD có linear convergence
                "loss_reduction_ratio": float(self.loss_history[0] / self.loss_history[-1]) if len(self.loss_history) > 0 else 1.0
            },
            "sparsity_analysis": {
                "final_sparsity": int(self.sparsity_history[-1]) if self.sparsity_history else 0,
                "total_features": len(self.weights) - 1,  # Không tính bias
                "sparsity_ratio": float(self.sparsity_history[-1] / (len(self.weights) - 1)) if self.sparsity_history else 0,
                "non_zero_weights": int((len(self.weights) - 1) - self.sparsity_history[-1]) if self.sparsity_history else (len(self.weights) - 1),
                "sparsity_evolution": "L1_regularization_induced_sparsity",
                "regularization_effect": "Feature_selection_via_soft_thresholding"
            },
            "algorithm_specific": {
                "method_type": "proximal_gradient",
                "regularization_type": self.ham_loss,
                "proximal_operator": "soft_thresholding" if self.ham_loss == "lasso" else "elastic_net_prox",
                "sparsity_inducing": True,
                "feature_selection": True,
                "non_smooth_optimization": True
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
        
        print(f"\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/proximal_gd"):
        """
        Tạo các biểu đồ visualization
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Tạo biểu đồ...")
        
        # 1. Convergence curves với sparsity
        print("   - Vẽ đường hội tụ với sparsity analysis")
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
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Proximal GD {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mực)
        print("   - Vẽ đường đồng mực optimization")
        sample_frequency = max(1, len(self.weights_history) // 50)
        sampled_weights = self.weights_history[::sample_frequency]
        
        # Chuẩn bị X_test với bias cho visualization
        X_test_with_bias = add_bias_column(X_test)
        
        # Use OLS loss for smooth part (Proximal GD separates smooth and non-smooth)
        def smooth_loss_func(X, y, w):
            return tinh_gia_tri_ham_loss('ols', X, y, w, None)
        
        ve_duong_dong_muc_optimization(
            loss_function=smooth_loss_func,
            weights_history=sampled_weights,
            X=X_test_with_bias, y=y_test,
            bias_history=None,  # Không cần bias riêng nữa
            title=f"Proximal GD {self.ham_loss.upper()} - Optimization Path (Smooth Part)",
            save_path=str(results_dir / "optimization_trajectory.png")
        )
        
        print(f"✅ Biểu đồ đã lưu vào: {results_dir.absolute()}")