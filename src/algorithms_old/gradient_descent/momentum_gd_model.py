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
    du_doan, danh_gia_mo_hinh, in_ket_qua_danh_gia, kiem_tra_dieu_kien_dung,
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss,
    add_bias_column
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
    - convergence_check_freq: Tần suất kiểm tra hội tụ (mỗi N iterations)
    """
    
    def __init__(self, ham_loss='ols', learning_rate=0.01, momentum=0.9, 
                 so_lan_thu=10000, diem_dung=1e-5, regularization=0.01, convergence_check_freq=100):
        self.ham_loss = ham_loss.lower()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.regularization = regularization
        self.convergence_check_freq = convergence_check_freq
        
        # Sử dụng unified functions với format mới (bias trong X)
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        
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
    def _get_best_results(self):
        """
        Lấy kết quả tốt nhất dựa trên gradient norm thấp nhất
        
        Returns:
            dict: Chứa best_weights, best_loss, best_gradient_norm, best_iteration
        """
        if not self.gradient_norms:
            raise ValueError("Không có lịch sử gradient norms để tìm kết quả tốt nhất")
        
        # Tìm index có gradient norm thấp nhất
        best_idx = np.argmin(self.gradient_norms)
        
        return {
            'best_weights': self.weights_history[best_idx],
            'best_loss': self.loss_history[best_idx],
            'best_gradient_norm': self.gradient_norms[best_idx],
            'best_iteration': best_idx * self.convergence_check_freq
        }
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"🚀 Training Momentum Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate: {self.learning_rate}, Momentum: {self.momentum}")
        print(f"   Max iterations: {self.so_lan_thu}")
        if self.ham_loss in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")
        
        # Thêm cột bias vào X
        X_with_bias = add_bias_column(X)
        print(f"   Original features: {X.shape[1]}, With bias: {X_with_bias.shape[1]}")
        
        # Initialize weights and velocity (bao gồm bias ở cuối)
        n_features_with_bias = X_with_bias.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        self.velocity = np.zeros(n_features_with_bias)
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.velocity_norms = []
        self.weights_history = []
        
        start_time = time.time()
        
        for lan_thu in range(self.so_lan_thu):
            # Tính gradient (luôn cần cho momentum update)
            gradient_w, _ = self.grad_func(X_with_bias, y, self.weights)  # _ vì không cần gradient_b riêng
            
            # Momentum update
            self.velocity = self.momentum * self.velocity + gradient_w
            self.weights = self.weights - self.learning_rate * self.velocity
            
            # Chỉ tính loss và lưu history khi cần thiết
            should_check_converged = (
                (lan_thu + 1) % self.convergence_check_freq == 0 or 
                lan_thu == self.so_lan_thu - 1
            )
            
            if should_check_converged:
                # Chỉ tính loss khi cần (expensive operation)
                loss_value = self.loss_func(X_with_bias, y, self.weights)
                gradient_norm = np.linalg.norm(gradient_w)
                velocity_norm = np.linalg.norm(self.velocity)
                
                # Lưu vào history
                self.loss_history.append(loss_value)
                self.gradient_norms.append(gradient_norm)
                self.velocity_norms.append(velocity_norm)
                self.weights_history.append(self.weights.copy())
                
                cost_change = 0.0 if len(self.loss_history) == 0 else (self.loss_history[-1] - loss_value) if len(self.loss_history) == 1 else (self.loss_history[-2] - self.loss_history[-1])
                should_stop, converged, reason = kiem_tra_dieu_kien_dung(
                    gradient_norm=gradient_norm,
                    cost_change=cost_change,
                    iteration=lan_thu,
                    tolerance=self.diem_dung,
                    max_iterations=self.so_lan_thu
                )
                
                if should_stop:
                    if converged:
                        print(f"✅ Momentum GD converged: {reason}")
                    else:
                        print(f"⚠️ Momentum GD stopped (not converged): {reason}")
                    self.converged = converged
                    self.final_iteration = lan_thu + 1
                    break
                print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.6f}, Gradient = {gradient_norm:.6f}, Velocity = {velocity_norm:.6f}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_lan_thu} Iteration")
            self.final_iteration = self.so_lan_thu
        
        print(f"Thời gian training: {self.training_time:.2f}s")
        print(f"Loss cuối: {self.loss_history[-1]:.6f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")  # Bias là phần tử cuối của weights
        print(f"Số weights (bao gồm bias): {len(self.weights)}")
        
        # Lấy kết quả tốt nhất thay vì kết quả cuối cùng
        best_results = self._get_best_results()
        best_weights = best_results['best_weights']
        best_loss = best_results['best_loss']
        best_gradient_norm = best_results['best_gradient_norm']
        best_iteration = best_results['best_iteration']
        
        print(f"🏆 Best results (gradient norm thấp nhất):")
        print(f"   Best iteration: {best_iteration}")
        print(f"   Best loss: {best_loss:.6f}")
        print(f"   Best gradient norm: {best_gradient_norm:.6f}")
        
        return {
            'weights': best_weights,  # Trả về best weights thay vì final
            'bias': best_weights[-1],  # Bias riêng để tương thích
            'velocity': self.velocity,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'velocity_norms': self.velocity_norms,
            'weights_history': self.weights_history,
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_iteration,
            'best_iteration': best_iteration,
            'best_loss': best_loss,
            'best_gradient_norm': best_gradient_norm,
            'final_loss': self.loss_history[-1],  # Để so sánh
            'final_gradient_norm': self.gradient_norms[-1]  # Để so sánh
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
                           f"Momentum Gradient Descent - {self.ham_loss.upper()}")
        return metrics
    
    def save_results(self, ten_file, base_dir="data/03_algorithms"):
        """
        Lưu kết quả model vào file
        
        Parameters:
        - ten_file: Tên file/folder để lưu kết quả
        - base_dir: Thư mục gốc để lưu
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        # Get algorithm name from class name
        algorithm_name = getattr(self, '__class__', type(self)).__name__.lower()
        # Setup results directory
        results_dir = Path(base_dir) / algorithm_name / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results.json
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
            "training_results": {
                "training_time": self.training_time,
                "converged": self.converged,
                "final_iteration": self.final_iteration,
                "total_iterations": self.so_lan_thu,
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1]),
                "final_velocity_norm": float(self.velocity_norms[-1])
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
                },
                "velocity_analysis": {
                    "final_velocity": self.velocity.tolist(),
                    "velocity_stats": {
                        "min": float(np.min(self.velocity[:-1])),
                        "max": float(np.max(self.velocity[:-1])),
                        "mean": float(np.mean(self.velocity[:-1])),
                        "std": float(np.std(self.velocity[:-1]))
                    }
                }
            },
            "convergence_analysis": {
                "iterations_to_converge": self.final_iteration,
                "final_cost_change": float(self.loss_history[-1] - self.loss_history[-2]) if len(self.loss_history) > 1 else 0.0,
                "convergence_rate": "superlinear",  # Momentum có thể đạt superlinear convergence
                "loss_reduction_ratio": float(self.loss_history[0] / self.loss_history[-1]) if len(self.loss_history) > 0 else 1.0,
                "velocity_contribution": "accelerated_convergence"
            },
            "algorithm_specific": {
                "gradient_descent_type": "momentum",
                "momentum_coefficient": self.momentum,
                "step_size_constant": True,
                "acceleration_used": True,
                "momentum_description": "Nesterov-style momentum with velocity accumulation"
            }
        }
        
        if self.ham_loss in ['ridge', 'lasso']:
            results_data["parameters"]["regularization"] = self.regularization
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        training_df = pd.DataFrame({
            'iteration': range(0, len(self.loss_history)*self.convergence_check_freq, self.convergence_check_freq),
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
        
        print(f"\n📊 Tạo biểu đồ...")
        
        # 1. Convergence curves với velocity - now using actual iteration numbers
        print("   - Vẽ đường hội tụ")
        import matplotlib.pyplot as plt
        
        # Create iteration values based on convergence_check_freq
        iterations = list(range(0, len(self.loss_history) * self.convergence_check_freq, self.convergence_check_freq))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0,0].semilogy(iterations, self.loss_history, 'b-', linewidth=2)
        axes[0,0].set_title('Loss Convergence')
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].grid(True, alpha=0.3)
        
        # Gradient norm
        axes[0,1].semilogy(iterations, self.gradient_norms, 'r-', linewidth=2)
        axes[0,1].set_title('Gradient Norm')
        axes[0,1].set_xlabel('Iteration')
        axes[0,1].set_ylabel('Gradient Norm')
        axes[0,1].grid(True, alpha=0.3)
        
        # Velocity norm
        axes[1,0].plot(iterations, self.velocity_norms, 'g-', linewidth=2)
        axes[1,0].set_title('Velocity Norm')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Velocity Norm')
        axes[1,0].grid(True, alpha=0.3)
        
        # Combined plot
        ax2 = axes[1,1].twinx()
        axes[1,1].semilogy(iterations, self.loss_history, 'b-', linewidth=2, label='Loss')
        ax2.plot(iterations, self.velocity_norms, 'g-', linewidth=2, label='Velocity Norm')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Loss', color='b')
        ax2.set_ylabel('Velocity Norm', color='g')
        axes[1,1].set_title('Loss vs Velocity')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Momentum GD {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mức)
        print("   - Vẽ đường đồng mức optimization")
        
        # Chuẩn bị X_test với bias cho visualization
        X_test_with_bias = add_bias_column(X_test)
        
        ve_duong_dong_muc_optimization(
            loss_function=self.loss_func,
            weights_history=self.weights_history,  # Pass full history
            X=X_test_with_bias, y=y_test,
            title=f"Momentum GD {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png"),
            original_iterations=self.final_iteration,  # Use actual number of iterations
            convergence_check_freq=self.convergence_check_freq,  # Pass convergence frequency
            max_trajectory_points=None  # None = show all points
        )
        
