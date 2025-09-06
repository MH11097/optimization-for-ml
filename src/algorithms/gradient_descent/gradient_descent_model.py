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
    du_doan, danh_gia_mo_hinh, in_ket_qua_danh_gia, kiem_tra_dieu_kien_dung,
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss,
    add_bias_column
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te
)


class GradientDescentModel:
    """   
    Parameters:
    - ham_loss: 'ols', 'ridge', 'lasso'
    - learning_rate: Tỷ lệ học
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ
    - regularization: Tham số regularization cho Ridge/Lasso
    - convergence_check_freq: Tần suất kiểm tra hội tụ (mỗi N iterations)
    """
    
    def __init__(self, ham_loss='ols', learning_rate=0.1, so_lan_thu=100000, diem_dung=1e-5, regularization=0.01, 
                 convergence_check_freq=100, step_size_method='constant', backtrack_c1=1e-4, backtrack_rho=0.8, 
                 adaptive_beta1=0.9, adaptive_beta2=0.999, adaptive_eps=1e-8, wolfe_c2=0.9, decay_gamma=0.95):
        self.ham_loss = ham_loss.lower()
        self.learning_rate = learning_rate
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.regularization = regularization
        self.convergence_check_freq = convergence_check_freq
        
        # Step size method parameters
        self.step_size_method = step_size_method  # 'constant', 'backtracking', 'decreasing_linear', 'decreasing_sqrt', 'adaptive'
        self.backtrack_c1 = backtrack_c1  # Armijo constant
        self.backtrack_rho = backtrack_rho  # Backtracking reduction factor
        self.adaptive_beta1 = adaptive_beta1  # Adam-like momentum parameter
        self.adaptive_beta2 = adaptive_beta2  # Adam-like second moment parameter
        self.adaptive_eps = adaptive_eps  # Adam-like epsilon
        self.wolfe_c2 = wolfe_c2  # Wolfe curvature condition parameter
        self.decay_gamma = decay_gamma  # Exponential decay factor
        
        # Adaptive step size variables
        if self.step_size_method == 'adaptive':
            self.m = None  # First moment estimate
            self.v = None  # Second moment estimate
            self.t = 0  # Time step
        
        # Sử dụng unified functions với format mới (bias trong X)
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes_history = []  # Lưu step size cho mỗi iteration
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0
        
        # Khởi tạo complexity tracker
        self.complexity_tracker = None  # Will be initialized in fit()

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
    
    def _backtracking_line_search(self, X, y, weights, gradient, direction):
        """
        Thực hiện backtracking line search theo thuật toán Armijo
        
        Args:
            X, y: Dữ liệu
            weights: Trọng số hiện tại
            gradient: Gradient tại weights hiện tại
            direction: Hướng tìm kiếm (thường là -gradient)
        
        Returns:
            alpha: Step size phù hợp
        """
        alpha = self.learning_rate
        current_loss = self.loss_func(X, y, weights)
        
        # Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
        armijo_lhs = lambda a: self.loss_func(X, y, weights + a * direction)
        armijo_rhs = lambda a: current_loss + self.backtrack_c1 * a * np.dot(gradient, direction)
        
        # Backtracking loop
        max_backtracks = 50
        for _ in range(max_backtracks):
            if armijo_lhs(alpha) <= armijo_rhs(alpha):
                break
            alpha *= self.backtrack_rho
        
        return alpha

    
    def _wolfe_line_search(self, X, y, weights, gradient, direction):
        """
        Thực hiện Wolfe line search với cả Armijo và curvature conditions
        
        Args:
            X, y: Dữ liệu
            weights: Trọng số hiện tại
            gradient: Gradient tại weights hiện tại
            direction: Hướng tìm kiếm (thường là -gradient)
        
        Returns:
            alpha: Step size phù hợp
        """
        alpha = self.learning_rate
        current_loss = self.loss_func(X, y, weights)
        
        # Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
        def armijo_condition(a):
            new_weights = weights + a * direction
            new_loss = self.loss_func(X, y, new_weights)
            return new_loss <= current_loss + self.backtrack_c1 * a * np.dot(gradient, direction)
        
        # Curvature condition (strong Wolfe): |grad(x + alpha*d)^T * d| <= c2 * |grad(x)^T * d|
        def curvature_condition(a):
            new_weights = weights + a * direction
            new_gradient, _ = self.grad_func(X, y, new_weights)  # Extract only gradient_w, ignore gradient_b
            return abs(np.dot(new_gradient, direction)) <= self.wolfe_c2 * abs(np.dot(gradient, direction))
        
        # Simple backtracking with both conditions
        max_backtracks = 50
        for _ in range(max_backtracks):
            if armijo_condition(alpha) and curvature_condition(alpha):
                break
            alpha *= self.backtrack_rho
            
            # Prevent alpha from becoming too small
            if alpha < 1e-10:
                alpha = 1e-10
                break
        
        return alpha
    
    def _get_step_size(self, iteration, gradient, X=None, y=None, weights=None):
        """
        Tính step size theo method được chọn
        
        Args:
            iteration: Iteration hiện tại (bắt đầu từ 0)
            gradient: Gradient hiện tại
            X, y, weights: Cần thiết cho backtracking
        
        Returns:
            step_size: Step size cho iteration này
        """
        if self.step_size_method == 'constant':
            return self.learning_rate
        
        elif self.step_size_method == 'decreasing_linear':
            # Alpha / (iteration + 1)
            return self.learning_rate / (iteration + 1)
        
        elif self.step_size_method == 'decreasing_sqrt':
            # Alpha / sqrt(iteration + 1)
            return self.learning_rate / np.sqrt(iteration + 1)
        
        elif self.step_size_method == 'decreasing_exponential':
            # Alpha * gamma^iteration
            return self.learning_rate * (self.decay_gamma ** iteration)
        
        elif self.step_size_method == 'backtracking':
            # Line search với Armijo condition
            direction = -gradient  # Steepest descent direction
            return self._backtracking_line_search(X, y, weights, gradient, direction)
        
        elif self.step_size_method == 'wolfe_conditions':
            # Wolfe line search với both Armijo và curvature conditions
            direction = -gradient  # Steepest descent direction
            return self._wolfe_line_search(X, y, weights, gradient, direction)
        
        elif self.step_size_method == 'adaptive':
            # Adam-like adaptive learning rate
            self.t += 1
            
            # Initialize moments on first iteration
            if self.m is None:
                self.m = np.zeros_like(gradient)
                self.v = np.zeros_like(gradient)
            
            # Update biased first and second moments
            self.m = self.adaptive_beta1 * self.m + (1 - self.adaptive_beta1) * gradient
            self.v = self.adaptive_beta2 * self.v + (1 - self.adaptive_beta2) * (gradient ** 2)
            
            # Bias correction
            m_hat = self.m / (1 - self.adaptive_beta1 ** self.t)
            v_hat = self.v / (1 - self.adaptive_beta2 ** self.t)
            
            # Return adaptive step size scaled by base learning rate
            return self.learning_rate / (np.sqrt(v_hat) + self.adaptive_eps)
        
        else:
            raise ValueError(f"Unknown step_size_method: {self.step_size_method}")
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, bias, loss_history, etc.
        """
        print(f"🚀 Training Gradient Descent - {self.ham_loss.upper()} - ")
        print(f"   Step size method: {self.step_size_method} - Base learning rate: {self.learning_rate} - Max iterations: {self.so_lan_thu}")
        if self.ham_loss in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")
        
        # Thêm cột bias vào X
        X_with_bias = add_bias_column(X)
        print(f"   Num of features: {X.shape[1]} (+1)")
        
        # Initialize weights (bao gồm bias ở cuối)
        n_features_with_bias = X_with_bias.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        
        # Initialize complexity tracker
        from utils.computational_complexity import ComputationalComplexityTracker
        self.complexity_tracker = ComputationalComplexityTracker(
            problem_size=(X.shape[0], X.shape[1])
        )
        self.complexity_tracker.start_tracking()
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes_history = []
        
        # Reset adaptive parameters if needed
        if self.step_size_method == 'adaptive':
            self.m = None
            self.v = None
            self.t = 0
        
        start_time = time.time()
        
        for lan_thu in range(self.so_lan_thu):
            # Tính gradient (luôn cần cho weight update)
            gradient_w, _ = self.grad_func(X_with_bias, y, self.weights)
            
            # Record complexity metrics
            self.complexity_tracker.record_gradient_evaluation(X_with_bias.shape)
            self.complexity_tracker.record_vector_operation(len(gradient_w), "basic")  # Weight update
            
            # Tính step size cho iteration này
            if self.step_size_method == 'adaptive':
                # For adaptive, step_size is a vector
                step_size = self._get_step_size(lan_thu, gradient_w, X_with_bias, y, self.weights)
                # Update weights with element-wise step size
                self.weights = self.weights - step_size * gradient_w
                # Lưu average step size cho visualization (ensure it's a scalar)
                avg_step_size = float(np.mean(step_size))
                self.step_sizes_history.append(avg_step_size)
            else:
                # For other methods, step_size is a scalar
                step_size = self._get_step_size(lan_thu, gradient_w, X_with_bias, y, self.weights)
                # Update weights with scalar step size
                self.weights = self.weights - step_size * gradient_w
                # Ensure step_size is a scalar before appending
                if np.isscalar(step_size):
                    self.step_sizes_history.append(float(step_size))
                else:
                    # If somehow we get a vector, take the mean
                    self.step_sizes_history.append(float(np.mean(step_size)))
            
            # Record memory allocation for weight copy
            self.complexity_tracker.record_memory_allocation(len(self.weights))
            
            # Chỉ tính loss và lưu history khi cần thiết
            should_check_converged = (
                (lan_thu + 1) % self.convergence_check_freq == 0 or 
                lan_thu == self.so_lan_thu - 1
            )
            
            if should_check_converged:
                # Chỉ tính loss khi cần (expensive operation)
                loss_value = self.loss_func(X_with_bias, y, self.weights)
                gradient_norm = np.linalg.norm(gradient_w)
                
                # Record complexity metrics for loss computation and norm
                self.complexity_tracker.record_function_evaluation(X_with_bias.shape)
                self.complexity_tracker.record_vector_operation(len(gradient_w), "norm")
                
                # Lưu vào history
                self.loss_history.append(loss_value)
                self.gradient_norms.append(gradient_norm)
                self.weights_history.append(self.weights.copy())                    
                cost_change = 0.0 if len(self.loss_history) == 0 else (self.loss_history[-1] - loss_value) if len(self.loss_history) == 1 else (self.loss_history[-2] - self.loss_history[-1])
                should_stop, converged, reason = kiem_tra_dieu_kien_dung(
                    gradient_norm=gradient_norm,
                    cost_change=cost_change,
                    iteration=lan_thu,
                    tolerance=self.diem_dung,
                    max_iterations=self.so_lan_thu,
                    loss_value=loss_value,
                    weights=self.weights
                )
                
                if should_stop:
                    if converged:
                        print(f"✅ Gradient Descent converged: {reason}")
                        self.complexity_tracker.mark_convergence(lan_thu + 1)
                    else:
                        print(f"⚠️ Gradient Descent stopped (not converged): {reason}")
                    self.converged = converged
                    self.final_iteration = lan_thu + 1
                    break

                # In thêm step size
                current_step_size = self.step_sizes_history[-1] if self.step_sizes_history else 0
                print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.6f}, Gradient = {gradient_norm:.6f}, Step size = {current_step_size}")
            
            # End iteration tracking
            self.complexity_tracker.end_iteration()
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"✅ Gradient Descent stopped: Đạt tối đa {self.so_lan_thu} vòng lặp")
            self.final_iteration = self.so_lan_thu
        
        print(f"Thời gian training: {self.training_time:.2f}s")
        print(f"Loss cuối: {self.loss_history[-1]:.6f}")
        print(f"Gradient norm cuối: {self.gradient_norms[-1]:.6f}")
        
        # Print complexity summary
        complexity_summary = self.complexity_tracker.get_summary_stats()
        print(f"📊 Complexity Summary:")
        print(f"   Total operations: {complexity_summary['total_operations']:,}")
        print(f"   Function evaluations: {complexity_summary['function_evaluations']}")
        print(f"   Gradient evaluations: {complexity_summary['gradient_evaluations']}")
        
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
            'bias': best_weights[-1], 
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'step_sizes_history': self.step_sizes_history,
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_iteration,
            'best_iteration': best_iteration,
            'best_loss': best_loss,
            'best_gradient_norm': best_gradient_norm,
            'final_loss': self.loss_history[-1],  # Để so sánh
            'final_gradient_norm': self.gradient_norms[-1],  # Để so sánh
            'complexity_metrics': self.complexity_tracker.get_complexity_analysis(self.final_iteration, self.converged)
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
        
        # Sử dụng bias từ weights (phần tử cuối) để tương thích với hàm cũ
        bias_value = self.weights[-1]
        weights_without_bias = self.weights[:-1]
        metrics = danh_gia_mo_hinh(weights_without_bias, X_test, y_test, bias_value)
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
        
        # Get complexity analysis
        complexity_analysis = self.complexity_tracker.get_complexity_analysis(
            self.final_iteration, self.converged
        ) if hasattr(self, 'complexity_tracker') and self.complexity_tracker else None
        
        # Lấy kết quả tốt nhất
        best_results = self._get_best_results()
        best_weights = best_results['best_weights']
        best_loss = best_results['best_loss']
        best_gradient_norm = best_results['best_gradient_norm'] 
        best_iteration = best_results['best_iteration']
        
        # Save comprehensive results.json
        results_data = {
            "algorithm": f"Gradient Descent - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "learning_rate": self.learning_rate,
                "max_iterations": self.so_lan_thu,
                "tolerance": self.diem_dung,
                "step_size_method": self.step_size_method
            },
            "training_results": {
                "training_time": self.training_time,
                "converged": self.converged,
                "final_iteration": self.final_iteration,
                "total_iterations": self.so_lan_thu,
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1]),
                # Thêm thông tin best results
                "best_iteration": best_iteration,
                "best_loss": float(best_loss),
                "best_gradient_norm": float(best_gradient_norm),
                "improvement_from_final": {
                    "loss_improvement": float(self.loss_history[-1] - best_loss),
                    "gradient_improvement": float(self.gradient_norms[-1] - best_gradient_norm),
                    "iterations_earlier": self.final_iteration - best_iteration
                }
            },
            "weights_analysis": {
                "n_features": len(best_weights) - 1,  # Không tính bias
                "n_weights_total": len(best_weights),  # Tính cả bias
                "bias_value": float(best_weights[-1]),
                "weights_without_bias": best_weights[:-1].tolist(),
                "complete_weight_vector": best_weights.tolist(),
                "weights_stats": {
                    "min": float(np.min(best_weights[:-1])),  # Stats chỉ của weights, không tính bias
                    "max": float(np.max(best_weights[:-1])),
                    "mean": float(np.mean(best_weights[:-1])),
                    "std": float(np.std(best_weights[:-1]))
                }
            },
            "convergence_analysis": {
                "iterations_to_converge": self.final_iteration,
                "best_iteration_found": best_iteration,
                "final_cost_change": float(self.loss_history[-1] - self.loss_history[-2]) if len(self.loss_history) > 1 else 0.0,
                "convergence_rate": "linear",  # Gradient Descent có convergence rate tuyến tính
                "loss_reduction_ratio": float(self.loss_history[0] / best_loss) if len(self.loss_history) > 0 else 1.0
            },
            "algorithm_specific": {
                "gradient_descent_type": "standard",
                "step_size_method": self.step_size_method,
                "step_size_constant": self.step_size_method == 'constant',
                "momentum_used": False,
                "returns_best_result": True,  # Đánh dấu rằng trả về best result
                "backtracking_parameters": {
                    "c1": self.backtrack_c1,
                    "rho": self.backtrack_rho
                } if self.step_size_method == 'backtracking' else None,
                "adaptive_parameters": {
                    "beta1": self.adaptive_beta1,
                    "beta2": self.adaptive_beta2,
                    "eps": self.adaptive_eps
                } if self.step_size_method == 'adaptive' else None
            }
        }
        
        # Add complexity metrics if available
        if complexity_analysis:
            results_data["computational_complexity"] = complexity_analysis
        
        if self.ham_loss in ['ridge', 'lasso']:
            results_data["parameters"]["regularization"] = self.regularization
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        training_df = pd.DataFrame({
            'iteration': range(0, len(self.loss_history)*self.convergence_check_freq, self.convergence_check_freq),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms
        })
        
        # Save step sizes history separately if available
        if self.step_sizes_history:
            step_sizes_df = pd.DataFrame({
                'iteration': range(len(self.step_sizes_history)),
                'step_size': self.step_sizes_history
            })
            step_sizes_df.to_csv(results_dir / "step_sizes_history.csv", index=False)
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        # Save complexity metrics separately for detailed analysis
        if complexity_analysis:
            with open(results_dir / "complexity_analysis.json", 'w') as f:
                json.dump(complexity_analysis, f, indent=2)
        
        print(f"\n✅ Kết quả đã được lưu vào: {results_dir.absolute()}")
        print(f"🏆 Sử dụng best results từ iteration {best_iteration} (gradient norm: {best_gradient_norm:.6f})")
        if complexity_analysis:
            print(f"📊 Complexity metrics saved to: complexity_analysis.json")
        
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
        
        # 1. Convergence curves - now with actual iteration numbers
        print("   - Vẽ đường hội tụ")
        # Create iteration values based on convergence_check_freq
        iterations = list(range(0, len(self.loss_history) * self.convergence_check_freq, self.convergence_check_freq))
        
        ve_duong_hoi_tu(self.loss_history, self.gradient_norms, 
                        iterations=iterations,
                        title=f"Gradient Descent {self.ham_loss.upper()} - Hội tụ",
                        save_path=str(results_dir / "convergence_analysis.png"))
        
        # 2. Predictions vs Actual
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Gradient Descent {self.ham_loss.upper()} - Dự đoán vs Thực tế",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mức)
        print("   - Vẽ đường đồng mức optimization")
        
        # Chuẩn bị X_test với bias cho visualization
        X_test_with_bias = add_bias_column(X_test)
        
        ve_duong_dong_muc_optimization(
            loss_function=self.loss_func,
            weights_history=self.weights_history,  # Pass full history
            X=X_test_with_bias, y=y_test,
            title=f"Gradient Descent {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png"),
            original_iterations=self.final_iteration,  # Use actual number of iterations
            convergence_check_freq=self.convergence_check_freq,  # Pass convergence frequency
            max_trajectory_points=None  # None = show all points
        )
        
