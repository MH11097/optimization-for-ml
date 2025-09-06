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
    du_doan, danh_gia_mo_hinh, in_ket_qua_danh_gia, kiem_tra_dieu_kien_dung,
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss,
    add_bias_column
)
from utils.visualization_utils import (
    ve_duong_hoi_tu, ve_du_doan_vs_thuc_te, ve_duong_dong_muc_optimization
)


class SGDModel:
    """   
    Stochastic Gradient Descent với hỗ trợ:
    - Multiple learning rate schedules
    - Batch processing với size tùy chọn
    - Momentum support
    - Enhanced shuffling strategies
    - Computational complexity tracking
    
    Parameters:
    - ham_loss: 'ols', 'ridge', 'lasso'  
    - learning_rate: Tỷ lệ học ban đầu
    - so_epochs: Số epochs tối đa
    - batch_size: Kích thước batch
    - diem_dung: Ngưỡng hội tụ
    - learning_rate_schedule: Phương pháp điều chỉnh learning rate
    - momentum: Hệ số momentum (0 = không dùng momentum)
    - convergence_check_freq: Tần suất kiểm tra hội tụ (mỗi N epochs)
    - shuffle_each_epoch: Enhanced shuffling với seed mới mỗi epoch
    - randomize_each_epoch: Full randomization với replacement mỗi epoch
    """
    
    def __init__(self, ham_loss='ols', learning_rate=0.01, so_epochs=100, batch_size=32, 
                 diem_dung=1e-5, regularization=0.01, learning_rate_schedule='constant',
                 momentum=0.0, convergence_check_freq=10, random_state=42,
                 # Enhanced shuffling strategies
                 shuffle_each_epoch=False, randomize_each_epoch=False,
                 # Learning rate schedule parameters
                 decay_rate=0.95, decay_steps=100,
                 # Fixed step length option
                 use_fixed_step_length=False, step_length=0.01):
        
        self.ham_loss = ham_loss.lower()
        self.learning_rate = learning_rate
        self.so_epochs = so_epochs
        self.batch_size = batch_size
        self.diem_dung = diem_dung
        self.regularization = regularization
        self.learning_rate_schedule = learning_rate_schedule
        self.momentum = momentum
        self.use_momentum = momentum > 0
        self.convergence_check_freq = convergence_check_freq
        self.random_state = random_state
        
        # Enhanced shuffling options
        self.shuffle_each_epoch = shuffle_each_epoch
        self.randomize_each_epoch = randomize_each_epoch
        
        # Learning rate schedule parameters
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
        # Fixed step length parameters
        self.use_fixed_step_length = use_fixed_step_length
        self.step_length = step_length
        
        # Sử dụng unified functions với format mới (bias trong X)
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        
        # Initialize attributes to store results
        self.weights = None
        self.velocity = None  # For momentum
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.learning_rates_history = []
        self.training_time = 0
        self.converged = False
        self.final_epoch = 0
        self.final_cost = 0
        
        print(f"SGD Model initialized:")
        print(f"   Loss function: {self.ham_loss.upper()}")
        if self.use_fixed_step_length:
            print(f"   Step length: {self.step_length} (fixed step length mode)")
        else:
            print(f"   Learning rate: {self.learning_rate} ({self.learning_rate_schedule})")
        print(f"   Epochs: {self.so_epochs}, Batch size: {self.batch_size}")
        print(f"   Momentum: {self.momentum}")
        print(f"   Random state: {self.random_state}")
        if self.use_fixed_step_length:
            print(f"   Using fixed step length instead of fixed step size")
        if regularization and ham_loss in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")

    def _get_best_results(self):
        """
        Lấy kết quả tốt nhất dựa trên gradient norm thấp nhất
        
        Returns:
            dict: Chứa best_weights, best_loss, best_gradient_norm, best_epoch
        """
        if not self.gradient_norms:
            raise ValueError("Không có lịch sử gradient norms để tìm kết quả tốt nhất")
        
        if len(self.gradient_norms) != len(self.weights_history) or len(self.gradient_norms) != len(self.loss_history):
            raise ValueError("Lịch sử gradient norms, weights và loss không cùng độ dài")
        
        # Tìm index có gradient norm thấp nhất
        best_idx = np.argmin(self.gradient_norms)
        
        return {
            'best_weights': self.weights_history[best_idx],
            'best_loss': float(self.loss_history[best_idx]),
            'best_gradient_norm': float(self.gradient_norms[best_idx]),
            'best_epoch': int(best_idx * self.convergence_check_freq)
        }
    
    def _get_learning_rate(self, epoch):
        """
        Tính learning rate theo schedule được chọn
        
        Args:
            epoch: Epoch hiện tại (0-indexed)
        """
        if self.learning_rate_schedule == 'constant':
            return self.learning_rate
        elif self.learning_rate_schedule == 'linear_decay':
            # Giảm tuyến tính: lr * (1 - epoch/max_epochs)
            decay_factor = 1.0 - (epoch / max(self.so_epochs, 1))
            return self.learning_rate * max(decay_factor, 0.01)  # Minimum 1% of original
        elif self.learning_rate_schedule == 'exponential_decay':
            # Giảm exponential: lr * decay_rate^(epoch/decay_steps)
            return self.learning_rate * (self.decay_rate ** (epoch / self.decay_steps))
        elif self.learning_rate_schedule == 'sqrt_decay':
            # Giảm theo sqrt: lr / sqrt(epoch + 1)
            return self.learning_rate / np.sqrt(epoch + 1)
        elif self.learning_rate_schedule == 'backtracking':
            # Simple backtracking implementation for SGD (placeholder)
            return self.learning_rate * (0.9 ** (epoch // 10))
        else:
            return self.learning_rate
    
    def _tinh_gradient_sample(self, xi, yi, weights):
        """
        Tính gradient cho 1 sample - optimized cho SGD
        
        Args:
            xi: Feature vector for sample i (đã có bias)
            yi: Target value for sample i  
            weights: Weight vector hiện tại
        """
        # Direct gradient computation for single sample (more efficient)
        if self.ham_loss == 'ols':
            # OLS: gradient = X^T * (X*w - y) / n = xi * (xi^T * w - yi)
            prediction = np.dot(xi, weights)
            residual = prediction - yi
            gradient = residual * xi
        elif self.ham_loss == 'ridge':
            # Ridge: gradient = X^T * (X*w - y) / n + lambda * w
            prediction = np.dot(xi, weights)
            residual = prediction - yi
            gradient = residual * xi + self.regularization * weights
            # Don't regularize bias term (last element)
            gradient[-1] -= self.regularization * weights[-1]  # Remove regularization from bias
        elif self.ham_loss == 'lasso':
            # Lasso: gradient = X^T * (X*w - y) / n + lambda * sign(w)
            prediction = np.dot(xi, weights)
            residual = prediction - yi
            gradient = residual * xi + self.regularization * np.sign(weights)
            # Don't regularize bias term (last element)
            gradient[-1] -= self.regularization * np.sign(weights[-1])  # Remove regularization from bias
        else:
            # Fallback to unified function for other loss functions
            X_sample = xi.reshape(1, -1)
            y_sample = np.array([yi])
            gradient, _ = tinh_gradient_ham_loss(self.ham_loss, X_sample, y_sample, weights, None, self.regularization)
        
        return gradient
    
    def _tinh_chi_phi(self, X, y, weights):
        """
        Tính chi phí (cost) cho toàn bộ dataset sử dụng unified function
        
        Args:
            X: Feature matrix (đã có bias)
            y: Target vector
            weights: Weight vector
        """
        return self.loss_func(X, y, weights)
    
    def _check_sgd_convergence(self, gradient_norm, cost_change, iteration, epoch_cost, loss_history):
        """
        Kiểm tra điều kiện dừng cho SGD với enhanced logic
        
        Returns:
            (should_stop, converged, reason)
        """
        # Use unified convergence check with SGD adaptations
        # Adjust tolerance for SGD noise (more lenient)
        sgd_tolerance = self.diem_dung * 10  # 10x more lenient for cost changes
        
        should_stop, converged, reason = kiem_tra_dieu_kien_dung(
            gradient_norm=gradient_norm,
            cost_change=cost_change,
            iteration=iteration,
            tolerance=self.diem_dung,  # Keep original tolerance for gradient
            max_iterations=self.so_epochs,
            loss_value=epoch_cost,
            weights=self.weights
        )
        
        # Additional SGD-specific divergence check
        if not should_stop and len(loss_history) >= 5:
            recent_costs = loss_history[-5:]
            if len(recent_costs) >= 2 and recent_costs[-1] > recent_costs[0] * 2:
                return True, False, "Cost increased significantly (possible divergence in SGD)"
        
        return should_stop, converged, reason
    
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"Training Stochastic Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate schedule: {self.learning_rate_schedule} - Base learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.so_epochs}, Batch size: {self.batch_size}")
        print(f"   Random state: {self.random_state}")
        
        if self.shuffle_each_epoch:
            print(f"   Enhanced shuffling: New random seed each epoch")
        if self.randomize_each_epoch:
            print(f"   Full randomization: Sample with replacement each epoch")
        
        if self.momentum > 0 or self.use_momentum:
            print(f"   Using momentum: {self.momentum}")
        
        # Initialize complexity tracker
        from utils.computational_complexity import ComputationalComplexityTracker
        self.complexity_tracker = ComputationalComplexityTracker(
            problem_size=(X.shape[0], X.shape[1])
        )
        self.complexity_tracker.start_tracking()
        
        np.random.seed(self.random_state)
        
        # Thêm cột bias vào X
        X_with_bias = add_bias_column(X)
        print(f"   Original features: {X.shape[1]}, With bias: {X_with_bias.shape[1]}")
        
        n_samples, n_features_with_bias = X_with_bias.shape
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        
        # Track initial memory allocation
        self.complexity_tracker.record_memory_allocation(len(self.weights))
        
        # Initialize velocity for momentum
        if self.momentum > 0 or self.use_momentum:
            self.velocity = np.zeros(n_features_with_bias)
            self.complexity_tracker.record_memory_allocation(len(self.velocity))
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.learning_rates_history = []
        
        # Add convergence tracking
        self.converged = False
        self.final_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(self.so_epochs):
            # Get learning rate for this epoch
            current_lr = self._get_learning_rate(epoch)
            self.learning_rates_history.append(current_lr)
            
            # Enhanced shuffling/randomization logic
            if self.randomize_each_epoch:
                # Full randomization with replacement - different samples each epoch
                epoch_seed = self.random_state + epoch * 1000  # Different seed each epoch
                np.random.seed(epoch_seed)
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_shuffled = X_with_bias[indices]
                y_shuffled = y[indices]
            elif self.shuffle_each_epoch:
                # Enhanced shuffling - new permutation seed each epoch
                epoch_seed = self.random_state + epoch * 100  # Different seed each epoch
                np.random.seed(epoch_seed)
                indices = np.random.permutation(n_samples)
                X_shuffled = X_with_bias[indices]
                y_shuffled = y[indices]
            else:
                # Standard shuffling (existing behavior)
                indices = np.random.permutation(n_samples)
                X_shuffled = X_with_bias[indices]
                y_shuffled = y[indices]
            
            epoch_gradients = []
            
            # Process in batches
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Track batch processing operations
                batch_size_actual = len(X_batch)
                
                # Tính gradient cho batch
                batch_gradient = np.zeros(n_features_with_bias)
                
                for j in range(len(X_batch)):
                    xi = X_batch[j]
                    yi = y_batch[j]
                    sample_gradient = self._tinh_gradient_sample(xi, yi, self.weights)
                    batch_gradient += sample_gradient
                    
                    # Track gradient computation for each sample
                    self.complexity_tracker.record_gradient_evaluation((1, n_features_with_bias))
                    self.complexity_tracker.record_vector_operation(n_features_with_bias, "basic")
                
                # Average gradient over batch
                batch_gradient /= len(X_batch)
                epoch_gradients.append(batch_gradient)
                
                # Track averaging operation
                self.complexity_tracker.record_vector_operation(n_features_with_bias, "basic")
                
                # Update weights with momentum if enabled
                if self.momentum > 0 or self.use_momentum:
                    # Momentum update: v = β * v + ∇L
                    self.velocity = self.momentum * self.velocity + batch_gradient
                    
                    if self.use_fixed_step_length:
                        # Fixed step length with momentum: normalize velocity then scale
                        velocity_norm = np.linalg.norm(self.velocity)
                        if velocity_norm > 1e-10:  # Avoid division by zero
                            unit_velocity = self.velocity / velocity_norm
                            self.weights -= self.step_length * unit_velocity
                        # else: velocity is zero, no update needed
                    else:
                        # Standard momentum: w = w - α * v
                        self.weights -= current_lr * self.velocity
                    
                    # Track momentum operations
                    self.complexity_tracker.record_vector_operation(n_features_with_bias, "basic")  # momentum update
                    self.complexity_tracker.record_vector_operation(n_features_with_bias, "basic")  # weight update
                    if self.use_fixed_step_length:
                        self.complexity_tracker.record_vector_operation(n_features_with_bias, "norm")  # normalization
                else:
                    if self.use_fixed_step_length:
                        # Fixed step length: normalize gradient then scale
                        gradient_norm = np.linalg.norm(batch_gradient)
                        if gradient_norm > 1e-10:  # Avoid division by zero
                            unit_gradient = batch_gradient / gradient_norm
                            self.weights -= self.step_length * unit_gradient
                        # else: gradient is zero, no update needed
                        
                        # Track fixed step length operations
                        self.complexity_tracker.record_vector_operation(n_features_with_bias, "norm")  # normalization
                        self.complexity_tracker.record_vector_operation(n_features_with_bias, "basic")  # weight update
                    else:
                        # Standard SGD update: w = w - α * ∇L
                        self.weights -= current_lr * batch_gradient
                        
                        # Track weight update
                        self.complexity_tracker.record_vector_operation(n_features_with_bias, "basic")
                
                # Track weight copy for history
                self.complexity_tracker.record_memory_allocation(len(self.weights))
            
            # Chỉ tính cost và lưu history khi cần thiết  
            should_log = (
                (epoch + 1) % self.convergence_check_freq == 0 or
                epoch == self.so_epochs - 1 or
                (epoch + 1) % 20 == 0  # Progress logging
            )
            
            if should_log:
                # Chỉ tính cost khi cần (expensive operation)
                epoch_cost = self._tinh_chi_phi(X_with_bias, y, self.weights)
                epoch_gradient_avg = np.mean(epoch_gradients, axis=0)
                gradient_norm = np.linalg.norm(epoch_gradient_avg)
                
                # Track cost and norm computations
                self.complexity_tracker.record_function_evaluation(X_with_bias.shape)
                self.complexity_tracker.record_vector_operation(len(epoch_gradient_avg), "norm")
                
                # Lưu vào history
                self.loss_history.append(epoch_cost)
                self.gradient_norms.append(gradient_norm)
                self.weights_history.append(self.weights.copy())
            
            # Check convergence với tần suất định sẵn hoặc ở epoch cuối
            if (epoch + 1) % self.convergence_check_freq == 0 or epoch == self.so_epochs - 1:
                # Đảm bảo có gradient_norm và epoch_cost cho convergence check
                if not should_log:
                    epoch_cost = self._tinh_chi_phi(X_with_bias, y, self.weights)
                    epoch_gradient_avg = np.mean(epoch_gradients, axis=0)
                    gradient_norm = np.linalg.norm(epoch_gradient_avg)
                    
                    # Track additional computations
                    self.complexity_tracker.record_function_evaluation(X_with_bias.shape)
                    self.complexity_tracker.record_vector_operation(len(epoch_gradient_avg), "norm")
                    
                # Calculate cost change safely
                if len(self.loss_history) <= 1:
                    cost_change = 0.0  # No previous cost to compare
                else:
                    cost_change = self.loss_history[-2] - self.loss_history[-1]  # Previous - current
                
                # Use SGD-specific convergence check
                should_stop, converged, reason = self._check_sgd_convergence(
                    gradient_norm=gradient_norm,
                    cost_change=cost_change,
                    iteration=epoch,
                    epoch_cost=epoch_cost,
                    loss_history=self.loss_history
                )
                
                if should_stop:
                    if converged:
                        print(f"SGD converged: {reason}")
                        self.complexity_tracker.mark_convergence(epoch + 1)
                    else:
                        print(f"SGD stopped (not converged): {reason}")
                    self.converged = converged
                    self.final_epoch = epoch + 1
                    break
            
            # Progress update - chỉ print khi đã có data
            if (epoch + 1) % 20 == 0 and should_log:
                print(f"   Epoch {epoch + 1}: Cost = {epoch_cost:.6f}, Gradient = {gradient_norm:.6f}, LR = {current_lr}")
            
            # End epoch tracking
            self.complexity_tracker.end_iteration()
        
        self.training_time = time.time() - start_time
        self.final_cost = self.loss_history[-1]
        
        if not self.converged:
            print(f"Reached maximum {self.so_epochs} epochs")
            self.final_epoch = self.so_epochs
            
        print(f"Thời gian training: {self.training_time:.2f}s")
        print(f"Loss cuối: {self.final_cost:.6f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")  # Bias là phần tử cuối của weights
        print(f"Số weights (bao gồm bias): {len(self.weights)}")
        
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
        best_epoch = best_results['best_epoch']
        
        print(f"🏆 Best results (gradient norm thấp nhất):")
        print(f"   Best epoch: {best_epoch}")
        print(f"   Best loss: {best_loss:.6f}")
        print(f"   Best gradient norm: {best_gradient_norm:.6f}")
        
        return {
            'weights': best_weights,  # Trả về best weights thay vì final
            'bias': best_weights[-1],  # Bias riêng để tương thích
            'velocity': getattr(self, 'velocity', None),  # SGD-specific: Include velocity if using momentum
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'step_sizes_history': self.learning_rates_history,  # Renamed for consistency with GD
            'learning_rates_history': self.learning_rates_history,  # Keep SGD-specific name too
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_epoch,  # Renamed for consistency with GD
            'final_epoch': self.final_epoch,  # Keep SGD-specific name too
            'best_iteration': best_epoch,  # Renamed for consistency with GD
            'best_epoch': best_epoch,  # Keep SGD-specific name too
            'best_loss': best_loss,
            'best_gradient_norm': best_gradient_norm,
            'final_loss': self.loss_history[-1],  # Để so sánh
            'final_gradient_norm': self.gradient_norms[-1],  # Để so sánh
            'complexity_metrics': self.complexity_tracker.get_complexity_analysis(self.final_epoch, self.converged) if hasattr(self, 'complexity_tracker') and self.complexity_tracker else None
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
        
        if X.shape[1] != len(self.weights) - 1:  # -1 for bias
            raise ValueError(f"Số features không khớp: X có {X.shape[1]} features, model được train với {len(self.weights) - 1} features")
        
        # Thêm cột bias vào X cho prediction
        X_with_bias = add_bias_column(X)
        return du_doan(X_with_bias, self.weights, None)

    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        if X_test.shape[1] != len(self.weights) - 1:  # -1 for bias
            raise ValueError(f"Số features không khớp: X_test có {X_test.shape[1]} features, model được train với {len(self.weights) - 1} features")
        
        if len(X_test) != len(y_test):
            raise ValueError(f"X_test và y_test phải có cùng số samples: {len(X_test)} vs {len(y_test)}")
        
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
        
        # Get complexity analysis
        complexity_analysis = self.complexity_tracker.get_complexity_analysis(
            self.final_epoch, self.converged
        ) if hasattr(self, 'complexity_tracker') and self.complexity_tracker else None
        
        # Lấy kết quả tốt nhất
        best_results = self._get_best_results()
        best_weights = best_results['best_weights']
        best_loss = best_results['best_loss']
        best_gradient_norm = best_results['best_gradient_norm'] 
        best_epoch = best_results['best_epoch']
        
        # Save comprehensive results.json
        results_data = {
            "algorithm": f"Stochastic Gradient Descent - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "learning_rate": self.learning_rate,
                "max_epochs": self.so_epochs,
                "batch_size": self.batch_size,
                "tolerance": self.diem_dung,
                "learning_rate_schedule": self.learning_rate_schedule,
                "momentum": self.momentum,
                "random_state": self.random_state,
                "shuffle_each_epoch": self.shuffle_each_epoch,
                "randomize_each_epoch": self.randomize_each_epoch
            },
            "training_results": {
                "training_time": self.training_time,
                "converged": self.converged,
                "final_epoch": int(self.final_epoch),
                "total_epochs": int(self.so_epochs),
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1]),
                # Thêm thông tin best results
                "best_epoch": best_epoch,
                "best_loss": float(best_loss),
                "best_gradient_norm": float(best_gradient_norm),
                "improvement_from_final": {
                    "loss_improvement": float(self.loss_history[-1] - best_loss),
                    "gradient_improvement": float(self.gradient_norms[-1] - best_gradient_norm),
                    "epochs_earlier": int(self.final_epoch - best_epoch)
                }
            },
            "weights_analysis": {
                "n_features": int(len(best_weights) - 1),  # Không tính bias
                "n_weights_total": int(len(best_weights)),  # Tính cả bias
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
                "epochs_to_converge": int(self.final_epoch),
                "best_epoch_found": best_epoch,
                "final_cost_change": float(self.loss_history[-1] - self.loss_history[-2]) if len(self.loss_history) > 1 else 0.0,
                "convergence_rate": "sublinear",  # SGD có convergence rate sublinear
                "loss_reduction_ratio": float(self.loss_history[0] / best_loss) if len(self.loss_history) > 0 else 1.0
            },
            "algorithm_specific": {
                "sgd_type": "mini_batch" if self.batch_size > 1 else "pure_sgd",
                "batch_size": self.batch_size,
                "learning_rate_schedule": self.learning_rate_schedule,
                "momentum_used": self.momentum > 0,
                "momentum_value": self.momentum if self.momentum > 0 else None,
                "shuffling_strategy": {
                    "shuffle_each_epoch": self.shuffle_each_epoch,
                    "randomize_each_epoch": self.randomize_each_epoch
                },
                "returns_best_result": True,  # Đánh dấu rằng trả về best result
                "stochastic_variance": True  # Đặc trưng của SGD
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
            'epoch': range(0, len(self.loss_history)*self.convergence_check_freq, self.convergence_check_freq),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms
        })
        
        # Save learning rates history for SGD
        if hasattr(self, 'learning_rates_history') and self.learning_rates_history:
            lr_df = pd.DataFrame({
                'epoch': range(len(self.learning_rates_history)),
                'learning_rate': self.learning_rates_history
            })
            lr_df.to_csv(results_dir / "learning_rates_history.csv", index=False)
            
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        # Save complexity metrics separately for detailed analysis
        if complexity_analysis:
            with open(results_dir / "complexity_analysis.json", 'w') as f:
                json.dump(complexity_analysis, f, indent=2)
        
        print(f"\n✅ Kết quả đã được lưu vào: {results_dir.absolute()}")
        print(f"🏆 Sử dụng best results từ epoch {best_epoch} (gradient norm: {best_gradient_norm:.6f})")
        if complexity_analysis:
            print(f"Complexity metrics saved to: complexity_analysis.json")
        
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
        
        # 1. Convergence curves - now with actual epoch numbers
        print("   - Vẽ đường hội tụ")
        # Create epoch values based on convergence_check_freq
        epochs = list(range(0, len(self.loss_history) * self.convergence_check_freq, self.convergence_check_freq))
        
        ve_duong_hoi_tu(self.loss_history, self.gradient_norms, 
                        iterations=epochs,
                        title=f"Stochastic Gradient Descent {self.ham_loss.upper()} - Hội tụ",
                        save_path=str(results_dir / "convergence_analysis.png"))
        
        # 2. Predictions vs Actual
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"Stochastic Gradient Descent {self.ham_loss.upper()} - Dự đoán vs Thực tế",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mức)
        print("   - Vẽ đường đồng mức optimization")
        
        # Chuẩn bị X_test với bias cho visualization
        X_test_with_bias = add_bias_column(X_test)
        
        # Create loss function compatible with visualization
        loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        
        ve_duong_dong_muc_optimization(
            loss_function=loss_func,
            weights_history=self.weights_history,  # Pass full history
            X=X_test_with_bias, y=y_test,
            title=f"Stochastic Gradient Descent {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png"),
            original_iterations=self.final_epoch,  # Use actual number of epochs
            convergence_check_freq=self.convergence_check_freq,  # Pass convergence frequency
            max_trajectory_points=50  # Limit points for SGD (can be noisy)
        )
        
