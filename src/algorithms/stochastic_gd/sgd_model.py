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


from utils.model_mixins import ComplexityTrackingMixin, OptimizationResultsMixin

class SGDModel(ComplexityTrackingMixin, OptimizationResultsMixin):
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
                 decay_rate=0.95, decay_steps=100):
        
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
        
        print(f"🔧 SGD Model initialized:")
        print(f"   Loss function: {self.ham_loss.upper()}")
        print(f"   Learning rate: {self.learning_rate} ({self.learning_rate_schedule})")
        print(f"   Epochs: {self.so_epochs}, Batch size: {self.batch_size}")
        print(f"   Momentum: {self.momentum}")
        print(f"   Random state: {self.random_state}")
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
        
        # Tìm index có gradient norm thấp nhất
        best_idx = np.argmin(self.gradient_norms)
        
        return {
            'best_weights': self.weights_history[best_idx],
            'best_loss': self.loss_history[best_idx],
            'best_gradient_norm': self.gradient_norms[best_idx],
            'best_epoch': best_idx * self.convergence_check_freq
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
        Tính gradient cho 1 sample
        
        Args:
            xi: Feature vector for sample i (đã có bias)
            yi: Target value for sample i  
            weights: Weight vector hiện tại
        """
        # Prediction
        pred = np.dot(xi, weights)
        residual = pred - yi
        
        if self.ham_loss == 'ols':
            # MSE gradient: 2 * (pred - y) * x
            gradient = 2 * residual * xi
        elif self.ham_loss == 'ridge':
            # Ridge gradient: MSE + L2 regularization
            gradient = 2 * residual * xi + 2 * self.regularization * weights
            # Don't regularize bias (last element)
            gradient[-1] -= 2 * self.regularization * weights[-1]
        elif self.ham_loss == 'lasso':
            # Lasso gradient: MSE + L1 regularization (simplified)
            gradient = 2 * residual * xi + self.regularization * np.sign(weights)
            # Don't regularize bias
            gradient[-1] -= self.regularization * np.sign(weights[-1])
        else:
            raise ValueError(f"Unsupported loss function: {self.ham_loss}")
            
        return gradient
    
    def _tinh_chi_phi(self, X, y, weights):
        """
        Tính chi phí (cost) cho toàn bộ dataset
        
        Args:
            X: Feature matrix (đã có bias)
            y: Target vector
            weights: Weight vector
        """
        predictions = X @ weights
        residuals = predictions - y
        
        if self.ham_loss == 'ols':
            cost = np.mean(residuals ** 2)
        elif self.ham_loss == 'ridge':
            mse_cost = np.mean(residuals ** 2)
            l2_penalty = self.regularization * np.sum(weights[:-1] ** 2)  # Không regularize bias
            cost = mse_cost + l2_penalty
        elif self.ham_loss == 'lasso':
            mse_cost = np.mean(residuals ** 2)
            l1_penalty = self.regularization * np.sum(np.abs(weights[:-1]))  # Không regularize bias
            cost = mse_cost + l1_penalty
        else:
            raise ValueError(f"Unsupported loss function: {self.ham_loss}")
            
        return cost
    
    def _check_sgd_convergence(self, gradient_norm, cost_change, iteration, epoch_cost, loss_history):
        """
        Kiểm tra điều kiện dừng cho SGD
        
        Returns:
            (should_stop, converged, reason)
        """
        # Gradient norm convergence
        if gradient_norm < self.diem_dung:
            return True, True, f"Gradient norm {gradient_norm:.2e} < tolerance {self.diem_dung:.2e}"
        
        # Cost change convergence (less strict for SGD due to noise)
        if abs(cost_change) < self.diem_dung * 10:  # 10x more lenient for SGD
            return True, True, f"Cost change {abs(cost_change):.2e} < tolerance {self.diem_dung * 10:.2e}"
        
        # Max iterations reached
        if iteration >= self.so_epochs - 1:
            return True, False, f"Reached maximum epochs {self.so_epochs}"
        
        # Check for divergence (cost increasing significantly)
        if len(loss_history) >= 5:
            recent_costs = loss_history[-5:]
            if recent_costs[-1] > recent_costs[0] * 2:  # Cost doubled in last 5 checks
                return True, False, "Cost is increasing significantly (possible divergence)"
        
        return False, False, "Continuing training"
    
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"🚀 Training Stochastic Gradient Descent - {self.ham_loss.upper()}")
        print(f"   Learning rate schedule: {self.learning_rate_schedule} - Base learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.so_epochs}, Batch size: {self.batch_size}")
        print(f"   Random state: {self.random_state}")
        
        if self.shuffle_each_epoch:
            print(f"   🔀 Enhanced shuffling: New random seed each epoch")
        if self.randomize_each_epoch:
            print(f"   🎲 Full randomization: Sample with replacement each epoch")
        
        if self.momentum > 0 or self.use_momentum:
            print(f"   Using momentum: {self.momentum}")
        
        # Initialize complexity tracking
        self.init_complexity_tracker(X, y)
        
        np.random.seed(self.random_state)
        
        # Thêm cột bias vào X
        X_with_bias = add_bias_column(X)
        print(f"   Original features: {X.shape[1]}, With bias: {X_with_bias.shape[1]}")
        
        n_samples, n_features_with_bias = X_with_bias.shape
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        
        # Track initial memory allocation
        self.track_memory_allocation(len(self.weights))
        
        # Initialize velocity for momentum
        if self.momentum > 0 or self.use_momentum:
            self.velocity = np.zeros(n_features_with_bias)
            self.track_memory_allocation(len(self.velocity))
        
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
                    self.track_gradient_evaluation((1, n_features_with_bias))
                    self.track_vector_operation(n_features_with_bias, "basic")
                
                # Average gradient over batch
                batch_gradient /= len(X_batch)
                epoch_gradients.append(batch_gradient)
                
                # Track averaging operation
                self.track_vector_operation(n_features_with_bias, "basic")
                
                # Update weights with momentum if enabled
                if self.momentum > 0 or self.use_momentum:
                    # Momentum update: v = β * v + ∇L
                    self.velocity = self.momentum * self.velocity + batch_gradient
                    # Weight update: w = w - α * v
                    self.weights -= current_lr * self.velocity
                    
                    # Track momentum operations
                    self.track_vector_operation(n_features_with_bias, "basic")  # momentum update
                    self.track_vector_operation(n_features_with_bias, "basic")  # weight update
                else:
                    # Standard SGD update: w = w - α * ∇L
                    self.weights -= current_lr * batch_gradient
                    
                    # Track weight update
                    self.track_vector_operation(n_features_with_bias, "basic")
                
                # Track weight copy for history
                self.track_memory_allocation(len(self.weights))
            
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
                self.track_function_evaluation(X_with_bias.shape)
                self.track_vector_operation(len(epoch_gradient_avg), "norm")
                
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
                    self.track_function_evaluation(X_with_bias.shape)
                    self.track_vector_operation(len(epoch_gradient_avg), "norm")
                    
                cost_change = 0.0 if len(self.loss_history) == 0 else (self.loss_history[-1] - epoch_cost) if len(self.loss_history) == 1 else (self.loss_history[-2] - self.loss_history[-1])
                
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
                        print(f"✅ SGD converged: {reason}")
                        self.mark_convergence_tracking(epoch + 1)
                    else:
                        print(f"⚠️ SGD stopped (not converged): {reason}")
                    self.converged = converged
                    self.final_epoch = epoch + 1
                    break
            
            # Progress update - chỉ print khi đã có data
            if (epoch + 1) % 20 == 0 and should_log:
                print(f"   Epoch {epoch + 1}: Cost = {epoch_cost:.6f}, Gradient = {gradient_norm:.6f}, LR = {current_lr}")
            
            # End epoch tracking
            self.end_iteration_tracking()
        
        self.training_time = time.time() - start_time
        self.final_cost = self.loss_history[-1]
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_epochs} epochs")
            self.final_epoch = self.so_epochs
            
        print(f"Thời gian training: {self.training_time:.2f}s")
        print(f"Loss cuối: {self.final_cost:.6f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")  # Bias là phần tử cuối của weights
        print(f"Số weights (bao gồm bias): {len(self.weights)}")
        
        # Print complexity summary
        self.print_complexity_summary()
        
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
            'velocity': getattr(self, 'velocity', None),  # Include velocity if using momentum
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'learning_rates_history': self.learning_rates_history,
            'training_time': self.training_time,
            'final_cost': self.final_cost,
            'converged': self.converged,
            'final_epoch': self.final_epoch,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_gradient_norm': best_gradient_norm,
            'final_loss': self.loss_history[-1],  # Để so sánh
            'final_gradient_norm': self.gradient_norms[-1],  # Để so sánh
            'complexity_metrics': self.get_complexity_analysis(self.final_epoch, self.converged)
        }
        
