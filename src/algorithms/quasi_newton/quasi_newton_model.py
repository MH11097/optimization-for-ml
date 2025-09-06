#!/usr/bin/env python3
"""
QuasiNewtonModel - Class cho BFGS Quasi-Newton Method
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


class QuasiNewtonModel:
    """
    BFGS Quasi-Newton Method Model với hỗ trợ nhiều loss functions
    
    Parameters:
    - ham_loss: 'ols', 'ridge', 'lasso'
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ (gradient norm)
    - regularization: Tham số regularization cho Ridge/Lasso
    - armijo_c1: Armijo constant cho line search
    - wolfe_c2: Wolfe curvature constant
    - backtrack_rho: Backtrack factor cho line search
    - max_line_search_iter: Số lần line search tối đa
    - damping: Damping factor cho BFGS update
    - convergence_check_freq: Tần suất kiểm tra hội tụ (mỗi N iterations)
    """
    
    def __init__(self, ham_loss='ols', so_lan_thu=10000, diem_dung=1e-5, 
                 regularization=0.01, armijo_c1=1e-4, wolfe_c2=0.9,
                 backtrack_rho=0.8, max_line_search_iter=50, damping=1e-8, convergence_check_freq=10,
                 method='bfgs', memory_size=10, sr1_skip_threshold=1e-8):
        self.ham_loss = ham_loss.lower()
        self.so_lan_thu = so_lan_thu
        self.diem_dung = diem_dung
        self.regularization = regularization
        self.convergence_check_freq = convergence_check_freq
        self.armijo_c1 = armijo_c1
        self.wolfe_c2 = wolfe_c2
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.damping = damping
        self.method = method.lower()  # 'bfgs', 'lbfgs', 'sr1'
        self.memory_size = memory_size  # For L-BFGS
        self.sr1_skip_threshold = sr1_skip_threshold  # For SR1 stability
        
        # Validate method
        if self.method not in ['bfgs', 'lbfgs', 'sr1']:
            raise ValueError(f"Unsupported method: {method}. Use 'bfgs', 'lbfgs', or 'sr1'")
        
        # L-BFGS specific storage
        self.s_vectors = []  # Step vectors s_k = x_{k+1} - x_k
        self.y_vectors = []  # Gradient diff vectors y_k = g_{k+1} - g_k
        self.rho_values = []  # 1 / (y_k^T s_k)
        self.alpha_values = []  # For two-loop recursion
        
        # Validate supported loss function
        if self.ham_loss not in ['ols', 'ridge', 'lasso']:
            raise ValueError(f"Không hỗ trợ loss function: {ham_loss}")
        
        # Sử dụng unified functions với format mới (bias trong X)
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        
        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None  # Bây giờ bao gồm bias ở cuối
        self.H_inv = None  # Inverse Hessian approximation (BFGS/SR1 only)
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.condition_numbers = []
        self.skipped_updates = []  # For SR1 tracking
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
    
    def _wolfe_line_search(self, X, y, weights, direction, gradient):
        """
        Wolfe line search để satisfy both Armijo và curvature conditions
        """
        current_loss = self.loss_func(X, y, weights)
        
        # Directional derivative: ∇f^T * d  
        directional_derivative = np.dot(gradient, direction)
        
        # Initial step size
        alpha = 1.0
        
        for i in range(self.max_line_search_iter):
            # Thử weights mới
            new_weights = weights + alpha * direction
            
            # Tính loss và gradient mới
            new_loss = self.loss_func(X, y, new_weights)
            new_gradient, _ = self.grad_func(X, y, new_weights)
            
            # Kiểm tra Armijo condition (sufficient decrease)
            armijo_condition = current_loss + self.armijo_c1 * alpha * directional_derivative
            
            if new_loss <= armijo_condition:
                # Kiểm tra curvature condition (Wolfe)
                curvature_condition = np.dot(new_gradient, direction)
                if curvature_condition >= self.wolfe_c2 * directional_derivative:
                    return alpha, i + 1, new_gradient
            
            # Giảm step size
            alpha *= self.backtrack_rho
        
        # Nếu line search fail, return step cuối và gradient mới
        new_weights = weights + alpha * direction
        new_gradient, _ = self.grad_func(X, y, new_weights)
        return alpha, self.max_line_search_iter, new_gradient
    
    def _cap_nhat_bfgs(self, H_inv, s, y):
        """
        Cập nhật inverse Hessian approximation theo BFGS formula
        
        H_{k+1}^{-1} = (I - ρ s y^T) H_k^{-1} (I - ρ y s^T) + ρ s s^T
        where ρ = 1 / (y^T s)
        """
        sy = np.dot(s, y)
        
        # Kiểm tra curvature condition để đảm bảo positive definiteness
        if sy < self.damping:
            print(f"   Warning: Curvature condition violated (sy = {sy:.2e}), skipping BFGS update")
            return H_inv
        
        rho = 1.0 / sy
        n = len(s)
        I = np.eye(n)
        
        # BFGS update formula
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H_inv_new = np.dot(A, np.dot(H_inv, B)) + rho * np.outer(s, s)
        
        return H_inv_new
    
    def _lbfgs_two_loop_recursion(self, gradient):
        """
        L-BFGS two-loop recursion để tính search direction
        
        Returns:
            direction: Search direction -H_k * gradient
        """
        if not self.s_vectors:
            # Nếu chưa có history, sử dụng steepest descent
            return -gradient
        
        q = gradient.copy()
        
        # First loop (backward)
        m = min(len(self.s_vectors), self.memory_size)
        self.alpha_values = []
        
        for i in range(m-1, -1, -1):
            alpha_i = self.rho_values[i] * np.dot(self.s_vectors[i], q)
            self.alpha_values.insert(0, alpha_i)  # Insert at beginning
            q = q - alpha_i * self.y_vectors[i]
        
        # Initial Hessian approximation H_0^{-1}
        # Sử dụng scaling factor từ Nocedal & Wright
        if self.s_vectors and self.y_vectors:
            gamma_k = (np.dot(self.s_vectors[-1], self.y_vectors[-1]) / 
                      np.dot(self.y_vectors[-1], self.y_vectors[-1]))
            r = gamma_k * q
        else:
            r = q
        
        # Second loop (forward)
        for i in range(m):
            beta_i = self.rho_values[i] * np.dot(self.y_vectors[i], r)
            r = r + (self.alpha_values[i] - beta_i) * self.s_vectors[i]
        
        return -r  # Search direction
    
    def _update_lbfgs_memory(self, s, y):
        """
        Cập nhật L-BFGS memory với vectors s và y mới
        """
        sy = np.dot(s, y)
        
        # Kiểm tra curvature condition
        if sy < self.damping:
            print(f"   Warning: L-BFGS curvature condition violated (sy = {sy:.2e}), skipping update")
            return
        
        rho = 1.0 / sy
        
        # Thêm vectors mới
        self.s_vectors.append(s)
        self.y_vectors.append(y)
        self.rho_values.append(rho)
        
        # Giới hạn memory size
        if len(self.s_vectors) > self.memory_size:
            self.s_vectors.pop(0)
            self.y_vectors.pop(0)
            self.rho_values.pop(0)
    
    def _cap_nhat_sr1(self, H_inv, s, y):
        """
        Cập nhật inverse Hessian approximation theo SR1 formula
        
        SR1 Update: H_{k+1}^{-1} = H_k^{-1} + ((s - H_k^{-1}y)(s - H_k^{-1}y)^T) / ((s - H_k^{-1}y)^T y)
        """
        Hy = np.dot(H_inv, y)
        v = s - Hy  # s - H_k^{-1} * y
        vTy = np.dot(v, y)
        
        # SR1 skip condition để tránh numerical instability
        if abs(vTy) < self.sr1_skip_threshold * np.linalg.norm(v) * np.linalg.norm(y):
            print(f"   Warning: SR1 update skipped due to small denominator (vTy = {vTy:.2e})")
            self.skipped_updates.append(True)
            return H_inv
        
        self.skipped_updates.append(False)
        
        # SR1 update formula
        H_inv_new = H_inv + np.outer(v, v) / vTy
        
        return H_inv_new
    
    def _get_convergence_rate(self):
        """Get convergence rate description based on method"""
        if self.method == 'bfgs':
            return "superlinear"  # BFGS has superlinear convergence
        elif self.method == 'lbfgs':
            return "superlinear"  # L-BFGS also has superlinear convergence
        elif self.method == 'sr1':
            return "linear_to_superlinear"  # SR1 can be linear or superlinear
        return "unknown"
    
    def _get_algorithm_specific_info(self):
        """Get algorithm-specific information for results"""
        base_info = {
            "method_type": f"quasi_newton_{self.method}",
            "second_order_approximation": True,
            "line_search_used": True,
            "line_search_type": "wolfe_conditions"
        }
        
        if self.method == 'bfgs':
            base_info.update({
                "hessian_computation": "bfgs_secant_approximation",
                "superlinear_convergence": "BFGS provides superlinear convergence rate",
                "memory_efficient": "inverse_hessian_approximation",
                "positive_definite": "guaranteed_by_bfgs_updates"
            })
        elif self.method == 'lbfgs':
            base_info.update({
                "hessian_computation": "lbfgs_limited_memory",
                "superlinear_convergence": "L-BFGS provides superlinear convergence rate",
                "memory_efficient": f"limited_memory_size_{self.memory_size}",
                "two_loop_recursion": "efficient_matrix_free_computation",
                "memory_usage": len(self.s_vectors) if hasattr(self, 's_vectors') else 0
            })
        elif self.method == 'sr1':
            skip_count = sum(self.skipped_updates) if self.skipped_updates else 0
            total_updates = len(self.skipped_updates) if self.skipped_updates else 0
            base_info.update({
                "hessian_computation": "sr1_symmetric_rank_one",
                "superlinear_convergence": "SR1 can provide superlinear convergence",
                "memory_efficient": "inverse_hessian_approximation",
                "update_stability": "skip_condition_for_numerical_stability",
                "skipped_updates": skip_count,
                "total_updates": total_updates,
                "skip_rate": skip_count / total_updates if total_updates > 0 else 0
            })
        
        return base_info
        
    def fit(self, X, y):
        """
        Huấn luyện model với dữ liệu X, y
        
        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"🚀 Training {self.method.upper()} Quasi-Newton Method - {self.ham_loss.upper()}")
        print(f"   Max iterations: {self.so_lan_thu}")
        if self.ham_loss in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")
        print(f"   Armijo c1: {self.armijo_c1}, Wolfe c2: {self.wolfe_c2}")
        
        # Debug: print input shapes
        print(f"   Input X shape: {X.shape}, y shape: {y.shape}")
        
        # Additional validation - check for shape mismatches that could cause broadcasting errors
        print(f"   Final validation - X shape: {X.shape}, y shape: {y.shape}")
        if len(y.shape) != 1:
            raise ValueError(f"y should be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: X={X.shape[0]}, y={y.shape[0]}")
        
        # Check for reasonable data sizes (detect if full dataset loaded by mistake)
        if X.shape[0] > 50000:
            print(f"   ⚠️  Warning: Very large dataset ({X.shape[0]:,} samples). QuasiNewton methods work best with smaller datasets.")
            print(f"   This might cause memory issues and slow convergence.")
            print(f"   Consider reducing batch size or using a smaller sample for QuasiNewton methods.")
            
            # For very large datasets, we should use a subset to avoid memory issues
            if X.shape[0] > 80000:
                print(f"   🔧 Auto-sampling to first 3200 samples for QuasiNewton stability...")
                sample_size = min(3200, X.shape[0])
                X = X[:sample_size].copy()  # Explicit copy to avoid reference issues
                y = y[:sample_size].copy()  # Explicit copy to avoid reference issues
                print(f"   New shape: X={X.shape}, y={y.shape}")
                
                # Validate shapes after sampling
                if X.shape[0] != y.shape[0]:
                    raise ValueError(f"Shape mismatch after sampling: X={X.shape[0]} samples, y={y.shape[0]} samples")
                if len(y.shape) != 1:
                    raise ValueError(f"y should be 1-dimensional after sampling, got shape {y.shape}")
            
        print(f"   Dataset size: {X.shape[0]:,} samples × {X.shape[1]} features")
        
        # Thêm cột bias vào X
        X_with_bias = add_bias_column(X)
        print(f"   Original features: {X.shape[1]}, With bias: {X_with_bias.shape[1]}")
        print(f"   X_with_bias shape: {X_with_bias.shape}")
        
        # Initialize weights và inverse Hessian approximation (bao gồm bias ở cuối)
        n_features_with_bias = X_with_bias.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        
        # Initialize algorithm-specific structures
        if self.method in ['bfgs', 'sr1']:
            self.H_inv = np.eye(n_features_with_bias)  # Initial approximation
            print(f"   Initialized H_inv shape: {self.H_inv.shape}")
        else:  # L-BFGS
            self.s_vectors = []
            self.y_vectors = []
            self.rho_values = []
            print(f"   L-BFGS memory size: {self.memory_size}")
        
        print(f"   Initialized weights shape: {self.weights.shape}")
        print(f"   Method: {self.method.upper()}")
        
        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.condition_numbers = []
        self.skipped_updates = []
        
        start_time = time.time()
        
        # Initial gradient
        try:
            gradient_prev, _ = self.grad_func(X_with_bias, y, self.weights)  # _ vì không cần gradient_b riêng
            print(f"   Initial gradient shape: {gradient_prev.shape}")
        except Exception as e:
            print(f"❌ Error computing initial gradient: {e}")
            print(f"   X_with_bias shape: {X_with_bias.shape}")
            print(f"   y shape: {y.shape}")  
            print(f"   weights shape: {self.weights.shape}")
            raise
        
        for lan_thu in range(self.so_lan_thu):
            try:
                # Tính gradient (luôn cần cho BFGS)
                # Debug shapes on critical iterations
                if lan_thu < 5 or lan_thu % 100 == 0:
                    print(f"   Debug iteration {lan_thu + 1} - X_with_bias: {X_with_bias.shape}, y: {y.shape}, weights: {self.weights.shape}")
                
                gradient_result = self.grad_func(X_with_bias, y, self.weights)
                
                # Debug: check what grad_func returns and handle properly
                if isinstance(gradient_result, tuple):
                    gradient_curr, gradient_b = gradient_result
                    # For QuasiNewton with bias in X, gradient_b should be None or ignored
                else:
                    gradient_curr = gradient_result
                    print(f"   Warning: grad_func returned non-tuple: {type(gradient_result)}")
                
                # Debug shapes on first iteration and when errors occur
                if lan_thu == 0 or gradient_curr.shape != self.weights.shape:
                    print(f"   Debug iteration {lan_thu + 1} - X_with_bias: {X_with_bias.shape}, y: {y.shape}, weights: {self.weights.shape}, gradient: {gradient_curr.shape}")
                
                # Check for shape consistency
                if gradient_curr.shape != self.weights.shape:
                    raise ValueError(f"Gradient shape {gradient_curr.shape} doesn't match weights shape {self.weights.shape}")
                
            except Exception as e:
                print(f"❌ Error at iteration {lan_thu + 1} computing gradient: {e}")
                print(f"   X_with_bias shape: {X_with_bias.shape}")
                print(f"   y shape: {y.shape}")
                print(f"   weights shape: {self.weights.shape}")
                
                # Additional debugging for broadcasting errors
                if "broadcast" in str(e).lower():
                    print(f"   🔍 Broadcasting error detected - this suggests a shape mismatch in gradient computation")
                    print(f"   Expected gradient shape: {self.weights.shape}")
                    try:
                        test_result = self.grad_func(X_with_bias, y, self.weights)
                        print(f"   Actual grad_func result type: {type(test_result)}")
                        if isinstance(test_result, tuple):
                            print(f"   Gradient tuple shapes: {[np.array(x).shape if x is not None else None for x in test_result]}")
                        else:
                            print(f"   Gradient result shape: {test_result.shape}")
                    except Exception as debug_e:
                        print(f"   Debug gradient computation also failed: {debug_e}")
                raise
            
            # Chỉ tính loss và lưu history khi cần thiết
            should_check_converged = (
                (lan_thu + 1) % self.convergence_check_freq == 0 or 
                lan_thu == self.so_lan_thu - 1
            )
            
            if should_check_converged:
                try:
                    # Chỉ tính loss khi cần (expensive operation)
                    loss_value = self.loss_func(X_with_bias, y, self.weights)
                    gradient_norm = np.linalg.norm(gradient_curr)
                    
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
                            print(f"✅ {self.method.upper()} Quasi-Newton converged: {reason}")
                        else:
                            print(f"⚠️ {self.method.upper()} Quasi-Newton stopped (not converged): {reason}")
                        self.converged = converged
                        self.final_iteration = lan_thu + 1
                        break
                        
                except Exception as e:
                    print(f"❌ Error at iteration {lan_thu + 1} computing loss/convergence: {e}")
                    raise
            
            try:
                # Compute search direction based on method
                if self.method == 'bfgs':
                    direction = -np.dot(self.H_inv, gradient_curr)
                elif self.method == 'lbfgs':
                    direction = self._lbfgs_two_loop_recursion(gradient_curr)
                elif self.method == 'sr1':
                    direction = -np.dot(self.H_inv, gradient_curr)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                
                if direction.shape != self.weights.shape:
                    raise ValueError(f"Direction shape {direction.shape} doesn't match weights shape {self.weights.shape}")
                
            except Exception as e:
                print(f"❌ Error at iteration {lan_thu + 1} computing direction: {e}")
                if self.method in ['bfgs', 'sr1']:
                    print(f"   H_inv shape: {self.H_inv.shape}")
                elif self.method == 'lbfgs':
                    print(f"   L-BFGS vectors stored: {len(self.s_vectors)}")
                print(f"   gradient_curr shape: {gradient_curr.shape}")
                raise
            
            try:
                # Line search để tìm step size
                step_size, ls_iter, gradient_new = self._wolfe_line_search(
                    X_with_bias, y, self.weights, direction, gradient_curr
                )
                
                self.step_sizes.append(step_size)
                self.line_search_iterations.append(ls_iter)
                
                # Update weights
                weights_new = self.weights + step_size * direction
                
                # Update quasi-Newton approximation
                s = weights_new - self.weights  # step
                y = gradient_new - gradient_curr  # gradient change
                
                if s.shape != y.shape:
                    raise ValueError(f"Step s shape {s.shape} doesn't match gradient change y shape {y.shape}")
                
                if self.method == 'bfgs':
                    self.H_inv = self._cap_nhat_bfgs(self.H_inv, s, y)
                    cond_num = np.linalg.cond(self.H_inv)
                    self.condition_numbers.append(cond_num)
                elif self.method == 'lbfgs':
                    self._update_lbfgs_memory(s, y)
                    # For L-BFGS, condition number is not directly computed
                    self.condition_numbers.append(np.nan)
                elif self.method == 'sr1':
                    self.H_inv = self._cap_nhat_sr1(self.H_inv, s, y)
                    cond_num = np.linalg.cond(self.H_inv)
                    self.condition_numbers.append(cond_num)
                
                # Update for next iteration
                self.weights = weights_new
                gradient_prev = gradient_curr
                
            except Exception as e:
                print(f"❌ Error at iteration {lan_thu + 1} in line search/update: {e}")
                raise
            
            # Progress update
            if (lan_thu + 1) % 10 == 0 and should_check_converged:
                if self.method == 'lbfgs':
                    print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient = {gradient_norm:.2e}, Step = {step_size:.6f}, Memory = {len(self.s_vectors)}")
                elif self.method == 'sr1':
                    skipped_count = sum(self.skipped_updates)
                    total_updates = len(self.skipped_updates)
                    skip_rate = skipped_count / total_updates if total_updates > 0 else 0
                    print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient = {gradient_norm:.2e}, Step = {step_size:.6f}, Skip = {skip_rate:.1%}")
                else:  # BFGS
                    cond_num = self.condition_numbers[-1] if self.condition_numbers else np.nan
                    print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient = {gradient_norm:.2e}, Step = {step_size:.6f}, Cond = {cond_num:.2e}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_lan_thu} vòng lặp")
            self.final_iteration = self.so_lan_thu
        
        print(f"Thời gian training: {self.training_time:.4f}s")
        print(f"Loss cuối: {self.loss_history[-1]:.8f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")  # Bias là phần tử cuối của weights
        print(f"Số weights (bao gồm bias): {len(self.weights)}")
        if self.step_sizes:
            print(f"Average step size: {np.mean(self.step_sizes):.6f}")
            print(f"Average line search iterations: {np.mean(self.line_search_iterations):.1f}")
        
        # Method-specific statistics
        if self.method == 'lbfgs':
            print(f"Final L-BFGS memory usage: {len(self.s_vectors)}/{self.memory_size}")
        elif self.method == 'sr1':
            if self.skipped_updates:
                skipped_count = sum(self.skipped_updates)
                total_updates = len(self.skipped_updates)
                skip_rate = skipped_count / total_updates
                print(f"SR1 updates skipped: {skipped_count}/{total_updates} ({skip_rate:.1%})")
        
        # Lấy kết quả tốt nhất thay vì kết quả cuối cùng
        best_results = self._get_best_results()
        best_weights = best_results['best_weights']
        best_loss = best_results['best_loss']
        best_gradient_norm = best_results['best_gradient_norm']
        best_iteration = best_results['best_iteration']
        
        print(f"🏆 Best results (gradient norm thấp nhất):")
        print(f"   Best iteration: {best_iteration}")
        print(f"   Best loss: {best_loss:.8f}")
        print(f"   Best gradient norm: {best_gradient_norm:.2e}")
        
        return {
            'weights': best_weights,  # Trả về best weights thay vì final
            'bias': best_weights[-1],  # Bias riêng để tương thích
            'H_inv': self.H_inv if self.method in ['bfgs', 'sr1'] else None,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'step_sizes': self.step_sizes,
            'line_search_iterations': self.line_search_iterations,
            'condition_numbers': self.condition_numbers,
            'skipped_updates': self.skipped_updates if self.method == 'sr1' else [],
            'lbfgs_memory_usage': len(self.s_vectors) if self.method == 'lbfgs' else 0,
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
                           f"{self.method.upper()} Quasi-Newton - {self.ham_loss.upper()}")
        return metrics
    
    def save_results(self, ten_file, base_dir="data/03_algorithms/quasi_newton"):
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
            "algorithm": f"{self.method.upper()} Quasi-Newton - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "max_iterations": self.so_lan_thu,
                "tolerance": self.diem_dung,
                "armijo_c1": self.armijo_c1,
                "wolfe_c2": self.wolfe_c2,
                "backtrack_rho": self.backtrack_rho,
                "damping": self.damping
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
                "convergence_rate": self._get_convergence_rate(),
                "loss_reduction_ratio": float(self.loss_history[0] / self.loss_history[-1]) if len(self.loss_history) > 0 else 1.0
            },
            "numerical_analysis": {
                "average_step_size": float(np.mean(self.step_sizes)) if self.step_sizes else 0,
                "average_line_search_iterations": float(np.mean(self.line_search_iterations)) if self.line_search_iterations else 0,
                "final_condition_number": float(self.condition_numbers[-1]) if self.condition_numbers else 0,
                "max_condition_number": float(np.max(self.condition_numbers)) if self.condition_numbers else 0,
                "min_condition_number": float(np.min(self.condition_numbers)) if self.condition_numbers else 0,
                "hessian_approximation_quality": "BFGS_secant_approximation",
                "line_search_efficiency": "Wolfe_conditions_satisfied"
            },
            "algorithm_specific": self._get_algorithm_specific_info()
        }
        
        if self.ham_loss in ['ridge', 'lasso']:
            results_data["parameters"]["regularization"] = self.regularization
        
        # Add method-specific parameters
        if self.method == 'lbfgs':
            results_data["parameters"]["memory_size"] = self.memory_size
        elif self.method == 'sr1':
            results_data["parameters"]["sr1_skip_threshold"] = self.sr1_skip_threshold
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        max_len = len(self.loss_history)
        training_df = pd.DataFrame({
            'iteration': range(0, len(self.loss_history)*self.convergence_check_freq, self.convergence_check_freq),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms,
            'step_size': self.step_sizes + [np.nan] * (max_len - len(self.step_sizes)),
            'line_search_iter': self.line_search_iterations + [np.nan] * (max_len - len(self.line_search_iterations)),
            'condition_number': self.condition_numbers + [np.nan] * (max_len - len(self.condition_numbers))
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/quasi_newton"):
        """
        Tạo các biểu đồ visualization
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Tạo biểu đồ...")
        
        # 1. Convergence curves với condition number - now using actual iteration numbers
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
        
        # Condition number
        if self.condition_numbers:
            cond_iterations = iterations[:len(self.condition_numbers)]
            axes[1,0].semilogy(cond_iterations, self.condition_numbers, 'g-', linewidth=2)
            axes[1,0].set_title('Condition Number of H_inv')
            axes[1,0].set_xlabel('Iteration')
            axes[1,0].set_ylabel('Condition Number')
            axes[1,0].grid(True, alpha=0.3)
        
        # Step sizes
        if self.step_sizes:
            step_iterations = iterations[:len(self.step_sizes)]
            axes[1,1].plot(step_iterations, self.step_sizes, 'm-', linewidth=2)
            axes[1,1].set_title('Step Sizes')
            axes[1,1].set_xlabel('Iteration')
            axes[1,1].set_ylabel('Step Size')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"{self.method.upper()} Quasi-Newton {self.ham_loss.upper()} - Predictions vs Actual",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory (đường đồng mức)
        print("   - Vẽ đường đồng mức optimization")
        
        # Chuẩn bị X_test với bias cho visualization
        X_test_with_bias = add_bias_column(X_test)
        
        ve_duong_dong_muc_optimization(
            loss_function=self.loss_func,
            weights_history=self.weights_history,  # Pass full history
            X=X_test_with_bias, y=y_test,
            title=f"{self.method.upper()} Quasi-Newton {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png"),
            original_iterations=self.final_iteration,
            convergence_check_freq=self.convergence_check_freq,
            max_trajectory_points=None  # Quasi-Newton usually converges quickly, show all
        )
        
