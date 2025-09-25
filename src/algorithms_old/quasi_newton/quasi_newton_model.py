#!/usr/bin/env python3
"""
QuasiNewtonModel - Class cho BFGS Quasi-Newton Method
Hỗ trợ các loss functions: OLS, Ridge (LASSO bị chặn)
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
    Quasi-Newton Method Model với hỗ trợ BFGS, L-BFGS, SR1
    
    Parameters:
    - ham_loss: 'ols', 'ridge' (LASSO bị chặn)
    - so_lan_thu: Số lần lặp tối đa
    - diem_dung: Ngưỡng hội tụ (gradient norm)
    - regularization: Tham số regularization cho Ridge
    - method: 'bfgs', 'lbfgs', 'sr1'
    - auto_subsample: Cho phép auto-subsample khi dataset lớn
    - use_powell_damping: Sử dụng Powell damping cho BFGS
    """
    
    def __init__(self, ham_loss='ols', so_lan_thu=10000, diem_dung=1e-5, 
                 regularization=0.01, armijo_c1=1e-4, wolfe_c2=0.9,
                 backtrack_rho=0.8, max_line_search_iter=50, damping=1e-8, convergence_check_freq=10,
                 method='bfgs', memory_size=10, sr1_skip_threshold=1e-8, 
                 auto_subsample=True, use_powell_damping=True):
        
        # Chặn LASSO cho quasi-Newton
        if ham_loss.lower() == 'lasso':
            raise ValueError("LASSO không phù hợp với quasi-Newton methods. Sử dụng 'ols' hoặc 'ridge'.")
        
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
        
        # Algorithm selection
        self.method = method.lower()  # 'bfgs', 'lbfgs', 'sr1'
        if self.method not in ['bfgs', 'lbfgs', 'sr1']:
            raise ValueError("Method must be one of: 'bfgs', 'lbfgs', 'sr1'")
        
        # Algorithm-specific parameters
        self.memory_size = memory_size  # For L-BFGS
        self.sr1_skip_threshold = sr1_skip_threshold  # For SR1 stability
        
        # Control flags
        self.auto_subsample = auto_subsample  # Không auto-subsample ngầm
        self.use_powell_damping = use_powell_damping  # Powell damping chuẩn cho BFGS
        
        # Initialize algorithm-specific structures
        self.H_inv = None  # For BFGS and SR1
        self.s_vectors = []  # For L-BFGS
        self.y_vectors = []  # For L-BFGS
        self.rho_values = []  # For L-BFGS
        
        # Training state
        self.weights = None
        self.bias = None
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0
        
        # Training history
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.step_sizes = []
        self.line_search_iterations = []
        self.condition_numbers = []
        self.skipped_updates = []
        
        # Setup loss and gradient functions
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(self.ham_loss, X, y, w, None, self.regularization)
    
    def _get_best_results(self):
        """Lấy kết quả tốt nhất dựa trên gradient norm thấp nhất"""
        if not self.gradient_norms:
            raise ValueError("Không có lịch sử gradient norms để tìm kết quả tốt nhất")
        
        best_idx = np.argmin(self.gradient_norms)
        return {
            'best_weights': self.weights_history[best_idx],
            'best_loss': self.loss_history[best_idx],
            'best_gradient_norm': self.gradient_norms[best_idx],
            'best_iteration': best_idx * self.convergence_check_freq
        }
    
    def _wolfe_line_search(self, X, y, weights, direction, gradient):
        """Wolfe line search với descent guard và Armijo-only fallback"""
        current_loss = self.loss_func(X, y, weights)
        
        # Descent guard: kiểm tra hướng descent
        directional_derivative = np.dot(gradient, direction)
        if directional_derivative >= 0:
            print(f"   Warning: Not a descent direction (∇f^T d = {directional_derivative:.2e} ≥ 0)")
            direction = -gradient
            directional_derivative = np.dot(gradient, direction)
        
        alpha = 1.0
        for i in range(self.max_line_search_iter):
            new_weights = weights + alpha * direction
            new_loss = self.loss_func(X, y, new_weights)
            new_gradient, _ = self.grad_func(X, y, new_weights)
            
            # Armijo condition
            armijo_condition = current_loss + self.armijo_c1 * alpha * directional_derivative
            if new_loss <= armijo_condition:
                # Wolfe curvature condition
                curvature_condition = np.dot(new_gradient, direction)
                if curvature_condition >= self.wolfe_c2 * directional_derivative:
                    return alpha, i + 1, new_gradient
            
            alpha *= self.backtrack_rho
        
        # Armijo-only fallback
        alpha = 1.0
        for i in range(self.max_line_search_iter):
            new_weights = weights + alpha * direction
            new_loss = self.loss_func(X, y, new_weights)
            armijo_condition = current_loss + self.armijo_c1 * alpha * directional_derivative
            if new_loss <= armijo_condition:
                new_gradient, _ = self.grad_func(X, y, new_weights)
                print(f"   Using Armijo-only fallback (α = {alpha:.2e})")
                return alpha, i + 1, new_gradient
            alpha *= self.backtrack_rho
        
        # Final fallback
        new_weights = weights + alpha * direction
        new_gradient, _ = self.grad_func(X, y, new_weights)
        return alpha, self.max_line_search_iter, new_gradient
    
    def _cap_nhat_bfgs(self, H_inv, s, y):
        """BFGS update với Powell damping chuẩn"""
        sy = np.dot(s, y)
        
        # Standard curvature condition
        if sy <= 0:
            print(f"   Warning: Curvature condition violated (sy = {sy:.2e} ≤ 0), skipping BFGS update")
            return H_inv
        
        # Powell damping chuẩn
        if self.use_powell_damping and sy < self.damping:
            Bs = np.dot(H_inv, s)
            sBs = np.dot(s, Bs)
            
            if sBs > 0:
                theta = 0.8 * sBs / (sBs - sy)
                theta = max(0.2, min(0.8, theta))  # Clamp theta
                
                y_damped = theta * y + (1 - theta) * Bs
                sy_damped = np.dot(s, y_damped)
                
                if sy_damped > self.damping:
                    y = y_damped
                    sy = sy_damped
                    print(f"   Applied Powell damping (θ = {theta:.3f}), sy = {sy:.2e}")
                else:
                    print(f"   Warning: BFGS update skipped after damping (sy = {sy:.2e})")
                    return H_inv
            else:
                print(f"   Warning: BFGS update skipped (sBs = {sBs:.2e} ≤ 0)")
                return H_inv
        elif sy < self.damping:
            print(f"   Warning: BFGS update skipped (sy = {sy:.2e} < damping)")
            return H_inv
        
        # Standard BFGS update
        rho = 1.0 / sy
        n = len(s)
        I = np.eye(n)
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H_inv_new = np.dot(A, np.dot(H_inv, B)) + rho * np.outer(s, s)
        return H_inv_new
    
    def _lbfgs_two_loop_recursion(self, g):
        """L-BFGS two-loop recursion với H_0 = γI (γ > 0)"""
        if len(self.s_vectors) == 0:
            return -g
        
        # γ_0 từ cặp (s,y) gần nhất: γ = (s^T y) / (y^T y)
        s_k = self.s_vectors[-1]
        y_k = self.y_vectors[-1]
        gamma_0 = np.dot(s_k, y_k) / np.dot(y_k, y_k)
        gamma_0 = max(gamma_0, 1e-8)  # Ép γ > 0
        
        m = len(self.s_vectors)
        alpha = np.zeros(m)
        
        # First loop (backward)
        q = g.copy()
        for i in range(m - 1, -1, -1):
            alpha[i] = self.rho_values[i] * np.dot(self.s_vectors[i], q)
            q = q - alpha[i] * self.y_vectors[i]
        
        # H_0 = γI
        r = gamma_0 * q
        
        # Second loop (forward)
        for i in range(m):
            beta = self.rho_values[i] * np.dot(self.y_vectors[i], r)
            r = r + (alpha[i] - beta) * self.s_vectors[i]
        
        return -r
    
    def _update_lbfgs_memory(self, s, y):
        """L-BFGS memory update - skip khi s^T y ≤ 0"""
        sy = np.dot(s, y)
        
        # Skip khi s^T y ≤ 0
        if sy <= 0:
            print(f"   Warning: L-BFGS pair (s,y) skipped (s^T y = {sy:.2e} ≤ 0)")
            return
        
        rho = 1.0 / sy
        self.s_vectors.append(s)
        self.y_vectors.append(y)
        self.rho_values.append(rho)
        
        # Memory limit
        if len(self.s_vectors) > self.memory_size:
            self.s_vectors.pop(0)
            self.y_vectors.pop(0)
            self.rho_values.pop(0)
    
    def _cap_nhat_sr1(self, H_inv, s, y):
        """SR1 update - giữ nguyên như cũ (đã hoạt động tốt)"""
        Hy = np.dot(H_inv, y)
        v = s - Hy
        vTy = np.dot(v, y)
        
        # SR1 skip condition
        if abs(vTy) < self.sr1_skip_threshold * np.linalg.norm(v) * np.linalg.norm(y):
            print(f"   Warning: SR1 update skipped due to small denominator (vTy = {vTy:.2e})")
            self.skipped_updates.append(True)
            return H_inv
        
        self.skipped_updates.append(False)
        H_inv_new = H_inv + np.outer(v, v) / vTy
        return H_inv_new
    
    def _get_convergence_rate(self):
        """Get convergence rate"""
        rates = {"bfgs": "superlinear", "lbfgs": "superlinear", "sr1": "linear_to_superlinear"}
        return rates.get(self.method, "unknown")
    
    def _get_algorithm_specific_info(self):
        """Algorithm-specific info"""
        base_info = {
            "method_type": f"quasi_newton_{self.method}",
            "second_order_approximation": True,
            "line_search_used": True
        }
        
        if self.method == 'sr1' and self.skipped_updates:
            skip_count = sum(self.skipped_updates)
            total_updates = len(self.skipped_updates)
            base_info.update({
                "skipped_updates": skip_count,
                "total_updates": total_updates,
                "skip_rate": skip_count / total_updates if total_updates > 0 else 0
            })
        
        return base_info
    
    def fit(self, X, y):
        """Huấn luyện model với dữ liệu X, y"""
        print(f"🚀 Training {self.method.upper()} Quasi-Newton Method - {self.ham_loss.upper()}")
        print(f"   Max iterations: {self.so_lan_thu}")
        if self.ham_loss == 'ridge':
            print(f"   Regularization: {self.regularization}")
        print(f"   Armijo c1: {self.armijo_c1}, Wolfe c2: {self.wolfe_c2}")
        
        # Validation
        print(f"   Input X shape: {X.shape}, y shape: {y.shape}")
        if len(y.shape) != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Invalid input shapes")
        
        # Auto-subsample check
        if X.shape[0] > 50000:
            print(f"   ⚠️  Warning: Very large dataset ({X.shape[0]:,} samples)")
            if self.auto_subsample and X.shape[0] > 80000:
                print(f"   🔧 Auto-sampling to first 3200 samples...")
                X = X[:3200].copy()
                y = y[:3200].copy()
                print(f"   New shape: X={X.shape}, y={y.shape}")
            elif X.shape[0] > 80000:
                print(f"   ⚠️  Dataset có {X.shape[0]:,} samples. Set auto_subsample=False để tắt.")
        
        print(f"   Dataset size: {X.shape[0]:,} samples × {X.shape[1]} features")
        
        # Store y to avoid variable shadowing
        y_data = y.copy()
        
        # Add bias column
        X_with_bias = add_bias_column(X)
        print(f"   Features with bias: {X_with_bias.shape[1]}")
        
        # Initialize
        n_features_with_bias = X_with_bias.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        
        if self.method in ['bfgs', 'sr1']:
            self.H_inv = np.eye(n_features_with_bias)
        else:  # L-BFGS
            self.s_vectors = []
            self.y_vectors = []
            self.rho_values = []
        
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
        
        # Main optimization loop
        for lan_thu in range(self.so_lan_thu):
            # Compute gradient
            gradient_result = self.grad_func(X_with_bias, y_data, self.weights)
            if isinstance(gradient_result, tuple):
                gradient_curr, _ = gradient_result
            else:
                gradient_curr = gradient_result
            
            # Convergence check
            should_check_converged = ((lan_thu + 1) % self.convergence_check_freq == 0 or 
                                    lan_thu == self.so_lan_thu - 1)
            
            if should_check_converged:
                loss_value = self.loss_func(X_with_bias, y_data, self.weights)
                gradient_norm = np.linalg.norm(gradient_curr)
                
                self.loss_history.append(loss_value)
                self.gradient_norms.append(gradient_norm)
                self.weights_history.append(self.weights.copy())
                
                cost_change = (0.0 if len(self.loss_history) <= 1 else 
                             self.loss_history[-2] - self.loss_history[-1])
                
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
                    print(f"✅ {self.method.upper()} converged: {reason}" if converged else 
                          f"⚠️ {self.method.upper()} stopped: {reason}")
                    self.converged = converged
                    self.final_iteration = lan_thu + 1
                    break
            
            # Compute search direction
            if self.method == 'bfgs':
                direction = -np.dot(self.H_inv, gradient_curr)
            elif self.method == 'lbfgs':
                direction = self._lbfgs_two_loop_recursion(gradient_curr)
            elif self.method == 'sr1':
                direction = -np.dot(self.H_inv, gradient_curr)
            
            # Line search
            step_size, ls_iter, gradient_new = self._wolfe_line_search(
                X_with_bias, y_data, self.weights, direction, gradient_curr)
            
            self.step_sizes.append(step_size)
            self.line_search_iterations.append(ls_iter)
            
            # Update weights
            weights_new = self.weights + step_size * direction
            
            # Update quasi-Newton approximation
            s = weights_new - self.weights
            y_grad = gradient_new - gradient_curr  # Avoid shadowing y_data
            
            if self.method == 'bfgs':
                self.H_inv = self._cap_nhat_bfgs(self.H_inv, s, y_grad)
                cond_num = np.linalg.cond(self.H_inv)
                self.condition_numbers.append(cond_num)
            elif self.method == 'lbfgs':
                self._update_lbfgs_memory(s, y_grad)
                self.condition_numbers.append(np.nan)
            elif self.method == 'sr1':
                self.H_inv = self._cap_nhat_sr1(self.H_inv, s, y_grad)
                cond_num = np.linalg.cond(self.H_inv)
                self.condition_numbers.append(cond_num)
            
            self.weights = weights_new
            
            # Progress update
            if (lan_thu + 1) % 10 == 0 and should_check_converged:
                if self.method == 'lbfgs':
                    print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient = {gradient_norm:.2e}, Memory = {len(self.s_vectors)}")
                elif self.method == 'sr1':
                    skip_rate = sum(self.skipped_updates) / len(self.skipped_updates) if self.skipped_updates else 0
                    print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient = {gradient_norm:.2e}, Skip = {skip_rate:.1%}")
                else:  # BFGS
                    cond_num = self.condition_numbers[-1] if self.condition_numbers else np.nan
                    print(f"   Vòng {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient = {gradient_norm:.2e}, Cond = {cond_num:.2e}")
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"⏹️ Đạt tối đa {self.so_lan_thu} Iteration")
            self.final_iteration = self.so_lan_thu
        
        print(f"Thời gian training: {self.training_time:.4f}s")
        print(f"Loss cuối: {self.loss_history[-1]:.8f}")
        print(f"Bias cuối: {self.weights[-1]:.6f}")
        
        # Get best results
        best_results = self._get_best_results()
        print(f"🏆 Best results (gradient norm thấp nhất):")
        print(f"   Best iteration: {best_results['best_iteration']}")
        print(f"   Best loss: {best_results['best_loss']:.8f}")
        print(f"   Best gradient norm: {best_results['best_gradient_norm']:.2e}")
        
        return {
            'weights': best_results['best_weights'],
            'bias': best_results['best_weights'][-1],
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
            'best_iteration': best_results['best_iteration'],
            'best_loss': best_results['best_loss'],
            'best_gradient_norm': best_results['best_gradient_norm'],
            'final_loss': self.loss_history[-1],
            'final_gradient_norm': self.gradient_norms[-1]
        }
    
    def predict(self, X):
        """Dự đoán với dữ liệu X"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện")
        X_with_bias = add_bias_column(X)
        return du_doan(X_with_bias, self.weights, None)
    
    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện")
        
        print(f"\n📋 Đánh giá model...")
        bias_value = self.weights[-1]
        weights_without_bias = self.weights[:-1]
        metrics = danh_gia_mo_hinh(weights_without_bias, X_test, y_test, bias_value)
        in_ket_qua_danh_gia(metrics, self.training_time, 
                           f"{self.method.upper()} Quasi-Newton - {self.ham_loss.upper()}")
        return metrics
    
    def save_results(self, ten_file, base_dir="data/03_algorithms"):
        """Lưu kết quả model"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện")
        # Get algorithm name from class name
        algorithm_name = getattr(self, '__class__', type(self)).__name__.lower()
        results_dir = Path(base_dir) / algorithm_name / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        print(f"   Lưu kết quả vào {results_dir}/results.json")
        results_data = {
            "algorithm": f"{self.method.upper()} Quasi-Newton - {self.ham_loss.upper()}",
            "loss_function": self.ham_loss.upper(),
            "parameters": {
                "max_iterations": self.so_lan_thu,
                "tolerance": self.diem_dung,
                "regularization": self.regularization if self.ham_loss == 'ridge' else None
            },
            "training_results": {
                "training_time": self.training_time,
                "converged": self.converged,
                "final_iteration": self.final_iteration
            },
            "algorithm_specific": self._get_algorithm_specific_info()
        }
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        max_len = len(self.loss_history)
        step_sizes_padded = self.step_sizes[:max_len] + [np.nan] * max(0, max_len - len(self.step_sizes))
        line_search_padded = self.line_search_iterations[:max_len] + [np.nan] * max(0, max_len - len(self.line_search_iterations))
        condition_nums_padded = self.condition_numbers[:max_len] + [np.nan] * max(0, max_len - len(self.condition_numbers))
        
        training_df = pd.DataFrame({
            'iteration': range(0, len(self.loss_history)*self.convergence_check_freq, self.convergence_check_freq),
            'loss': self.loss_history,
            'gradient_norm': self.gradient_norms,
            'step_size': step_sizes_padded,
            'line_search_iter': line_search_padded,
            'condition_number': condition_nums_padded
        })
        training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        print(f"\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir
    
    def plot_results(self, X_test, y_test, ten_file, base_dir="data/03_algorithms/quasi_newton"):
        """Tạo biểu đồ visualization"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện")
        
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Tạo biểu đồ...")
        
        # 1. Convergence curves
        print("   - Vẽ đường hội tụ")
        iterations = list(range(0, len(self.loss_history) * self.convergence_check_freq, self.convergence_check_freq))
        
        ve_duong_hoi_tu(self.loss_history, self.gradient_norms, 
                        iterations=iterations,
                        title=f"{self.method.upper()} Quasi-Newton {self.ham_loss.upper()} - Hội tụ",
                        save_path=str(results_dir / "convergence_analysis.png"))
        
        # 2. Predictions vs Actual
        print("   - So sánh dự đoán vs thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                             title=f"{self.method.upper()} Quasi-Newton {self.ham_loss.upper()} - Dự đoán vs Thực tế",
                             save_path=str(results_dir / "predictions_vs_actual.png"))
        
        # 3. Optimization trajectory
        print("   - Vẽ đường đồng mức optimization")
        X_test_with_bias = add_bias_column(X_test)
        
        ve_duong_dong_muc_optimization(
            loss_function=self.loss_func,
            weights_history=self.weights_history,
            X=X_test_with_bias, y=y_test,
            title=f"{self.method.upper()} Quasi-Newton {self.ham_loss.upper()} - Optimization Path",
            save_path=str(results_dir / "optimization_trajectory.png"),
            original_iterations=self.final_iteration,
            convergence_check_freq=self.convergence_check_freq,
            max_trajectory_points=None
        )