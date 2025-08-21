#!/usr/bin/env python3
"""
- Ham loss: OLS = (1/2n) * ||y - Xw||²
- Gradient = X^T(Xw - y) / n
- L-BFGS: Limited memory BFGS với history size m
- Two-loop recursion để compute search direction
- Regularization: 1e-8
- Max Iterations: 200
- Tolerance: 1e-6
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from collections import deque
import sys
import os
import json

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.optimization_utils import (
    tinh_mse, du_doan, 
    tinh_gia_tri_ham_OLS, tinh_gradient_OLS,
    danh_gia_mo_hinh, in_ket_qua_danh_gia
)
from utils.data_process_utils import load_du_lieu
from utils.visualization_utils import ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te



def wolfe_line_search(X, y, weights, direction, gradient,
                     armijo_c1=1e-4, wolfe_c2=0.9, backtrack_rho=0.8, max_line_search_iter=50):
    """
    Wolfe line search để satisfy both Armijo và curvature conditions
    """
    current_loss = tinh_gia_tri_ham_OLS(X, y, weights)
    
    # Directional derivative: ∇f^T * d  
    directional_derivative = np.dot(gradient, direction)
    
    # Initial step size
    alpha = 1.0
    
    for i in range(max_line_search_iter):
        # Thử weights mới
        new_weights = weights + alpha * direction
        
        # Tính loss và gradient mới
        new_loss = tinh_gia_tri_ham_OLS(X, y, new_weights)
        new_gradient = tinh_gradient_OLS(X, y, new_weights)
        
        # Kiểm tra Armijo condition (sufficient decrease)
        armijo_condition = current_loss + armijo_c1 * alpha * directional_derivative
        
        if new_loss <= armijo_condition:
            # Kiểm tra curvature condition (Wolfe)
            curvature_condition = np.dot(new_gradient, direction)
            if curvature_condition >= wolfe_c2 * directional_derivative:
                return alpha, i + 1, new_gradient
        
        # Giảm step size
        alpha *= backtrack_rho
    
    # Nếu line search fail, return step cuối và gradient mới
    new_weights = weights + alpha * direction
    new_gradient = tinh_gradient_OLS(X, y, new_weights)
    return alpha, max_line_search_iter, new_gradient

def lbfgs_two_loop_recursion(gradient, s_history, y_history, gamma_k=1.0):
    """
    L-BFGS two-loop recursion để compute search direction
    
    Args:
        gradient: current gradient
        s_history: deque của step vectors s_i = x_{i+1} - x_i
        y_history: deque của gradient differences y_i = ∇f_{i+1} - ∇f_i  
        gamma_k: scaling factor (thường = s^T y / y^T y)
    
    Returns:
        direction: L-BFGS search direction
    """
    if len(s_history) == 0:
        return -gradient  # Steepest descent for first iteration
    
    m = len(s_history)  # current history size
    q = gradient.copy()
    alphas = np.zeros(m)
    
    # First loop (backward)
    for i in range(m - 1, -1, -1):
        s_i = s_history[i]
        y_i = y_history[i]
        
        rho_i = 1.0 / np.dot(y_i, s_i)
        alphas[i] = rho_i * np.dot(s_i, q)
        q = q - alphas[i] * y_i
    
    # Scaling
    r = gamma_k * q
    
    # Second loop (forward)
    for i in range(m):
        s_i = s_history[i]
        y_i = y_history[i]
        
        rho_i = 1.0 / np.dot(y_i, s_i)
        beta = rho_i * np.dot(y_i, r)
        r = r + (alphas[i] - beta) * s_i
    
    return -r  # Search direction

def lbfgs_method(X, y, memory_size=10, regularization=1e-8, max_lan_thu=200, diem_dung=1e-6,
                armijo_c1=1e-4, wolfe_c2=0.9, backtrack_rho=0.8):
    print("Training L-BFGS Quasi-Newton Method for OLS...")
    print(f"   Memory size: {memory_size}")
    print(f"   Regularization: {regularization}")
    print(f"   Max iterations: {max_lan_thu}")
    print(f"   Tolerance: {diem_dung}")
    print(f"   Armijo constant: {armijo_c1}")
    print(f"   Wolfe constant: {wolfe_c2}")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    # L-BFGS history storage
    s_history = deque(maxlen=memory_size)  # step vectors
    y_history = deque(maxlen=memory_size)  # gradient differences
    
    loss_history = []
    gradient_norms = []
    weights_history = []
    step_sizes = []
    line_search_iterations = []
    
    start_time = time.time()
    
    # Initial gradient
    gradient_curr = tinh_gradient_OLS(X, y, weights)
    
    for lan_thu in range(max_lan_thu):
        # Compute loss
        loss_value = tinh_gia_tri_ham_OLS(X, y, weights)
        
        # Store history
        loss_history.append(loss_value)
        gradient_norm = np.linalg.norm(gradient_curr)
        gradient_norms.append(gradient_norm)
        weights_history.append(weights.copy())
        
        # Check convergence
        if gradient_norm < diem_dung:
            print(f"Converged after {lan_thu + 1} iterations (gradient norm: {gradient_norm:.2e})")
            break
        
        # Compute scaling factor gamma_k
        if len(y_history) > 0:
            y_recent = y_history[-1]
            s_recent = s_history[-1]
            gamma_k = np.dot(s_recent, y_recent) / np.dot(y_recent, y_recent)
        else:
            gamma_k = 1.0
        
        # L-BFGS direction using two-loop recursion
        direction = lbfgs_two_loop_recursion(gradient_curr, s_history, y_history, gamma_k)
        
        # Line search với Wolfe conditions
        step_size, ls_iter, gradient_new = wolfe_line_search(
            X, y, weights, direction, gradient_curr, 
            armijo_c1, wolfe_c2, backtrack_rho
        )
        
        step_sizes.append(step_size)
        line_search_iterations.append(ls_iter)
        
        # Update weights
        weights_new = weights + step_size * direction
        
        # Update L-BFGS history
        if lan_thu > 0:  # Skip first iteration
            s_k = weights_new - weights  # step
            y_k = gradient_new - gradient_curr  # gradient change
            
            # Only add to history if curvature condition is satisfied
            if np.dot(s_k, y_k) > 1e-12:
                s_history.append(s_k)
                y_history.append(y_k)
        
        # Update for next iteration
        weights = weights_new
        gradient_curr = gradient_new
        
        # Progress update
        if (lan_thu + 1) % 20 == 0:
            print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient norm = {gradient_norm:.2e}, Step size = {step_size:.6f}, Memory used = {len(s_history)}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Final loss: {loss_history[-1]:.8f}")
    print(f"Final gradient norm: {gradient_norms[-1]:.2e}")
    print(f"Average step size: {np.mean(step_sizes):.6f}")
    print(f"Average line search iterations: {np.mean(line_search_iterations):.1f}")
    print(f"Final memory usage: {len(s_history)}/{memory_size}")
    
    # Condition number estimate (not directly available for L-BFGS)
    condition_number = 0.0  # L-BFGS doesn't explicitly form Hessian
    
    return weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number


def main():
    """Chạy L-BFGS Quasi-Newton Method cho OLS"""
    print("L-BFGS QUASI-NEWTON METHOD - OLS SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/quasi_newton/lbfgs")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number = lbfgs_method(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set...")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "L-BFGS Quasi-Newton Method - OLS")
    
    # Save results.json
    print("   Lưu kết quả vào results.json...")
    results_data = {
        "algorithm": "L-BFGS Quasi-Newton Method - OLS",
        "loss_function": "OLS (Ordinary Least Squares)",
        "parameters": {
            "memory_size": 10,
            "regularization": 1e-8,
            "max_iterations": 200,
            "tolerance": 1e-6,
            "armijo_constant": 1e-4,
            "wolfe_constant": 0.9,
            "backtrack_factor": 0.8
        },
        "metrics": metrics,
        "training_time": training_time,
        "convergence": {
            "iterations": len(loss_history),
            "final_loss": float(loss_history[-1]),
            "final_gradient_norm": float(gradient_norms[-1])
        },
        "numerical_analysis": {
            "memory_efficient": "O(mn) instead of O(n²) storage",
            "average_step_size": float(np.mean(step_sizes)) if step_sizes else 0,
            "quasi_newton_efficiency": "L-BFGS with limited memory",
            "line_search_method": "Wolfe conditions for robust convergence"
        }
    }
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save training history
    print("   Lưu lịch sử training vào training_history.csv...")
    max_len = len(loss_history)
    training_df = pd.DataFrame({
        'iteration': range(max_len),
        'loss': loss_history,
        'gradient_norm': gradient_norms,
        'step_size': step_sizes + [np.nan] * (max_len - len(step_sizes))
    })
    training_df.to_csv(results_dir / "training_history.csv", index=False)
    
    print(f"\nTạo các biểu đồ visualization...")
    
    # 1. Convergence curves
    print("   Vẽ đường hội tụ...")
    ve_duong_hoi_tu(loss_history, gradient_norms, 
                    title="L-BFGS Quasi-Newton Method OLS - Convergence Analysis",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế...")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="L-BFGS Quasi-Newton Method OLS - Predictions vs Actual",
                         save_path=str(results_dir / "predictions_vs_actual.png"))
    
    # 3. Optimization trajectory
    print("   Vẽ đường đẳng mực optimization...")
    sample_frequency = max(1, len(weights_history) // 50)
    sampled_weights = weights_history[::sample_frequency]
    
    ve_duong_dong_muc_optimization(
        loss_function=tinh_gia_tri_ham_OLS,
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="L-BFGS Quasi-Newton Method OLS - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()