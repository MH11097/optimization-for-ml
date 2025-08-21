#!/usr/bin/env python3
"""
- Ham loss: OLS = (1/2n) * ||y - Xw||²
- Gradient = X^T(Xw - y) / n
- BFGS approximates inverse Hessian: H_k ≈ H^(-1)
- Update: w_{k+1} = w_k - α_k * H_k * ∇L(w_k)
- Regularization: 1e-8
- Max Iterations: 100
- Tolerance: 1e-6
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

def cap_nhat_bfgs(H_inv, s, y, damping=1e-8):
    """
    Cập nhật inverse Hessian approximation theo BFGS formula
    
    H_{k+1}^{-1} = (I - ρ s y^T) H_k^{-1} (I - ρ y s^T) + ρ s s^T
    where ρ = 1 / (y^T s)
    """
    sy = np.dot(s, y)
    
    # Kiểm tra curvature condition để đảm bảo positive definiteness
    if sy < damping:
        # Powell's damping trick
        theta = (1 - damping) / (sy - damping) if sy < damping else 1
        y = theta * y + (1 - theta) * H_inv @ s
        sy = np.dot(s, y)
    
    if abs(sy) < 1e-12:  # Avoid division by zero
        return H_inv
    
    rho = 1.0 / sy
    
    # BFGS update formula
    I = np.eye(len(s))
    A1 = I - rho * np.outer(s, y)
    A2 = I - rho * np.outer(y, s)
    
    H_inv_new = A1 @ H_inv @ A2 + rho * np.outer(s, s)
    
    return H_inv_new

def bfgs_method(X, y, regularization=1e-8, max_lan_thu=100, diem_dung=1e-6,
               armijo_c1=1e-4, wolfe_c2=0.9, backtrack_rho=0.8, restart_freq=None):
    print("Training BFGS Quasi-Newton Method for OLS...")
    print(f"   Regularization: {regularization}")
    print(f"   Max iterations: {max_lan_thu}")
    print(f"   Tolerance: {diem_dung}")
    print(f"   Armijo constant: {armijo_c1}")
    print(f"   Wolfe constant: {wolfe_c2}")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    # Initialize inverse Hessian approximation as identity
    H_inv = np.eye(n_features)
    
    loss_history = []
    gradient_norms = []
    weights_history = []
    step_sizes = []
    line_search_iterations = []
    
    start_time = time.time()
    
    # Initial gradient
    gradient_prev = tinh_gradient_OLS(X, y, weights)
    
    for lan_thu in range(max_lan_thu):
        # Compute loss and gradient
        loss_value = tinh_gia_tri_ham_OLS(X, y, weights)
        gradient_curr = tinh_gradient_OLS(X, y, weights)
        
        # Store history
        loss_history.append(loss_value)
        gradient_norm = np.linalg.norm(gradient_curr)
        gradient_norms.append(gradient_norm)
        weights_history.append(weights.copy())
        
        # Check convergence
        if gradient_norm < diem_dung:
            print(f"Converged after {lan_thu + 1} iterations (gradient norm: {gradient_norm:.2e})")
            break
        
        # Restart BFGS if needed
        if restart_freq and (lan_thu + 1) % restart_freq == 0:
            H_inv = np.eye(n_features)
            print(f"BFGS restart at iteration {lan_thu + 1}")
        
        # BFGS direction: d = -H_inv * gradient
        direction = -H_inv @ gradient_curr
        
        # Line search với Wolfe conditions
        step_size, ls_iter, gradient_new = wolfe_line_search(
            X, y, weights, direction, gradient_curr, 
            armijo_c1, wolfe_c2, backtrack_rho
        )
        
        step_sizes.append(step_size)
        line_search_iterations.append(ls_iter)
        
        # Update weights
        weights_new = weights + step_size * direction
        
        # BFGS update
        if lan_thu > 0:  # Skip first iteration
            s = weights_new - weights  # step
            y = gradient_new - gradient_curr  # gradient change
            
            # Update inverse Hessian approximation
            H_inv = cap_nhat_bfgs(H_inv, s, y)
        
        # Update for next iteration
        weights = weights_new
        gradient_prev = gradient_curr
        
        # Progress update
        if (lan_thu + 1) % 20 == 0:
            print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient norm = {gradient_norm:.2e}, Step size = {step_size:.6f}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Final loss: {loss_history[-1]:.8f}")
    print(f"Final gradient norm: {gradient_norms[-1]:.2e}")
    print(f"Average step size: {np.mean(step_sizes):.6f}")
    print(f"Average line search iterations: {np.mean(line_search_iterations):.1f}")
    
    # Final Hessian condition
    condition_number = np.linalg.cond(H_inv)
    
    return weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number


def main():
    """Chạy BFGS Quasi-Newton Method cho OLS"""
    print("BFGS QUASI-NEWTON METHOD - OLS SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/quasi_newton/bfgs")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number = bfgs_method(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set...")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "BFGS Quasi-Newton Method - OLS")
    
    # Save results.json
    print("   Lưu kết quả vào results.json...")
    results_data = {
        "algorithm": "BFGS Quasi-Newton Method - OLS",
        "loss_function": "OLS (Ordinary Least Squares)",
        "parameters": {
            "regularization": 1e-8,
            "max_iterations": 100,
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
            "inverse_hessian_condition_number": float(condition_number),
            "average_step_size": float(np.mean(step_sizes)) if step_sizes else 0,
            "quasi_newton_efficiency": "BFGS approximates inverse Hessian",
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
                    title="BFGS Quasi-Newton Method OLS - Convergence Analysis",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế...")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="BFGS Quasi-Newton Method OLS - Predictions vs Actual",
                         save_path=str(results_dir / "predictions_vs_actual.png"))
    
    # 3. Optimization trajectory
    print("   Vẽ đường đẳng mực optimization...")
    sample_frequency = max(1, len(weights_history) // 50)
    sampled_weights = weights_history[::sample_frequency]
    
    ve_duong_dong_muc_optimization(
        loss_function=tinh_gia_tri_ham_OLS,
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="BFGS Quasi-Newton Method OLS - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()