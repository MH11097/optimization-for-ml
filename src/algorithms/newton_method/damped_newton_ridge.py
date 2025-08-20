#!/usr/bin/env python3
"""
- Ham loss: Ridge = (1/2n) * ||y - Xw||² + λ||w||²
- Gradient = X^T(Xw - y) / n + λw
- Hessian = X^T*X / n + λI
- Damped Newton với line search: w_{k+1} = w_k - α_k * H^{-1} * ∇L(w_k)
- Regularization: 1e-6
- Max Iterations: 100
- Tolerance: 1e-8
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
    tinh_loss_ridge, tinh_gradient_ridge, tinh_hessian_ridge,
    giai_he_phuong_trinh_tuyen_tinh,
    danh_gia_mo_hinh, in_ket_qua_danh_gia
)
from utils.visualization_utils import ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te

def load_du_lieu():
    data_dir = Path("data/02.1_sampled")
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test = pd.read_csv(data_dir / "X_test.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    print(f"Loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def backtracking_line_search_ridge(X, y, weights, direction, gradient, lam,
                                 armijo_c1=1e-4, backtrack_rho=0.8, max_line_search_iter=50):
    """
    Backtracking line search với Armijo condition cho Ridge
    """
    current_loss = tinh_loss_ridge(X, y, weights, lam)
    
    # Directional derivative: ∇f^T * d
    directional_derivative = np.dot(gradient, direction)
    
    # Initial step size
    alpha = 1.0
    
    for i in range(max_line_search_iter):
        # Thử weights mới
        new_weights = weights + alpha * direction
        
        # Tính loss mới
        new_loss = tinh_loss_ridge(X, y, new_weights, lam)
        
        # Kiểm tra Armijo condition: f(x + αd) ≤ f(x) + c₁α∇f^T d
        armijo_condition = current_loss + armijo_c1 * alpha * directional_derivative
        
        if new_loss <= armijo_condition:
            return alpha, i + 1
        
        # Giảm step size
        alpha *= backtrack_rho
    
    # Nếu line search fail, return very small step
    return alpha, max_line_search_iter

def damped_newton_method(X, y, regularization=1e-6, max_lan_thu=100, diem_dung=1e-8,
                        armijo_c1=1e-4, backtrack_rho=0.8, max_line_search_iter=50):
    print("Training Damped Newton Method v2 for Ridge...")
    print(f"   Regularization: {regularization}")
    print(f"   Max iterations: {max_lan_thu}")
    print(f"   Tolerance: {diem_dung}")
    print(f"   Armijo constant: {armijo_c1}")
    print(f"   Backtrack factor: {backtrack_rho}")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    loss_history = []
    gradient_norms = []
    weights_history = []
    step_sizes = []
    line_search_iterations = []
    
    start_time = time.time()
    
    # Precompute Hessian (constant for Ridge)
    H = tinh_hessian_ridge(X, regularization)
    condition_number = np.linalg.cond(H)
    
    print(f"   Hessian condition number: {condition_number:.2e}")
    
    for lan_thu in range(max_lan_thu):
        # Compute loss and gradient
        loss_value = tinh_loss_ridge(X, y, weights, regularization)
        gradient_w = tinh_gradient_ridge(X, y, weights, regularization)
        
        # Store history
        loss_history.append(loss_value)
        gradient_norm = np.linalg.norm(gradient_w)
        gradient_norms.append(gradient_norm)
        weights_history.append(weights.copy())
        
        # Check convergence
        if gradient_norm < diem_dung:
            print(f"Converged after {lan_thu + 1} iterations (gradient norm: {gradient_norm:.2e})")
            break
        
        # Newton direction: giải H * d = -gradient
        try:
            direction = -giai_he_phuong_trinh_tuyen_tinh(H, gradient_w)
            
            # Line search để tìm step size
            step_size, ls_iter = backtracking_line_search_ridge(
                X, y, weights, direction, gradient_w, regularization,
                armijo_c1, backtrack_rho, max_line_search_iter
            )
            
            step_sizes.append(step_size)
            line_search_iterations.append(ls_iter)
            
            # Update weights
            weights = weights + step_size * direction
            
        except np.linalg.LinAlgError:
            print(f"Linear algebra error at iteration {lan_thu + 1}")
            break
        
        # Progress update
        if (lan_thu + 1) % 10 == 0:
            print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient norm = {gradient_norm:.2e}, Step size = {step_size:.6f}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Final loss: {loss_history[-1]:.8f}")
    print(f"Final gradient norm: {gradient_norms[-1]:.2e}")
    print(f"Average step size: {np.mean(step_sizes):.6f}")
    print(f"Average line search iterations: {np.mean(line_search_iterations):.1f}")
    
    return weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number


def main():
    """Chạy Damped Newton Method v2 cho Ridge"""
    print("DAMPED NEWTON METHOD V2 - RIDGE SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/newton_method/damped_newton_v2_ridge")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number = damped_newton_method(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set...")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "Damped Newton Method v2 - Ridge")
    
    # Save results.json
    print("   Lưu kết quả vào results.json...")
    results_data = {
        "algorithm": "Damped Newton Method v2 - Ridge", 
        "loss_function": "Ridge Regression (L2 Regularization)",
        "parameters": {
            "regularization": 1e-6,
            "max_iterations": 100,
            "tolerance": 1e-8,
            "armijo_constant": 1e-4,
            "backtrack_factor": 0.8,
            "max_line_search_iter": 50
        },
        "metrics": metrics,
        "training_time": training_time,
        "convergence": {
            "iterations": len(loss_history),
            "final_loss": float(loss_history[-1]),
            "final_gradient_norm": float(gradient_norms[-1])
        },
        "numerical_analysis": {
            "hessian_condition_number": float(condition_number),
            "average_step_size": float(np.mean(step_sizes)) if step_sizes else 0,
            "line_search_efficiency": "Adaptive step size with Armijo condition",
            "regularization_effect": "L2 regularization improves condition number"
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
                    title="Damped Newton Method v2 Ridge - Convergence Analysis",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế...")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="Damped Newton Method v2 Ridge - Predictions vs Actual",
                         save_path=str(results_dir / "predictions_vs_actual.png"))
    
    # 3. Optimization trajectory
    print("   Vẽ đường đẳng mực optimization...")
    sample_frequency = max(1, len(weights_history) // 50)
    sampled_weights = weights_history[::sample_frequency]
    
    # Create a wrapper function for Ridge loss
    def ridge_loss_wrapper(X, y):
        def loss_func(weights):
            return tinh_loss_ridge(X, y, weights, 1e-6)
        return loss_func
    
    ve_duong_dong_muc_optimization(
        loss_function=ridge_loss_wrapper(X_train, y_train),
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="Damped Newton Method v2 Ridge - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()