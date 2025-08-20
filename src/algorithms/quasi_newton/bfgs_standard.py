#!/usr/bin/env python3
"""
- Ham loss: OLS = (1/2n) * ||y - Xw||²
- Update: BFGS formula
- Line search: Armijo condition
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
from utils.visualization_utils import ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te

def load_du_lieu():
    data_dir = Path("data/02.1_sampled")
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test = pd.read_csv(data_dir / "X_test.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    print(f"Loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def armijo_line_search(X, y, weights, p, gradient, c1=1e-4, max_backtracks=20):
    """Simple Armijo line search"""
    alpha = 1.0
    f0 = tinh_gia_tri_ham_OLS(X, y, weights)
    slope = c1 * np.dot(gradient, p)
    
    for _ in range(max_backtracks):
        weights_new = weights + alpha * p
        f_new = tinh_gia_tri_ham_OLS(X, y, weights_new)
        
        if f_new <= f0 + alpha * slope:
            return alpha
        alpha *= 0.5
    
    return alpha

def bfgs_method(X, y, max_lan_thu=100, diem_dung=1e-6):


    print("Training BFGS Quasi-Newton Method...")
    print(f"   Max iterations: {max_lan_thu}")
    print(f"   Tolerance: {diem_dung}")
    
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    # Initialize inverse Hessian approximation as identity
    B_inv = np.eye(n_features)
    
    loss_history = []
    gradient_norms = []
    weights_history = []
    step_sizes = []
    curvature_info = []
    
    start_time = time.time()
    
    for lan_thu in range(max_lan_thu):
        # Compute loss and gradient
        loss_value = tinh_gia_tri_ham_OLS(X, y, weights)
        gradient = tinh_gradient_OLS(X, y, weights)
        
        # Store history
        loss_history.append(loss_value)
        gradient_norm = np.linalg.norm(gradient)
        gradient_norms.append(gradient_norm)
        weights_history.append(weights.copy())
        
        # Check convergence
        if gradient_norm < diem_dung:
            print(f"Converged after {lan_thu + 1} iterations (gradient norm: {gradient_norm:.2e})")
            break
        
        # BFGS search direction
        p = -B_inv @ gradient
        
        # Line search
        alpha = armijo_line_search(X, y, weights, p, gradient)
        step_sizes.append(alpha)
        
        # Update weights
        s = alpha * p
        weights_new = weights + s
        
        # Compute new gradient for BFGS update
        gradient_new = tinh_gradient_OLS(X, y, weights_new)
        y_k = gradient_new - gradient
        
        # BFGS update of inverse Hessian approximation
        rho = np.dot(y_k, s)
        curvature_info.append(rho)
        
        if rho > 1e-12:  # Curvature condition
            # Sherman-Morrison formula for BFGS update
            rho_inv = 1.0 / rho
            A1 = np.eye(n_features) - rho_inv * np.outer(s, y_k)
            A2 = np.eye(n_features) - rho_inv * np.outer(y_k, s)
            B_inv = A1 @ B_inv @ A2 + rho_inv * np.outer(s, s)
        else:
            print(f"   Warning: Curvature condition violated at iteration {lan_thu + 1}")
        
        weights = weights_new
        
        # Progress update
        if (lan_thu + 1) % 10 == 0:
            print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient norm = {gradient_norm:.2e}, Step size = {alpha:.4f}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Final loss: {loss_history[-1]:.8f}")
    print(f"Final gradient norm: {gradient_norms[-1]:.2e}")
    print(f"Average step size: {np.mean(step_sizes):.4f}")
    print(f"Curvature violations: {sum(1 for c in curvature_info if c <= 1e-12)}")
    
    return weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, curvature_info


def main():
    """Chạy BFGS Standard"""
    print("BFGS QUASI-NEWTON - STANDARD SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/quasi_newton/bfgs_standard")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, curvature_info = bfgs_method(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set...")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "BFGS Quasi-Newton - Standard")
    
    # Save results.json
    print("   Lưu kết quả vào results.json...")
    results_data = {
        "algorithm": "BFGS Quasi-Newton - Standard",
        "loss_function": "OLS (Ordinary Least Squares)",
        "parameters": {
            "max_iterations": 100,
            "tolerance": 1e-6,
            "line_search": "Armijo condition",
            "c1": 1e-4
        },
        "metrics": metrics,
        "training_time": training_time,
        "convergence": {
            "iterations": len(loss_history),
            "final_loss": float(loss_history[-1]),
            "final_gradient_norm": float(gradient_norms[-1])
        },
        "bfgs_analysis": {
            "average_step_size": float(np.mean(step_sizes)),
            "curvature_violations": sum(1 for c in curvature_info if c <= 1e-12),
            "super_linear_convergence": len(loss_history) <= 30,
            "hessian_approximation": "Updated via BFGS formula",
            "memory_usage": "O(n²) for inverse Hessian approximation"
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
        'step_size': step_sizes + [np.nan] * (max_len - len(step_sizes)),
        'curvature_info': curvature_info + [np.nan] * (max_len - len(curvature_info))
    })
    training_df.to_csv(results_dir / "training_history.csv", index=False)
    
    print(f"\nTạo các biểu đồ visualization...")
    
    # 1. Convergence curves
    print("   Vẽ đường hội tụ...")
    ve_duong_hoi_tu(loss_history, gradient_norms, 
                    title="BFGS Standard - Convergence Analysis",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế...")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="BFGS Standard - Predictions vs Actual",
                         save_path=str(results_dir / "predictions_vs_actual.png"))
    
    # 3. Optimization trajectory
    print("   Vẽ đường đẳng mực optimization...")
    sample_frequency = max(1, len(weights_history) // 50)
    sampled_weights = weights_history[::sample_frequency]
    
    ve_duong_dong_muc_optimization(
        loss_function=tinh_gia_tri_ham_OLS,
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="BFGS Standard - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()