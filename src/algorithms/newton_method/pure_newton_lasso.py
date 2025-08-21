#!/usr/bin/env python3
"""
- Ham loss: Lasso (Smooth) = (1/2n) * ||y - Xw||² + λ * smooth_l1(w, μ)
- Gradient = X^T(Xw - y) / n + λ * smooth_l1_gradient(w, μ)
- Hessian = X^T*X / n + λ * smooth_l1_hessian(w, μ)
- Regularization: 1e-4
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
    tinh_loss_lasso_smooth, tinh_gradient_lasso_smooth,
    tinh_hessian_OLS, giai_he_phuong_trinh_tuyen_tinh,
    danh_gia_mo_hinh, in_ket_qua_danh_gia
)
from utils.visualization_utils import ve_duong_hoi_tu, ve_duong_dong_muc_optimization, ve_du_doan_vs_thuc_te
from utils.data_process_utils import load_du_lieu

def tinh_hessian_lasso_smooth(X, weights, lam=1e-4, mu=1e-3):
    """Tính Hessian cho smooth Lasso"""
    # Base Hessian from OLS
    H_ols = tinh_hessian_OLS(X)
    
    # Smooth L1 Hessian approximation
    # For smooth L1: ψ(w) ≈ |w| for |w| > μ, else w²/(2μ)
    # Hessian: λ * diag(1/μ) for |w| ≤ μ, else 0
    n_features = len(weights)
    H_l1 = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        if abs(weights[i]) <= mu:
            H_l1[i, i] = lam / mu
        # else remains 0 for |w| > μ
    
    return H_ols + H_l1

def newton_method(X, y, regularization=1e-4, max_lan_thu=100, diem_dung=1e-8, mu=1e-3):
    print("Training Pure Newton Method v3 for Lasso (Smooth)...")
    print(f"   Regularization: {regularization}")
    print(f"   Smoothing parameter: {mu}")
    print(f"   Max iterations: {max_lan_thu}")
    print(f"   Tolerance: {diem_dung}")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    loss_history = []
    gradient_norms = []
    weights_history = []
    step_sizes = []
    
    start_time = time.time()
    
    for lan_thu in range(max_lan_thu):
        # Compute loss and gradient
        loss_value = tinh_loss_lasso_smooth(X, y, weights, regularization, mu)
        gradient_w = tinh_gradient_lasso_smooth(X, y, weights, regularization, mu)
        
        # Store history
        loss_history.append(loss_value)
        gradient_norm = np.linalg.norm(gradient_w)
        gradient_norms.append(gradient_norm)
        weights_history.append(weights.copy())
        
        # Check convergence
        if gradient_norm < diem_dung:
            print(f"Converged after {lan_thu + 1} iterations (gradient norm: {gradient_norm:.2e})")
            break
        
        # Compute Hessian (depends on current weights for smooth L1)
        H = tinh_hessian_lasso_smooth(X, weights, regularization, mu)
        condition_number = np.linalg.cond(H)
        
        # Newton step
        try:
            delta_w = giai_he_phuong_trinh_tuyen_tinh(H, gradient_w)
            step_size = np.linalg.norm(delta_w)
            step_sizes.append(step_size)
            
            weights = weights - delta_w
            
        except np.linalg.LinAlgError:
            print(f"Linear algebra error at iteration {lan_thu + 1}")
            break
        
        # Progress update
        if (lan_thu + 1) % 20 == 0:
            print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient norm = {gradient_norm:.2e}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Final loss: {loss_history[-1]:.8f}")
    print(f"Final gradient norm: {gradient_norms[-1]:.2e}")
    print(f"Final Hessian condition number: {condition_number:.2e}")
    
    return weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number


def main():
    """Chạy Pure Newton Method v3 cho Lasso (Smooth)"""
    print("PURE NEWTON METHOD V3 - LASSO (SMOOTH) SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/newton_method/pure_newton_v3_lasso")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number = newton_method(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set...")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "Pure Newton Method v3 - Lasso (Smooth)")
    
    # Save results.json
    print("   Lưu kết quả vào results.json...")
    results_data = {
        "algorithm": "Pure Newton Method v3 - Lasso (Smooth)",
        "loss_function": "Smooth Lasso (L1 Regularization with smoothing)",
        "parameters": {
            "regularization": 1e-4,
            "smoothing_parameter": 1e-3,
            "max_iterations": 100,
            "tolerance": 1e-8
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
            "sparsity_effect": "L1 regularization promotes sparsity",
            "smoothing_effect": "Smoothing makes Lasso differentiable"
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
                    title="Pure Newton Method v3 Lasso - Convergence Analysis",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế...")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="Pure Newton Method v3 Lasso - Predictions vs Actual",
                         save_path=str(results_dir / "predictions_vs_actual.png"))
    
    # 3. Optimization trajectory
    print("   Vẽ đường đẳng mực optimization...")
    sample_frequency = max(1, len(weights_history) // 50)
    sampled_weights = weights_history[::sample_frequency]
    
    # Create a wrapper function for Lasso loss
    def lasso_loss_wrapper(X, y):
        def loss_func(weights):
            return tinh_loss_lasso_smooth(X, y, weights, 1e-4, 1e-3)
        return loss_func
    
    ve_duong_dong_muc_optimization(
        loss_function=lasso_loss_wrapper(X_train, y_train),
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="Pure Newton Method v3 Lasso - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    # Print sparsity analysis
    print(f"\n=== SPARSITY ANALYSIS ===")
    print(f"Non-zero weights: {np.sum(np.abs(weights) > 1e-6)}/{len(weights)}")
    print(f"Sparsity ratio: {(len(weights) - np.sum(np.abs(weights) > 1e-6))/len(weights)*100:.1f}%")

if __name__ == "__main__":
    main()