#!/usr/bin/env python3
"""
- Ham loss: Ridge = (1/2n) * ||y - Xw||² + λ||w||²
- Gradient = X^T(Xw - y) / n + λw
- Hessian = X^T*X / n + λI
- Regularization: 1e-6
- Max Iterations: 50
- Tolerance: 1e-10
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

def newton_method(X, y, regularization=1e-6, max_lan_thu=50, diem_dung=1e-10):
    print("Training Pure Newton Method v2 for Ridge...")
    print(f"   Regularization: {regularization}")
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
        if (lan_thu + 1) % 10 == 0:
            print(f"Iteration {lan_thu + 1}: Loss = {loss_value:.8f}, Gradient norm = {gradient_norm:.2e}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Final loss: {loss_history[-1]:.8f}")
    print(f"Final gradient norm: {gradient_norms[-1]:.2e}")
    
    return weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number


def main():
    """Chạy Pure Newton Method v2 cho Ridge"""
    print("PURE NEWTON METHOD V2 - RIDGE SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/newton_method/pure_newton_v2_ridge")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, loss_history, gradient_norms, weights_history, training_time, step_sizes, condition_number = newton_method(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set...")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "Pure Newton Method v2 - Ridge")
    
    # Save results.json
    print("   Lưu kết quả vào results.json...")
    results_data = {
        "algorithm": "Pure Newton Method v2 - Ridge",
        "loss_function": "Ridge Regression (L2 Regularization)",
        "parameters": {
            "regularization": 1e-6,
            "max_iterations": 50,
            "tolerance": 1e-10
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
            "quadratic_convergence": len(loss_history) <= 20,
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
                    title="Pure Newton Method v2 Ridge - Convergence Analysis",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế...")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="Pure Newton Method v2 Ridge - Predictions vs Actual",
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
        title="Pure Newton Method v2 Ridge - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()