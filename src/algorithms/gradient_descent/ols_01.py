#!/usr/bin/env python3
"""
- Ham loss: OLS = (1/2n) * ||y - Xw||²
- Gradient = X^T(Xw - y) / n
- Learning Rate: 0.1
- Max Iterations (số lần lặp tối đa): 500
- Tolerance (điểm dừng): 1e-5
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
from utils.data_process_utils import load_du_lieu



def gradient_descent(X, y, learning_rate=0.1, max_lan_thu=500, diem_dung=1e-5):


    print("Training Fast OLS Gradient Descent")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max iterations: {max_lan_thu}")
    print(f"   Tolerance: {diem_dung}")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    OLS_history = []
    gradient_norms = []
    weights_history = []
    
    start_time = time.time()
    
    for lan_thu in range(max_lan_thu):
        # Compute OLS loss and gradient
        ols_value = tinh_gia_tri_ham_OLS(X, y, weights)
        gradient_w = tinh_gradient_OLS(X, y, weights)
        
        # Update weights
        weights = weights - learning_rate * gradient_w
        
        # Store history
        OLS_history.append(ols_value)
        gradient_norm = np.linalg.norm(gradient_w)
        gradient_norms.append(gradient_norm)
        weights_history.append(weights.copy())
        
        # Check convergence (check điều kiện điểm dừng)
        if lan_thu > 0 and abs(OLS_history[-1] - OLS_history[-2]) < diem_dung:
            print(f"Converged after {lan_thu + 1} iterations")
            break
        
        # Progress update
        if (lan_thu + 1) % 50 == 0:
            print(f"Iteration {lan_thu + 1}: OLS = {ols_value:.6f}, Gradient norm = {gradient_norm:.6f}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final OLS: {OLS_history[-1]:.6f}")
    print(f"Final gradient norm: {gradient_norm:.6f}")
    
    return weights, OLS_history, gradient_norms, weights_history, training_time


def main():
    """Chạy Fast OLS Gradient Descent"""
    print("GRADIENT DESCENT - FAST OLS SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/gradient_descent/fast_ols_gd")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, OLS_history, gradient_norms, weights_history, training_time = gradient_descent(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "Fast OLS Gradient Descent")
    
    # Save results.json
    print("   Lưu kết quả vào results.json")
    results_data = {
        "algorithm": "Fast OLS Gradient Descent",
        "loss_function": "OLS (Ordinary Least Squares)",
        "parameters": {
            "learning_rate": 0.1,
            "max_iterations": 500,
            "tolerance": 1e-5
        },
        "metrics": metrics,
        "training_time": training_time,
        "convergence": {
            "iterations": len(OLS_history),
            "final_loss": float(OLS_history[-1]),
            "final_gradient_norm": float(gradient_norms[-1])
        }
    }
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save training history
    print("   Lưu lịch sử training vào training_history.csv")
    training_df = pd.DataFrame({
        'iteration': range(len(OLS_history)),
        'loss': OLS_history,
        'gradient_norm': gradient_norms
    })
    training_df.to_csv(results_dir / "training_history.csv", index=False)
    
    print(f"\n Tạo các biểu đồ visualization")
    # 1. Convergence curves
    print("   Vẽ đường hội tụ")
    ve_duong_hoi_tu(OLS_history, gradient_norms, 
                    title="Fast OLS GD - Convergence Analysis",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="Fast OLS GD - Predictions vs Actual",
                         save_path=str(results_dir / "predictions_vs_actual.png"))
    
    # 3. Optimization trajectory
    print("   Vẽ đường đẳng mực optimization")
    sample_frequency = max(1, len(weights_history) // 100)
    sampled_weights = weights_history[::sample_frequency]
    
    ve_duong_dong_muc_optimization(
        loss_function=tinh_gia_tri_ham_OLS,
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="Fast OLS GD - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()