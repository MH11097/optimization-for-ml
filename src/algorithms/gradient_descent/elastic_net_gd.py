#!/usr/bin/env python3
"""
- Ham loss: Elastic Net = MSE + α * (l1_ratio * L1 + (1-l1_ratio) * L2)
- Gradient = X^T(Xw - y) / n + α * [l1_ratio * sign(w) + (1-l1_ratio) * w]
- Learning Rate: 0.01
- Max Iterations: 1000
- Tolerance: 1e-6
- Alpha: 0.1, L1 ratio: 0.5

ĐẶC ĐIỂM: Kết hợp Ridge và Lasso - balanced regularization
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
    tinh_loss_elastic_net, 
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

def tinh_gradient_elastic_net(X, y, w, alpha, l1_ratio, epsilon=1e-8):
    """
    Tính gradient cho Elastic Net (smooth approximation for L1)
    """
    n = X.shape[0]
    predictions = X @ w
    errors = predictions - y
    mse_gradient = (X.T @ errors) / n
    
    # L1 gradient (smooth approximation)
    l1_gradient = alpha * l1_ratio * w / np.sqrt(w ** 2 + epsilon)
    # L2 gradient
    l2_gradient = alpha * (1 - l1_ratio) * w
    
    return mse_gradient + l1_gradient + l2_gradient

def gradient_descent(X, y, learning_rate=0.01, alpha=0.1, l1_ratio=0.5, 
                                max_lan_thu=1000, diem_dung=1e-6):


    print("Training Elastic Net Gradient Descent...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Alpha (total regularization): {alpha}")
    print(f"   L1 ratio: {l1_ratio}")
    print(f"   L2 ratio: {1 - l1_ratio}")
    print(f"   Max iterations: {max_lan_thu}")
    print(f"   Tolerance: {diem_dung}")
    print(f"   Loss function: Elastic Net (MSE + L1 + L2)")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    loss_history = []
    gradient_norms = []
    weights_history = []
    
    start_time = time.time()
    
    for lan_thu in range(max_lan_thu):
        # Compute Elastic Net loss and gradient
        loss_value = tinh_loss_elastic_net(X, y, weights, 0, alpha, l1_ratio)
        gradient_w = tinh_gradient_elastic_net(X, y, weights, alpha, l1_ratio)
        
        # Update weights
        weights = weights - learning_rate * gradient_w
        
        # Store history
        loss_history.append(loss_value)
        gradient_norm = np.linalg.norm(gradient_w)
        gradient_norms.append(gradient_norm)
        weights_history.append(weights.copy())
        
        # Check convergence
        if lan_thu > 0 and abs(loss_history[-1] - loss_history[-2]) < diem_dung:
            print(f"Converged after {lan_thu + 1} iterations")
            break
        
        # Progress update
        if (lan_thu + 1) % 100 == 0:
            print(f"Iteration {lan_thu + 1}: Elastic Net Loss = {loss_value:.6f}, Gradient norm = {gradient_norm:.6f}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final Elastic Net Loss: {loss_history[-1]:.6f}")
    print(f"Final gradient norm: {gradient_norm:.6f}")
    print(f"Final weights L1 norm: {np.sum(np.abs(weights)):.6f}")
    print(f"Final weights L2 norm: {np.linalg.norm(weights):.6f}")
    print(f"Sparsity (weights near zero): {np.sum(np.abs(weights) < 0.01) / len(weights):.1%}")
    
    return weights, loss_history, gradient_norms, weights_history, training_time


def main():
    """Chạy Elastic Net Gradient Descent"""
    print("GRADIENT DESCENT - ELASTIC NET SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/gradient_descent/elastic_net_gd")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, loss_history, gradient_norms, weights_history, training_time = gradient_descent(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set...")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "Elastic Net Gradient Descent")
    
    # Save results.json
    print("   Lưu kết quả vào results.json...")
    results_data = {
        "algorithm": "Elastic Net Gradient Descent",
        "loss_function": "Elastic Net (MSE + L1 + L2 regularization)",
        "parameters": {
            "learning_rate": 0.01,
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "max_iterations": 1000,
            "tolerance": 1e-6
        },
        "metrics": metrics,
        "training_time": training_time,
        "convergence": {
            "iterations": len(loss_history),
            "final_loss": float(loss_history[-1]),
            "final_gradient_norm": float(gradient_norms[-1])
        },
        "regularization_analysis": {
            "weights_l1_norm": float(np.sum(np.abs(weights))),
            "weights_l2_norm": float(np.linalg.norm(weights)),
            "sparsity_ratio": float(np.sum(np.abs(weights) < 0.01) / len(weights)),
            "l1_contribution": "50%",
            "l2_contribution": "50%",
            "regularization_balance": "Equal L1/L2 mix"
        }
    }
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save training history
    print("   Lưu lịch sử training vào training_history.csv...")
    training_df = pd.DataFrame({
        'iteration': range(len(loss_history)),
        'loss': loss_history,
        'gradient_norm': gradient_norms
    })
    training_df.to_csv(results_dir / "training_history.csv", index=False)
    
    print(f"\nTạo các biểu đồ visualization...")
    
    # Create custom loss function for visualization
    def elastic_net_loss_func(X, y, w):
        return tinh_loss_elastic_net(X, y, w, 0, 0.1, 0.5)
    
    # 1. Convergence curves
    print("   Vẽ đường hội tụ...")
    ve_duong_hoi_tu(loss_history, gradient_norms, 
                    title="Elastic Net GD - Convergence Analysis",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế...")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="Elastic Net GD - Predictions vs Actual",
                         save_path=str(results_dir / "predictions_vs_actual.png"))
    
    # 3. Optimization trajectory
    print("   Vẽ đường đẳng mực optimization...")
    sample_frequency = max(1, len(weights_history) // 100)
    sampled_weights = weights_history[::sample_frequency]
    
    ve_duong_dong_muc_optimization(
        loss_function=elastic_net_loss_func,
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="Elastic Net GD - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()