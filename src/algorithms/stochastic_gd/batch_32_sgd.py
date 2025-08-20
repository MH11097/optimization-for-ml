#!/usr/bin/env python3
"""
- Ham loss: OLS = (1/2n) * ||y - Xw||²
- Gradient = X^T(Xw - y) / n
- Batch Size: 32
- Learning Rate: 0.01
- Max Epochs: 100
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

def stochastic_gradient_descent(X, y, batch_size=32, learning_rate=0.01, max_epochs=100, diem_dung=1e-6):


    print("Training Stochastic Gradient Descent...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Tolerance: {diem_dung}")
    
    n_samples, n_features = X.shape
    weights = np.random.normal(0, 0.01, n_features)
    
    loss_history = []
    gradient_norms = []
    weights_history = []
    epoch_losses = []
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Shuffle data at the beginning of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        epoch_gradient_norms = []
        n_batches = 0
        
        # Mini-batch updates
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:batch_end]
            y_batch = y_shuffled[i:batch_end]
            
            # Compute gradient on batch
            gradient_w = tinh_gradient_OLS(X_batch, y_batch, weights)
            
            # Update weights
            weights = weights - learning_rate * gradient_w
            
            # Track metrics
            batch_loss = tinh_gia_tri_ham_OLS(X_batch, y_batch, weights)
            epoch_loss += batch_loss
            gradient_norm = np.linalg.norm(gradient_w)
            epoch_gradient_norms.append(gradient_norm)
            n_batches += 1
            
            # Store history (every few batches to avoid too much data)
            if len(loss_history) < 1000:  # Limit history size
                loss_history.append(batch_loss)
                gradient_norms.append(gradient_norm)
                weights_history.append(weights.copy())
        
        # Epoch-level metrics
        avg_epoch_loss = epoch_loss / n_batches
        avg_gradient_norm = np.mean(epoch_gradient_norms)
        epoch_losses.append(avg_epoch_loss)
        
        # Check convergence (based on epoch loss)
        if epoch > 5 and abs(epoch_losses[-1] - epoch_losses[-6]) < diem_dung:
            print(f"Converged after {epoch + 1} epochs")
            break
        
        # Progress update
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Avg Loss = {avg_epoch_loss:.6f}, Avg Gradient norm = {avg_gradient_norm:.6f}, Batches = {n_batches}")
    
    training_time = time.time() - start_time
    
    if epoch == max_epochs - 1:
        print(f"Reached maximum epochs ({max_epochs})")
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final epoch loss: {epoch_losses[-1]:.6f}")
    print(f"Total batches processed: {(epoch + 1) * n_batches}")
    
    return weights, loss_history, gradient_norms, weights_history, training_time, epoch_losses, n_batches


def main():
    """Chạy SGD với batch size 32"""
    print("STOCHASTIC GRADIENT DESCENT - BATCH 32 SETUP")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/stochastic_gd/batch_32_sgd")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, loss_history, gradient_norms, weights_history, training_time, epoch_losses, batches_per_epoch = stochastic_gradient_descent(X_train, y_train)
    
    # Đánh giá model
    print(f"\nĐánh giá model trên test set...")
    metrics = danh_gia_mo_hinh(weights, X_test, y_test)
    in_ket_qua_danh_gia(metrics, training_time, "SGD - Batch Size 32")
    
    # Save results.json
    print("   Lưu kết quả vào results.json...")
    results_data = {
        "algorithm": "Stochastic Gradient Descent - Batch Size 32",
        "loss_function": "OLS (Ordinary Least Squares)",
        "parameters": {
            "batch_size": 32,
            "learning_rate": 0.01,
            "max_epochs": 100,
            "tolerance": 1e-6
        },
        "metrics": metrics,
        "training_time": training_time,
        "convergence": {
            "epochs": len(epoch_losses),
            "batches_per_epoch": batches_per_epoch,
            "total_batches": len(epoch_losses) * batches_per_epoch,
            "final_epoch_loss": float(epoch_losses[-1])
        },
        "sgd_analysis": {
            "batch_size_effect": "Good balance between gradient noise and computational efficiency",
            "variance_vs_efficiency": "Medium variance, good convergence speed",
            "memory_usage": "Moderate - processes 32 samples at a time"
        }
    }
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save training history
    print("   Lưu lịch sử training vào training_history.csv...")
    
    # Batch-level history
    max_len = min(len(loss_history), 1000)  # Limit size
    batch_df = pd.DataFrame({
        'batch': range(max_len),
        'loss': loss_history[:max_len],
        'gradient_norm': gradient_norms[:max_len]
    })
    batch_df.to_csv(results_dir / "batch_history.csv", index=False)
    
    # Epoch-level history
    epoch_df = pd.DataFrame({
        'epoch': range(len(epoch_losses)),
        'avg_loss': epoch_losses
    })
    epoch_df.to_csv(results_dir / "epoch_history.csv", index=False)
    
    print(f"\nTạo các biểu đồ visualization...")
    
    # 1. Convergence curves (use epoch losses for cleaner view)
    print("   Vẽ đường hội tụ...")
    ve_duong_hoi_tu(epoch_losses, None, 
                    title="SGD Batch-32 - Epoch Convergence",
                    save_path=str(results_dir / "convergence_analysis.png"))
    
    # 2. Predictions vs Actual
    print("   Vẽ so sánh dự đoán với thực tế...")
    y_pred_test = du_doan(X_test, weights, 0)
    ve_du_doan_vs_thuc_te(y_test, y_pred_test, 
                         title="SGD Batch-32 - Predictions vs Actual",
                         save_path=str(results_dir / "predictions_vs_actual.png"))
    
    # 3. Optimization trajectory (sample weights history)
    print("   Vẽ đường đẳng mực optimization...")
    sample_frequency = max(1, len(weights_history) // 100)
    sampled_weights = weights_history[::sample_frequency]
    
    ve_duong_dong_muc_optimization(
        loss_function=tinh_gia_tri_ham_OLS,
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="SGD Batch-32 - Optimization Path",
        save_path=str(results_dir / "optimization_trajectory.png")
    )
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()