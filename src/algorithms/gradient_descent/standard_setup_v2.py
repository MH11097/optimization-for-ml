#!/usr/bin/env python3
"""
Gradient Descent - Standard Setup

=== ỨNG DỤNG THỰC TẾ: GRADIENT DESCENT CỔ ĐIỂN ===

THAM SỐ TỐI ƯU:
- Learning Rate: 0.01 (vừa phải, ổn định)
- Max Iterations: 1000 (đủ để hội tụ)
- Tolerance: 1e-6 (chính xác cao)

ĐẶC ĐIỂM:
- Hội tụ ổn định và có thể dự đoán
- Phù hợp cho người mới bắt đầu
- Setup cơ bản, đáng tin cậy
- Sử dụng dữ liệu từ 02.1_sampled


- Ham loss: OLS = (1/2n) * ||y - Xw||²
- ∇L(w): Gradient = X^T(Xw - y) / n
- Regularization: 1e-12
- Max Iterations: 50
- Tolerance: 1e-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys
import os

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.optimization_utils import tinh_mse,  du_doan
from utils.visualization_utils import ve_duong_hoi_tu, ve_duong_dong_muc_optimization

def load_du_lieu():
    data_dir = Path("data/02.1_sampled")
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test = pd.read_csv(data_dir / "X_test.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    print(f"Loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def tinh_gia_tri_ham_OLS(X, y, w):
    """
    Tính giá trị hàm OLS tại điểm w
    
    Hàm OLS: L(w) = (1/2n) * ||Xw - y||²
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        y: vector target (n_samples,)
        w: vector weights (n_features,)
    
    Trả về:
        float: giá trị hàm OLS tại w
    """
    n_samples = X.shape[0]
    
    # Dự đoán: ŷ = Xw (không có bias)
    predictions = X @ w
    
    # Residuals: e = ŷ - y
    residuals = predictions - y
    
    # OLS loss: L(w) = (1/2n) * ||e||²
    ols_value = (1 / (2 * n_samples)) * np.sum(residuals ** 2)
    
    return ols_value

def tinh_gradient(X, y, w, he_so_tu_do=0):
    """Tính gradient của MSE cost function cho weights và bias"""
    n_samples = X.shape[0]
    du_doan_values = du_doan(X, w, he_so_tu_do)
    errors = du_doan_values - y
    
    # Gradient cho weights
    gradient_w = (1 / n_samples) * X.T.dot(errors)
    
    # Gradient cho bias
    gradient_b = (2 / n_samples) * np.sum(errors)
    
    return gradient_w, gradient_b

def gradient_descent(X, y, learning_rate=0.01, max_lan_thu=10000, diem_dung=1e-6):

    print("🚀 Training Standard Gradient Descent...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max iterations: {max_lan_thu}")
    print(f"   Tolerance: {diem_dung}")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features) # chon mot diem bat ki de bat dau
    
    OLS_history = []
    gradient_norms = []
    weights_history = []
    
    start_time = time.time()
    
    for lan_thu in range(max_lan_thu):
        # Compute cost, OLS value, and gradient
        ols_value = tinh_gia_tri_ham_OLS(X, y, weights)
        gradient_w, _ = tinh_gradient(X, y, weights)  # Only use weight gradient
        
        # Update weights
        weights = weights - learning_rate * gradient_w
        
        # Store history
        OLS_history.append(ols_value)
        gradient_norm = np.linalg.norm(gradient_w)
        gradient_norms.append(gradient_norm)
        weights_history.append(weights.copy())
        
        # Check convergence based on OLS value change
        if lan_thu > 0 and abs(OLS_history[-1] - OLS_history[-2]) < diem_dung:
            print(f"Converged after {lan_thu + 1} lan_thu (OLS change: {abs(OLS_history[-1] - OLS_history[-2]):.2e})")
            break
        
        # Progress update with monitoring
        if (lan_thu + 1) % 100 == 0:
            print(f"Iteration {lan_thu + 1}: OLS = {ols_value:.6f}, Gradient norm = {gradient_norm:.6f}")
    
    training_time = time.time() - start_time
    
    if lan_thu == max_lan_thu - 1:
        print(f"Reached maximum iterations ({max_lan_thu})")
    
    print(f"⏱️ Training time: {training_time:.2f} seconds")
    print(f"📉 Final OLS: {OLS_history[-1]:.6f}")
    print(f"📏 Final gradient norm: {gradient_norm:.6f}")
    
    return weights, OLS_history, gradient_norms, weights_history, training_time


def main():
    """Chạy Gradient Descent với Standard Setup"""
    print("GRADIENT DESCENT - STANDARD SETUP")
    
    # Setup
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Train model
    weights, OLS_history, gradient_norms, weights_history, training_time = gradient_descent(X_train, y_train)
    
    # Evaluate
    # metrics = evaluate_model(weights, X_test, y_test)
    
    print(f"\n🎨 Creating visualizations...")
    
    # 1. Convergence curves
    ve_duong_hoi_tu(OLS_history, gradient_norms, 
                    title="Gradient Descent Convergence Analysis")
    
    # 2. Contour plot with trajectory (sample every 10th point for performance)
    sample_frequency = max(1, len(weights_history) // 100)  # Max 100 points
    sampled_weights = weights_history[::sample_frequency]
    
    ve_duong_dong_muc_optimization(
        loss_function=tinh_gia_tri_ham_OLS,
        weights_history=sampled_weights,
        X=X_train, y=y_train,
        title="Gradient Descent Optimization Path"
    )
    
    print(f"\n✅ Training and visualization completed!")

if __name__ == "__main__":
    main()