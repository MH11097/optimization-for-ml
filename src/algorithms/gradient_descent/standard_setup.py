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

from utils.optimization_utils import tinh_mse, compute_r2_score, predict
# Removed old import - using load_sampled_data instead
from utils.visualization_utils import ve_duong_hoi_tu, ve_so_sanh_thuc_te_du_doan

def setup_output_dir():
    """Tạo thư mục output"""
    output_dir = Path("data/03_algorithms/gradient_descent/standard_setup")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_sampled_data():
    """Load dữ liệu từ 02.1_sampled (consistent với workflow hiện tại)"""
    data_dir = Path("data/02.1_sampled")
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Sampled data not found: {data_dir / file}")
    
    print("📂 Loading sampled data...")
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test = pd.read_csv(data_dir / "X_test.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    print(f"✅ Loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def tinh_chi_phi(X, y, trong_so, he_so_tu_do):
    """Tính Mean Squared Error cost với bias"""
    du_doan_values = predict(X, trong_so, he_so_tu_do)
    cost = tinh_mse(y, du_doan_values)
    return cost

def tinh_gradient(X, y, trong_so, he_so_tu_do):
    """Tính gradient của MSE cost function cho weights và bias"""
    n_samples = X.shape[0]
    du_doan_values = predict(X, trong_so, he_so_tu_do)
    errors = du_doan_values - y
    
    # Gradient cho weights
    gradient_w = (2 / n_samples) * X.T.dot(errors)
    
    # Gradient cho bias
    gradient_b = (2 / n_samples) * np.sum(errors)
    
    return gradient_w, gradient_b

def gd_toi_uu_hoa_co_ban(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Standard Gradient Descent Implementation
    
    Setup Parameters:
    - learning_rate: 0.01 (moderate, safe choice)
    - max_iterations: 1000 (enough for most cases)
    - tolerance: 1e-6 (good balance precision vs speed)
    """
    print("🚀 Training Standard Gradient Descent...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Tolerance: {tolerance}")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    cost_history = []
    gradient_norms = []
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Compute cost and gradient
        cost = compute_cost(X, y, weights)
        gradient = compute_gradient(X, y, weights)
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Store history
        cost_history.append(cost)
        gradient_norms.append(np.linalg.norm(gradient))
        
        # Check convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"✅ Converged after {iteration + 1} iterations")
            break
        
        # Progress update
        if (iteration + 1) % 200 == 0:
            print(f"   Iteration {iteration + 1}: Cost = {cost:.6f}")
    
    training_time = time.time() - start_time
    
    if iteration == max_iterations - 1:
        print(f"⚠️ Reached maximum iterations ({max_iterations})")
    
    print(f"⏱️ Training time: {training_time:.2f} seconds")
    print(f"📉 Final cost: {cost_history[-1]:.6f}")
    print(f"📏 Final gradient norm: {gradient_norms[-1]:.6f}")
    
    return weights, cost_history, gradient_norms, training_time

def evaluate_model(weights, X_test, y_test):
    """Đánh giá model trên test set"""
    predictions = X_test.dot(weights)
    
    # Metrics
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    
    # R-squared
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_results(cost_history, gradient_norms, weights, X_test, y_test, output_dir):
    """Vẽ các biểu đồ kết quả"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gradient Descent - Standard Setup Results', fontsize=16)
    
    # 1. Training curve
    ax1 = axes[0, 0]
    ax1.plot(cost_history, 'b-', linewidth=2)
    ax1.set_title('Training Cost Over Time')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('MSE Cost')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Gradient norms
    ax2 = axes[0, 1]
    ax2.plot(gradient_norms, 'r-', linewidth=2)
    ax2.set_title('Gradient Magnitude Over Time')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Gradient Norm')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Predictions vs Actual
    ax3 = axes[1, 0]
    predictions = X_test.dot(weights)
    ax3.scatter(y_test, predictions, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title('Predictions vs Actual')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals
    ax4 = axes[1, 1]
    residuals = y_test - predictions
    ax4.scatter(predictions, residuals, alpha=0.6)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Predicted Values')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Plot')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "standard_setup_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(weights, metrics, cost_history, gradient_norms, training_time, output_dir):
    """Lưu kết quả"""
    results = {
        'setup_name': 'Standard Setup',
        'algorithm': 'Gradient Descent',
        'parameters': {
            'learning_rate': 0.01,
            'max_iterations': 1000,
            'tolerance': 1e-6
        },
        'metrics': metrics,
        'training_time': training_time,
        'convergence': {
            'iterations': len(cost_history),
            'final_cost': cost_history[-1],
            'final_gradient_norm': gradient_norms[-1]
        },
        'notes': {
            'setup_description': 'Standard, safe setup with moderate learning rate',
            'pros': ['Stable convergence', 'Good for beginners', 'Reliable'],
            'cons': ['May be slow', 'Not optimized for speed']
        }
    }
    
    # Save detailed results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save weights
    np.save(output_dir / "weights.npy", weights)
    
    # Save training history
    history_df = pd.DataFrame({
        'iteration': range(len(cost_history)),
        'cost': cost_history,
        'gradient_norm': gradient_norms
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)

def print_results(metrics):
    """In kết quả đánh giá"""
    print("\n" + "="*50)
    print("📊 STANDARD SETUP - EVALUATION RESULTS")
    print("="*50)
    print(f"Test MSE:  {metrics['mse']:.6f}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")
    print(f"Test MAE:  {metrics['mae']:.6f}")
    print(f"R² Score:  {metrics['r2']:.4f}")
    print(f"MAPE:      {metrics['mape']:.2f}%")
    
    print(f"\n💡 SETUP CHARACTERISTICS:")
    print(f"   ✅ Learning Rate: 0.01 (moderate, stable)")
    print(f"   ✅ Max Iterations: 1000 (sufficient)")
    print(f"   ✅ Tolerance: 1e-6 (good precision)")
    
    print(f"\n🎯 WHEN TO USE THIS SETUP:")
    print(f"   • First time implementing GD")
    print(f"   • Want stable, predictable results")
    print(f"   • Not concerned about training speed")
    print(f"   • Educational purposes")

def main():
    """Chạy Gradient Descent với Standard Setup"""
    print("🎯 GRADIENT DESCENT - STANDARD SETUP")
    print("Safe, reliable configuration for learning and production")
    
    # Setup
    output_dir = setup_output_dir()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    weights, cost_history, gradient_norms, training_time = gradient_descent_fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(weights, X_test, y_test)
    
    # Plot results
    plot_results(cost_history, gradient_norms, weights, X_test, y_test, output_dir)
    
    # Save everything
    save_results(weights, metrics, cost_history, gradient_norms, training_time, output_dir)
    
    # Print results
    print_results(metrics)
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()