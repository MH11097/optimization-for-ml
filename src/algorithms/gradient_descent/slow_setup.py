#!/usr/bin/env python3
"""
Gradient Descent - Slow Setup

=== ỨNG DỤNG THỰC TẾ: GRADIENT DESCENT CHẬM ===

THAM SỐ:
- Learning Rate: 0.001 (thấp)
- Max Iterations: 2000
- Tolerance: 1e-6

ĐẶC ĐIỂM:
- Hội tụ chậm nhưng ổn định
- Learning rate thấp, ít risk overshoot  
- Phù hợp khi cần độ chính xác cao
- An toàn, ít oscillation
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
from utils.visualization_utils import ve_duong_hoi_tu, ve_so_sanh_thuc_te_du_doan

def setup_output_dir():
    """Tạo thư mục output"""
    output_dir = Path("data/03_algorithms/gradient_descent/slow_setup")
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

def tinh_chi_phi(X, y, weights, bias=0.0):
    """Tính Mean Squared Error cost function"""
    predictions = X.dot(weights) + bias
    errors = predictions - y
    cost = np.mean(errors ** 2)
    return cost

def tinh_gradient(X, y, weights, bias=0.0):
    """Tính gradient của MSE cost function"""
    n_samples = X.shape[0]
    predictions = X.dot(weights) + bias
    errors = predictions - y
    gradient_w = (2 / n_samples) * X.T.dot(errors)
    gradient_b = (2 / n_samples) * np.sum(errors)
    return gradient_w, gradient_b

def gradient_descent_cham(X, y, learning_rate=0.001, max_iterations=2000, tolerance=1e-6):
    """
    Gradient Descent Chậm - Ổn định và Chính xác
    
    Tham số:
    - learning_rate: 0.001 (thấp, cho ổn định)
    - max_iterations: 2000 (nhiều, chậm hội tụ)
    - tolerance: 1e-6 (đòi hỏi chính xác)
    """
    print("🐌 Training Slow Gradient Descent...")
    print(f"   Learning rate: {learning_rate} (THẤP - cho ổn định)")
    print(f"   Max iterations: {max_iterations} (nhiều, mong đợi chậm)")
    print(f"   Tolerance: {tolerance} (đòi hỏi chính xác)")
    
    # Khởi tạo parameters
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    bias = 0.0
    
    cost_history = []
    gradient_norms = []
    train_mse_history = []
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Tính cost và gradient
        cost = tinh_chi_phi(X, y, weights, bias)
        gradient_w, gradient_b = tinh_gradient(X, y, weights, bias)
        
        # Cập nhật parameters
        weights = weights - learning_rate * gradient_w
        bias = bias - learning_rate * gradient_b
        
        # Lưu lịch sử
        cost_history.append(cost)
        gradient_norm = np.sqrt(np.sum(gradient_w**2) + gradient_b**2)
        gradient_norms.append(gradient_norm)
        train_mse_history.append(cost)
        
        # In tiến trình mỗi 200 iterations
        if iteration % 200 == 0:
            print(f"   Iteration {iteration:4d}: Cost = {cost:.6f}, Gradient norm = {gradient_norm:.6f}")
        
        # Kiểm tra hội tụ
        if gradient_norm < tolerance:
            print(f"   ✅ Hội tụ tại iteration {iteration} (gradient norm < {tolerance})")
            break
    
    training_time = time.time() - start_time
    
    print(f"⏱️ Training hoàn thành trong {training_time:.3f}s")
    print(f"📊 Final cost: {cost:.6f}")
    print(f"📈 Tổng iterations: {len(cost_history)}")
    
    return weights, bias, cost_history, gradient_norms, training_time

def du_doan(X, weights, bias):
    """Dự đoán kết quả với trọng số và bias"""
    return X.dot(weights) + bias

def save_results(results, output_dir):
    """Lưu kết quả vào files"""
    
    # 1. Save training history
    history_df = pd.DataFrame({
        'iteration': range(len(results['cost_history'])),
        'cost': results['cost_history'],
        'gradient_norm': results['gradient_norms'],
        'train_mse': results['train_mse_history']
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    # 2. Save results summary
    summary = {
        'method': 'Gradient Descent',
        'setup': 'slow_setup',
        'final_train_mse': results['final_train_mse'],
        'final_test_mse': results['final_test_mse'],
        'final_train_r2': results['final_train_r2'],
        'final_test_r2': results['final_test_r2'],
        'optimization_time': results['optimization_time'],
        'convergence_iterations': results['convergence_iterations'],
        'final_gradient_norm': results['final_gradient_norm'],
        'n_weights': len(results['weights']),
        'bias': results['bias']
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 3. Save weights
    weights_df = pd.DataFrame({
        'feature_index': range(len(results['weights'])),
        'weight_value': results['weights']
    })
    weights_df.to_csv(output_dir / "learned_weights.csv", index=False)
    
    print(f"💾 Results saved to {output_dir}")

def main():
    """Chạy Gradient Descent với Slow Setup - Ổn định & Chính xác"""
    print("🐌 GRADIENT DESCENT - SLOW SETUP")
    print("Low learning rate configuration for stability")
    
    try:
        # Setup
        output_dir = setup_output_dir()
        
        # Load data
        X_train, X_test, y_train, y_test = load_sampled_data()
        
        # Run optimization
        print("🚀 Starting Gradient Descent slow optimization...")
        weights, bias, cost_history, gradient_norms, training_time = gradient_descent_cham(
            X_train, y_train,
            learning_rate=0.001,    # Slow setup
            max_iterations=2000,
            tolerance=1e-6
        )
        
        # Final evaluation
        train_predictions = du_doan(X_train, weights, bias)
        test_predictions = du_doan(X_test, weights, bias)
        
        final_train_mse = tinh_mse(y_train, train_predictions)
        final_test_mse = tinh_mse(y_test, test_predictions)
        
        final_train_r2 = compute_r2_score(y_train, train_predictions)
        final_test_r2 = compute_r2_score(y_test, test_predictions)
        
        results = {
            'weights': weights,
            'bias': bias,
            'cost_history': cost_history,
            'gradient_norms': gradient_norms,
            'train_mse_history': cost_history,  # Same as cost for MSE
            'final_train_mse': final_train_mse,
            'final_test_mse': final_test_mse,
            'final_train_r2': final_train_r2,
            'final_test_r2': final_test_r2,
            'optimization_time': training_time,
            'convergence_iterations': len(cost_history),
            'final_gradient_norm': gradient_norms[-1] if gradient_norms else 0.0
        }
        
        # Save results
        save_results(results, output_dir)
        
        # Plot results - use Vietnamese visualization functions
        print("📈 Creating visualization...")
        ve_duong_hoi_tu(cost_history, gradient_norms, "Gradient Descent - Slow Setup")
        ve_so_sanh_thuc_te_du_doan(y_test, test_predictions, "Slow Setup Test Predictions")
        
        print("\n" + "="*50)
        print("🐌 SLOW SETUP - EVALUATION RESULTS")
        print("="*50)
        print(f"Test MSE:  {final_test_mse:.6f}")
        print(f"Test R²:   {final_test_r2:.4f}")
        
        print(f"\n🐌 STABILITY CHARACTERISTICS:")
        print(f"   📉 Learning Rate: 0.001 (THẤP cho ổn định)")
        print(f"   ⏱️ Training Time: {training_time:.3f}s (chậm nhưng ổn định)")
        print(f"   🎯 Convergence: {len(cost_history)} iterations")
        
        print(f"\n✅ Results saved to: {output_dir}")
        print("🎉 Gradient Descent slow optimization completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in Gradient Descent slow optimization: {e}")
        raise

if __name__ == "__main__":
    results = main()