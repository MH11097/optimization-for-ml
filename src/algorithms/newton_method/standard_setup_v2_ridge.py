#!/usr/bin/env python3
"""
Standard Setup v2 - Ridge Regression

=== ỨNG DỤNG THỰC TẾ: STANDARD SETUP + RIDGE ===

HÀM LOSS: Ridge Regression
Công thức: L(w) = (1/2n) * Σ(y_i - ŷ_i)² + λ * ||w||²
Bao gồm L2 regularization để tránh overfitting

THAM SỐ TỐI ƯU:
- Regularization: 1e-6 (cho Ridge regression)
- Max Iterations: 50
- Tolerance: 1e-10

ĐẶC ĐIỂM:
- Hội tụ rất nhanh (quadratic convergence)
- Dùng thông tin bậc 2 (ma trận Hessian)
- Tốt cho bài toán convex và ill-conditioned
- Tránh overfitting với L2 regularization
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

from utils.optimization_utils import (
    tinh_gradient_hoi_quy_tuyen_tinh,
    tinh_ma_tran_hessian_hoi_quy_tuyen_tinh,
    giai_he_phuong_trinh_tuyen_tinh,
    kiem_tra_positive_definite,
    tinh_condition_number
)
from utils.optimization_utils import tinh_mse, compute_r2_score, predict

def setup_output_dir():
    """Tạo thư mục output"""
    output_dir = Path("data/03_algorithms/newton_method/standard_setup")
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

def newton_method_optimization(X_train, y_train, X_test, y_test, 
                             regularization=1e-8, max_iterations=50, tolerance=1e-10):
    """
    Newton Method optimization cho linear regression
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        regularization: λ cho regularization term
        max_iterations: số iteration tối đa
        tolerance: tolerance cho convergence
    
    Returns:
        results: dict chứa kết quả optimization
    """
    n_samples, n_features = X_train.shape
    
    # Khởi tạo parameters
    weights = np.zeros(n_features)
    bias = 0.0
    
    # Tracking
    cost_history = []
    gradient_norms = []
    train_mse_history = []
    test_mse_history = []
    
    start_time = time.time()
    
    print(f"🧮 Newton Method Optimization")
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Regularization: {regularization}")
    print(f"Max iterations: {max_iterations}")
    print(f"Tolerance: {tolerance}")
    print()
    
    # Pre-compute Hessian (constant cho linear regression)
    hessian = tinh_ma_tran_hessian_hoi_quy_tuyen_tinh(X_train, regularization)
    condition_number = tinh_condition_number(hessian)
    is_positive_definite = kiem_tra_positive_definite(hessian)
    
    print(f"📊 Hessian Analysis:")
    print(f"  Condition number: {condition_number:.2e}")
    print(f"  Is positive definite: {is_positive_definite}")
    
    if not is_positive_definite:
        print("⚠️  WARNING: Hessian is not positive definite!")
        print("   Consider increasing regularization parameter.")
    print()
    
    # Main optimization loop
    for iteration in range(max_iterations + 1):
        # Tính cost function
        predictions_train = du_doan(X_train, weights, bias)
        train_mse = tinh_mse(y_train, predictions_train)
        
        # Regularization term
        regularization_term = 0.5 * regularization * np.sum(weights**2)
        cost = train_mse + regularization_term
        
        # Test performance
        predictions_test = du_doan(X_test, weights, bias)
        test_mse = tinh_mse(y_test, predictions_test)
        
        # Tính gradient
        gradient_w, gradient_b = tinh_gradient_hoi_quy_tuyen_tinh(
            X_train, y_train, weights, bias, regularization
        )
        
        # Gradient norm
        gradient_norm = np.sqrt(np.sum(gradient_w**2) + gradient_b**2)
        
        # Lưu tracking info
        cost_history.append(cost)
        gradient_norms.append(gradient_norm)
        train_mse_history.append(train_mse)
        test_mse_history.append(test_mse)
        
        if iteration % 10 == 0 or iteration < 5:
            print(f"Iteration {iteration:3d}: "
                  f"Cost = {cost:.8f}, "
                  f"Train MSE = {train_mse:.8f}, "
                  f"Test MSE = {test_mse:.8f}, "
                  f"||grad|| = {gradient_norm:.2e}")
        
        # Kiểm tra convergence
        if gradient_norm < tolerance:
            print(f"\n✅ Converged after {iteration} iterations!")
            print(f"   Reason: Gradient norm {gradient_norm:.2e} < {tolerance:.2e}")
            break
        
        if iteration > 0:
            cost_change = abs(cost_history[-2] - cost)
            if cost_change < tolerance:
                print(f"\n✅ Converged after {iteration} iterations!")
                print(f"   Reason: Cost change {cost_change:.2e} < {tolerance:.2e}")
                break
        
        if iteration >= max_iterations:
            print(f"\n⏰ Reached max iterations ({max_iterations})")
            break
        
        # Newton step: giải H * step = gradient
        try:
            # Giải cho weights step
            weights_step = giai_he_phuong_trinh_tuyen_tinh(hessian, gradient_w)
            
            # Bias step (bias không có regularization cross-terms)
            bias_step = gradient_b
            
            # Update parameters
            weights = weights - weights_step
            bias = bias - bias_step
            
        except Exception as e:
            print(f"❌ Error in Newton step at iteration {iteration}: {e}")
            break
    
    end_time = time.time()
    optimization_time = end_time - start_time
    
    # Final metrics
    final_predictions_train = du_doan(X_train, weights, bias)
    final_predictions_test = du_doan(X_test, weights, bias)
    
    final_train_mse = tinh_mse(y_train, final_predictions_train)
    final_test_mse = tinh_mse(y_test, final_predictions_test)
    
    final_train_r2 = compute_r2_score(y_train, final_predictions_train)
    final_test_r2 = compute_r2_score(y_test, final_predictions_test)
    
    print(f"\n📈 Final Results:")
    print(f"  Optimization time: {optimization_time:.4f} seconds")
    print(f"  Final train MSE: {final_train_mse:.8f}")
    print(f"  Final test MSE: {final_test_mse:.8f}")
    print(f"  Final train R²: {final_train_r2:.6f}")
    print(f"  Final test R²: {final_test_r2:.6f}")
    print(f"  Condition number: {condition_number:.2e}")
    
    return {
        'weights': weights,
        'bias': bias,
        'cost_history': cost_history,
        'gradient_norms': gradient_norms,
        'train_mse_history': train_mse_history,
        'test_mse_history': test_mse_history,
        'final_train_mse': final_train_mse,
        'final_test_mse': final_test_mse,
        'final_train_r2': final_train_r2,
        'final_test_r2': final_test_r2,
        'optimization_time': optimization_time,
        'condition_number': condition_number,
        'convergence_iterations': iteration,
        'final_gradient_norm': gradient_norm,
        'method': 'Newton Method',
        'setup': 'standard'
    }

def save_results(results, output_dir):
    """Lưu kết quả vào files"""
    
    # 1. Save training history
    history_df = pd.DataFrame({
        'iteration': range(len(results['cost_history'])),
        'cost': results['cost_history'],
        'gradient_norm': results['gradient_norms'],
        'train_mse': results['train_mse_history'],
        'test_mse': results['test_mse_history']
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    # 2. Save results summary
    summary = {
        'method': results['method'],
        'setup': results['setup'],
        'final_train_mse': results['final_train_mse'],
        'final_test_mse': results['final_test_mse'],
        'final_train_r2': results['final_train_r2'],
        'final_test_r2': results['final_test_r2'],
        'optimization_time': results['optimization_time'],
        'condition_number': results['condition_number'],
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

def plot_convergence(results, output_dir):
    """Vẽ biểu đồ convergence"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    iterations = range(len(results['cost_history']))
    
    # Cost history
    axes[0, 0].plot(iterations, results['cost_history'])
    axes[0, 0].set_title('Cost Function Convergence')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Cost (MSE + Regularization)')
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # Gradient norm
    axes[0, 1].plot(iterations, results['gradient_norms'])
    axes[0, 1].set_title('Gradient Norm')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('||Gradient||')
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')
    
    # MSE comparison
    axes[1, 0].plot(iterations, results['train_mse_history'], label='Train MSE')
    axes[1, 0].plot(iterations, results['test_mse_history'], label='Test MSE')
    axes[1, 0].set_title('Train vs Test MSE')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Weight distribution
    axes[1, 1].hist(results['weights'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Weight Distribution')
    axes[1, 1].set_xlabel('Weight Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.suptitle(f'Newton Method - Standard Setup\nFinal Test MSE: {results["final_test_mse"]:.6f}', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "standard_setup_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function để chạy Newton Method standard setup"""
    
    try:
        # Setup
        output_dir = setup_output_dir()
        
        # Load data
        X_train, X_test, y_train, y_test = load_sampled_data()
        
        # Run optimization
        print("🚀 Starting Newton Method optimization...")
        results = newton_method_optimization(
            X_train, y_train, X_test, y_test,
            regularization=1e-8,    # Standard setup
            max_iterations=50,
            tolerance=1e-10
        )
        
        # Save results
        save_results(results, output_dir)
        
        # Plot results
        plot_convergence(results, output_dir)
        
        print("\n🎉 Newton Method optimization completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in Newton Method optimization: {e}")
        raise

if __name__ == "__main__":
    results = main()