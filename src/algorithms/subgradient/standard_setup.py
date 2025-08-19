#!/usr/bin/env python3
"""
Subgradient Method - Standard Setup cho Non-Smooth Functions

=== THAM SỐ SETUP & HÀM LOSS ===

CÁC HÀM LOSS HỖ TRỢ:
1. L1 Norm: |w|_1 (non-smooth)
2. Hinge Loss: max(0, 1-y*f(x)) (SVM)
3. Non-smooth optimization problems

CÁC SETUP KHÁC NHAU:
Standard Setup (Subgradient):
- Learning Rate: 0.01 (constant, diminishing)
- Max Iterations: 2000 (cần nhiều hơn do convergence chậm)
- Tolerance: 1e-6
- Sử dụng cho: Non-smooth functions

ĐẶC ĐIỂM:
- Cho functions không smooth (không có gradient everywhere)
- Dùng subgradient thay vì gradient
- Convergence chậm hơn GD (O(1/√t))
- Không guarantee monotonic decrease
- Sử dụng dữ liệu từ 02.1_sampled

TOÁN HỌC:
- w_t+1 = w_t - α_t * g_t
- g_t ∈ ∂f(w_t) (subgradient)
- Cho L1: subgradient = sign(w) when w≠0, [-1,1] when w=0
- Learning rate schedule: α_t = α_0 / sqrt(t)
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
    output_dir = Path("data/03_algorithms/subgradient/standard_setup")
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

def compute_smooth_cost_and_gradient(X, y, weights):
    """Tính smooth part (MSE) và gradient của nó"""
    predictions = X.dot(weights)
    errors = predictions - y
    smooth_cost = np.mean(errors ** 2)
    smooth_gradient = (2 / len(y)) * X.T.dot(errors)
    return smooth_cost, smooth_gradient

def compute_l1_subgradient(weights, lambda_l1):
    """
    Tính subgradient của L1 norm
    
    ∂|w|/∂w = sign(w) if w ≠ 0
             ∈ [-1, 1] if w = 0
    
    Ở đây ta chọn 0 khi w = 0 (một choice trong subdifferential)
    """
    subgradient = np.zeros_like(weights)
    subgradient[weights > 0] = lambda_l1
    subgradient[weights < 0] = -lambda_l1
    # Khi weights[i] = 0, subgradient[i] = 0 (one choice)
    return subgradient

def subgradient_method_fit(X, y, learning_rate=0.01, lambda_l1=0.01, 
                          max_iterations=2000, tolerance=1e-6):
    """
    Subgradient Method Implementation
    
    For f(w) = MSE(w) + λ||w||₁
    
    Algorithm:
    1. Compute subgradient: g = ∇MSE(w) + ∂(λ||w||₁)
    2. Update: w = w - α * g
    
    Note: Cost may not decrease monotonically!
    """
    print("📉 Training Subgradient Method...")
    print(f"   Learning rate: {learning_rate} (constant)")
    print(f"   L1 regularization (λ): {lambda_l1}")
    print(f"   Max iterations: {max_iterations}")
    print("   Note: Non-smooth optimization, cost may oscillate")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    cost_history = []
    smooth_cost_history = []
    l1_penalty_history = []
    subgradient_norms = []
    best_weights = weights.copy()
    best_cost = float('inf')
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Compute smooth part
        smooth_cost, smooth_gradient = compute_smooth_cost_and_gradient(X, y, weights)
        
        # Compute L1 subgradient
        l1_subgradient = compute_l1_subgradient(weights, lambda_l1)
        
        # Total subgradient
        total_subgradient = smooth_gradient + l1_subgradient
        
        # Update weights
        weights -= learning_rate * total_subgradient
        
        # Compute costs
        l1_penalty = lambda_l1 * np.sum(np.abs(weights))
        total_cost = smooth_cost + l1_penalty
        
        # Track best solution (important for subgradient method!)
        if total_cost < best_cost:
            best_cost = total_cost
            best_weights = weights.copy()
        
        # Store history
        cost_history.append(total_cost)
        smooth_cost_history.append(smooth_cost)
        l1_penalty_history.append(l1_penalty)
        subgradient_norms.append(np.linalg.norm(total_subgradient))
        
        # Progress update
        if (iteration + 1) % 400 == 0:
            sparsity = np.sum(np.abs(weights) < 1e-6)
            print(f"   Iteration {iteration + 1}: Cost = {total_cost:.6f}, "
                  f"Best = {best_cost:.6f}, Sparsity = {sparsity}")
    
    training_time = time.time() - start_time
    
    # Use best weights found
    weights = best_weights
    final_smooth_cost, _ = compute_smooth_cost_and_gradient(X, y, weights)
    final_l1_penalty = lambda_l1 * np.sum(np.abs(weights))
    final_total_cost = final_smooth_cost + final_l1_penalty
    
    print(f"⏱️ Training time: {training_time:.3f} seconds")
    print(f"📉 Final cost: {final_total_cost:.6f}")
    print(f"🏆 Best cost found: {best_cost:.6f}")
    print(f"📊 Final sparsity: {np.sum(np.abs(weights) < 1e-6)}/{n_features}")
    
    # Convergence analysis for subgradient method
    recent_costs = cost_history[-100:] if len(cost_history) >= 100 else cost_history
    cost_std = np.std(recent_costs)
    print(f"📈 Recent cost std: {cost_std:.6f} (oscillation measure)")
    
    return (weights, cost_history, smooth_cost_history, l1_penalty_history, 
            subgradient_norms, training_time, best_cost)

def evaluate_model(weights, X_test, y_test, lambda_l1):
    """Đánh giá model"""
    predictions = X_test.dot(weights)
    
    # Standard metrics
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    
    # R-squared
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # MAPE
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    # Sparsity metrics
    sparsity_count = np.sum(np.abs(weights) < 1e-6)
    sparsity_ratio = sparsity_count / len(weights)
    l1_norm = np.sum(np.abs(weights))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'sparsity_count': sparsity_count,
        'sparsity_ratio': sparsity_ratio,
        'l1_norm': l1_norm
    }

def plot_results(cost_history, smooth_cost_history, l1_penalty_history, 
                subgradient_norms, weights, X_test, y_test, best_cost, output_dir):
    """Vẽ các biểu đồ đặc trưng của Subgradient Method"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Subgradient Method - Non-Smooth Optimization Analysis', fontsize=16)
    
    # 1. Cost evolution (với oscillations)
    ax1 = axes[0, 0]
    ax1.plot(cost_history, 'b-', linewidth=1, alpha=0.7, label='Actual Cost')
    ax1.axhline(y=best_cost, color='red', linestyle='--', linewidth=2, label=f'Best Cost: {best_cost:.4f}')
    
    # Running average để thấy trend
    window = 50
    if len(cost_history) > window:
        running_avg = np.convolve(cost_history, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(cost_history)), running_avg, 'green', linewidth=2, label='Running Average')
    
    ax1.set_title('Cost Evolution (Non-Monotonic)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Total Cost')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Subgradient norms
    ax2 = axes[0, 1]
    ax2.plot(subgradient_norms, 'purple', linewidth=1, alpha=0.8)
    ax2.set_title('Subgradient Magnitude')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('||Subgradient||')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Cost components
    ax3 = axes[0, 2]
    ax3.plot(smooth_cost_history, 'g-', linewidth=2, label='Smooth (MSE)')
    ax3.plot(l1_penalty_history, 'r-', linewidth=2, label='L1 Penalty')
    ax3.set_title('Cost Components')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Cost')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Predictions vs Actual
    ax4 = axes[1, 0]
    predictions = X_test.dot(weights)
    ax4.scatter(y_test, predictions, alpha=0.6, color='orange')
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax4.set_xlabel('Actual Values')
    ax4.set_ylabel('Predicted Values')
    ax4.set_title('Predictions vs Actual')
    ax4.grid(True, alpha=0.3)
    
    # 5. Convergence analysis
    ax5 = axes[1, 1]
    # Plot cost improvement over time
    if len(cost_history) > 1:
        cost_improvements = [best_cost - cost for cost in cost_history]
        ax5.plot(cost_improvements, 'blue', linewidth=2)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax5.set_title('Distance from Best Cost')
        ax5.set_xlabel('Iterations')
        ax5.set_ylabel('Best Cost - Current Cost')
        ax5.grid(True, alpha=0.3)
    
    # 6. Weight distribution
    ax6 = axes[1, 2]
    weights_nonzero = weights[np.abs(weights) > 1e-6]
    if len(weights_nonzero) > 0:
        ax6.hist(weights_nonzero, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        ax6.axvline(0, color='red', linestyle='--', linewidth=2)
        ax6.set_title(f'Non-Zero Weights\n(Active: {len(weights_nonzero)}/{len(weights)})')
        ax6.set_xlabel('Weight Values')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'All weights are zero', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Weight Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / "standard_setup_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(weights, metrics, cost_history, smooth_cost_history, 
                l1_penalty_history, subgradient_norms, training_time, best_cost, output_dir):
    """Lưu kết quả với Subgradient Method analysis"""
    
    # Subgradient-specific analysis
    subgradient_analysis = {
        'best_cost_found': float(best_cost),
        'final_cost': float(cost_history[-1]),
        'cost_oscillation_std': float(np.std(cost_history[-100:])) if len(cost_history) >= 100 else float(np.std(cost_history)),
        'convergence_type': 'non_monotonic',
        'avg_subgradient_norm': float(np.mean(subgradient_norms)),
        'final_sparsity': int(np.sum(np.abs(weights) < 1e-6))
    }
    
    results = {
        'setup_name': 'Standard Setup',
        'algorithm': 'Subgradient Method',
        'parameters': {
            'learning_rate': 0.01,
            'lambda_l1': 0.01,
            'max_iterations': 2000,
            'tolerance': 1e-6
        },
        'metrics': metrics,
        'training_time': training_time,
        'convergence': {
            'iterations': len(cost_history),
            'final_cost': cost_history[-1],
            'best_cost': best_cost
        },
        'subgradient_analysis': subgradient_analysis,
        'notes': {
            'setup_description': 'Standard Subgradient Method for non-smooth L1 regularized problems',
            'pros': ['Handles non-smooth functions', 'Simple implementation', 'General purpose'],
            'cons': ['Slow convergence O(1/√t)', 'Non-monotonic', 'Requires best-iterate tracking'],
            'mathematical_principle': 'Uses subgradients instead of gradients for non-smooth optimization',
            'key_insight': 'Cost may oscillate but converges to optimal neighborhood in expectation'
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
        'total_cost': cost_history,
        'smooth_cost': smooth_cost_history,
        'l1_penalty': l1_penalty_history,
        'subgradient_norm': subgradient_norms
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)

def print_results(metrics, training_time, iterations, best_cost, final_cost):
    """In kết quả với focus trên non-smooth optimization"""
    print("\n" + "="*60)
    print("📉 SUBGRADIENT METHOD - STANDARD SETUP RESULTS")
    print("="*60)
    print(f"Test MSE:      {metrics['mse']:.6f}")
    print(f"Test RMSE:     {metrics['rmse']:.6f}")
    print(f"Test MAE:      {metrics['mae']:.6f}")
    print(f"R² Score:      {metrics['r2']:.6f}")
    print(f"MAPE:          {metrics['mape']:.2f}%")
    
    print(f"\n📉 NON-SMOOTH OPTIMIZATION CHARACTERISTICS:")
    print(f"   🎯 Best Cost Found:    {best_cost:.6f}")
    print(f"   📊 Final Cost:         {final_cost:.6f}")
    print(f"   📈 Cost Gap:           {final_cost - best_cost:.6f}")
    print(f"   🔄 Iterations:         {iterations}")
    print(f"   ⏱️ Training Time:      {training_time:.3f}s")
    
    print(f"\n🎯 SPARSITY RESULTS:")
    print(f"   ❌ Zero Weights:       {metrics['sparsity_count']}")
    print(f"   📊 Sparsity Ratio:     {metrics['sparsity_ratio']*100:.1f}%")
    print(f"   📏 L1 Norm:            {metrics['l1_norm']:.6f}")
    
    print(f"\n🧮 SUBGRADIENT METHOD INSIGHTS:")
    print(f"   • Handles non-smooth functions (L1 regularization)")
    print(f"   • Uses subgradients: ∂f ⊇ {{g : f(y) ≥ f(x) + g^T(y-x)}}")
    print(f"   • For L1: ∂|w| = sign(w) if w≠0, [-1,1] if w=0")
    print(f"   • Convergence: O(1/√t) - slower than gradient descent")
    print(f"   • Non-monotonic: cost may oscillate")
    
    print(f"\n⚠️ IMPORTANT CHARACTERISTICS:")
    print(f"   • Cost không giảm monotonic như GD")
    print(f"   • Cần track best iterate, không dùng final iterate")
    print(f"   • Convergence rate chậm hơn smooth methods")
    print(f"   • Nhưng handle được non-smooth problems")
    
    print(f"\n🎯 KHI NÀO DÙNG SUBGRADIENT METHOD:")
    print(f"   • Function không smooth (có L1, L∞, max, etc.)")
    print(f"   • Không có smooth approximation")
    print(f"   • Constraint optimization với non-smooth penalties")
    print(f"   • Robust optimization problems")
    
    print(f"\n💡 SO SÁNH VỚI CÁC METHODS KHÁC:")
    print(f"   • vs Gradient Descent: Slower but handles non-smooth")
    print(f"   • vs Proximal GD: Simpler but less efficient")
    print(f"   • vs Newton: Much simpler, no Hessian needed")

def main():
    """Chạy Subgradient Method với Standard Setup"""
    print("📉 SUBGRADIENT METHOD - STANDARD SETUP")
    print("Non-smooth optimization for L1 regularized problems")
    
    # Setup
    output_dir = setup_output_dir()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    weights, cost_history, smooth_cost_history, l1_penalty_history, subgradient_norms, training_time, best_cost = subgradient_method_fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(weights, X_test, y_test, lambda_l1=0.01)
    
    # Plot results
    plot_results(cost_history, smooth_cost_history, l1_penalty_history, 
                subgradient_norms, weights, X_test, y_test, best_cost, output_dir)
    
    # Save everything
    save_results(weights, metrics, cost_history, smooth_cost_history, 
                l1_penalty_history, subgradient_norms, training_time, best_cost, output_dir)
    
    # Print results
    print_results(metrics, training_time, len(cost_history), best_cost, cost_history[-1])
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()