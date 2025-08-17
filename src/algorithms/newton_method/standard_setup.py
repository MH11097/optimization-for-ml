from src.utils.data_loader import load_data_chunked
#!/usr/bin/env python3
"""
Newton Method - Standard Setup
Regularization: 1e-8 (minimal)
Max Iterations: 50

Đặc điểm:
- Hội tụ rất nhanh (quadratic convergence)
- Dùng thông tin bậc 2 (Hessian)
- Tốt cho bài toán convex
- Cần ít iterations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

def setup_output_dir():
    """Tạo thư mục output"""
    output_dir = Path("data/03_algorithms/newton_method/standard_setup")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_processed_data():
    """Load dữ liệu đã xử lý"""
    data_dir = Path("data/02_processed")
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Processed data not found: {data_dir / file}")
    
    print("📂 Loading processed data...")
    X_train = load_data_chunked(data_dir / "X_train.csv").values
    X_test = load_data_chunked(data_dir / "X_test.csv").values
    y_train = load_data_chunked(data_dir / "y_train.csv").values.ravel()
    y_test = load_data_chunked(data_dir / "y_test.csv").values.ravel()
    
    print(f"✅ Loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def compute_cost(X, y, weights):
    """Tính Mean Squared Error cost"""
    predictions = X.dot(weights)
    errors = predictions - y
    cost = np.mean(errors ** 2)
    return cost

def compute_gradient(X, y, weights):
    """Tính gradient của MSE cost function"""
    n_samples = X.shape[0]
    predictions = X.dot(weights)
    errors = predictions - y
    gradient = (2 / n_samples) * X.T.dot(errors)
    return gradient

def compute_hessian(X):
    """Tính Hessian matrix cho linear regression"""
    n_samples = X.shape[0]
    hessian = (2 / n_samples) * X.T.dot(X)
    return hessian

def newton_method_fit(X, y, regularization=1e-8, max_iterations=50):
    """
    Newton Method Implementation
    
    Setup Parameters:
    - regularization: 1e-8 (minimal for numerical stability)
    - max_iterations: 50 (usually converges very fast)
    
    Newton update: w = w - H^(-1) * ∇J
    """
    print("🎯 Training Newton Method...")
    print(f"   Regularization: {regularization}")
    print(f"   Max iterations: {max_iterations}")
    print("   Using second-order information (Hessian)")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    cost_history = []
    gradient_norms = []
    step_sizes = []  # Track actual step sizes taken
    
    start_time = time.time()
    
    # Compute Hessian (constant for linear regression)
    print("📐 Computing Hessian matrix...")
    H = compute_hessian(X)
    
    # Add regularization for numerical stability
    H_reg = H + regularization * np.eye(H.shape[0])
    
    # Check if Hessian is invertible
    try:
        H_inv = np.linalg.inv(H_reg)
        print("✅ Hessian successfully inverted")
    except np.linalg.LinAlgError:
        print("⚠️ Hessian singular, adding more regularization...")
        H_reg = H + (regularization * 1000) * np.eye(H.shape[0])
        H_inv = np.linalg.inv(H_reg)
    
    # Check condition number
    cond_number = np.linalg.cond(H_reg)
    print(f"📊 Hessian condition number: {cond_number:.2e}")
    
    for iteration in range(max_iterations):
        # Compute cost and gradient
        cost = compute_cost(X, y, weights)
        gradient = compute_gradient(X, y, weights)
        
        # Newton step
        newton_step = H_inv.dot(gradient)
        step_size = np.linalg.norm(newton_step)
        
        # Update weights
        weights_new = weights - newton_step
        
        # Store history
        cost_history.append(cost)
        gradient_norms.append(np.linalg.norm(gradient))
        step_sizes.append(step_size)
        
        # Check convergence
        if iteration > 0:
            cost_change = abs(cost_history[-1] - cost_history[-2])
            if cost_change < 1e-10:
                print(f"✅ Converged after {iteration + 1} iterations")
                print(f"   Cost change: {cost_change:.2e}")
                break
        
        weights = weights_new
        
        # Progress update
        if (iteration + 1) % 10 == 0:
            print(f"   Iteration {iteration + 1}: Cost = {cost:.8f}, Step size = {step_size:.2e}")
    
    training_time = time.time() - start_time
    
    if iteration == max_iterations - 1:
        print(f"⚠️ Reached maximum iterations ({max_iterations})")
    
    print(f"⚡ Training time: {training_time:.4f} seconds (VERY FAST!)")
    print(f"📉 Final cost: {cost_history[-1]:.8f}")
    print(f"📏 Final gradient norm: {gradient_norms[-1]:.2e}")
    print(f"📐 Final step size: {step_sizes[-1]:.2e}")
    
    return weights, cost_history, gradient_norms, step_sizes, training_time, cond_number

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
    
    # MAPE
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_results(cost_history, gradient_norms, step_sizes, weights, X_test, y_test, cond_number, output_dir):
    """Vẽ các biểu đồ kết quả với focus trên Newton Method characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Newton Method - Standard Setup Results', fontsize=16)
    
    # 1. Training curve - rapid convergence
    ax1 = axes[0, 0]
    ax1.plot(cost_history, 'b-', linewidth=3, marker='o', markersize=6)
    ax1.set_title('Rapid Convergence (Newton Method)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('MSE Cost')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Annotate convergence speed
    if len(cost_history) > 1:
        convergence_rate = cost_history[0] / cost_history[-1]
        ax1.text(0.6, 0.8, f'Convergence Rate: {convergence_rate:.1e}x', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 2. Gradient norms - should drop very quickly
    ax2 = axes[0, 1]
    ax2.plot(gradient_norms, 'r-', linewidth=3, marker='s', markersize=6)
    ax2.set_title('Gradient Norm Decay')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Gradient Norm')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Newton step sizes
    ax3 = axes[1, 0]
    ax3.plot(step_sizes, 'g-', linewidth=3, marker='^', markersize=6)
    ax3.set_title('Newton Step Sizes')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Step Size')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Add condition number info
    ax3.text(0.05, 0.95, f'Hessian Condition Number: {cond_number:.1e}', 
             transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    # 4. Predictions vs Actual
    ax4 = axes[1, 1]
    predictions = X_test.dot(weights)
    ax4.scatter(y_test, predictions, alpha=0.6, color='purple')
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax4.set_xlabel('Actual Values')
    ax4.set_ylabel('Predicted Values')
    ax4.set_title('Predictions vs Actual')
    ax4.grid(True, alpha=0.3)
    
    # Calculate and show R²
    r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    ax4.text(0.05, 0.95, f'R² = {r2:.6f}', transform=ax4.transAxes,
             bbox=dict(boxstyle="round", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig(output_dir / "standard_setup_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(weights, metrics, cost_history, gradient_norms, step_sizes, training_time, cond_number, output_dir):
    """Lưu kết quả với Newton Method analysis"""
    
    # Newton-specific analysis
    newton_analysis = {
        'convergence_rate': cost_history[0] / cost_history[-1] if len(cost_history) > 1 else 1,
        'quadratic_convergence_indicator': len(cost_history) < 20,  # Converged very fast
        'hessian_condition_number': float(cond_number),
        'final_step_size': float(step_sizes[-1]) if step_sizes else 0,
        'gradient_reduction_ratio': float(gradient_norms[0] / gradient_norms[-1]) if len(gradient_norms) > 1 else 1
    }
    
    results = {
        'setup_name': 'Standard Setup',
        'algorithm': 'Newton Method',
        'parameters': {
            'regularization': 1e-8,
            'max_iterations': 50
        },
        'metrics': metrics,
        'training_time': training_time,
        'convergence': {
            'iterations': len(cost_history),
            'final_cost': cost_history[-1],
            'final_gradient_norm': gradient_norms[-1]
        },
        'newton_analysis': newton_analysis,
        'notes': {
            'setup_description': 'Standard Newton Method with minimal regularization',
            'pros': ['Extremely fast convergence', 'Uses second-order info', 'Optimal for convex problems'],
            'cons': ['Requires Hessian computation', 'Memory intensive', 'May fail if Hessian singular'],
            'mathematical_principle': 'Uses Hessian (second derivatives) for optimal step direction'
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
        'gradient_norm': gradient_norms,
        'step_size': step_sizes
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)

def print_results(metrics, training_time, iterations, cond_number):
    """In kết quả đánh giá với Newton Method insights"""
    print("\n" + "="*50)
    print("🎯 NEWTON METHOD - STANDARD SETUP RESULTS")
    print("="*50)
    print(f"Test MSE:  {metrics['mse']:.8f}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")
    print(f"Test MAE:  {metrics['mae']:.6f}")
    print(f"R² Score:  {metrics['r2']:.6f}")
    print(f"MAPE:      {metrics['mape']:.2f}%")
    
    print(f"\n⚡ NEWTON METHOD CHARACTERISTICS:")
    print(f"   🚀 Training Time: {training_time:.4f}s (extremely fast!)")
    print(f"   🔄 Iterations: {iterations} (quadratic convergence)")
    print(f"   📐 Hessian Condition: {cond_number:.2e}")
    print(f"   ✅ Regularization: 1e-8 (minimal)")
    
    # Convergence analysis
    if iterations < 10:
        convergence_rating = "EXCELLENT (< 10 iterations)"
        color = "🟢"
    elif iterations < 20:
        convergence_rating = "VERY GOOD (< 20 iterations)"
        color = "🟡"
    else:
        convergence_rating = "SLOWER THAN EXPECTED"
        color = "🔴"
    
    print(f"   {color} Convergence: {convergence_rating}")
    
    # Condition number analysis
    if cond_number < 1e6:
        condition_status = "WELL-CONDITIONED"
        cond_color = "🟢"
    elif cond_number < 1e12:
        condition_status = "MODERATELY CONDITIONED"
        cond_color = "🟡"
    else:
        condition_status = "ILL-CONDITIONED"
        cond_color = "🔴"
    
    print(f"   {cond_color} Matrix Condition: {condition_status}")
    
    print(f"\n🧮 MATHEMATICAL INSIGHTS:")
    print(f"   • Newton Method uses Hessian (second derivatives)")
    print(f"   • Update: w = w - H⁻¹∇J (optimal step direction)")
    print(f"   • Quadratic convergence near optimum")
    print(f"   • Perfect for convex problems like linear regression")
    
    print(f"\n🎯 WHEN TO USE NEWTON METHOD:")
    print(f"   • Small to medium datasets (Hessian computation)")
    print(f"   • Convex optimization problems")
    print(f"   • When you need very fast convergence")
    print(f"   • Research/analysis requiring optimal solutions")
    
    print(f"\n⚠️ LIMITATIONS:")
    print(f"   • Requires Hessian computation O(n²) memory")
    print(f"   • Matrix inversion O(n³) time complexity")
    print(f"   • May fail on non-convex or ill-conditioned problems")
    print(f"   • Not suitable for very large datasets")

def main():
    """Chạy Newton Method với Standard Setup"""
    print("🎯 NEWTON METHOD - STANDARD SETUP")
    print("Second-order optimization using Hessian information")
    
    # Setup
    output_dir = setup_output_dir()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    weights, cost_history, gradient_norms, step_sizes, training_time, cond_number = newton_method_fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(weights, X_test, y_test)
    
    # Plot results
    plot_results(cost_history, gradient_norms, step_sizes, weights, X_test, y_test, cond_number, output_dir)
    
    # Save everything
    save_results(weights, metrics, cost_history, gradient_norms, step_sizes, training_time, cond_number, output_dir)
    
    # Print results
    print_results(metrics, training_time, len(cost_history), cond_number)
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()