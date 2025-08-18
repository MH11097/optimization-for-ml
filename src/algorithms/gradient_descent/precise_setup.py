from src.utils.data_loader import load_data_chunked
#!/usr/bin/env python3
"""
Gradient Descent - Precise Setup
Learning Rate: 0.001 (low)
Max Iterations: 2000
Tolerance: 1e-8

Đặc điểm:
- Hội tụ chậm nhưng rất chính xác
- Learning rate thấp, rất ổn định
- Tìm được minimum tốt nhất
- Phù hợp cho production, nghiên cứu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

def setup_output_dir():
    """Tạo thư mục output"""
    output_dir = Path("data/03_algorithms/gradient_descent/precise_setup")
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

def gradient_descent_fit(X, y, learning_rate=0.001, max_iterations=2000, tolerance=1e-8):
    """
    Precise Gradient Descent Implementation
    
    Setup Parameters:
    - learning_rate: 0.001 (low, for precision)
    - max_iterations: 2000 (more, expect slow convergence)
    - tolerance: 1e-8 (very strict, for precision)
    """
    print("🎯 Training Precise Gradient Descent...")
    print(f"   Learning rate: {learning_rate} (LOW - for precision)")
    print(f"   Max iterations: {max_iterations} (more for thorough search)")
    print(f"   Tolerance: {tolerance} (very strict)")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    cost_history = []
    gradient_norms = []
    cost_improvements = []  # Track improvement per iteration
    
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
        
        # Track improvement
        if iteration > 0:
            improvement = cost_history[-2] - cost_history[-1]
            cost_improvements.append(improvement)
        
        # Check convergence (very strict)
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"✅ High precision convergence after {iteration + 1} iterations")
            break
        
        # Progress update (more frequent for long training)
        if (iteration + 1) % 400 == 0:
            recent_improvement = np.mean(cost_improvements[-100:]) if len(cost_improvements) >= 100 else 0
            print(f"   Iteration {iteration + 1}: Cost = {cost:.8f}, Avg improvement = {recent_improvement:.2e}")
    
    training_time = time.time() - start_time
    
    if iteration == max_iterations - 1:
        print(f"⚠️ Reached maximum iterations ({max_iterations})")
        print("   Consider increasing max_iterations for even better precision")
    
    print(f"⏳ Training time: {training_time:.2f} seconds (thorough)")
    print(f"📉 Final cost: {cost_history[-1]:.8f} (high precision)")
    print(f"📏 Final gradient norm: {gradient_norms[-1]:.2e}")
    
    # Precision analysis
    if len(cost_improvements) > 100:
        recent_improvements = cost_improvements[-100:]
        avg_recent_improvement = np.mean(recent_improvements)
        print(f"🔍 Recent improvement rate: {avg_recent_improvement:.2e} per iteration")
        
        stability = np.std(recent_improvements) / np.mean(recent_improvements) if np.mean(recent_improvements) > 0 else 0
        print(f"📊 Stability score: {1/(1+stability):.4f} (closer to 1 = more stable)")
    
    return weights, cost_history, gradient_norms, cost_improvements, training_time

def evaluate_model(weights, X_test, y_test):
    """Đánh giá model trên test set với precision metrics"""
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
    
    # Additional precision metrics
    residuals = y_test - predictions
    
    # Max absolute error
    max_error = np.max(np.abs(residuals))
    
    # 95th percentile error
    error_95th = np.percentile(np.abs(residuals), 95)
    
    # Prediction confidence interval (approximate)
    prediction_std = np.std(residuals)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'max_error': max_error,
        'error_95th': error_95th,
        'prediction_std': prediction_std,
        'residuals': residuals
    }

def plot_results(cost_history, gradient_norms, cost_improvements, weights, X_test, y_test, output_dir):
    """Vẽ các biểu đồ kết quả với focus trên precision analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Gradient Descent - Precise Setup Results (High Precision Analysis)', fontsize=16)
    
    # 1. Training curve - high precision view
    ax1 = axes[0, 0]
    ax1.plot(cost_history, 'b-', linewidth=1.5, alpha=0.8)
    ax1.set_title('Precise Convergence Curve')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('MSE Cost (High Precision)')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 2. Gradient norms - precision tracking
    ax2 = axes[0, 1]
    ax2.plot(gradient_norms, 'r-', linewidth=1.5, alpha=0.8)
    ax2.set_title('Gradient Norm Decay')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Gradient Norm (log scale)')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Cost improvements per iteration
    ax3 = axes[0, 2]
    if cost_improvements:
        ax3.plot(cost_improvements, 'g-', linewidth=1, alpha=0.7)
        ax3.set_title('Cost Improvement per Iteration')
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Cost Reduction')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # 4. Predictions vs Actual - precision view
    ax4 = axes[1, 0]
    predictions = X_test.dot(weights)
    ax4.scatter(y_test, predictions, alpha=0.6, s=20, color='purple')
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Calculate and show R²
    r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    ax4.set_xlabel('Actual Values')
    ax4.set_ylabel('Predicted Values')
    ax4.set_title(f'Predictions vs Actual (R² = {r2:.6f})')
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuals distribution
    ax5 = axes[1, 1]
    residuals = y_test - predictions
    ax5.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax5.set_title('Residuals Distribution')
    ax5.set_xlabel('Residuals')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax5.axvline(mean_residual, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_residual:.2e}')
    ax5.legend()
    
    # 6. Convergence analysis - last 20% of training
    ax6 = axes[1, 2]
    if len(cost_history) > 100:
        final_portion = cost_history[-len(cost_history)//5:]  # Last 20%
        ax6.plot(final_portion, 'b-', linewidth=2)
        ax6.set_title('Final Convergence (Last 20%)')
        ax6.set_xlabel('Iterations (final phase)')
        ax6.set_ylabel('Cost (fine-tuning)')
        ax6.grid(True, alpha=0.3)
        ax6.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_dir / "precise_setup_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(weights, metrics, cost_history, gradient_norms, cost_improvements, training_time, output_dir):
    """Lưu kết quả với precision analysis"""
    
    # Advanced precision analysis
    precision_analysis = {}
    if len(cost_improvements) > 100:
        recent_improvements = cost_improvements[-100:]
        precision_analysis = {
            'final_convergence_rate': np.mean(recent_improvements),
            'convergence_stability': np.std(recent_improvements),
            'total_cost_reduction': cost_history[0] - cost_history[-1],
            'precision_score': len(cost_history) / 2000,  # How much of max iterations used
            'gradient_reduction_ratio': gradient_norms[0] / gradient_norms[-1]
        }
    
    results = {
        'setup_name': 'Precise Setup',
        'algorithm': 'Gradient Descent',
        'parameters': {
            'learning_rate': 0.001,
            'max_iterations': 2000,
            'tolerance': 1e-8
        },
        'metrics': metrics,
        'training_time': training_time,
        'convergence': {
            'iterations': len(cost_history),
            'final_cost': cost_history[-1],
            'final_gradient_norm': gradient_norms[-1]
        },
        'precision_analysis': precision_analysis,
        'notes': {
            'setup_description': 'Low learning rate for maximum precision',
            'pros': ['Highest precision', 'Most stable', 'Best final result', 'Research quality'],
            'cons': ['Slow training', 'More computation', 'May be overkill'],
            'recommendations': 'Use when: precision is critical, have time, production systems'
        }
    }
    
    # Save detailed results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save weights
    np.save(output_dir / "weights.npy", weights)
    
    # Save comprehensive training history
    history_df = pd.DataFrame({
        'iteration': range(len(cost_history)),
        'cost': cost_history,
        'gradient_norm': gradient_norms
    })
    
    if cost_improvements:
        improvements_padded = [0] + cost_improvements  # Add 0 for first iteration
        history_df['cost_improvement'] = improvements_padded
    
    history_df.to_csv(output_dir / "training_history.csv", index=False)

def print_results(metrics, training_time, iterations):
    """In kết quả đánh giá với focus trên precision"""
    print("\n" + "="*50)
    print("🎯 PRECISE SETUP - EVALUATION RESULTS")
    print("="*50)
    print(f"Test MSE:      {metrics['mse']:.8f} (high precision)")
    print(f"Test RMSE:     {metrics['rmse']:.8f}")
    print(f"Test MAE:      {metrics['mae']:.8f}")
    print(f"R² Score:      {metrics['r2']:.6f}")
    print(f"MAPE:          {metrics['mape']:.4f}%")
    
    print(f"\n🔍 PRECISION METRICS:")
    print(f"   📏 Max Error:     {metrics['max_error']:.6f}")
    print(f"   📊 95th %ile Err: {metrics['error_95th']:.6f}")
    print(f"   📐 Pred Std:      {metrics['prediction_std']:.6f}")
    
    print(f"\n🎯 PRECISION CHARACTERISTICS:")
    print(f"   🐌 Learning Rate: 0.001 (LOW for precision)")
    print(f"   ⏱️ Training Time: {training_time:.2f}s (thorough)")
    print(f"   🔄 Iterations Used: {iterations}/2000")
    print(f"   ✅ Tolerance: 1e-8 (very strict)")
    
    # Precision score
    precision_score = min(metrics['r2'], 1.0)
    if precision_score > 0.95:
        precision_rating = "EXCELLENT"
        color = "🟢"
    elif precision_score > 0.90:
        precision_rating = "VERY GOOD"
        color = "🟡"
    elif precision_score > 0.85:
        precision_rating = "GOOD"
        color = "🟠"
    else:
        precision_rating = "NEEDS IMPROVEMENT"
        color = "🔴"
    
    print(f"   {color} Precision Rating: {precision_rating}")
    
    print(f"\n🎯 WHEN TO USE PRECISE SETUP:")
    print(f"   • Production systems requiring high accuracy")
    print(f"   • Research where precision matters")
    print(f"   • Final model training")
    print(f"   • When computational time is not a constraint")
    
    print(f"\n💡 OPTIMIZATION INSIGHTS:")
    print(f"   • Very stable convergence")
    print(f"   • Minimal risk of overshooting")
    print(f"   • Best for finding global minimum")
    print(f"   • Suitable for sensitive applications")

def main():
    """Chạy Gradient Descent với Precise Setup"""
    print("🎯 GRADIENT DESCENT - PRECISE SETUP")
    print("Low learning rate configuration for maximum precision")
    
    # Setup
    output_dir = setup_output_dir()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    weights, cost_history, gradient_norms, cost_improvements, training_time = gradient_descent_fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(weights, X_test, y_test)
    
    # Plot results
    plot_results(cost_history, gradient_norms, cost_improvements, weights, X_test, y_test, output_dir)
    
    # Save everything
    save_results(weights, metrics, cost_history, gradient_norms, cost_improvements, training_time, output_dir)
    
    # Print results
    print_results(metrics, training_time, len(cost_history))
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()