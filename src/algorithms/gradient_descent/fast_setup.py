#!/usr/bin/env python3
"""
Gradient Descent - Fast Setup

=== ỨNG DỤNG THỰC TẾ: GRADIENT DESCENT NHANH ===

THAM SỐ TỐI ƯU:
- Learning Rate: 0.1 (cao, cho tốc độ)
- Max Iterations: 500 (ít hơn, kỳ vọng hội tụ nhanh)
- Tolerance: 1e-5 (thoải mái cho tốc độ)

ĐẶC ĐIỂM:
- Hội tụ nhanh nhưng có thể không ổn định
- Phù hợp cho thí nghiệm nhanh
- Risk: có thể overshoot minimum
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
    output_dir = Path("data/03_algorithms/gradient_descent/fast_setup")
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

def gd_toi_uu_hoa_nhanh(X, y, learning_rate=0.1, max_iterations=500, tolerance=1e-5):
    """
    Fast Gradient Descent Implementation
    
    Setup Parameters:
    - learning_rate: 0.1 (high, for speed)
    - max_iterations: 500 (fewer, expect fast convergence)
    - tolerance: 1e-5 (less strict, for speed)
    """
    print("⚡ Training Fast Gradient Descent...")
    print(f"   Learning rate: {learning_rate} (HIGH - for speed)")
    print(f"   Max iterations: {max_iterations} (fewer expected)")
    print(f"   Tolerance: {tolerance} (relaxed for speed)")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    cost_history = []
    gradient_norms = []
    oscillation_count = 0  # Track potential oscillations
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Compute cost and gradient
        cost = tinh_chi_phi(X, y, weights, bias)
        gradient_w, gradient_b = tinh_gradient(X, y, weights, bias)
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Store history
        cost_history.append(cost)
        gradient_norms.append(np.linalg.norm(gradient))
        
        # Check for oscillation (cost increases)
        if iteration > 5 and cost > cost_history[-2]:
            oscillation_count += 1
            if oscillation_count > 3:
                print(f"⚠️ Oscillation detected! May need lower learning rate")
        
        # Check convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"✅ Fast convergence after {iteration + 1} iterations")
            break
        
        # Progress update
        if (iteration + 1) % 100 == 0:
            print(f"   Iteration {iteration + 1}: Cost = {cost:.6f}")
    
    training_time = time.time() - start_time
    
    if iteration == max_iterations - 1:
        print(f"⚠️ Reached maximum iterations ({max_iterations})")
    
    print(f"⚡ Training time: {training_time:.3f} seconds (FAST!)")
    print(f"📉 Final cost: {cost_history[-1]:.6f}")
    print(f"📏 Final gradient norm: {gradient_norms[-1]:.6f}")
    print(f"🔄 Oscillations detected: {oscillation_count}")
    
    return weights, cost_history, gradient_norms, training_time, oscillation_count

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

def plot_results(cost_history, gradient_norms, weights, X_test, y_test, output_dir, oscillation_count):
    """Vẽ các biểu đồ kết quả với focus trên speed analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gradient Descent - Fast Setup Results (High LR Analysis)', fontsize=16)
    
    # 1. Training curve với oscillation analysis
    ax1 = axes[0, 0]
    ax1.plot(cost_history, 'b-', linewidth=2, label='Training Cost')
    ax1.set_title(f'Fast Convergence (Oscillations: {oscillation_count})')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('MSE Cost')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight oscillations if any
    if len(cost_history) > 1:
        for i in range(1, len(cost_history)):
            if cost_history[i] > cost_history[i-1]:
                ax1.scatter(i, cost_history[i], color='red', s=20, alpha=0.7)
    
    # 2. Gradient norms - steep descent analysis
    ax2 = axes[0, 1]
    ax2.plot(gradient_norms, 'r-', linewidth=2)
    ax2.set_title('Gradient Magnitude (Fast Descent)')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Gradient Norm')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Predictions vs Actual
    ax3 = axes[1, 0]
    predictions = X_test.dot(weights)
    ax3.scatter(y_test, predictions, alpha=0.6, color='orange')
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title('Predictions vs Actual (Fast Training)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning rate impact analysis
    ax4 = axes[1, 1]
    if len(cost_history) > 10:
        # Show first vs last 10 iterations improvement
        early_costs = cost_history[:10]
        late_costs = cost_history[-10:]
        
        x_early = range(len(early_costs))
        x_late = range(len(cost_history)-10, len(cost_history))
        
        ax4.plot(x_early, early_costs, 'g-', linewidth=3, label='Early (Fast Drop)')
        ax4.plot(x_late, late_costs, 'b-', linewidth=3, label='Late (Fine-tuning)')
        ax4.set_title('Learning Rate Impact')
        ax4.set_xlabel('Iterations')
        ax4.set_ylabel('Cost')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fast_setup_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(weights, metrics, cost_history, gradient_norms, training_time, oscillation_count, output_dir):
    """Lưu kết quả với analysis cho fast setup"""
    results = {
        'setup_name': 'Fast Setup',
        'algorithm': 'Gradient Descent',
        'parameters': {
            'learning_rate': 0.1,
            'max_iterations': 500,
            'tolerance': 1e-5
        },
        'metrics': metrics,
        'training_time': training_time,
        'convergence': {
            'iterations': len(cost_history),
            'final_cost': cost_history[-1],
            'final_gradient_norm': gradient_norms[-1],
            'oscillation_count': oscillation_count
        },
        'speed_analysis': {
            'avg_cost_drop_per_iteration': (cost_history[0] - cost_history[-1]) / len(cost_history),
            'early_convergence_rate': cost_history[0] / cost_history[min(10, len(cost_history)-1)],
            'stability_score': 1 - (oscillation_count / len(cost_history))
        },
        'notes': {
            'setup_description': 'High learning rate for fast convergence',
            'pros': ['Very fast training', 'Quick results', 'Good for experimentation'],
            'cons': ['May oscillate', 'Less stable', 'Risk of overshooting'],
            'recommendations': 'Use when: speed is priority, can tolerate some instability'
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

def print_results(metrics, training_time, oscillation_count):
    """In kết quả đánh giá với focus trên speed"""
    print("\n" + "="*50)
    print("⚡ FAST SETUP - EVALUATION RESULTS")
    print("="*50)
    print(f"Test MSE:  {metrics['mse']:.6f}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")
    print(f"Test MAE:  {metrics['mae']:.6f}")
    print(f"R² Score:  {metrics['r2']:.4f}")
    print(f"MAPE:      {metrics['mape']:.2f}%")
    
    print(f"\n⚡ SPEED CHARACTERISTICS:")
    print(f"   🚀 Learning Rate: 0.1 (HIGH for speed)")
    print(f"   ⏱️ Training Time: {training_time:.3f}s (fast!)")
    print(f"   🔄 Oscillations: {oscillation_count} (stability indicator)")
    
    # Stability assessment
    if oscillation_count == 0:
        stability = "EXCELLENT"
        color = "🟢"
    elif oscillation_count <= 3:
        stability = "GOOD"
        color = "🟡"
    else:
        stability = "UNSTABLE"
        color = "🔴"
    
    print(f"   {color} Stability: {stability}")
    
    print(f"\n🎯 WHEN TO USE FAST SETUP:")
    print(f"   • Need quick experiments")
    print(f"   • Prototyping phase")
    print(f"   • Time-constrained situations")
    print(f"   • Can tolerate some instability")
    
    print(f"\n⚠️ CAUTIONS:")
    print(f"   • May miss optimal solution")
    print(f"   • Check for oscillations")
    print(f"   • Consider standard setup if unstable")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    """Chạy Gradient Descent với Fast Setup - Standardized Results"""
    print("⚡ GRADIENT DESCENT - FAST SETUP")
    print("High learning rate configuration for speed")
    
    # Memory monitoring
    initial_memory = get_memory_usage()
    
    # Load data
    X_train, X_test, y_train, y_test = load_sampled_data()
    
    # Configuration
    config = {
        'algorithm': 'gradient_descent',
        'setup': 'fast_setup',
        'learning_rate': 0.1,
        'max_iterations': 500,
        'tolerance': 1e-5,
        'loss_function': 'mse',
        'description': 'High learning rate for fast convergence'
    }
    
    # Train model
    weights, cost_history, gradient_norms, training_time, oscillation_count = gd_toi_uu_hoa_nhanh(X_train, y_train)
    
    # Calculate peak memory usage
    peak_memory = get_memory_usage()
    memory_usage = peak_memory - initial_memory
    
    # Standard evaluation
    train_pred = X_train.dot(weights)
    test_pred = X_test.dot(weights)
    
    # Standard ML metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    metrics = {
        'final_train_mse': float(mean_squared_error(y_train, train_pred)),
        'final_test_mse': float(mean_squared_error(y_test, test_pred)),
        'final_train_r2': float(r2_score(y_train, train_pred)),
        'final_test_r2': float(r2_score(y_test, test_pred)),
        'final_train_mae': float(mean_absolute_error(y_train, train_pred)),
        'final_test_mae': float(mean_absolute_error(y_test, test_pred)),
        'convergence_iteration': len(cost_history),
        'oscillation_count': oscillation_count,
        'final_gradient_norm': float(gradient_norms[-1]) if gradient_norms else 0.0
    }
    
    # Car price specific metrics
    car_price_metrics = calculate_price_metrics(y_test, test_pred)
    
    # Training history
    training_history = pd.DataFrame({
        'iteration': range(len(cost_history)),
        'train_loss': cost_history,
        'gradient_norm': gradient_norms,
        'test_loss': [mean_squared_error(y_test, X_test.dot(weights)) for _ in cost_history]  # Simplified
    })
    
    # Predictions for analysis
    predictions = {
        'y_train_actual': y_train,
        'y_train_pred': train_pred,
        'y_test_actual': y_test,
        'y_test_pred': test_pred
    }
    
    # Save results using standardized format
    output_dir = save_algorithm_results(
        algorithm='gradient_descent',
        setup='fast_setup',
        config=config,
        training_history=training_history,
        predictions=predictions,
        model_weights=weights,
        metrics=metrics,
        car_price_metrics=car_price_metrics,
        training_time=training_time,
        memory_usage=memory_usage
    )
    
    # Generate plots (legacy format for now)
    plot_results(cost_history, gradient_norms, weights, X_test, y_test, output_dir, oscillation_count)
    
    # Print results
    print_results(metrics, training_time, oscillation_count)
    print(f"\n📊 Car Price Metrics:")
    print(f"   💰 Mean Absolute Error: ${car_price_metrics['mean_absolute_error_dollars']:,.0f}")
    print(f"   📍 Predictions within 10%: {car_price_metrics['predictions_within_10pct']:.1%}")
    print(f"   🎯 Predictions within $5K: {car_price_metrics['predictions_within_5000_dollars']:.1%}")
    
    print(f"\n✅ Standardized results saved to: {output_dir}")

if __name__ == "__main__":
    main()