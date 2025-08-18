from src.utils.data_loader import load_data_chunked
#!/usr/bin/env python3
"""
Proximal Gradient Descent - Standard Setup
Learning Rate: 0.01
Lambda (L1): 0.01
Max Iterations: 1000

Đặc điểm:
- Kết hợp Gradient Descent + L1 regularization
- Proximal operator cho sparsity (feature selection)
- Soft thresholding function
- Tạo ra sparse solutions (nhiều weights = 0)

Toán học:
- Forward step: z = w - α∇f(w)  
- Proximal step: w = prox_λ(z) = soft_threshold(z, λα)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

def setup_output_dir():
    """Tạo thư mục output"""
    output_dir = Path("data/algorithms/proximal_gd/standard_setup")
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

def compute_smooth_cost(X, y, weights):
    """Tính smooth part của cost function (MSE)"""
    predictions = X.dot(weights)
    errors = predictions - y
    smooth_cost = np.mean(errors ** 2)
    return smooth_cost

def compute_smooth_gradient(X, y, weights):
    """Tính gradient của smooth part"""
    n_samples = X.shape[0]
    predictions = X.dot(weights)
    errors = predictions - y
    gradient = (2 / n_samples) * X.T.dot(errors)
    return gradient

def soft_threshold(z, threshold):
    """
    Soft thresholding operator - Proximal operator cho L1 norm
    
    prox_λ|·|(z) = sign(z) * max(|z| - λ, 0)
    
    Đây là key function của Proximal GD!
    """
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

def proximal_gd_fit(X, y, learning_rate=0.01, lambda_l1=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Proximal Gradient Descent Implementation
    
    Algorithm:
    1. Forward step: z = w - α∇f(w)  
    2. Proximal step: w = prox_λ(z)
    
    Setup Parameters:
    - learning_rate: 0.01 (step size)
    - lambda_l1: 0.01 (L1 regularization strength)
    - max_iterations: 1000
    """
    print("🎯 Training Proximal Gradient Descent...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   L1 regularization (λ): {lambda_l1}")
    print(f"   Max iterations: {max_iterations}")
    print("   Algorithm: Forward step + Proximal step")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    cost_history = []
    smooth_cost_history = []
    l1_penalty_history = []
    sparsity_history = []  # Track number of zero weights
    threshold = lambda_l1 * learning_rate  # Proximal threshold
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # 1. Compute smooth cost and gradient
        smooth_cost = compute_smooth_cost(X, y, weights)
        gradient = compute_smooth_gradient(X, y, weights)
        
        # 2. Forward step (standard gradient step)
        z = weights - learning_rate * gradient
        
        # 3. Proximal step (soft thresholding)
        weights_new = soft_threshold(z, threshold)
        
        # Compute costs for tracking
        l1_penalty = lambda_l1 * np.sum(np.abs(weights_new))
        total_cost = smooth_cost + l1_penalty
        
        # Track sparsity (how many weights are exactly zero)
        sparsity = np.sum(np.abs(weights_new) < 1e-10)
        
        # Store history
        cost_history.append(total_cost)
        smooth_cost_history.append(smooth_cost)
        l1_penalty_history.append(l1_penalty)
        sparsity_history.append(sparsity)
        
        # Check convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"✅ Converged after {iteration + 1} iterations")
            break
        
        weights = weights_new
        
        # Progress update
        if (iteration + 1) % 200 == 0:
            print(f"   Iteration {iteration + 1}: Total cost = {total_cost:.6f}, "
                  f"Sparsity = {sparsity}/{n_features}")
    
    training_time = time.time() - start_time
    
    if iteration == max_iterations - 1:
        print(f"⚠️ Reached maximum iterations ({max_iterations})")
    
    print(f"⏱️ Training time: {training_time:.3f} seconds")
    print(f"📉 Final total cost: {cost_history[-1]:.6f}")
    print(f"📉 Final smooth cost: {smooth_cost_history[-1]:.6f}")
    print(f"📏 Final L1 penalty: {l1_penalty_history[-1]:.6f}")
    print(f"🎯 Final sparsity: {sparsity_history[-1]}/{n_features} weights = 0")
    print(f"📊 Sparsity ratio: {sparsity_history[-1]/n_features*100:.1f}%")
    
    return (weights, cost_history, smooth_cost_history, l1_penalty_history, 
            sparsity_history, training_time)

def evaluate_model(weights, X_test, y_test, lambda_l1):
    """Đánh giá model với L1 regularization"""
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
    sparsity_count = np.sum(np.abs(weights) < 1e-10)
    sparsity_ratio = sparsity_count / len(weights)
    l1_norm = np.sum(np.abs(weights))
    
    # Total cost with L1 penalty
    total_cost = mse + lambda_l1 * l1_norm
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'sparsity_count': sparsity_count,
        'sparsity_ratio': sparsity_ratio,
        'l1_norm': l1_norm,
        'total_cost': total_cost,
        'active_features': len(weights) - sparsity_count
    }

def plot_results(cost_history, smooth_cost_history, l1_penalty_history, 
                sparsity_history, weights, X_test, y_test, output_dir):
    """Vẽ các biểu đồ đặc trưng của Proximal GD"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Proximal Gradient Descent - Sparsity & Regularization Analysis', fontsize=16)
    
    # 1. Cost components
    ax1 = axes[0, 0]
    ax1.plot(cost_history, 'b-', linewidth=2, label='Total Cost')
    ax1.plot(smooth_cost_history, 'g-', linewidth=2, label='Smooth Cost (MSE)')
    ax1.plot(l1_penalty_history, 'r-', linewidth=2, label='L1 Penalty')
    ax1.set_title('Cost Components Over Time')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Sparsity evolution
    ax2 = axes[0, 1]
    ax2.plot(sparsity_history, 'purple', linewidth=3)
    ax2.set_title('Sparsity Evolution (Zero Weights)')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Number of Zero Weights')
    ax2.grid(True, alpha=0.3)
    
    # Add final sparsity annotation
    final_sparsity = sparsity_history[-1]
    total_features = len(weights)
    ax2.axhline(y=final_sparsity, color='red', linestyle='--', alpha=0.7)
    ax2.text(0.6, 0.8, f'Final: {final_sparsity}/{total_features}\n({final_sparsity/total_features*100:.1f}%)', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. Weight histogram
    ax3 = axes[0, 2]
    weights_nonzero = weights[np.abs(weights) > 1e-10]
    ax3.hist(weights_nonzero, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_title(f'Non-Zero Weights Distribution\n(Active: {len(weights_nonzero)}/{len(weights)})')
    ax3.set_xlabel('Weight Values')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. Predictions vs Actual
    ax4 = axes[1, 0]
    predictions = X_test.dot(weights)
    ax4.scatter(y_test, predictions, alpha=0.6, color='green')
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax4.set_xlabel('Actual Values')
    ax4.set_ylabel('Predicted Values')
    ax4.set_title('Predictions vs Actual (Sparse Model)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature importance (non-zero weights)
    ax5 = axes[1, 1]
    feature_importance = np.abs(weights)
    sorted_indices = np.argsort(feature_importance)[::-1]
    top_features = sorted_indices[:20]  # Top 20 features
    
    ax5.bar(range(len(top_features)), feature_importance[top_features], 
            color='skyblue', edgecolor='navy', alpha=0.7)
    ax5.set_title('Top 20 Feature Importance (|weights|)')
    ax5.set_xlabel('Feature Index (sorted)')
    ax5.set_ylabel('|Weight|')
    ax5.grid(True, alpha=0.3)
    
    # 6. Regularization path effect
    ax6 = axes[1, 2]
    # Show how L1 penalty affects final sparsity
    iterations = range(len(sparsity_history))
    sparsity_ratio = [s/len(weights)*100 for s in sparsity_history]
    
    ax6.plot(iterations, sparsity_ratio, 'red', linewidth=3)
    ax6.set_title('Sparsity Ratio Evolution')
    ax6.set_xlabel('Iterations')
    ax6.set_ylabel('Sparsity Ratio (%)')
    ax6.grid(True, alpha=0.3)
    
    # Add lambda info
    ax6.text(0.05, 0.95, f'λ = {0.01}', transform=ax6.transAxes,
             bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    plt.tight_layout()
    plt.savefig(output_dir / "standard_setup_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(weights, metrics, cost_history, smooth_cost_history, 
                l1_penalty_history, sparsity_history, training_time, output_dir):
    """Lưu kết quả với Proximal GD analysis"""
    
    # Proximal GD specific analysis
    proximal_analysis = {
        'final_sparsity_count': int(sparsity_history[-1]),
        'final_sparsity_ratio': float(sparsity_history[-1] / len(weights)),
        'sparsity_evolution_rate': float(np.mean(np.diff(sparsity_history))),
        'l1_regularization_effect': float(l1_penalty_history[-1] / smooth_cost_history[-1]),
        'active_features': int(len(weights) - sparsity_history[-1]),
        'feature_selection_efficiency': float(sparsity_history[-1] / len(weights) * 100)
    }
    
    # Find most important features
    feature_importance = np.abs(weights)
    top_feature_indices = np.argsort(feature_importance)[::-1][:10].tolist()
    top_feature_weights = [float(weights[i]) for i in top_feature_indices]
    
    results = {
        'setup_name': 'Standard Setup',
        'algorithm': 'Proximal Gradient Descent',
        'parameters': {
            'learning_rate': 0.01,
            'lambda_l1': 0.01,
            'max_iterations': 1000,
            'tolerance': 1e-6
        },
        'metrics': metrics,
        'training_time': training_time,
        'convergence': {
            'iterations': len(cost_history),
            'final_total_cost': cost_history[-1],
            'final_smooth_cost': smooth_cost_history[-1],
            'final_l1_penalty': l1_penalty_history[-1]
        },
        'proximal_analysis': proximal_analysis,
        'feature_selection': {
            'top_feature_indices': top_feature_indices,
            'top_feature_weights': top_feature_weights
        },
        'notes': {
            'setup_description': 'Standard Proximal GD with L1 regularization for sparsity',
            'pros': ['Automatic feature selection', 'Sparse solutions', 'Handles high-dimensional data'],
            'cons': ['Extra hyperparameter (λ)', 'May remove important features', 'Slower than standard GD'],
            'mathematical_principle': 'Combines gradient descent with proximal operator for non-smooth regularization',
            'key_insight': 'Soft thresholding creates sparsity by setting small weights to exactly zero'
        }
    }
    
    # Save detailed results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save weights and feature info
    np.save(output_dir / "weights.npy", weights)
    
    # Save comprehensive training history
    history_df = pd.DataFrame({
        'iteration': range(len(cost_history)),
        'total_cost': cost_history,
        'smooth_cost': smooth_cost_history,
        'l1_penalty': l1_penalty_history,
        'sparsity_count': sparsity_history
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    # Save feature analysis
    feature_df = pd.DataFrame({
        'feature_index': range(len(weights)),
        'weight': weights,
        'abs_weight': np.abs(weights),
        'is_active': np.abs(weights) > 1e-10
    })
    feature_df = feature_df.sort_values('abs_weight', ascending=False)
    feature_df.to_csv(output_dir / "feature_analysis.csv", index=False)

def print_results(metrics, training_time, iterations):
    """In kết quả với focus trên sparsity và feature selection"""
    print("\n" + "="*60)
    print("🎯 PROXIMAL GRADIENT DESCENT - STANDARD SETUP RESULTS")
    print("="*60)
    print(f"Test MSE:        {metrics['mse']:.6f}")
    print(f"Test RMSE:       {metrics['rmse']:.6f}")
    print(f"Test MAE:        {metrics['mae']:.6f}")
    print(f"R² Score:        {metrics['r2']:.6f}")
    print(f"MAPE:            {metrics['mape']:.2f}%")
    print(f"Total Cost (w/ L1): {metrics['total_cost']:.6f}")
    
    print(f"\n🎯 SPARSITY & FEATURE SELECTION:")
    print(f"   🔍 Active Features: {metrics['active_features']} (selected)")
    print(f"   ❌ Zero Features:   {metrics['sparsity_count']} (removed)")
    print(f"   📊 Sparsity Ratio:  {metrics['sparsity_ratio']*100:.1f}%")
    print(f"   📏 L1 Norm:         {metrics['l1_norm']:.6f}")
    
    # Feature selection assessment
    if metrics['sparsity_ratio'] > 0.5:
        selection_rating = "HIGH SPARSITY"
        color = "🟢"
    elif metrics['sparsity_ratio'] > 0.2:
        selection_rating = "MODERATE SPARSITY"
        color = "🟡"
    else:
        selection_rating = "LOW SPARSITY"
        color = "🔴"
    
    print(f"   {color} Feature Selection: {selection_rating}")
    
    print(f"\n⚙️ ALGORITHM CHARACTERISTICS:")
    print(f"   📐 Learning Rate: 0.01")
    print(f"   🎯 L1 Regularization (λ): 0.01")
    print(f"   ⏱️ Training Time: {training_time:.3f}s")
    print(f"   🔄 Iterations: {iterations}")
    
    print(f"\n🧮 PROXIMAL GRADIENT DESCENT INSIGHTS:")
    print(f"   • Forward step: z = w - α∇f(w)")
    print(f"   • Proximal step: w = soft_threshold(z, λα)")
    print(f"   • Soft threshold tạo sparsity (weights → 0)")
    print(f"   • L1 regularization encourages feature selection")
    print(f"   • Tự động loại bỏ features không quan trọng")
    
    print(f"\n🎯 KHI NÀO DÙNG PROXIMAL GD:")
    print(f"   • High-dimensional data (nhiều features)")
    print(f"   • Cần feature selection tự động")
    print(f"   • Muốn interpretable models")
    print(f"   • Overfitting do too many features")
    print(f"   • Sparse solutions preferred")
    
    print(f"\n💡 SO SÁNH VỚI STANDARD GD:")
    print(f"   • Standard GD: Keeps all features")
    print(f"   • Proximal GD: Automatic feature selection")
    print(f"   • Trade-off: Thêm hyperparameter λ")
    print(f"   • Benefit: Simpler, more interpretable models")

def main():
    """Chạy Proximal Gradient Descent với Standard Setup"""
    print("🎯 PROXIMAL GRADIENT DESCENT - STANDARD SETUP")
    print("Gradient descent with L1 regularization for automatic feature selection")
    
    # Setup
    output_dir = setup_output_dir()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    weights, cost_history, smooth_cost_history, l1_penalty_history, sparsity_history, training_time = proximal_gd_fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(weights, X_test, y_test, lambda_l1=0.01)
    
    # Plot results
    plot_results(cost_history, smooth_cost_history, l1_penalty_history, 
                sparsity_history, weights, X_test, y_test, output_dir)
    
    # Save everything
    save_results(weights, metrics, cost_history, smooth_cost_history, 
                l1_penalty_history, sparsity_history, training_time, output_dir)
    
    # Print results
    print_results(metrics, training_time, len(cost_history))
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()