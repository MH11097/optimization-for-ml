from src.utils.data_loader import load_data_chunked
#!/usr/bin/env python3
"""
03. Gradient Descent Algorithm
Input: data/02_processed/ (training data)
Output: data/03_algorithms/gradient_descent/ (model, metrics, plots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
import time

def setup_output_dir():
    """Create output directory for gradient descent results"""
    output_dir = Path("data/03_algorithms/gradient_descent")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_processed_data():
    """Load processed training data"""
    data_dir = Path("data/02_processed")
    
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Processed data not found: {data_dir / file}")
    
    print("📂 Loading processed data...")
    X_train = load_data_chunked(data_dir / "X_train.csv")
    X_test = load_data_chunked(data_dir / "X_test.csv")
    y_train = load_data_chunked(data_dir / "y_train.csv").values.ravel()
    y_test = load_data_chunked(data_dir / "y_test.csv").values.ravel()
    
    print(f"✅ Loaded: Train {X_train.shape}, Test {X_test.shape}")
    
    return X_train.values, X_test.values, y_train, y_test

def compute_cost(X, y, weights):
    """Compute Mean Squared Error cost"""
    predictions = X.dot(weights)
    errors = predictions - y
    cost = np.mean(errors ** 2)
    return cost

def compute_gradient(X, y, weights):
    """Compute gradient of MSE cost function"""
    n_samples = X.shape[0]
    predictions = X.dot(weights)
    errors = predictions - y
    gradient = (2 / n_samples) * X.T.dot(errors)
    return gradient

def gradient_descent_fit(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Gradient Descent algorithm implementation
    
    Parameters:
    - X: feature matrix (n_samples, n_features)
    - y: target vector (n_samples,)
    - learning_rate: step size for updates
    - max_iterations: maximum number of iterations
    - tolerance: convergence tolerance
    
    Returns:
    - weights: learned weights
    - cost_history: cost at each iteration
    """
    print("🚀 Training Gradient Descent...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max iterations: {max_iterations}")
    
    # Initialize weights
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    cost_history = []
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Compute cost and gradient
        cost = compute_cost(X, y, weights)
        gradient = compute_gradient(X, y, weights)
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Store cost
        cost_history.append(cost)
        
        # Check convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"✅ Converged after {iteration + 1} iterations")
            break
        
        # Progress update
        if (iteration + 1) % 100 == 0:
            print(f"   Iteration {iteration + 1}: Cost = {cost:.6f}")
    
    training_time = time.time() - start_time
    
    if iteration == max_iterations - 1:
        print(f"⚠️  Reached maximum iterations ({max_iterations})")
    
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    print(f"📉 Final cost: {cost_history[-1]:.6f}")
    
    return weights, cost_history, training_time

def gradient_descent_predict(X, weights):
    """Make predictions using learned weights"""
    return X.dot(weights)

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }

def plot_training_curve(cost_history, output_dir):
    """Plot training cost curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.title('Gradient Descent - Training Curve', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true, y_pred, split_name, output_dir):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Gradient Descent - Predictions vs Actual ({split_name})\nR² = {r2:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    filename = f"predictions_{split_name.lower()}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true, y_pred, split_name, output_dir):
    """Plot residuals analysis"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residuals vs Predicted ({split_name})')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residuals Distribution ({split_name})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"residuals_{split_name.lower()}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_model_results(weights, cost_history, train_metrics, test_metrics, training_time, output_dir):
    """Save model weights and results"""
    print("\n💾 SAVING RESULTS")
    print("=" * 50)
    
    # Save model weights
    model_data = {
        'algorithm': 'Gradient Descent',
        'weights': weights.tolist(),
        'cost_history': cost_history,
        'training_time': training_time,
        'converged': len(cost_history) < 1000,  # Assume max_iter = 1000
        'final_cost': cost_history[-1]
    }
    
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # Save metrics
    results = {
        'algorithm': 'Gradient Descent',
        'training_time': training_time,
        'iterations': len(cost_history),
        'final_cost': cost_history[-1],
        'converged': len(cost_history) < 1000,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save simple CSV for comparison
    comparison_data = {
        'algorithm': 'Gradient Descent',
        'train_mse': train_metrics['mse'],
        'test_mse': test_metrics['mse'],
        'train_r2': train_metrics['r2'],
        'test_r2': test_metrics['r2'],
        'training_time': training_time
    }
    
    pd.DataFrame([comparison_data]).to_csv(output_dir / "results_summary.csv", index=False)
    
    print("✅ Saved files:")
    print(f"  - model.pkl: Model weights and training info")
    print(f"  - metrics.json: Detailed metrics")
    print(f"  - results_summary.csv: Summary for comparison")

def print_results(train_metrics, test_metrics, training_time):
    """Print training results"""
    print("\n📊 TRAINING RESULTS")
    print("=" * 50)
    
    print("Training Set Performance:")
    print(f"  MSE:  {train_metrics['mse']:.6f}")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  MAE:  {train_metrics['mae']:.6f}")
    print(f"  R²:   {train_metrics['r2']:.6f}")
    print(f"  MAPE: {train_metrics['mape']:.2f}%")
    
    print("\nTest Set Performance:")
    print(f"  MSE:  {test_metrics['mse']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.6f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    
    print(f"\nTraining Time: {training_time:.2f} seconds")

def main():
    """Main gradient descent pipeline"""
    print("🎯 Gradient Descent Algorithm")
    print("=" * 60)
    
    # Setup
    output_dir = setup_output_dir()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    weights, cost_history, training_time = gradient_descent_fit(
        X_train, y_train,
        learning_rate=0.01,
        max_iterations=1000,
        tolerance=1e-6
    )
    
    # Make predictions
    y_pred_train = gradient_descent_predict(X_train, weights)
    y_pred_test = gradient_descent_predict(X_test, weights)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Print results
    print_results(train_metrics, test_metrics, training_time)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    plot_training_curve(cost_history, output_dir)
    plot_predictions(y_train, y_pred_train, "Train", output_dir)
    plot_predictions(y_test, y_pred_test, "Test", output_dir)
    plot_residuals(y_train, y_pred_train, "Train", output_dir)
    plot_residuals(y_test, y_pred_test, "Test", output_dir)
    
    # Save results
    save_model_results(weights, cost_history, train_metrics, test_metrics, training_time, output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ GRADIENT DESCENT COMPLETED!")
    print("=" * 60)
    print(f"🎯 Test R²: {test_metrics['r2']:.4f}")
    print(f"📉 Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"⏱️  Training Time: {training_time:.2f}s")
    print(f"📁 Results saved to: {output_dir}")
    print("▶️  Next: python src/03_algorithms/newton_method.py")

if __name__ == "__main__":
    main()