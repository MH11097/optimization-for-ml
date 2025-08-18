from src.utils.data_loader import load_data_chunked
#!/usr/bin/env python3
"""
03. Ridge Regression Algorithm
Input: data/02_processed/ (training data)
Output: data/03_algorithms/ridge_regression/ (model, metrics, plots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
import time

def setup_output_dir():
    """Create output directory for Ridge results"""
    output_dir = Path("data/03_algorithms/ridge_regression")
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

def ridge_fit(X, y, lambda_l2=0.01):
    """
    Ridge Regression implementation using Normal Equation
    Solves: min ||y - Xw||^2 + λ||w||^2
    
    Parameters:
    - X: feature matrix (n_samples, n_features)
    - y: target vector (n_samples,)
    - lambda_l2: L2 regularization parameter
    
    Returns:
    - weights: learned weights
    """
    print("🎯 Training Ridge Regression...")
    print(f"   L2 regularization: {lambda_l2}")
    
    start_time = time.time()
    
    n_samples, n_features = X.shape
    
    # Ridge regression solution: w = (X^T X + λI)^(-1) X^T y
    XTX = X.T.dot(X)
    identity = np.eye(n_features)
    regularized_matrix = XTX + lambda_l2 * identity
    
    try:
        # Solve using normal equation
        weights = np.linalg.solve(regularized_matrix, X.T.dot(y))
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if matrix is singular
        print("⚠️  Using pseudo-inverse due to numerical issues")
        weights = np.linalg.pinv(regularized_matrix).dot(X.T.dot(y))
    
    training_time = time.time() - start_time
    
    print(f"⏱️  Training time: {training_time:.4f} seconds")
    print("✅ Ridge regression training completed (direct solution)")
    
    return weights, training_time

def ridge_predict(X, weights):
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

def plot_predictions(y_true, y_pred, split_name, output_dir):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=20, color='purple')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Ridge Regression - Predictions vs Actual ({split_name})\nR² = {r2:.4f}')
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
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=20, color='purple')
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residuals vs Predicted ({split_name})')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='purple')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residuals Distribution ({split_name})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"residuals_{split_name.lower()}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(weights, output_dir):
    """Plot feature importance based on weights"""
    # Get top 15 features by absolute weight
    abs_weights = np.abs(weights)
    top_indices = np.argsort(abs_weights)[-15:]
    
    top_weights = weights[top_indices]
    feature_names = [f"Feature_{i}" for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if w < 0 else 'purple' for w in top_weights]
    bars = plt.barh(range(len(top_weights)), top_weights, color=colors, alpha=0.7)
    
    plt.yticks(range(len(top_weights)), feature_names)
    plt.xlabel('Weight Value')
    plt.title('Top 15 Feature Weights (Ridge Regression)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_regularization_effect(weights, lambda_l2, output_dir):
    """Plot regularization effect on weights"""
    plt.figure(figsize=(10, 6))
    
    # Plot weight distribution
    plt.hist(weights, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    plt.axvline(x=np.mean(weights), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(weights):.4f}')
    
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title(f'Weight Distribution (λ = {lambda_l2})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "weight_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_model_results(weights, train_metrics, test_metrics, training_time, lambda_l2, output_dir):
    """Save model weights and results"""
    print("\n💾 SAVING RESULTS")
    print("=" * 50)
    
    # Save model weights
    model_data = {
        'algorithm': 'Ridge Regression',
        'weights': weights.tolist(),
        'lambda_l2': lambda_l2,
        'training_time': training_time,
        'method': 'normal_equation'
    }
    
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # Save metrics
    results = {
        'algorithm': 'Ridge Regression',
        'training_time': training_time,
        'lambda_l2': lambda_l2,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save simple CSV for comparison
    comparison_data = {
        'algorithm': 'Ridge Regression',
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

def print_results(train_metrics, test_metrics, training_time, lambda_l2):
    """Print training results"""
    print("\n📊 TRAINING RESULTS")
    print("=" * 50)
    
    print(f"Regularization (λ): {lambda_l2}")
    
    print("\nTraining Set Performance:")
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
    
    print(f"\nTraining Time: {training_time:.4f} seconds")

def main():
    """Main ridge regression pipeline"""
    print("🎯 Ridge Regression Algorithm")
    print("=" * 60)
    
    # Setup
    output_dir = setup_output_dir()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    lambda_l2 = 0.01
    weights, training_time = ridge_fit(X_train, y_train, lambda_l2=lambda_l2)
    
    # Make predictions
    y_pred_train = ridge_predict(X_train, weights)
    y_pred_test = ridge_predict(X_test, weights)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Print results
    print_results(train_metrics, test_metrics, training_time, lambda_l2)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    plot_predictions(y_train, y_pred_train, "Train", output_dir)
    plot_predictions(y_test, y_pred_test, "Test", output_dir)
    plot_residuals(y_train, y_pred_train, "Train", output_dir)
    plot_residuals(y_test, y_pred_test, "Test", output_dir)
    plot_feature_importance(weights, output_dir)
    plot_regularization_effect(weights, lambda_l2, output_dir)
    
    # Save results
    save_model_results(weights, train_metrics, test_metrics, training_time, lambda_l2, output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ RIDGE REGRESSION COMPLETED!")
    print("=" * 60)
    print(f"🎯 Test R²: {test_metrics['r2']:.4f}")
    print(f"📉 Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"⏱️  Training Time: {training_time:.4f}s")
    print(f"🔧 L2 Regularization: {lambda_l2}")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()