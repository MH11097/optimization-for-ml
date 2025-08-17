#!/usr/bin/env python3
"""
Gradient Descent - Medium Setup
Learning Rate: 0.01 (medium)
Max Iterations: 1000
Tolerance: 1e-5

Đặc điểm:
- Cân bằng giữa tốc độ và ổn định
- Learning rate vừa phải
- Số iterations hợp lý
- Setup tiêu chuẩn cho hầu hết trường hợp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import psutil
import os
from src.utils.data_loader import load_data_chunked
from src.algorithms.core.results_manager import save_algorithm_results
from src.algorithms.core.car_price_metrics import calculate_price_metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

def gradient_descent_fit(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-5):
    """
    Medium Gradient Descent Implementation - Balanced
    
    Setup Parameters:
    - learning_rate: 0.01 (medium, balanced)
    - max_iterations: 1000 (reasonable)
    - tolerance: 1e-5 (standard)
    """
    print("⚖️ Training Medium Gradient Descent...")
    print(f"   Learning rate: {learning_rate} (MEDIUM - balanced)")
    print(f"   Max iterations: {max_iterations} (reasonable)")
    print(f"   Tolerance: {tolerance} (standard)")
    
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
        
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"   Iteration {iteration:4d}: Cost = {cost:.6f}, Gradient norm = {gradient_norms[-1]:.6f}")
        
        # Convergence check
        if len(gradient_norms) > 1 and gradient_norms[-1] < tolerance:
            print(f"   ✅ Converged at iteration {iteration} (gradient norm < {tolerance})")
            break
    
    training_time = time.time() - start_time
    
    print(f"⏱️ Training completed in {training_time:.3f}s")
    print(f"📊 Final cost: {cost:.6f}")
    print(f"📈 Total iterations: {len(cost_history)}")
    
    return weights, cost_history, gradient_norms, training_time

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    """Chạy Gradient Descent với Medium Setup - Cân bằng"""
    print("⚖️ GRADIENT DESCENT - MEDIUM SETUP")
    print("Balanced learning rate configuration")
    
    # Memory monitoring
    initial_memory = get_memory_usage()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Configuration
    config = {
        'algorithm': 'gradient_descent',
        'setup': 'medium_setup',
        'learning_rate': 0.01,
        'max_iterations': 1000,
        'tolerance': 1e-5,
        'loss_function': 'mse',
        'description': 'Balanced learning rate for general use'
    }
    
    # Train model
    weights, cost_history, gradient_norms, training_time = gradient_descent_fit(X_train, y_train)
    
    # Calculate peak memory usage
    peak_memory = get_memory_usage()
    memory_usage = peak_memory - initial_memory
    
    # Standard evaluation
    train_pred = X_train.dot(weights)
    test_pred = X_test.dot(weights)
    
    # Standard ML metrics
    metrics = {
        'final_train_mse': float(mean_squared_error(y_train, train_pred)),
        'final_test_mse': float(mean_squared_error(y_test, test_pred)),
        'final_train_r2': float(r2_score(y_train, train_pred)),
        'final_test_r2': float(r2_score(y_test, test_pred)),
        'final_train_mae': float(mean_absolute_error(y_train, train_pred)),
        'final_test_mae': float(mean_absolute_error(y_test, test_pred)),
        'convergence_iteration': len(cost_history),
        'final_gradient_norm': float(gradient_norms[-1]) if gradient_norms else 0.0
    }
    
    # Car price specific metrics
    car_price_metrics = calculate_price_metrics(y_test, test_pred)
    
    # Training history
    training_history = pd.DataFrame({
        'iteration': range(len(cost_history)),
        'train_loss': cost_history,
        'gradient_norm': gradient_norms,
        'test_loss': [mean_squared_error(y_test, X_test.dot(weights)) for _ in cost_history]
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
        setup='medium_setup',
        config=config,
        training_history=training_history,
        predictions=predictions,
        model_weights=weights,
        metrics=metrics,
        car_price_metrics=car_price_metrics,
        training_time=training_time,
        memory_usage=memory_usage
    )
    
    # Print results
    print("\n" + "="*50)
    print("⚖️ MEDIUM SETUP - EVALUATION RESULTS")
    print("="*50)
    print(f"Test MSE:  {metrics['final_test_mse']:.6f}")
    print(f"Test R²:   {metrics['final_test_r2']:.4f}")
    print(f"Test MAE:  {metrics['final_test_mae']:.6f}")
    
    print(f"\n⚖️ BALANCED CHARACTERISTICS:")
    print(f"   📊 Learning Rate: 0.01 (MEDIUM - balanced)")
    print(f"   ⏱️ Training Time: {training_time:.3f}s (reasonable)")
    print(f"   🎯 Convergence: {len(cost_history)} iterations")
    
    print(f"\n📊 Car Price Metrics:")
    print(f"   💰 Mean Absolute Error: ${car_price_metrics['mean_absolute_error_dollars']:,.0f}")
    print(f"   📍 Predictions within 10%: {car_price_metrics['predictions_within_10pct']:.1%}")
    print(f"   🎯 Predictions within $5K: {car_price_metrics['predictions_within_5000_dollars']:.1%}")
    
    print(f"\n✅ Standardized results saved to: {output_dir}")

if __name__ == "__main__":
    main()