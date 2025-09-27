#!/usr/bin/env python3
"""
Test script để kiểm tra subgradient implementation sau khi sửa chữa
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from optimization.algorithms.subgradient import SubgradientOptimizer
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def test_subgradient_implementation():
    """Test the corrected subgradient implementation"""
    print("=== Testing Corrected Subgradient Implementation ===")

    # Generate test data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Standardize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    print(f"Data shape: X={X_scaled.shape}, y={y_scaled.shape}")

    # Test different step size strategies
    step_strategies = ['constant', 'square_summable', 'non_summable_diminishing']

    for strategy in step_strategies:
        print(f"\n--- Testing {strategy} step size strategy ---")

        try:
            # Create optimizer with strategy-specific parameters
            if strategy == 'constant':
                optimizer = SubgradientOptimizer(
                    ham_loss='lasso',
                    lambda_penalty=0.1,
                    step_size_method=strategy,
                    step_size=0.001,  # Smaller step size
                    max_iterations=500,
                    diem_dung=1e-6
                )
            elif strategy == 'square_summable':
                optimizer = SubgradientOptimizer(
                    ham_loss='lasso',
                    lambda_penalty=0.1,
                    step_size_method=strategy,
                    square_summable_a=0.1,  # Smaller initial step
                    square_summable_b=1.0,  # Add offset
                    max_iterations=500,
                    diem_dung=1e-6
                )
            elif strategy == 'non_summable_diminishing':
                optimizer = SubgradientOptimizer(
                    ham_loss='lasso',
                    lambda_penalty=0.1,
                    step_size_method=strategy,
                    non_summable_a=0.1,  # Smaller initial step
                    max_iterations=500,
                    diem_dung=1e-6
                )

            # Fit model
            result = optimizer.fit(X_scaled, y_scaled)

            # Check results
            print(f"Converged: {result.get('converged', False)}")
            print(f"Final iteration: {result.get('final_iteration', 'N/A')}")
            print(f"Final loss: {result.get('final_loss', 'N/A'):.6f}")

            # Check weights sparsity
            weights = optimizer.weights
            sparsity = np.sum(np.abs(weights) < 1e-3) / len(weights)
            print(f"Sparsity level: {sparsity:.3f}")
            print(f"Non-zero weights: {np.sum(np.abs(weights) >= 1e-3)}")

            # Test prediction
            predictions = optimizer.predict(X_scaled[:5])
            print(f"Sample predictions: {predictions[:3]}")

        except Exception as e:
            print(f"ERROR in {strategy}: {e}")

    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_subgradient_implementation()