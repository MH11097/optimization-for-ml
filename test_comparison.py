#!/usr/bin/env python3
"""
So sánh implementation mới với base implementation
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "algorithms_old", "subgradient"))

from optimization.algorithms.subgradient import SubgradientOptimizer

# Import base implementation (we'll create a simple constant step version)
class SimpleSubgradient:
    """Simplified version based on base_subgradient.py logic"""

    def __init__(self, lambda_penalty=0.1, step_size=0.001, max_iterations=500, tolerance=1e-8):
        self.lambda_penalty = lambda_penalty
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def _compute_cost(self, X, y, weights):
        """Cost function matching base implementation"""
        predictions = X @ weights
        mse = np.mean((y - predictions) ** 2) / 2
        regularization_term = self.lambda_penalty * np.linalg.norm(weights, 1)
        return mse + regularization_term

    def _compute_gradient(self, X, y, weights):
        """Gradient matching base implementation"""
        n_samples = X.shape[0]
        # Gradient of squared loss
        grad = (1 / n_samples) * X.T @ (X @ weights - y)
        # Subgradient of L1 norm
        subgrad = np.sign(weights)
        # Full subgradient
        return grad + self.lambda_penalty * subgrad

    def fit(self, X, y):
        """Simple fit following base logic"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # Minimum loss tracking
        BASE_LOSS_VALUE = 1000
        min_loss_1 = {"iteration": 0, "loss_value": BASE_LOSS_VALUE, "weights": None}
        min_loss_2 = {"iteration": 0, "loss_value": BASE_LOSS_VALUE, "weights": None}

        loss_history = []

        for iteration in range(1, self.max_iterations + 1):
            # Compute loss and gradient
            loss_value = self._compute_cost(X, y, self.weights)
            gradient = self._compute_gradient(X, y, self.weights)

            # Update weights
            self.weights = self.weights - self.step_size * gradient

            loss_history.append(loss_value)

            # Update min loss
            if loss_value < min_loss_1["loss_value"]:
                min_loss_2 = min_loss_1.copy()
                min_loss_1 = {
                    "iteration": iteration,
                    "loss_value": loss_value,
                    "weights": self.weights.copy(),
                }

            # Check convergence (matching base logic)
            if (iteration > 1 and
                min_loss_1["loss_value"] != BASE_LOSS_VALUE and
                abs(min_loss_1["loss_value"] - min_loss_2["loss_value"]) < self.tolerance):
                print(f"Converged after {iteration} iterations")
                break

        # Set final weights to best weights
        if min_loss_1["weights"] is not None:
            self.weights = min_loss_1["weights"].copy()

        return {
            'converged': iteration < self.max_iterations,
            'final_iteration': iteration,
            'final_loss': loss_history[-1] if loss_history else float('inf'),
            'min_loss_1': min_loss_1
        }

    def predict(self, X):
        return X @ self.weights

def compare_implementations():
    """Compare both implementations"""
    print("=== Comparing Implementations ===")

    # Generate same test data
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.randn(50)

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Test with same parameters
    lambda_penalty = 0.05
    step_size = 0.001
    max_iterations = 200
    tolerance = 1e-6

    print(f"Lambda penalty: {lambda_penalty}")
    print(f"Step size: {step_size}")

    # Test simple implementation
    print("\n--- Simple Implementation (Base Logic) ---")
    simple = SimpleSubgradient(
        lambda_penalty=lambda_penalty,
        step_size=step_size,
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    result_simple = simple.fit(X, y)

    print(f"Converged: {result_simple.get('converged', False)}")
    print(f"Final iteration: {result_simple.get('final_iteration', 'N/A')}")
    print(f"Final loss: {result_simple.get('final_loss', 'N/A'):.6f}")
    print(f"Weights: {simple.weights}")
    print(f"Min loss: {result_simple['min_loss_1']['loss_value']:.6f}")

    # Test new implementation
    print("\n--- New Implementation ---")
    optimizer = SubgradientOptimizer(
        ham_loss='lasso',
        lambda_penalty=lambda_penalty,
        regularization=0.0,  # Force to use lambda_penalty only
        step_size_method='constant',
        step_size=step_size,
        max_iterations=max_iterations,
        diem_dung=tolerance
    )
    result_new = optimizer.fit(X, y)

    print(f"Actual lambda_penalty used: {optimizer.lambda_penalty}")
    print(f"Converged: {result_new.get('converged', False)}")
    print(f"Final iteration: {result_new.get('final_iteration', 'N/A')}")
    print(f"Final loss: {result_new.get('final_loss', 'N/A'):.6f}")
    print(f"Weights shape: {optimizer.weights.shape}")
    print(f"Weights: {optimizer.weights}")

    # Compare predictions
    pred_simple = simple.predict(X[:3])
    pred_new = optimizer.predict(X[:3])

    print(f"\n--- Predictions Comparison ---")
    print(f"Simple: {pred_simple}")
    print(f"New:    {pred_new}")
    print(f"Diff:   {np.abs(pred_simple - pred_new)}")
    print(f"Max diff: {np.max(np.abs(pred_simple - pred_new)):.8f}")

if __name__ == "__main__":
    compare_implementations()