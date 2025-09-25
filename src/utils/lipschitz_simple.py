"""
Simple Lipschitz Constant Calculator
====================================

Tính Lipschitz constant và suggest optimal learning rate
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

def compute_lipschitz_constant_and_optimal_lr(
    loss_type: str = 'ols',
    regularization: float = 0.01,
    verbose: bool = True
) -> Dict:
    """
    Tính Lipschitz constant và suggest learning rate optimal

    Args:
        loss_type: 'ols', 'ridge'
        regularization: Regularization parameter for Ridge
        verbose: Print results

    Returns:
        dict with Lipschitz constant and optimal learning rates
    """
    if verbose:
        print(f"🔍 Computing Lipschitz constant for {loss_type.upper()} loss...")

    try:
        # Load data using existing function
        from utils.data_process_utils import load_du_lieu
        from utils.data.loaders import add_bias_column

        # Load data
        X_train, X_test, y_train, y_test = load_du_lieu()
        # Add bias column
        X_train = add_bias_column(X_train)

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data for demonstration...")

        # Create synthetic data
        np.random.seed(42)
        n_samples, n_features = 1000, 10
        X_train = np.random.randn(n_samples, n_features)
        y_train = X_train @ np.random.randn(n_features) + 0.1 * np.random.randn(n_samples)
        X_test = np.random.randn(200, n_features)
        y_test = X_test @ np.random.randn(n_features) + 0.1 * np.random.randn(200)

        # Add bias column manually
        X_train = np.column_stack([X_train, np.ones(len(X_train))])
        print(f"   Created synthetic data: {X_train.shape}")

    if verbose:
        print(f"   Data shape: {X_train.shape}")
        print(f"   Samples (n): {len(y_train)}")
        print(f"   Features (p): {X_train.shape[1]}")

    # Compute Hessian matrix manually
    if loss_type == 'ols':
        # H = (1/n) * X^T * X
        H = X_train.T @ X_train / len(y_train)
    elif loss_type == 'ridge':
        # H = (1/n) * X^T * X + 2*alpha * I
        H = X_train.T @ X_train / len(y_train)
        # Add regularization (don't regularize bias - last element)
        reg_matrix = 2 * regularization * np.eye(H.shape[0])
        reg_matrix[-1, -1] = 0  # Don't regularize bias
        H = H + reg_matrix
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(H)
    eigenvals_real = eigenvals.real

    # Lipschitz constant = max eigenvalue
    L = np.max(eigenvals_real)
    min_eigenval = np.min(eigenvals_real)
    condition_number = L / min_eigenval if min_eigenval > 1e-12 else np.inf

    # Optimal learning rates
    conservative_lr = 1.0 / L  # Guaranteed convergence
    aggressive_lr = 2.0 / L    # Fastest convergence

    # Results
    results = {
        'lipschitz_constant': float(L),
        'conservative_lr': float(conservative_lr),
        'aggressive_lr': float(aggressive_lr),
        'max_eigenval': float(L),
        'min_eigenval': float(min_eigenval),
        'condition_number': float(condition_number),
        'data_shape': X_train.shape,
        'loss_type': loss_type,
        'regularization': regularization if loss_type == 'ridge' else None
    }

    if verbose:
        print_results(results)

    return results


def print_results(results: Dict):
    """Print analysis results"""
    L = results['lipschitz_constant']
    conservative_lr = results['conservative_lr']
    aggressive_lr = results['aggressive_lr']
    condition_number = results['condition_number']

    print(f"\n📊 Lipschitz Analysis Results for {results['loss_type'].upper()}:")
    print("=" * 55)

    # Core results
    print(f"🎯 Lipschitz constant (L):     {L:.6f}")
    print(f"📐 Conservative LR (1/L):      {conservative_lr:.6f}")
    print(f"⚡ Aggressive LR (2/L):        {aggressive_lr:.6f}")
    print(f"🔢 Condition number:           {condition_number:.2e}")

    # Eigenvalue analysis
    print(f"\n📈 Eigenvalue Analysis:")
    print(f"   Maximum eigenvalue:         {L:.6f}")
    print(f"   Minimum eigenvalue:         {results['min_eigenval']:.6f}")

    # Recommendations
    print(f"\n💡 Learning Rate Recommendations:")
    print(f"   For guaranteed convergence: α = {conservative_lr:.6f}")
    print(f"   For fastest convergence:    α = {aggressive_lr:.6f}")
    print(f"   Safe range:                 α ∈ [0.001, {conservative_lr:.6f}]")

    if results['loss_type'] == 'ridge':
        print(f"   Regularization λ:           {results['regularization']}")
        print(f"   Note: Ridge regularization increases Lipschitz constant")


def compare_ols_vs_ridge(regularizations: list = None):
    """Compare OLS vs Ridge with different regularizations"""
    if regularizations is None:
        regularizations = [0.001, 0.01, 0.1]

    print("🔄 Comparing OLS vs Ridge:")
    print("=" * 50)

    # Compute OLS baseline
    ols_results = compute_lipschitz_constant_and_optimal_lr('ols', verbose=False)
    ols_L = ols_results['lipschitz_constant']
    ols_lr = ols_results['conservative_lr']

    print(f"📊 OLS Baseline:")
    print(f"   Lipschitz L:     {ols_L:.6f}")
    print(f"   Optimal LR:      {ols_lr:.6f}")

    print(f"\n📊 Ridge Comparison:")
    for reg in regularizations:
        ridge_results = compute_lipschitz_constant_and_optimal_lr('ridge', reg, verbose=False)
        ridge_L = ridge_results['lipschitz_constant']
        ridge_lr = ridge_results['conservative_lr']

        improvement_L = ridge_L / ols_L
        improvement_lr = ols_lr / ridge_lr

        print(f"   λ = {reg}:")
        print(f"     Lipschitz L:   {ridge_L:.6f} ({improvement_L:.2f}x vs OLS)")
        print(f"     Optimal LR:    {ridge_lr:.6f} ({improvement_lr:.2f}x smaller)")
        print(f"     Condition #:   {ridge_results['condition_number']:.2e}")


if __name__ == "__main__":
    print("🚀 Simple Lipschitz Constant Analysis Tool")
    print("=" * 50)

    # Test OLS
    print("\n1️⃣ Testing OLS Loss:")
    ols_results = compute_lipschitz_constant_and_optimal_lr('ols')

    # Test Ridge
    print("\n2️⃣ Testing Ridge Loss:")
    ridge_results = compute_lipschitz_constant_and_optimal_lr('ridge', 0.01)

    # Comparison
    print("\n3️⃣ OLS vs Ridge Comparison:")
    compare_ols_vs_ridge([0.001, 0.01, 0.1])

    print(f"\n✅ Analysis complete!")
    print(f"💡 Use α = {ols_results['conservative_lr']:.6f} for OLS")
    print(f"💡 Use α = {ridge_results['conservative_lr']:.6f} for Ridge (λ=0.01)")