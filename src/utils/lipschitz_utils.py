"""
Lipschitz Constant Utilities
============================

This module provides utilities for computing Lipschitz constants of loss functions
and suggesting optimal learning rates for gradient descent algorithms.

Theory:
- For OLS: L = lambda_max(X^T X / n)
- For Ridge: L = lambda_max(X^T X / n + 2α I)
- Optimal learning rate: α ≤ 2/L (for strongly convex functions)
- Conservative learning rate: α = 1/L (guaranteed convergence)

Author: Optimization for ML Project
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Import with fallback for both relative and absolute imports
try:
    from .optimization_utils import tinh_hessian_ham_loss
    from .data_process_utils import load_and_split_data
    from .data.loaders import add_bias_column
except ImportError:
    # Fallback for direct execution
    from utils.optimization_utils import tinh_hessian_ham_loss
    from utils.data_process_utils import load_and_split_data
    from utils.data.loaders import add_bias_column


def compute_lipschitz_constant_and_optimal_lr(
    loss_type: str = 'ols',
    regularization: float = 0.01,
    verbose: bool = True
) -> Dict:
    """
    Tính Lipschitz constant và suggest learning rate optimal

    Args:
        loss_type: Type of loss function ('ols', 'ridge', 'lasso')
        regularization: Regularization parameter (for Ridge/Lasso)
        verbose: Whether to print detailed results

    Returns:
        dict: {
            'lipschitz_constant': Lipschitz constant L,
            'conservative_lr': α = 1/L (guaranteed convergence),
            'aggressive_lr': α = 2/L (fastest convergence),
            'max_eigenval': Maximum eigenvalue,
            'min_eigenval': Minimum eigenvalue,
            'condition_number': Condition number of Hessian,
            'eigenvals_stats': Statistics of eigenvalues,
            'theory_bounds': Theoretical analysis
        }

    Example:
        >>> results = compute_lipschitz_constant_and_optimal_lr('ols')
        >>> print(f"Optimal LR: {results['conservative_lr']:.6f}")
    """
    if verbose:
        print(f"🔍 Computing Lipschitz constant for {loss_type.upper()} loss...")

    # Load và preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_bias = add_bias_column(X_train)

    if verbose:
        print(f"   Data shape: {X_train_bias.shape}")
        print(f"   Samples (n): {len(y_train)}")
        print(f"   Features (p): {X_train_bias.shape[1]}")

    # Compute Hessian matrix
    H = tinh_hessian_ham_loss(X_train_bias, loss_type, regularization)

    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(H)
    eigenvals_real = eigenvals.real

    # Sort eigenvalues for analysis
    eigenvals_sorted = np.sort(eigenvals_real)

    # Lipschitz constant = max eigenvalue
    L = np.max(eigenvals_real)
    min_eigenval = np.min(eigenvals_real)
    condition_number = L / min_eigenval if min_eigenval > 1e-12 else np.inf

    # Optimal learning rates based on theory
    conservative_lr = 1.0 / L  # Guaranteed convergence
    aggressive_lr = 2.0 / L    # Fastest convergence (for strongly convex)

    # Eigenvalue statistics
    eigenvals_stats = {
        'mean': float(np.mean(eigenvals_real)),
        'std': float(np.std(eigenvals_real)),
        'median': float(np.median(eigenvals_real)),
        'q25': float(np.percentile(eigenvals_real, 25)),
        'q75': float(np.percentile(eigenvals_real, 75)),
        'range': float(L - min_eigenval)
    }

    # Theoretical analysis
    theory_bounds = {
        'is_positive_definite': bool(min_eigenval > 1e-12),
        'is_well_conditioned': bool(condition_number < 1e12),
        'spectral_radius': float(L),
        'recommended_lr_range': (float(conservative_lr), float(aggressive_lr)),
        'convergence_rate_estimate': float(1 - 2*min_eigenval/L) if min_eigenval > 0 else None
    }

    results = {
        'lipschitz_constant': float(L),
        'conservative_lr': float(conservative_lr),
        'aggressive_lr': float(aggressive_lr),
        'max_eigenval': float(L),
        'min_eigenval': float(min_eigenval),
        'condition_number': float(condition_number),
        'eigenvals_stats': eigenvals_stats,
        'eigenvals_full': eigenvals_real.tolist(),
        'hessian_shape': H.shape,
        'theory_bounds': theory_bounds
    }

    if verbose:
        print_lipschitz_analysis(results, loss_type, regularization)

    return results


def print_lipschitz_analysis(results: Dict, loss_type: str, regularization: float):
    """Print detailed Lipschitz analysis results"""
    L = results['lipschitz_constant']
    conservative_lr = results['conservative_lr']
    aggressive_lr = results['aggressive_lr']
    condition_number = results['condition_number']
    min_eigenval = results['min_eigenval']

    print(f"\n📊 Lipschitz Analysis Results for {loss_type.upper()}:")
    print("=" * 55)

    # Core results
    print(f"🎯 Lipschitz constant (L):     {L:.6f}")
    print(f"📐 Conservative LR (1/L):      {conservative_lr:.6f}")
    print(f"⚡ Aggressive LR (2/L):        {aggressive_lr:.6f}")
    print(f"🔢 Condition number:           {condition_number:.2e}")

    # Eigenvalue analysis
    print(f"\n📈 Eigenvalue Analysis:")
    print(f"   Maximum eigenvalue:         {L:.6f}")
    print(f"   Minimum eigenvalue:         {min_eigenval:.6f}")
    print(f"   Eigenvalue range:           [{min_eigenval:.6f}, {L:.6f}]")

    stats = results['eigenvals_stats']
    print(f"   Mean eigenvalue:            {stats['mean']:.6f}")
    print(f"   Median eigenvalue:          {stats['median']:.6f}")
    print(f"   Standard deviation:         {stats['std']:.6f}")

    # Theoretical insights
    theory = results['theory_bounds']
    print(f"\n🧮 Theoretical Analysis:")
    print(f"   Positive definite:          {'✅' if theory['is_positive_definite'] else '❌'}")
    print(f"   Well-conditioned:           {'✅' if theory['is_well_conditioned'] else '❌'}")

    if theory['convergence_rate_estimate']:
        conv_rate = theory['convergence_rate_estimate']
        print(f"   Est. convergence rate:      {conv_rate:.4f}")

    # Recommendations
    print(f"\n💡 Learning Rate Recommendations:")
    print(f"   For guaranteed convergence: α = {conservative_lr:.6f}")
    print(f"   For fastest convergence:    α = {aggressive_lr:.6f}")
    print(f"   Safe range:                 α ∈ [0.001, {conservative_lr:.6f}]")

    if loss_type == 'ridge':
        print(f"   Regularization α:           {regularization}")
        print(f"   Note: Ridge regularization increases Lipschitz constant")


def compare_lipschitz_constants(
    loss_types: list = None,
    regularizations: list = None,
    verbose: bool = True
) -> Dict:
    """
    So sánh Lipschitz constants giữa các loss functions khác nhau

    Args:
        loss_types: List of loss types to compare
        regularizations: List of regularization values for Ridge
        verbose: Whether to print comparison

    Returns:
        dict: Comparison results
    """
    if loss_types is None:
        loss_types = ['ols', 'ridge']

    if regularizations is None:
        regularizations = [0.001, 0.01, 0.1]

    results = {}

    # Compute OLS baseline
    if 'ols' in loss_types:
        if verbose:
            print("Computing OLS baseline...")
        results['ols'] = compute_lipschitz_constant_and_optimal_lr('ols', verbose=False)

    # Compute Ridge with different regularizations
    if 'ridge' in loss_types:
        results['ridge'] = {}
        for reg in regularizations:
            if verbose:
                print(f"Computing Ridge with α={reg}...")
            results['ridge'][f'alpha_{reg}'] = compute_lipschitz_constant_and_optimal_lr(
                'ridge', reg, verbose=False
            )

    if verbose:
        print_comparison_results(results)

    return results


def print_comparison_results(results: Dict):
    """Print comparison of different loss functions"""
    print(f"\n🔄 Lipschitz Constant Comparison:")
    print("=" * 60)

    if 'ols' in results:
        ols_L = results['ols']['lipschitz_constant']
        ols_lr = results['ols']['conservative_lr']
        print(f"📊 OLS:")
        print(f"   Lipschitz L:     {ols_L:.6f}")
        print(f"   Optimal LR:      {ols_lr:.6f}")

        if 'ridge' in results:
            print(f"\n📊 Ridge Comparison (vs OLS):")
            for key, ridge_result in results['ridge'].items():
                reg_val = key.replace('alpha_', '')
                ridge_L = ridge_result['lipschitz_constant']
                ridge_lr = ridge_result['conservative_lr']

                improvement_L = ridge_L / ols_L
                improvement_lr = ols_lr / ridge_lr

                print(f"   α = {reg_val}:")
                print(f"     Lipschitz L:   {ridge_L:.6f} ({improvement_L:.2f}x vs OLS)")
                print(f"     Optimal LR:    {ridge_lr:.6f} ({improvement_lr:.2f}x smaller)")
                print(f"     Condition #:   {ridge_result['condition_number']:.2e}")


# Example usage
if __name__ == "__main__":
    print("🚀 Lipschitz Constant Analysis Tool")
    print("=" * 50)

    # Test với OLS
    print("\n1️⃣ Testing OLS Loss:")
    ols_results = compute_lipschitz_constant_and_optimal_lr('ols')

    # Test với Ridge
    print("\n2️⃣ Testing Ridge Loss:")
    ridge_results = compute_lipschitz_constant_and_optimal_lr('ridge', 0.01)

    # Comparison
    print("\n3️⃣ Comprehensive Comparison:")
    comparison = compare_lipschitz_constants(
        loss_types=['ols', 'ridge'],
        regularizations=[0.001, 0.01, 0.1]
    )

    print(f"\n✅ Analysis complete! Use the suggested learning rates in your optimization algorithms.")