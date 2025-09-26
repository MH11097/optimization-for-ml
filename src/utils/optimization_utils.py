"""
Optimization Utilities - Backward Compatibility Module
This module provides backward compatibility with the original optimization_utils.py
by re-exporting functions with their original names and signatures.
"""
import numpy as np
from typing import Callable, Optional, Tuple, Any
# Import from refactored modules
from .core.loss_functions import LossFunction
from .core.gradients import GradientFunction
from .core.hessians import HessianFunction
from .core.metrics import MetricsCalculator
from .core.linear_algebra import LinearAlgebraUtils
from .data.loaders import add_bias_column

# Initialize function objects for backward compatibility
_loss_fn = LossFunction()
_grad_fn = GradientFunction()
_hess_fn = HessianFunction()
_metrics = MetricsCalculator()
_linalg = LinearAlgebraUtils()

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================
def tinh_gia_tri_ham_loss(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray,
                         loss_type: str = 'ols', regularization: float = 0.0) -> float:
    """Compute loss function value (backward compatibility)"""
    if loss_type == 'ols':
        return _loss_fn.compute(loss_type, X, y, trong_so)
    else:
        return _loss_fn.compute(loss_type, X, y, trong_so, alpha=regularization)

def tinh_gradient_ham_loss(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray,
                          loss_type: str = 'ols', regularization: float = 0.0) -> Tuple[np.ndarray, float]:
    """Compute gradient (backward compatibility)"""
    if loss_type == 'ols':
        gradient = _grad_fn.compute(loss_type, X, y, trong_so)
    else:
        gradient = _grad_fn.compute(loss_type, X, y, trong_so, alpha=regularization)
    # Return tuple format expected by old code: (full_gradient, dummy_bias)
    # The full gradient vector already includes bias as last element
    # The second element is ignored in the new codebase (just for compatibility)
    return gradient, 0.0

def tinh_hessian_ham_loss(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                         loss_type: str = 'ols', regularization: float = 0.0) -> np.ndarray:
    """Compute Hessian matrix (backward compatibility)"""
    if loss_type == 'ols':
        return _hess_fn.compute(loss_type, X, y, weights)
    else:
        return _hess_fn.compute(loss_type, X, y, weights, alpha=regularization)

def du_doan(X: np.ndarray, w: np.ndarray, bias: float = None) -> np.ndarray:
    """Make predictions (backward compatibility)"""
    if bias is not None:
        # If bias is provided separately, add it
        return X @ w + bias
    else:
        # Assume weights include bias (first element)
        return X @ w

def danh_gia_mo_hinh(weights: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                     bias: float = 0.0) -> dict:
    """Evaluate model performance with comprehensive ML metrics (backward compatibility)"""
    # Import MetricType enum
    from .core.metrics import MetricType
    # FIX: Use consistent prediction method with training
    # Training uses X_with_bias @ full_weights, so evaluation should too
    if bias != 0.0:
        # Old format: separate weights and bias - convert to new format
        X_test_with_bias = add_bias_column(X_test)
        full_weights = np.append(weights, bias)  # Add bias at the end
        y_pred = X_test_with_bias @ full_weights
    else:
        # New format: weights already include bias
        X_test_with_bias = add_bias_column(X_test)
        y_pred = X_test_with_bias @ weights
    # Calculate number of features for adjusted R² (original features, not including bias)
    n_samples, n_features = X_test.shape
    return {
        'mse': _metrics.compute(MetricType.MSE, y_test, y_pred),
        'rmse': _metrics.compute(MetricType.RMSE, y_test, y_pred),
        'mae': _metrics.compute(MetricType.MAE, y_test, y_pred),
        'r2': _metrics.compute(MetricType.R_SQUARED, y_test, y_pred),
        'adjusted_r2': _metrics.compute(MetricType.ADJUSTED_R_SQUARED, y_test, y_pred, n_features=n_features),
        'mape': _metrics.compute(MetricType.MAPE, y_test, y_pred),
        'smape': _metrics.compute(MetricType.SMAPE, y_test, y_pred)
    }

def in_ket_qua_danh_gia(metrics: dict, training_time: float = None,
                       algorithm_name: str = "Model") -> None:
    """Print evaluation results (backward compatibility)"""
    print(f"\n=== {algorithm_name} - EVALUATION RESULTS ===")
    for metric, value in metrics.items():
        print(f"   {metric.upper()}: {value:.6f}")
    if training_time is not None:
        print(f"   TRAINING TIME: {training_time:.2f}s")
    print("=" * 50)

# Additional commonly used functions
def tinh_gradient_hoi_quy_tuyen_tinh(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray,
                                   dieu_chinh: float = 0.0) -> np.ndarray:
    """Compute linear regression gradient (backward compatibility)"""
    return tinh_gradient_ham_loss(X, y, trong_so, 'ols', dieu_chinh)

def tinh_ma_tran_hessian_hoi_quy_tuyen_tinh(X: np.ndarray, dieu_chinh: float = 0.0) -> np.ndarray:
    """Compute linear regression Hessian (backward compatibility)"""
    return tinh_hessian_ham_loss(X, 'ols', dieu_chinh)

def giai_he_phuong_trinh_tuyen_tinh(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve linear system (backward compatibility)"""
    return _linalg.solve_linear_system(A, b)

def kiem_tra_positive_definite(matrix: np.ndarray) -> bool:
    """Check if matrix is positive definite (backward compatibility)"""
    return _linalg.check_positive_definite(matrix)

def tinh_condition_number(matrix: np.ndarray) -> float:
    """Compute condition number (backward compatibility)"""
    return _linalg.compute_condition_number(matrix)

def tinh_mse(y_that: np.ndarray, y_du_doan: np.ndarray) -> float:
    """Compute MSE (backward compatibility)"""
    return _metrics.compute('mse', y_that, y_du_doan)

def tinh_mae(y_that: np.ndarray, y_du_doan: np.ndarray) -> float:
    """Compute MAE (backward compatibility)"""
    return _metrics.compute('mae', y_that, y_du_doan)

def tinh_r2_score(y_that: np.ndarray, y_du_doan: np.ndarray) -> float:
    """Compute R² score (backward compatibility)"""
    return _metrics.compute('r2', y_that, y_du_doan)

def tinh_loss_ols(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray, he_so_tu_do: float = 0) -> float:
    """Compute OLS loss (backward compatibility)"""
    return tinh_gia_tri_ham_loss(X, y, trong_so, 'ols', he_so_tu_do)

def tinh_loss_ridge(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray,
                   he_so_chinh_quy: float = 0.01) -> float:
    """Compute Ridge loss (backward compatibility)"""
    return tinh_gia_tri_ham_loss(X, y, trong_so, 'ridge', he_so_chinh_quy)

def tinh_loss_lasso_smooth(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray,
                          he_so_chinh_quy: float = 0.01) -> float:
    """Compute smooth Lasso loss (backward compatibility)"""
    return tinh_gia_tri_ham_loss(X, y, trong_so, 'lasso', he_so_chinh_quy)

def kiem_tra_dieu_kien_dung(gradient_norm: float, cost_change: float, iteration: int,
                           tolerance: float = 1e-6, max_iterations: int = 100000,
                           loss_value: Optional[float] = None, weights: Optional[np.ndarray] = None,
                           divergence_threshold: float = 1e+5) -> Tuple[bool, bool, str]:
    """
    Check convergence conditions with early divergence detection (backward compatibility)
    
    Args:
        gradient_norm: Norm of the gradient
        cost_change: Change in cost from previous iteration
        iteration: Current iteration number
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations allowed
        loss_value: Current loss value (optional)
        weights: Current weights (optional)
        divergence_threshold: Threshold for early divergence detection (default: 1e+10)
        
    Returns:
        (should_stop, converged, reason): Tuple indicating whether to stop, if converged, and reason
    """
    # 1. Check for NaN/Inf values (existing logic)
    if weights is not None and (np.any(np.isnan(weights)) or np.any(np.isinf(weights))):
        return True, False, "numerical_instability: weights contain NaN/Inf"
    if loss_value is not None and (np.isnan(loss_value) or np.isinf(loss_value)):
        return True, False, "numerical_instability: loss is NaN/Inf"
    
    if np.isnan(gradient_norm) or np.isinf(gradient_norm):
        return True, False, "numerical_instability: gradient_norm is NaN/Inf"
    # 2. Check for early divergence (NEW - values > threshold)
    if loss_value is not None and abs(loss_value) > divergence_threshold:
        return True, False, f"early_divergence: loss |{loss_value:.2e}| > {divergence_threshold:.2e}"
    if gradient_norm > divergence_threshold:
        return True, False, f"early_divergence: gradient_norm {gradient_norm:.2e} > {divergence_threshold:.2e}"
    if weights is not None and np.any(np.abs(weights) > divergence_threshold):
        max_weight = np.max(np.abs(weights))
        return True, False, f"early_divergence: max |weight| {max_weight:.2e} > {divergence_threshold:.2e}"
    # 3. Check max iterations
    if iteration >= max_iterations:
        return True, False, "max_iterations"
    # 4. Check convergence conditions individually
    gradient_converged = gradient_norm < tolerance
    cost_converged = iteration > 0 and abs(cost_change) < tolerance
    # REQUIRE BOTH CONDITIONS TO BE MET SIMULTANEOUSLY (AND logic)
    if gradient_converged and cost_converged:
        return True, True, f"converged_both_conditions: gradient_norm={gradient_norm:.2e} < {tolerance:.2e} AND cost_change={abs(cost_change):.2e} < {tolerance:.2e}"
    # Show current status - continue optimization
    if iteration > 0:
        grad_status = "✓" if gradient_converged else "✗"
        cost_status = "✓" if cost_converged else "✗"
        return False, False, f"continuing: gradient_norm={gradient_norm:.2e} ({grad_status}), cost_change={abs(cost_change):.2e} ({cost_status})"
    else:
        grad_status = "✓" if gradient_converged else "✗"
        return False, False, f"continuing: gradient_norm={gradient_norm:.2e} ({grad_status}), cost_change=N/A (first_iteration)"

def backtracking_line_search(cost_func: Callable, gradient: np.ndarray,
                            weights: np.ndarray, direction: np.ndarray,
                            alpha: float = 1.0, rho: float = 0.5, c1: float = 1e-4) -> float:
    """Backtracking line search (backward compatibility)"""
    # Simple backtracking line search implementation
    current_cost = cost_func(weights)
    directional_derivative = np.dot(gradient, direction)
    while True:
        new_weights = weights + alpha * direction
        new_cost = cost_func(new_weights)
        # Armijo condition
        if new_cost <= current_cost + c1 * alpha * directional_derivative:
            return alpha
        alpha *= rho
        # Safety check
        if alpha < 1e-16:
            return alpha

def check_for_numerical_issues(gradient_norm: float, loss_value: Optional[float] = None,
                              weights: Optional[np.ndarray] = None) -> dict:
    """Check for numerical issues (backward compatibility)"""
    issues = {
        'has_issues': False,
        'gradient_issues': False,
        'loss_issues': False,
        'weight_issues': False,
        'messages': []
    }
    # Check gradient
    if np.isnan(gradient_norm) or np.isinf(gradient_norm):
        issues['gradient_issues'] = True
        issues['has_issues'] = True
        issues['messages'].append("Gradient contains NaN or Inf values")
    # Check loss
    if loss_value is not None:
        if np.isnan(loss_value) or np.isinf(loss_value):
            issues['loss_issues'] = True
            issues['has_issues'] = True
            issues['messages'].append("Loss contains NaN or Inf values")
    # Check weights
    if weights is not None:
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            issues['weight_issues'] = True
            issues['has_issues'] = True
            issues['messages'].append("Weights contain NaN or Inf values")
    return issues

def in_thong_tin_ma_tran(matrix: np.ndarray, name: str = "Matrix") -> None:
    """Print matrix information (backward compatibility)"""
    print(f"\n{name} Information:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Condition number: {tinh_condition_number(matrix):.2e}")
    print(f"  Min eigenvalue: {np.min(np.linalg.eigvals(matrix)):.2e}")
    print(f"  Max eigenvalue: {np.max(np.linalg.eigvals(matrix)):.2e}")

def in_thong_tin_gradient(gradient: np.ndarray) -> None:
    """Print gradient information (backward compatibility)"""
    print(f"\nGradient Information:")
    print(f"  Norm: {np.linalg.norm(gradient):.2e}")
    print(f"  Max component: {np.max(np.abs(gradient)):.2e}")
    print(f"  Mean component: {np.mean(np.abs(gradient)):.2e}")

# Export all backward compatibility functions
__all__ = [
    # Core functions
    'add_bias_column',
    'tinh_gia_tri_ham_loss',
    'tinh_gradient_ham_loss',
    'tinh_hessian_ham_loss',
    'du_doan',
    'danh_gia_mo_hinh',
    'in_ket_qua_danh_gia',
    # Linear regression specific
    'tinh_gradient_hoi_quy_tuyen_tinh',
    'tinh_ma_tran_hessian_hoi_quy_tuyen_tinh',
    'giai_he_phuong_trinh_tuyen_tinh',
    # Matrix operations
    'kiem_tra_positive_definite',
    'tinh_condition_number',
    # Metrics
    'tinh_mse',
    'tinh_mae',
    'tinh_r2_score',
    # Loss functions
    'tinh_loss_ols',
    'tinh_loss_ridge',
    'tinh_loss_lasso_smooth',
    # Optimization utilities
    'kiem_tra_dieu_kien_dung',
    'backtracking_line_search',
    'check_for_numerical_issues',
    # Debug utilities
    'in_thong_tin_ma_tran',
    'in_thong_tin_gradient',
]