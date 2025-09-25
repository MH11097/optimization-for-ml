"""
Unified Hessian Function Implementation
This module provides a single, clean interface for all Hessian computations,
replacing duplicate Hessian functions in the original utils.
Mathematical Formulations:
- OLS Hessian: ∇²L = (1/n) * X^T * X
- Ridge Hessian: ∇²L = (1/n) * X^T * X + 2α * I
- Lasso Hessian: ∇²L ≈ (1/n) * X^T * X (L1 term is non-differentiable)
- Huber Hessian: ∇²L = (1/n) * X^T * diag(huber_weight) * X
Note: For Newton methods, the Hessian represents the curvature of the loss function.
"""
import numpy as np
from typing import Union, Optional, Literal
from abc import ABC, abstractmethod
# Type aliases
LossType = Literal['ols', 'ridge', 'lasso', 'elastic_net', 'huber']
ArrayLike = Union[np.ndarray, list]

class BaseHessian(ABC):
    """Abstract base class for Hessian computations."""
    
    @abstractmethod
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix."""
        pass
    
    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Validate input arrays for Hessian computation."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1D array, got {weights.ndim}D")
        if X.shape[0] != len(y):
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {len(y)}")
        if X.shape[1] != len(weights):
            raise ValueError(f"X and weights dimensions mismatch: {X.shape[1]} vs {len(weights)}")

class OLSHessian(BaseHessian):
    """
    Ordinary Least Squares Hessian
    
    ∇²L(w) = (1/n) * X^T * X
    
    The Hessian is constant and independent of weights for OLS.
    It represents the curvature of the quadratic loss function.
    """
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        # For OLS, Hessian is simply X^T * X / n
        hessian = X.T @ X / len(y)
        
        return hessian

class RidgeHessian(BaseHessian):
    """
    Ridge Regression Hessian (L2 Regularization)
    
    ∇²L(w) = (1/n) * X^T * X + 2α * I
    
    Adds regularization to the diagonal, improving condition number
    and making the Hessian positive definite.
    """
    
    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError("Ridge alpha must be non-negative")
        self.alpha = alpha
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        # OLS Hessian term
        ols_hessian = X.T @ X / len(y)
        
        # L2 regularization term (don't regularize bias)
        n_features = len(weights)
        reg_hessian = np.zeros((n_features, n_features))
        
        if n_features > 1:
            # Assume bias is last element - don't regularize it
            reg_hessian[:-1, :-1] = 2 * self.alpha * np.eye(n_features - 1)
        else:
            reg_hessian = 2 * self.alpha * np.eye(n_features)
        
        return ols_hessian + reg_hessian

class LassoHessian(BaseHessian):
    """
    Lasso Regression Hessian (L1 Regularization)
    
    ∇²L(w) ≈ (1/n) * X^T * X
    
    The L1 penalty is not twice differentiable, so we use the OLS Hessian
    as an approximation. This is commonly done in practice for Newton methods.
    """
    
    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError("Lasso alpha must be non-negative")
        self.alpha = alpha
        
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        # For Lasso, we typically use the OLS Hessian as approximation
        # since L1 penalty is not twice differentiable
        hessian = X.T @ X / len(y)
        
        return hessian

class ElasticNetHessian(BaseHessian):
    """
    Elastic Net Hessian (L1 + L2 Regularization)
    
    ∇²L(w) ≈ (1/n) * X^T * X + 2α₂ * I
    
    Uses OLS Hessian plus L2 regularization term.
    L1 term is approximated by ignoring non-differentiable part.
    """
    
    def __init__(self, alpha_l1: float = 0.01, alpha_l2: float = 0.01):
        if alpha_l1 < 0 or alpha_l2 < 0:
            raise ValueError("ElasticNet alphas must be non-negative")
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        # OLS Hessian term
        ols_hessian = X.T @ X / len(y)
        
        # L2 regularization term only (L1 is non-differentiable)
        n_features = len(weights)
        reg_hessian = np.zeros((n_features, n_features))
        
        if n_features > 1:
            # Don't regularize bias (assume it's last element)
            reg_hessian[:-1, :-1] = 2 * self.alpha_l2 * np.eye(n_features - 1)
        else:
            reg_hessian = 2 * self.alpha_l2 * np.eye(n_features)
        
        return ols_hessian + reg_hessian

class HuberHessian(BaseHessian):
    """
    Huber Loss Hessian (Robust Hessian)
    
    ∇²L(w) = (1/n) * X^T * diag(huber_weight) * X
    
    where huber_weight(r, δ) = {
        1           if |r| ≤ δ
        0           if |r| > δ
    }
    
    The Hessian depends on the residuals and provides robust curvature information.
    """
    
    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError("Huber delta must be positive")
        self.delta = delta
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        predictions = X @ weights
        residuals = predictions - y
        abs_residuals = np.abs(residuals)
        
        # Huber weights: 1 for small residuals, 0 for large ones
        # This makes the Hessian adaptive to outliers
        huber_weights = (abs_residuals <= self.delta).astype(float)
        
        # Weighted Hessian
        W = np.diag(huber_weights)
        hessian = X.T @ W @ X / len(y)
        
        return hessian

class HessianFunction:
    """
    Unified Hessian Function Interface
    
    This class provides a single interface for all Hessian computations,
    replacing scattered Hessian functions in the original utils.
    
    Example:
        hess_fn = HessianFunction()
        
        # OLS Hessian
        hessian = hess_fn.compute('ols', X, y, weights)
        
        # Ridge Hessian with regularization
        hessian = hess_fn.compute('ridge', X, y, weights, alpha=0.01)
        
        # Custom Hessian object
        ridge_hess = hess_fn.get_hessian_object('ridge', alpha=0.1)
        hessian = ridge_hess.compute(X, y, weights)
    """
    
    def __init__(self):
        self._hessian_registry = {
            'ols': OLSHessian,
            'ridge': RidgeHessian,
            'lasso': LassoHessian,
            'elastic_net': ElasticNetHessian,
            'huber': HuberHessian
        }
    
    def compute(self, 
                loss_type: LossType, 
                X: np.ndarray, 
                y: np.ndarray, 
                weights: np.ndarray,
                **kwargs) -> np.ndarray:
        """
        Compute Hessian for specified loss type and parameters.
        
        Args:
            loss_type: Type of loss function
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            weights: Weight vector (n_features,)
            **kwargs: Loss-specific parameters (alpha, delta, etc.)
        
        Returns:
            Hessian matrix (n_features, n_features)
        
        Raises:
            ValueError: If loss_type is unknown or inputs are invalid
        """
        if loss_type not in self._hessian_registry:
            available = list(self._hessian_registry.keys())
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
        
        hessian_class = self._hessian_registry[loss_type]
        
        # Create Hessian object with parameters
        if kwargs:
            hessian_obj = hessian_class(**kwargs)
        else:
            hessian_obj = hessian_class()
        
        return hessian_obj.compute(X, y, weights)
    
    def get_hessian_object(self, loss_type: LossType, **kwargs) -> BaseHessian:
        """
        Get a configured Hessian object for repeated use.
        
        Args:
            loss_type: Type of loss function
            **kwargs: Loss-specific parameters
        
        Returns:
            Configured Hessian object
        """
        if loss_type not in self._hessian_registry:
            available = list(self._hessian_registry.keys())
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
        
        hessian_class = self._hessian_registry[loss_type]
        
        if kwargs:
            return hessian_class(**kwargs)
        else:
            return hessian_class()
    
    @property
    def available_hessians(self) -> list[str]:
        """Get list of available Hessian functions."""
        return list(self._hessian_registry.keys())

# Convenience function for backward compatibility
def compute_hessian(loss_type: LossType, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   weights: np.ndarray,
                   regularization: float = 0.0,
                   **kwargs) -> np.ndarray:
    """
    Convenience function for computing Hessians (backward compatibility).
    
    This function provides a direct interface similar to the original
    tinh_hessian_ham_loss function but with clean parameters.
    
    Args:
        loss_type: Type of loss ('ols', 'ridge', 'lasso', etc.)
        X: Feature matrix (includes bias column)
        y: Target vector
        weights: Weight vector (includes bias weight)
        regularization: Regularization parameter (for ridge/elastic_net)
        **kwargs: Additional parameters
    
    Returns:
        Hessian matrix (n_features, n_features)
    """
    hess_fn = HessianFunction()
    
    # Map common parameters
    if loss_type in ['ridge', 'elastic_net'] and regularization > 0:
        if loss_type == 'ridge':
            kwargs['alpha'] = regularization
        elif loss_type == 'elastic_net':
            kwargs['alpha_l2'] = regularization
    
    return hess_fn.compute(loss_type, X, y, weights, **kwargs)

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================
def tinh_Hessian_OLS(X: np.ndarray, y: np.ndarray = None, weights: np.ndarray = None) -> np.ndarray:
    """Compute OLS Hessian (backward compatibility)"""
    hess_fn = HessianFunction()
    # For OLS, Hessian doesn't depend on y or weights
    return hess_fn.compute('ols', X, np.zeros(X.shape[0]), np.zeros(X.shape[1]))

def tinh_Hessian_Ridge_with_bias(X: np.ndarray, y: np.ndarray = None, weights: np.ndarray = None,
                               alpha: float = 0.01) -> np.ndarray:
    """Compute Ridge Hessian with bias (backward compatibility)"""
    hess_fn = HessianFunction()
    # For Ridge, Hessian doesn't depend on y or weights
    return hess_fn.compute('ridge', X, np.zeros(X.shape[0]), np.zeros(X.shape[1]), alpha=alpha)