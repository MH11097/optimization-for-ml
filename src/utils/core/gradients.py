"""
Unified Gradient Function Implementation
This module provides a single, clean interface for all gradient computations,
replacing the 10+ duplicate gradient functions in the original utils.
Mathematical Formulations:
- OLS Gradient: ∇L = (1/n) * X^T * (Xw - y)
- Ridge Gradient: ∇L = (1/n) * X^T * (Xw - y) + 2α * w
- Lasso Gradient: ∇L = (1/n) * X^T * (Xw - y) + α * sign(w)
- Elastic Net Gradient: ∇L = (1/n) * X^T * (Xw - y) + α₁ * sign(w) + 2α₂ * w
- Huber Gradient: ∇L = (1/n) * X^T * huber_grad(Xw - y, δ)
"""
import numpy as np
from typing import Union, Optional, Literal, Tuple
from abc import ABC, abstractmethod
# Type aliases
LossType = Literal['ols', 'ridge', 'lasso', 'elastic_net', 'huber']
ArrayLike = Union[np.ndarray, list]

class BaseGradient(ABC):
    """Abstract base class for gradient computations."""
    
    @abstractmethod
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute gradient vector."""
        pass
    
    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Validate input arrays for gradient computation."""
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

class OLSGradient(BaseGradient):
    """
    Ordinary Least Squares Gradient
    
    ∇L(w) = (1/n) * X^T * (Xw - y)
    
    The gradient of the squared error loss. Points in the direction
    of steepest increase of the loss function.
    """
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        predictions = X @ weights
        residuals = predictions - y
        gradient = X.T @ residuals / len(y)
        
        return gradient

class RidgeGradient(BaseGradient):
    """
    Ridge Regression Gradient (L2 Regularization)
    
    ∇L(w) = (1/n) * X^T * (Xw - y) + 2α * w
    
    Adds L2 penalty gradient. The regularization term pulls weights toward zero.
    Note: Bias term (if present) should not be regularized.
    """
    
    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError("Ridge alpha must be non-negative")
        self.alpha = alpha
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        # OLS gradient term
        predictions = X @ weights
        residuals = predictions - y
        ols_gradient = X.T @ residuals / len(y)
        
        # L2 regularization gradient (don't regularize bias)
        reg_gradient = np.zeros_like(weights)
        if len(weights) > 1:
            # Assume bias is last element
            reg_gradient[:-1] = 2 * self.alpha * weights[:-1]
        else:
            reg_gradient = 2 * self.alpha * weights
        
        return ols_gradient + reg_gradient

class LassoGradient(BaseGradient):
    """
    Lasso Regression Gradient (L1 Regularization)
    
    ∇L(w) = (1/n) * X^T * (Xw - y) + α * sign(w)
    
    Uses subgradient for L1 penalty. The sign function creates
    sparse solutions by pushing small weights to exactly zero.
    """
    
    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError("Lasso alpha must be non-negative")
        self.alpha = alpha
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        # OLS gradient term
        predictions = X @ weights
        residuals = predictions - y
        ols_gradient = X.T @ residuals / len(y)
        
        # L1 regularization subgradient (don't regularize bias)
        reg_gradient = np.zeros_like(weights)
        if len(weights) > 1:
            # Assume bias is last element
            reg_gradient[:-1] = self.alpha * np.sign(weights[:-1])
        else:
            reg_gradient = self.alpha * np.sign(weights)
        
        return ols_gradient + reg_gradient

class ElasticNetGradient(BaseGradient):
    """
    Elastic Net Gradient (L1 + L2 Regularization)
    
    ∇L(w) = (1/n) * X^T * (Xw - y) + α₁ * sign(w) + 2α₂ * w
    
    Combines Ridge and Lasso gradients for balanced regularization.
    """
    
    def __init__(self, alpha_l1: float = 0.01, alpha_l2: float = 0.01):
        if alpha_l1 < 0 or alpha_l2 < 0:
            raise ValueError("ElasticNet alphas must be non-negative")
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        # OLS gradient term
        predictions = X @ weights
        residuals = predictions - y
        ols_gradient = X.T @ residuals / len(y)
        
        # Regularization gradients (exclude bias)
        reg_gradient = np.zeros_like(weights)
        if len(weights) > 1:
            # L1 term
            reg_gradient[:-1] += self.alpha_l1 * np.sign(weights[:-1])
            # L2 term
            reg_gradient[:-1] += 2 * self.alpha_l2 * weights[:-1]
        else:
            reg_gradient += self.alpha_l1 * np.sign(weights)
            reg_gradient += 2 * self.alpha_l2 * weights
        
        return ols_gradient + reg_gradient

class HuberGradient(BaseGradient):
    """
    Huber Loss Gradient (Robust Gradient)
    
    ∇L(w) = (1/n) * X^T * huber_grad(Xw - y, δ)
    
    where huber_grad(r, δ) = {
        r                if |r| ≤ δ
        δ * sign(r)      if |r| > δ
    }
    
    Robust gradient that's less sensitive to outliers.
    """
    
    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError("Huber delta must be positive")
        self.delta = delta
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self._validate_inputs(X, y, weights)
        
        predictions = X @ weights
        residuals = predictions - y
        
        # Huber gradient computation
        abs_residuals = np.abs(residuals)
        is_small = abs_residuals <= self.delta
        
        huber_grad = np.zeros_like(residuals)
        huber_grad[is_small] = residuals[is_small]
        huber_grad[~is_small] = self.delta * np.sign(residuals[~is_small])
        
        gradient = X.T @ huber_grad / len(y)
        return gradient

class GradientFunction:
    """
    Unified Gradient Function Interface
    
    This class replaces all duplicate gradient functions from the original utils
    with a single, clean, and extensible interface.
    
    Example:
        grad_fn = GradientFunction()
        
        # OLS gradient
        gradient = grad_fn.compute('ols', X, y, weights)
        
        # Ridge gradient with regularization
        gradient = grad_fn.compute('ridge', X, y, weights, alpha=0.01)
        
        # Custom gradient object
        ridge_grad = grad_fn.get_gradient_object('ridge', alpha=0.1)
        gradient = ridge_grad.compute(X, y, weights)
    """
    
    def __init__(self):
        self._gradient_registry = {
            'ols': OLSGradient,
            'ridge': RidgeGradient,
            'lasso': LassoGradient,
            'elastic_net': ElasticNetGradient,
            'huber': HuberGradient
        }
    
    def compute(self, 
                loss_type: LossType, 
                X: np.ndarray, 
                y: np.ndarray, 
                weights: np.ndarray,
                **kwargs) -> np.ndarray:
        """
        Compute gradient for specified loss type and parameters.
        
        Args:
            loss_type: Type of loss function
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            weights: Weight vector (n_features,)
            **kwargs: Loss-specific parameters (alpha, delta, etc.)
        
        Returns:
            Gradient vector (n_features,)
        
        Raises:
            ValueError: If loss_type is unknown or inputs are invalid
        """
        if loss_type not in self._gradient_registry:
            available = list(self._gradient_registry.keys())
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
        
        gradient_class = self._gradient_registry[loss_type]
        
        # Create gradient object with parameters
        if kwargs:
            gradient_obj = gradient_class(**kwargs)
        else:
            gradient_obj = gradient_class()
        
        return gradient_obj.compute(X, y, weights)
    
    def get_gradient_object(self, loss_type: LossType, **kwargs) -> BaseGradient:
        """
        Get a configured gradient object for repeated use.
        
        Args:
            loss_type: Type of loss function
            **kwargs: Loss-specific parameters
        
        Returns:
            Configured gradient object
        """
        if loss_type not in self._gradient_registry:
            available = list(self._gradient_registry.keys())
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
        
        gradient_class = self._gradient_registry[loss_type]
        
        if kwargs:
            return gradient_class(**kwargs)
        else:
            return gradient_class()
    
    @property
    def available_gradients(self) -> list[str]:
        """Get list of available gradient functions."""
        return list(self._gradient_registry.keys())

# Convenience function for backward compatibility
def compute_gradient(loss_type: LossType, 
                    X: np.ndarray, 
                    y: np.ndarray, 
                    weights: np.ndarray,
                    regularization: float = 0.0,
                    **kwargs) -> Tuple[np.ndarray, None]:
    """
    Convenience function for computing gradients (backward compatibility).
    
    This function provides a direct interface similar to the original
    tinh_gradient_ham_loss function but with clean parameters.
    
    Args:
        loss_type: Type of loss ('ols', 'ridge', 'lasso', etc.)
        X: Feature matrix (includes bias column)
        y: Target vector
        weights: Weight vector (includes bias weight)
        regularization: Regularization parameter (for ridge/lasso)
        **kwargs: Additional parameters
    
    Returns:
        Tuple of (gradient_vector, None) for compatibility with original signature
    """
    grad_fn = GradientFunction()
    
    # Map common parameters
    if loss_type in ['ridge', 'lasso'] and regularization > 0:
        kwargs['alpha'] = regularization
    
    gradient = grad_fn.compute(loss_type, X, y, weights, **kwargs)
    
    # Return tuple for backward compatibility (original returned gradient_w, gradient_b)
    # Since we use bias-in-weights format, gradient_b is None
    return gradient, None

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================
def tinh_gradient_OLS(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute OLS gradient (backward compatibility)"""
    grad_fn = GradientFunction()
    return grad_fn.compute('ols', X, y, weights)

def tinh_gradient_Ridge_with_bias(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                                alpha: float = 0.01) -> np.ndarray:
    """Compute Ridge gradient with bias (backward compatibility)"""
    grad_fn = GradientFunction()
    return grad_fn.compute('ridge', X, y, weights, alpha=alpha)