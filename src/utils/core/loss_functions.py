"""
Unified Loss Function Implementation
This module provides a single, clean interface for all loss functions,
replacing the 15+ duplicate functions in the original utils.
Mathematical Formulations:
- OLS (Ordinary Least Squares): L = (1/2n) * ||Xw - y||²
- Ridge Regression: L = (1/2n) * ||Xw - y||² + α * ||w||²
- Lasso Regression: L = (1/2n) * ||Xw - y||² + α * ||w||₁
- Elastic Net: L = (1/2n) * ||Xw - y||² + α₁ * ||w||₁ + α₂ * ||w||²
- Huber Loss: L = (1/2n) * Σ huber_loss(Xw - y, δ)
"""
import numpy as np
from typing import Union, Optional, Literal
from abc import ABC, abstractmethod
# Type aliases for clarity
LossType = Literal['ols', 'ridge', 'lasso', 'elastic_net', 'huber']
ArrayLike = Union[np.ndarray, list]

class BaseLoss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Compute loss value."""
        pass
    
    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Validate input arrays for mathematical operations."""
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

class OLSLoss(BaseLoss):
    """
    Ordinary Least Squares Loss
    
    L(w) = (1/2n) * ||Xw - y||²
    
    The classic squared error loss function. Assumes Gaussian noise
    and provides maximum likelihood estimation.
    """
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        self._validate_inputs(X, y, weights)
        
        predictions = X @ weights
        residuals = predictions - y
        loss = 0.5 * np.mean(residuals ** 2)
        
        return loss

class RidgeLoss(BaseLoss):
    """
    Ridge Regression Loss (L2 Regularization)
    
    L(w) = (1/2n) * ||Xw - y||² + α * ||w||²
    
    Adds L2 penalty to prevent overfitting. The regularization parameter α
    controls the trade-off between fitting the data and keeping weights small.
    """
    
    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError("Ridge alpha must be non-negative")
        self.alpha = alpha
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        self._validate_inputs(X, y, weights)
        
        # OLS term
        predictions = X @ weights
        residuals = predictions - y
        mse_term = 0.5 * np.mean(residuals ** 2)
        
        # L2 regularization term (don't regularize bias if present)
        # Assume bias is last element
        if len(weights) > 1:
            reg_term = self.alpha * np.sum(weights[:-1] ** 2)  # Exclude bias
        else:
            reg_term = self.alpha * np.sum(weights ** 2)
        
        return mse_term + reg_term

class LassoLoss(BaseLoss):
    """
    Lasso Regression Loss (L1 Regularization)
    
    L(w) = (1/2n) * ||Xw - y||² + α * ||w||₁
    
    Uses L1 penalty for feature selection. Can drive weights to exactly zero,
    performing automatic feature selection.
    """
    
    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError("Lasso alpha must be non-negative")
        self.alpha = alpha
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        self._validate_inputs(X, y, weights)
        
        # OLS term
        predictions = X @ weights
        residuals = predictions - y
        mse_term = 0.5 * np.mean(residuals ** 2)
        
        # L1 regularization term (don't regularize bias)
        if len(weights) > 1:
            reg_term = self.alpha * np.sum(np.abs(weights[:-1]))  # Exclude bias
        else:
            reg_term = self.alpha * np.sum(np.abs(weights))
        
        return mse_term + reg_term

class ElasticNetLoss(BaseLoss):
    """
    Elastic Net Loss (L1 + L2 Regularization)
    
    L(w) = (1/2n) * ||Xw - y||² + α₁ * ||w||₁ + α₂ * ||w||²
    
    Combines Ridge and Lasso penalties. Useful when you have correlated
    features and want both regularization and feature selection.
    """
    
    def __init__(self, alpha_l1: float = 0.01, alpha_l2: float = 0.01):
        if alpha_l1 < 0 or alpha_l2 < 0:
            raise ValueError("ElasticNet alphas must be non-negative")
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        self._validate_inputs(X, y, weights)
        
        # OLS term
        predictions = X @ weights
        residuals = predictions - y
        mse_term = 0.5 * np.mean(residuals ** 2)
        
        # Regularization terms (exclude bias)
        if len(weights) > 1:
            l1_term = self.alpha_l1 * np.sum(np.abs(weights[:-1]))
            l2_term = self.alpha_l2 * np.sum(weights[:-1] ** 2)
        else:
            l1_term = self.alpha_l1 * np.sum(np.abs(weights))
            l2_term = self.alpha_l2 * np.sum(weights ** 2)
        
        return mse_term + l1_term + l2_term

class HuberLoss(BaseLoss):
    """
    Huber Loss (Robust Loss Function)
    
    L(w) = (1/n) * Σ huber_loss(Xw - y, δ)
    
    where huber_loss(r, δ) = {
        (1/2) * r²           if |r| ≤ δ
        δ * (|r| - δ/2)      if |r| > δ
    }
    
    Robust to outliers. Quadratic for small residuals, linear for large ones.
    """
    
    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError("Huber delta must be positive")
        self.delta = delta
    
    def compute(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        self._validate_inputs(X, y, weights)
        
        predictions = X @ weights
        residuals = predictions - y
        abs_residuals = np.abs(residuals)
        
        # Huber loss computation
        is_small = abs_residuals <= self.delta
        loss_small = 0.5 * residuals[is_small] ** 2
        loss_large = self.delta * (abs_residuals[~is_small] - 0.5 * self.delta)
        
        total_loss = np.sum(loss_small) + np.sum(loss_large)
        return total_loss / len(y)

class LossFunction:
    """
    Unified Loss Function Interface
    
    This class replaces all duplicate loss functions from the original utils
    with a single, clean, and extensible interface.
    
    Example:
        loss_fn = LossFunction()
        
        # OLS
        loss = loss_fn.compute('ols', X, y, weights)
        
        # Ridge with regularization
        loss = loss_fn.compute('ridge', X, y, weights, alpha=0.01)
        
        # Custom loss object
        ridge_loss = loss_fn.get_loss_object('ridge', alpha=0.1)
        loss = ridge_loss.compute(X, y, weights)
    """
    
    def __init__(self):
        self._loss_registry = {
            'ols': OLSLoss,
            'ridge': RidgeLoss,
            'lasso': LassoLoss,
            'elastic_net': ElasticNetLoss,
            'huber': HuberLoss
        }
    
    def compute(self, 
                loss_type: LossType, 
                X: np.ndarray, 
                y: np.ndarray, 
                weights: np.ndarray,
                **kwargs) -> float:
        """
        Compute loss for specified type and parameters.
        
        Args:
            loss_type: Type of loss function
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            weights: Weight vector (n_features,)
            **kwargs: Loss-specific parameters (alpha, delta, etc.)
        
        Returns:
            Loss value as float
        
        Raises:
            ValueError: If loss_type is unknown or inputs are invalid
        """
        if loss_type not in self._loss_registry:
            available = list(self._loss_registry.keys())
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
        
        loss_class = self._loss_registry[loss_type]
        
        # Create loss object with parameters
        if kwargs:
            loss_obj = loss_class(**kwargs)
        else:
            loss_obj = loss_class()
        
        return loss_obj.compute(X, y, weights)
    
    def get_loss_object(self, loss_type: LossType, **kwargs) -> BaseLoss:
        """
        Get a configured loss object for repeated use.
        
        Useful when you need to compute the same loss multiple times
        with the same parameters.
        
        Args:
            loss_type: Type of loss function
            **kwargs: Loss-specific parameters
        
        Returns:
            Configured loss object
        """
        if loss_type not in self._loss_registry:
            available = list(self._loss_registry.keys())
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")
        
        loss_class = self._loss_registry[loss_type]
        
        if kwargs:
            return loss_class(**kwargs)
        else:
            return loss_class()
    
    @property
    def available_losses(self) -> list[str]:
        """Get list of available loss functions."""
        return list(self._loss_registry.keys())

# Convenience function for backward compatibility
def compute_loss(loss_type: LossType, 
                X: np.ndarray, 
                y: np.ndarray, 
                weights: np.ndarray,
                regularization: float = 0.0,
                **kwargs) -> float:
    """
    Convenience function for computing loss (backward compatibility).
    
    This function provides a direct interface similar to the original
    tinh_gia_tri_ham_loss function but with clean parameters.
    
    Args:
        loss_type: Type of loss ('ols', 'ridge', 'lasso', etc.)
        X: Feature matrix
        y: Target vector  
        weights: Weight vector
        regularization: Regularization parameter (for ridge/lasso)
        **kwargs: Additional parameters
    
    Returns:
        Loss value
    """
    loss_fn = LossFunction()
    
    # Map common parameters
    if loss_type in ['ridge', 'lasso'] and regularization > 0:
        kwargs['alpha'] = regularization
    
    return loss_fn.compute(loss_type, X, y, weights, **kwargs)

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================
def tinh_gia_tri_ham_OLS(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Compute OLS loss (backward compatibility)"""
    return compute_loss('ols', X, y, weights)

def tinh_gia_tri_ham_Ridge_with_bias(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                                   alpha: float = 0.01) -> float:
    """Compute Ridge loss with bias (backward compatibility)"""
    return compute_loss('ridge', X, y, weights, alpha=alpha)

def tinh_gia_tri_ham_Lasso_with_bias(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                                   alpha: float = 0.01) -> float:
    """Compute Lasso loss with bias (backward compatibility)"""
    return compute_loss('lasso', X, y, weights, alpha=alpha)

def tinh_gia_tri_ham_ElasticNet_with_bias(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                                        alpha_l1: float = 0.01, alpha_l2: float = 0.01) -> float:
    """Compute ElasticNet loss with bias (backward compatibility)"""
    return compute_loss('elastic_net', X, y, weights, alpha_l1=alpha_l1, alpha_l2=alpha_l2)