"""
Validation Mixins
This module provides mixins for input validation and data checking,
extending the capabilities of the optimization algorithms.
"""
import numpy as np
from typing import Union, Tuple, Optional

class ValidationMixin:
    """
    Mixin class for input validation and data checking
    """
    def validate_input_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Validate input data for training
        Args:
            X: Feature matrix
            y: Target vector
        Returns:
            True if data is valid
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("X and y cannot be empty")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")
        return True
    def validate_weights(self, weights: np.ndarray, expected_size: Optional[int] = None) -> bool:
        """
        Validate weight vector
        Args:
            weights: Weight vector to validate
            expected_size: Expected size of weights (optional)
        Returns:
            True if weights are valid
        Raises:
            ValueError: If weights are invalid
        """
        if not isinstance(weights, np.ndarray):
            raise ValueError("weights must be a numpy array")
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if expected_size is not None and len(weights) != expected_size:
            raise ValueError(f"weights must have size {expected_size}, got {len(weights)}")
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("weights contain NaN or infinite values")
        return True
    def validate_hyperparameters(self) -> bool:
        """
        Validate hyperparameters of the model
        Returns:
            True if hyperparameters are valid
        Raises:
            ValueError: If hyperparameters are invalid
        """
        if hasattr(self, 'learning_rate'):
            if self.learning_rate <= 0:
                raise ValueError("learning_rate must be positive")
        if hasattr(self, 'max_iterations'):
            if self.max_iterations <= 0:
                raise ValueError("max_iterations must be positive")
        if hasattr(self, 'convergence_tolerance'):
            if self.convergence_tolerance <= 0:
                raise ValueError("convergence_tolerance must be positive")
        if hasattr(self, 'regularization_strength'):
            if self.regularization_strength < 0:
                raise ValueError("regularization_strength must be non-negative")
        if hasattr(self, 'momentum'):
            if not (0 <= self.momentum < 1):
                raise ValueError("momentum must be in [0, 1)")
        if hasattr(self, 'batch_size'):
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive")
        return True
    def check_convergence_conditions(self, gradient_norm: float,
                                   cost_change: float, iteration: int) -> bool:
        """
        Check if convergence conditions are met
        Args:
            gradient_norm: Norm of the gradient
            cost_change: Change in cost function
            iteration: Current iteration number
        Returns:
            True if converged
        """
        # Check gradient norm convergence
        if hasattr(self, 'convergence_tolerance'):
            if gradient_norm < self.convergence_tolerance:
                return True
        # Check cost change convergence
        if hasattr(self, 'cost_tolerance'):
            if abs(cost_change) < self.cost_tolerance:
                return True
        # Check maximum iterations
        if hasattr(self, 'max_iterations'):
            if iteration >= self.max_iterations:
                return True
        return False
    def validate_prediction_input(self, X: np.ndarray) -> bool:
        """
        Validate input for prediction
        Args:
            X: Feature matrix for prediction
        Returns:
            True if input is valid
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if hasattr(self, 'weights') and self.weights is not None:
            expected_features = len(self.weights)
            if X.shape[1] != expected_features:
                raise ValueError(f"X must have {expected_features} features, got {X.shape[1]}")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")
        return True