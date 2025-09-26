"""
SKlearn Stochastic Gradient Descent variants wrapper.
Implements multiple SGD-based algorithms from sklearn for stochastic optimization.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Perceptron
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper


class SklearnSGDVariants(BaseLibraryWrapper):
    """
    Wrapper for sklearn SGD variants including SGDRegressor, PassiveAggressiveRegressor.
    Provides multiple stochastic optimization algorithms from scikit-learn.
    """
    
    def __init__(self,
                 variant: str = 'sgd',  # 'sgd', 'passive_aggressive', 'perceptron'
                 learning_rate: float = 0.01,
                 learning_rate_schedule: str = 'constant',  # 'constant', 'optimal', 'invscaling', 'adaptive'
                 momentum: float = 0.0,
                 alpha: float = 0.0001,  # L2 regularization
                 l1_ratio: float = 0.15,  # For elastic net
                 epsilon: float = 0.1,   # For epsilon-insensitive loss
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        Initialize sklearn SGD variants wrapper.
        
        Args:
            variant: Type of SGD variant ('sgd', 'passive_aggressive', 'perceptron')
            learning_rate: Learning rate value
            learning_rate_schedule: Learning rate schedule
            momentum: Momentum parameter (only for SGD)
            alpha: L2 regularization strength
            l1_ratio: Elastic net mixing parameter
            epsilon: Epsilon for epsilon-insensitive loss
            loss_type: Loss function type
            regularization: Regularization parameter
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            random_state: Random seed
        """
        super().__init__(
            library_name='sklearn',
            algorithm_name=f'SGD_{variant}',
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            random_state=random_state,
            variant=variant,
            learning_rate=learning_rate,
            learning_rate_schedule=learning_rate_schedule,
            momentum=momentum,
            alpha=alpha,
            l1_ratio=l1_ratio,
            epsilon=epsilon,
            **kwargs
        )
        
        self.variant = variant
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.momentum = momentum
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        
        # Scaler for feature standardization
        self.scaler = None
        
    def _get_sklearn_loss_name(self) -> str:
        """Map our loss type to sklearn loss name."""
        loss_mapping = {
            'ols': 'squared_error',
            'ridge': 'squared_error',  # Ridge handled by alpha parameter
            'lasso': 'squared_error',  # Lasso handled by penalty parameter
            'huber': 'huber',
            'epsilon_insensitive': 'epsilon_insensitive'
        }
        return loss_mapping.get(self.loss_type, 'squared_error')
    
    def _get_penalty_type(self) -> str:
        """Map our loss type to sklearn penalty type."""
        if self.loss_type == 'ridge':
            return 'l2'
        elif self.loss_type == 'lasso':
            return 'l1'
        elif self.loss_type == 'elastic_net':
            return 'elasticnet'
        else:
            return 'l2'  # Default to L2
    
    def _create_external_optimizer(self, n_features: int) -> Any:
        """
        Create sklearn SGD variant optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            Sklearn optimizer instance
        """
        # Common parameters
        common_params = {
            'alpha': max(self.alpha, self.regularization),  # Use the larger of the two
            'fit_intercept': True,  # We handle bias in our wrapper
            'max_iter': min(self.max_iterations, 10000),  # Sklearn has reasonable limits
            'tol': self.convergence_tolerance,
            'random_state': self.random_state,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10
        }
        
        if self.variant == 'sgd':
            return SGDRegressor(
                loss=self._get_sklearn_loss_name(),
                penalty=self._get_penalty_type(),
                learning_rate=self.learning_rate_schedule,
                eta0=self.learning_rate,
                momentum=self.momentum,
                l1_ratio=self.l1_ratio,
                epsilon=self.epsilon,
                **common_params
            )
        elif self.variant == 'passive_aggressive':
            return PassiveAggressiveRegressor(
                C=1.0 / max(self.alpha, self.regularization),  # C is inverse of regularization
                fit_intercept=True,
                max_iter=min(self.max_iterations, 10000),
                tol=self.convergence_tolerance,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=self.random_state,
                epsilon=self.epsilon
            )
        elif self.variant == 'perceptron':
            # Perceptron for regression (using SGDRegressor with perceptron loss)
            return SGDRegressor(
                loss='perceptron',
                penalty=self._get_penalty_type(),
                learning_rate=self.learning_rate_schedule,
                eta0=self.learning_rate,
                **common_params
            )
        else:
            raise ValueError(f"Unknown SGD variant: {self.variant}")
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run sklearn SGD optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        # Remove bias column for sklearn (it handles intercept internally)
        X_no_bias = X[:, :-1]
        
        # Standardize features for better convergence
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_no_bias)
        
        # Fit the model
        self.external_optimizer.fit(X_scaled, y)
        
        # Extract final weights
        # Sklearn stores coefficients and intercept separately
        coef = self.external_optimizer.coef_
        intercept = getattr(self.external_optimizer, 'intercept_', [0.0])
        
        # Handle different intercept formats
        if hasattr(intercept, '__len__') and len(intercept) > 0:
            intercept_value = intercept[0]
        else:
            intercept_value = float(intercept)
        
        # Transform coefficients back to original scale
        if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
            # Inverse transform the coefficients
            original_coef = coef / self.scaler.scale_
            # Adjust intercept for the scaling
            original_intercept = intercept_value - np.sum(original_coef * self.scaler.mean_)
        else:
            original_coef = coef
            original_intercept = intercept_value
        
        # Combine coefficients and intercept (bias)
        final_weights = np.append(original_coef, original_intercept)
        
        # Get iteration count
        n_iter = getattr(self.external_optimizer, 'n_iter_', 1)
        if hasattr(n_iter, '__len__'):
            n_iter = n_iter[0] if len(n_iter) > 0 else 1
        
        # Check convergence
        converged = (n_iter < self.external_optimizer.max_iter) if hasattr(self.external_optimizer, 'max_iter') else True
        
        return {
            'final_weights': final_weights,
            'iterations': int(n_iter),
            'converged': converged,
            'sklearn_coef': coef,
            'sklearn_intercept': intercept_value,
            'scaling_applied': True
        }
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get sklearn SGD specific results.
        
        Returns:
            Dictionary containing sklearn-specific results
        """
        base_results = super()._get_algorithm_specific_results()
        
        sklearn_specific = {
            'sklearn_sgd_specific': {
                'variant': self.variant,
                'learning_rate_schedule': self.learning_rate_schedule,
                'learning_rate_value': self.learning_rate,
                'momentum': self.momentum,
                'alpha_regularization': self.alpha,
                'l1_ratio': self.l1_ratio,
                'epsilon': self.epsilon,
                'penalty_type': self._get_penalty_type(),
                'sklearn_loss': self._get_sklearn_loss_name(),
                'feature_scaling': 'StandardScaler applied',
                'final_n_iter': getattr(self.external_optimizer, 'n_iter_', 'Unknown') if self.external_optimizer else 'Unknown'
            }
        }
        
        base_results.update(sklearn_specific)
        return base_results


# Convenience functions for specific variants
def create_sklearn_sgd_constant(learning_rate: float = 0.01, **kwargs) -> SklearnSGDVariants:
    """Create SGD with constant learning rate."""
    return SklearnSGDVariants(
        variant='sgd',
        learning_rate=learning_rate,
        learning_rate_schedule='constant',
        **kwargs
    )

def create_sklearn_sgd_optimal(alpha: float = 0.0001, **kwargs) -> SklearnSGDVariants:
    """Create SGD with optimal learning rate schedule."""
    return SklearnSGDVariants(
        variant='sgd',
        learning_rate_schedule='optimal',
        alpha=alpha,
        **kwargs
    )

def create_sklearn_sgd_adaptive(learning_rate: float = 0.01, **kwargs) -> SklearnSGDVariants:
    """Create SGD with adaptive learning rate."""
    return SklearnSGDVariants(
        variant='sgd',
        learning_rate=learning_rate,
        learning_rate_schedule='adaptive',
        **kwargs
    )

def create_sklearn_sgd_momentum(learning_rate: float = 0.01, momentum: float = 0.9, **kwargs) -> SklearnSGDVariants:
    """Create SGD with momentum."""
    return SklearnSGDVariants(
        variant='sgd',
        learning_rate=learning_rate,
        learning_rate_schedule='constant',
        momentum=momentum,
        **kwargs
    )

def create_sklearn_passive_aggressive(regularization: float = 0.01, **kwargs) -> SklearnSGDVariants:
    """Create Passive Aggressive regressor."""
    return SklearnSGDVariants(
        variant='passive_aggressive',
        regularization=regularization,
        **kwargs
    )