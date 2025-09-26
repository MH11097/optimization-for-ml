"""
Scikit-learn SGD wrapper for gradient descent comparison.
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

try:
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SGDRegressor = None
    StandardScaler = None

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper


class SklearnSGDWrapper(BaseLibraryWrapper):
    """
    Wrapper for scikit-learn SGDRegressor.
    
    Provides gradient descent functionality using sklearn's implementation
    with various loss functions and regularization options.
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 max_iterations: int = 100000,
                 convergence_tolerance: float = 1e-3,
                 random_state: Optional[int] = None,
                 # SGD-specific parameters
                 momentum: float = 0.0,
                 learning_rate_schedule: str = 'constant',
                 early_stopping: bool = False,
                 validation_fraction: float = 0.1,
                 **kwargs):
        """
        Initialize sklearn SGD wrapper.
        
        Args:
            learning_rate: Learning rate for SGD
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter (alpha in sklearn)
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
            random_state: Random seed
            momentum: Momentum parameter
            learning_rate_schedule: Learning rate schedule ('constant', 'adaptive', 'invscaling')
            early_stopping: Whether to use early stopping
            validation_fraction: Fraction of data for early stopping validation
            **kwargs: Additional parameters for SGDRegressor
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SklearnSGDWrapper")
        
        # Store SGD-specific parameters
        self.learning_rate = learning_rate
        self.momentum = momentum  
        self.learning_rate_schedule = learning_rate_schedule
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.sklearn_kwargs = kwargs
        
        super().__init__(
            library_name='sklearn',
            algorithm_name='SGD',
            loss_type=loss_type,
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            random_state=random_state,
            learning_rate=learning_rate,
            momentum=momentum,
            learning_rate_schedule=learning_rate_schedule,
            early_stopping=early_stopping,
            **kwargs
        )
        
        # Scaler for feature normalization (sklearn SGD works better with scaled features)
        self.scaler = StandardScaler()
        self.feature_scaling_enabled = kwargs.get('feature_scaling', True)
    
    def _create_external_optimizer(self, n_features: int) -> SGDRegressor:
        """
        Create sklearn SGDRegressor instance.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            SGDRegressor instance
        """
        # Map loss types to sklearn loss functions
        sklearn_loss_map = {
            'ols': 'squared_error',  # sklearn 1.0+ uses 'squared_error' instead of 'squared_loss'
            'ridge': 'squared_error',  # Ridge handled via penalty parameter
            'lasso': 'squared_error'   # Lasso handled via penalty parameter
        }
        
        # Map regularization to sklearn penalty
        sklearn_penalty_map = {
            'ols': None,
            'ridge': 'l2', 
            'lasso': 'l1'
        }
        
        sklearn_loss = sklearn_loss_map[self.loss_type]
        sklearn_penalty = sklearn_penalty_map[self.loss_type]
        
        # Create SGDRegressor
        sgd_params = {
            'loss': sklearn_loss,
            'penalty': sklearn_penalty,
            'alpha': self.regularization if sklearn_penalty else 0.0,
            'learning_rate': self.learning_rate_schedule,
            'eta0': self.learning_rate,
            'max_iter': self.max_iterations,
            'tol': self.convergence_tolerance,
            'random_state': self.random_state,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction if self.early_stopping else 0.1,
            'fit_intercept': False,  # We handle bias manually
            'shuffle': True,
            'verbose': 0
        }
        
        # Add momentum if specified
        if self.momentum > 0:
            sgd_params['momentum'] = self.momentum
        
        # Add any additional sklearn parameters
        sgd_params.update(self.sklearn_kwargs)
        
        return SGDRegressor(**sgd_params)
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run sklearn SGD optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Separate bias and features for sklearn (it expects no bias column)
            X_no_bias = X[:, :-1]  # All columns except last (bias)
            bias_column = X[:, -1]   # Last column (should be all ones)
            
            # Scale features if enabled
            if self.feature_scaling_enabled and X_no_bias.shape[1] > 0:
                X_scaled = self.scaler.fit_transform(X_no_bias)
            else:
                X_scaled = X_no_bias
            
            # Fit the model
            self.external_optimizer.fit(X_scaled, y)
            
            # Extract results
            sklearn_coef = self.external_optimizer.coef_
            sklearn_intercept = self.external_optimizer.intercept_
            
            # Reconstruct weights in original scale
            if self.feature_scaling_enabled and X_no_bias.shape[1] > 0:
                # Transform coefficients back to original scale
                original_coef = sklearn_coef / self.scaler.scale_
                original_intercept = sklearn_intercept - np.sum(original_coef * self.scaler.mean_)\n            else:\n                original_coef = sklearn_coef\n                original_intercept = sklearn_intercept\n            \n            # Combine into single weight vector (features + bias)\n            if len(original_coef.shape) > 0:\n                final_weights = np.concatenate([original_coef.flatten(), [original_intercept]])\n            else:\n                final_weights = np.array([original_coef, original_intercept])\n            \n            # Check convergence (sklearn doesn't always provide this info)\n            converged = (\n                hasattr(self.external_optimizer, 'n_iter_') and\n                self.external_optimizer.n_iter_ < self.max_iterations\n            )\n            \n            iterations = getattr(self.external_optimizer, 'n_iter_', self.max_iterations)\n            \n            return {\n                'final_weights': final_weights,\n                'converged': converged,\n                'iterations': int(iterations),\n                'sklearn_coef': sklearn_coef,\n                'sklearn_intercept': sklearn_intercept,\n                'feature_scaling_used': self.feature_scaling_enabled\n            }\n            \n        except Exception as e:\n            print(f\"sklearn SGD optimization failed: {str(e)}\")\n            # Return initial weights if optimization failed\n            return {\n                'final_weights': self.weights,\n                'converged': False,\n                'iterations': 0,\n                'error': str(e)\n            }\n    \n    def _get_algorithm_specific_results(self) -> Dict[str, Any]:\n        \"\"\"\n        Get sklearn SGD-specific results.\n        \n        Returns:\n            Dictionary containing sklearn-specific results\n        \"\"\"\n        base_results = super()._get_algorithm_specific_results()\n        \n        sklearn_specific = {\n            'sgd_params': {\n                'learning_rate': self.learning_rate,\n                'momentum': self.momentum,\n                'learning_rate_schedule': self.learning_rate_schedule,\n                'early_stopping': self.early_stopping,\n                'validation_fraction': self.validation_fraction,\n                'feature_scaling_enabled': self.feature_scaling_enabled\n            }\n        }\n        \n        # Add sklearn model information if available\n        if self.external_optimizer is not None:\n            sklearn_specific['sklearn_model'] = {\n                'n_iter_': getattr(self.external_optimizer, 'n_iter_', None),\n                'coef_': getattr(self.external_optimizer, 'coef_', None),\n                'intercept_': getattr(self.external_optimizer, 'intercept_', None),\n                't_': getattr(self.external_optimizer, 't_', None)  # Internal iteration counter\n            }\n        \n        base_results['algorithm_specific'].update(sklearn_specific)\n        return base_results\n\n\ndef create_sklearn_sgd_optimizer(learning_rate: float = 0.001,\n                                loss_type: str = 'ols',\n                                regularization: float = 0.01,\n                                max_iterations: int = 100000,\n                                random_state: Optional[int] = None,\n                                **kwargs) -> SklearnSGDWrapper:\n    \"\"\"\n    Factory function to create sklearn SGD optimizer.\n    \n    Args:\n        learning_rate: Learning rate for SGD\n        loss_type: Type of loss function ('ols', 'ridge', 'lasso')\n        regularization: Regularization parameter\n        max_iterations: Maximum number of iterations\n        random_state: Random seed\n        **kwargs: Additional parameters\n        \n    Returns:\n        SklearnSGDWrapper instance\n    \"\"\"\n    return SklearnSGDWrapper(\n        learning_rate=learning_rate,\n        loss_type=loss_type,\n        regularization=regularization,\n        max_iterations=max_iterations,\n        random_state=random_state,\n        **kwargs\n    )