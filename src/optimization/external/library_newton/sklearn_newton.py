"""
Scikit-learn Newton solver wrapper for Newton optimization comparison.
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

try:
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    Ridge = None
    LogisticRegression = None
    StandardScaler = None

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper


class SklearnNewtonWrapper(BaseLibraryWrapper):
    """
    Wrapper for scikit-learn Newton solvers.
    
    Provides Newton optimization functionality using sklearn's newton-cg solver
    available in Ridge regression and other models.
    """
    
    def __init__(self,
                 loss_type: str = 'ridge',  # Default to ridge since Newton-CG works well with it
                 regularization: float = 0.01,
                 max_iterations: int = 100000,
                 convergence_tolerance: float = 1e-3,
                 random_state: Optional[int] = None,
                 # sklearn Newton-specific parameters
                 solver: str = 'newton-cg',
                 feature_scaling: bool = True,
                 **kwargs):
        """
        Initialize sklearn Newton wrapper.
        
        Args:
            loss_type: Type of loss function ('ridge' works best, 'ols' uses Ridge with alpha=0)
            regularization: Regularization parameter (alpha in sklearn)
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
            random_state: Random seed
            solver: Solver to use ('newton-cg' for Newton, 'lbfgs' for quasi-Newton)
            feature_scaling: Whether to scale features
            **kwargs: Additional parameters for sklearn model
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SklearnNewtonWrapper")
        
        # Store sklearn-specific parameters
        self.solver = solver
        self.feature_scaling = feature_scaling
        self.sklearn_kwargs = kwargs
        
        super().__init__(
            library_name='sklearn',
            algorithm_name=f'Newton-{solver}',
            loss_type=loss_type,
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            random_state=random_state,
            solver=solver,
            feature_scaling=feature_scaling,
            **kwargs
        )
        
        # Scaler for feature normalization
        self.scaler = StandardScaler() if feature_scaling else None
    
    def _create_external_optimizer(self, n_features: int):
        """
        Create sklearn model with Newton solver.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            Sklearn model instance
        """
        if self.loss_type in ['ols', 'ridge']:
            # Use Ridge regression (alpha=0 for OLS)
            alpha = 0.0 if self.loss_type == 'ols' else self.regularization
            
            model_params = {
                'alpha': alpha,
                'solver': self.solver,
                'max_iter': self.max_iterations,
                'tol': self.convergence_tolerance,
                'random_state': self.random_state,
                'fit_intercept': False  # We handle bias manually
            }
            
            # Add any additional parameters
            model_params.update(self.sklearn_kwargs)
            
            return Ridge(**model_params)
        
        elif self.loss_type == 'lasso':
            print("Warning: Sklearn Newton solver doesn't directly support Lasso. Using Ridge approximation.")
            model_params = {
                'alpha': self.regularization,
                'solver': self.solver,
                'max_iter': self.max_iterations,
                'tol': self.convergence_tolerance,
                'random_state': self.random_state,
                'fit_intercept': False
            }
            
            model_params.update(self.sklearn_kwargs)
            return Ridge(**model_params)
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run sklearn Newton optimization.
        
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
            if self.feature_scaling and X_no_bias.shape[1] > 0:
                X_scaled = self.scaler.fit_transform(X_no_bias)
            else:
                X_scaled = X_no_bias
            
            # Fit the model
            self.external_optimizer.fit(X_scaled, y)
            
            # Extract results
            sklearn_coef = self.external_optimizer.coef_
            sklearn_intercept = self.external_optimizer.intercept_
            
            # Reconstruct weights in original scale
            if self.feature_scaling and X_no_bias.shape[1] > 0:
                # Transform coefficients back to original scale
                original_coef = sklearn_coef / self.scaler.scale_
                original_intercept = sklearn_intercept - np.sum(original_coef * self.scaler.mean_)
            else:
                original_coef = sklearn_coef
                original_intercept = sklearn_intercept
            
            # Combine into single weight vector (features + bias)
            if len(original_coef.shape) > 0:
                final_weights = np.concatenate([original_coef.flatten(), [original_intercept]])
            else:
                final_weights = np.array([original_coef, original_intercept])
            
            # Check convergence (sklearn doesn't always provide detailed convergence info)
            converged = True  # Assume convergence if no exception was raised
            
            # Get number of iterations if available
            iterations = getattr(self.external_optimizer, 'n_iter_', self.max_iterations)
            
            # Create minimal callback history since sklearn doesn't provide iteration details
            final_loss = self.loss_func(X, y, final_weights)
            final_gradient_w, _ = self.grad_func(X, y, final_weights)
            final_gradient_norm = np.linalg.norm(final_gradient_w)
            
            # Store final results in callback history
            self.callback_history['losses'] = [float(final_loss)]
            self.callback_history['gradient_norms'] = [float(final_gradient_norm)]
            self.callback_history['weights'] = [final_weights.copy()]
            self.callback_history['iterations'] = [int(iterations)]
            
            return {
                'final_weights': final_weights,
                'converged': converged,
                'iterations': int(iterations),
                'sklearn_coef': sklearn_coef,
                'sklearn_intercept': sklearn_intercept,
                'feature_scaling_used': self.feature_scaling
            }
            
        except Exception as e:
            print(f"sklearn Newton optimization failed: {str(e)}")
            # Return initial weights if optimization failed
            return {
                'final_weights': self.weights,
                'converged': False,
                'iterations': 0,
                'error': str(e)
            }
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get sklearn Newton-specific results.
        
        Returns:
            Dictionary containing sklearn-specific results
        """
        base_results = super()._get_algorithm_specific_results()
        
        sklearn_specific = {
            'sklearn_newton_params': {
                'solver': self.solver,
                'feature_scaling': self.feature_scaling,
                'regularization_alpha': self.regularization
            }
        }
        
        # Add sklearn model information if available
        if self.external_optimizer is not None:
            sklearn_specific['sklearn_model'] = {
                'model_type': type(self.external_optimizer).__name__,
                'n_iter_': getattr(self.external_optimizer, 'n_iter_', None),
                'coef_': getattr(self.external_optimizer, 'coef_', None),
                'intercept_': getattr(self.external_optimizer, 'intercept_', None),
                'solver_used': getattr(self.external_optimizer, 'solver', self.solver)
            }
        
        base_results['algorithm_specific'].update(sklearn_specific)
        return base_results


def create_sklearn_newton_optimizer(loss_type: str = 'ridge',
                                   regularization: float = 0.01,
                                   max_iterations: int = 100000,
                                   convergence_tolerance: float = 1e-3,
                                   random_state: Optional[int] = None,
                                   **kwargs) -> SklearnNewtonWrapper:
    """
    Factory function to create sklearn Newton optimizer.
    
    Args:
        loss_type: Type of loss function ('ridge' recommended, 'ols' uses Ridge with alpha=0)
        regularization: Regularization parameter
        max_iterations: Maximum number of iterations
        convergence_tolerance: Convergence tolerance
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        SklearnNewtonWrapper instance
    """
    return SklearnNewtonWrapper(
        loss_type=loss_type,
        regularization=regularization,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        random_state=random_state,
        **kwargs
    )