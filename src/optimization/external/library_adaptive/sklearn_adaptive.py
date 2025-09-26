"""
Sklearn Adaptive Learning Rate wrapper.
Implements adaptive learning rate techniques using sklearn's SGD with adaptive schedules.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper


class SklearnAdaptive(BaseLibraryWrapper):
    """
    Wrapper for sklearn adaptive learning rate techniques.
    Uses SGDRegressor with various adaptive learning rate schedules and online learning algorithms.
    """
    
    def __init__(self,
                 learning_rate_schedule: str = 'adaptive',  # 'adaptive', 'optimal', 'invscaling'
                 initial_learning_rate: float = 0.01,
                 power_t: float = 0.5,             # For invscaling
                 momentum: float = 0.0,
                 alpha: float = 0.0001,            # Regularization strength
                 epsilon: float = 0.1,             # Epsilon for epsilon-insensitive loss
                 n_iter_no_change: int = 5,        # For adaptive schedule
                 validation_fraction: float = 0.1,
                 tol: Optional[float] = None,      # Tolerance for stopping
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        Initialize sklearn adaptive learning rate wrapper.
        
        Args:
            learning_rate_schedule: Learning rate schedule type
            initial_learning_rate: Initial learning rate
            power_t: Exponent for inverse scaling learning rate
            momentum: Momentum parameter
            alpha: L2 regularization strength
            epsilon: Epsilon for epsilon-insensitive loss
            n_iter_no_change: Number of iterations with no improvement to wait
            validation_fraction: Fraction of training data for validation
            tol: Tolerance for stopping criterion
            loss_type: Loss function type
            regularization: Regularization parameter
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            random_state: Random seed
        """
        super().__init__(
            library_name='sklearn',
            algorithm_name=f'Adaptive_{learning_rate_schedule}',
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            random_state=random_state,
            learning_rate_schedule=learning_rate_schedule,
            initial_learning_rate=initial_learning_rate,
            power_t=power_t,
            momentum=momentum,
            alpha=alpha,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            tol=tol,
            **kwargs
        )
        
        self.learning_rate_schedule = learning_rate_schedule
        self.initial_learning_rate = initial_learning_rate
        self.power_t = power_t
        self.momentum = momentum
        self.alpha = max(alpha, regularization)  # Use larger regularization
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol if tol is not None else convergence_tolerance
        
        # Scaler for feature standardization
        self.scaler = None
        
        # Multiple SGD instances for ensemble-like adaptive behavior
        self.sgd_instances = []
        self.weights_history_detailed = []
    
    def _get_sklearn_loss_name(self) -> str:
        """Map our loss type to sklearn loss name."""
        loss_mapping = {
            'ols': 'squared_error',
            'ridge': 'squared_error',
            'lasso': 'squared_error',
            'huber': 'huber',
            'epsilon_insensitive': 'epsilon_insensitive',
            'mae': 'epsilon_insensitive'
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
            return 'l2'
    
    def _create_external_optimizer(self, n_features: int) -> SGDRegressor:
        """
        Create sklearn adaptive SGD optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            Sklearn SGD regressor with adaptive learning rate
        """
        # Common parameters
        common_params = {
            'loss': self._get_sklearn_loss_name(),
            'penalty': self._get_penalty_type(),
            'alpha': self.alpha,
            'fit_intercept': True,
            'max_iter': min(self.max_iterations, 10000),
            'tol': self.tol,
            'shuffle': True,
            'verbose': 0,
            'epsilon': self.epsilon,
            'random_state': self.random_state,
            'learning_rate': self.learning_rate_schedule,
            'eta0': self.initial_learning_rate,
            'power_t': self.power_t,
            'early_stopping': True,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'momentum': self.momentum,
        }
        
        # Create primary SGD instance
        primary_sgd = SGDRegressor(**common_params)
        
        # For adaptive behavior, create multiple instances with different parameters
        self.sgd_instances = [primary_sgd]
        
        if self.learning_rate_schedule == 'adaptive':
            # Create additional instances with different validation fractions and tolerances
            for val_frac in [0.05, 0.15, 0.2]:
                for n_iter_change in [3, 7, 10]:
                    adaptive_params = common_params.copy()
                    adaptive_params.update({
                        'validation_fraction': val_frac,
                        'n_iter_no_change': n_iter_change,
                        'random_state': self.random_state + len(self.sgd_instances) if self.random_state else None
                    })
                    self.sgd_instances.append(SGDRegressor(**adaptive_params))
        
        return primary_sgd
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run sklearn adaptive learning rate optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        # Remove bias column for sklearn
        X_no_bias = X[:, :-1]
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_no_bias)
        
        # Train multiple SGD instances and select the best one
        best_sgd = None
        best_score = float('-inf')
        all_results = []
        
        for i, sgd in enumerate(self.sgd_instances):
            try:
                # Fit the model
                sgd.fit(X_scaled, y)
                
                # Evaluate performance (negative loss as score)
                predictions = sgd.predict(X_scaled)
                mse = np.mean((predictions - y) ** 2)
                score = -mse  # Higher is better
                
                # Store results
                coef = sgd.coef_
                intercept = getattr(sgd, 'intercept_', [0.0])
                if hasattr(intercept, '__len__') and len(intercept) > 0:
                    intercept_value = intercept[0]
                else:
                    intercept_value = float(intercept)
                
                # Transform coefficients back to original scale
                if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
                    original_coef = coef / self.scaler.scale_
                    original_intercept = intercept_value - np.sum(original_coef * self.scaler.mean_)
                else:
                    original_coef = coef
                    original_intercept = intercept_value
                
                final_weights = np.append(original_coef, original_intercept)
                n_iter = getattr(sgd, 'n_iter_', 1)
                if hasattr(n_iter, '__len__'):
                    n_iter = n_iter[0] if len(n_iter) > 0 else 1
                
                result = {
                    'sgd_index': i,
                    'final_weights': final_weights,
                    'score': score,
                    'mse': mse,
                    'n_iter': int(n_iter),
                    'converged': int(n_iter) < sgd.max_iter
                }
                all_results.append(result)
                
                # Track the best model
                if score > best_score:
                    best_score = score
                    best_sgd = sgd
                    best_result = result
                
            except Exception as e:
                print(f\"[WARNING] SGD instance {i} failed: {str(e)}\")\
                continue\
        \
        if best_sgd is None:\
            # Fallback: use the primary SGD instance\
            self.external_optimizer.fit(X_scaled, y)\
            coef = self.external_optimizer.coef_\
            intercept = getattr(self.external_optimizer, 'intercept_', [0.0])\
            if hasattr(intercept, '__len__') and len(intercept) > 0:\
                intercept_value = intercept[0]\
            else:\
                intercept_value = float(intercept)\
            \
            if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:\
                original_coef = coef / self.scaler.scale_\
                original_intercept = intercept_value - np.sum(original_coef * self.scaler.mean_)\
            else:\
                original_coef = coef\
                original_intercept = intercept_value\
            \
            final_weights = np.append(original_coef, original_intercept)\
            n_iter = getattr(self.external_optimizer, 'n_iter_', 1)\
            if hasattr(n_iter, '__len__'):\
                n_iter = n_iter[0] if len(n_iter) > 0 else 1\
            \
            best_result = {\
                'sgd_index': 0,\
                'final_weights': final_weights,\
                'score': 0.0,\
                'mse': float('inf'),\
                'n_iter': int(n_iter),\
                'converged': False\
            }\
        \
        return {\
            'final_weights': best_result['final_weights'],\
            'iterations': best_result['n_iter'],\
            'converged': best_result['converged'],\
            'best_sgd_index': best_result['sgd_index'],\
            'best_score': best_result['score'],\
            'all_results': all_results,\
            'num_sgd_instances': len(self.sgd_instances)\
        }\
    \
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:\
        \"\"\"\
        Get sklearn adaptive learning rate specific results.\
        \
        Returns:\
            Dictionary containing sklearn-specific results\
        \"\"\"\
        base_results = super()._get_algorithm_specific_results()\
        \
        sklearn_specific = {\
            'sklearn_adaptive_specific': {\
                'learning_rate_schedule': self.learning_rate_schedule,\
                'initial_learning_rate': self.initial_learning_rate,\
                'power_t': self.power_t,\
                'momentum': self.momentum,\
                'alpha_regularization': self.alpha,\
                'epsilon': self.epsilon,\
                'n_iter_no_change': self.n_iter_no_change,\
                'validation_fraction': self.validation_fraction,\
                'tolerance': self.tol,\
                'penalty_type': self._get_penalty_type(),\
                'sklearn_loss': self._get_sklearn_loss_name(),\
                'feature_scaling': 'StandardScaler applied',\
                'num_sgd_instances': len(self.sgd_instances),\
                'ensemble_adaptive_behavior': len(self.sgd_instances) > 1\
            }\
        }\
        \
        base_results.update(sklearn_specific)\
        return base_results\
\
\
# Convenience functions for specific adaptive configurations\
def create_sklearn_adaptive_standard(initial_learning_rate: float = 0.01, **kwargs) -> SklearnAdaptive:\
    \"\"\"Create standard adaptive learning rate SGD.\"\"\"\
    return SklearnAdaptive(\
        learning_rate_schedule='adaptive',\
        initial_learning_rate=initial_learning_rate,\
        **kwargs\
    )\
\
def create_sklearn_optimal_lr(alpha: float = 0.0001, **kwargs) -> SklearnAdaptive:\
    \"\"\"Create SGD with optimal learning rate schedule.\"\"\"\
    return SklearnAdaptive(\
        learning_rate_schedule='optimal',\
        alpha=alpha,\
        **kwargs\
    )\
\
def create_sklearn_invscaling_lr(initial_learning_rate: float = 0.01, power_t: float = 0.5, **kwargs) -> SklearnAdaptive:\
    \"\"\"Create SGD with inverse scaling learning rate.\"\"\"\
    return SklearnAdaptive(\
        learning_rate_schedule='invscaling',\
        initial_learning_rate=initial_learning_rate,\
        power_t=power_t,\
        **kwargs\
    )\
\
def create_sklearn_adaptive_momentum(initial_learning_rate: float = 0.01, momentum: float = 0.9, **kwargs) -> SklearnAdaptive:\
    \"\"\"Create adaptive SGD with momentum.\"\"\"\
    return SklearnAdaptive(\
        learning_rate_schedule='adaptive',\
        initial_learning_rate=initial_learning_rate,\
        momentum=momentum,\
        **kwargs\
    )\
\
def create_sklearn_adaptive_aggressive(initial_learning_rate: float = 0.01, n_iter_no_change: int = 3, **kwargs) -> SklearnAdaptive:\
    \"\"\"Create aggressive adaptive SGD (stops early).\"\"\"\
    return SklearnAdaptive(\
        learning_rate_schedule='adaptive',\
        initial_learning_rate=initial_learning_rate,\
        n_iter_no_change=n_iter_no_change,\
        **kwargs\
    )