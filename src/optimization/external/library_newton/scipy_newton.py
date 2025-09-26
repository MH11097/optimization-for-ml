"""
SciPy Newton methods wrapper for Newton optimization comparison.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    minimize = None

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper


class SciPyNewtonWrapper(BaseLibraryWrapper):
    """
    Wrapper for SciPy Newton methods.
    
    Provides Newton optimization functionality using SciPy's implementations
    including Newton-CG, trust-ncg, and dogleg methods.
    """
    
    def __init__(self,
                 method: str = 'Newton-CG',
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 max_iterations: int = 100000,
                 convergence_tolerance: float = 1e-3,
                 random_state: Optional[int] = None,
                 # Newton-specific parameters
                 xtol: Optional[float] = None,
                 gtol: Optional[float] = None,
                 trust_radius: float = 1.0,
                 initial_trust_radius: Optional[float] = None,
                 max_trust_radius: float = 1000.0,
                 eta: float = 0.15,
                 **kwargs):
        """
        Initialize SciPy Newton wrapper.
        
        Args:
            method: Newton method ('Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact')
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
            random_state: Random seed
            xtol: Tolerance for termination by parameter change
            gtol: Gradient tolerance (None uses convergence_tolerance)
            trust_radius: Initial trust radius for trust region methods
            initial_trust_radius: Initial trust radius (alternative specification)
            max_trust_radius: Maximum trust radius
            eta: Trust region update parameter
            **kwargs: Additional parameters for scipy.optimize.minimize
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for SciPyNewtonWrapper")
        
        # Store Newton-specific parameters
        self.method = method
        self.xtol = xtol if xtol is not None else convergence_tolerance
        self.gtol = gtol if gtol is not None else convergence_tolerance
        self.trust_radius = initial_trust_radius if initial_trust_radius is not None else trust_radius
        self.max_trust_radius = max_trust_radius
        self.eta = eta
        self.scipy_kwargs = kwargs
        
        super().__init__(
            library_name='scipy',
            algorithm_name=f'Newton-{method}',
            loss_type=loss_type,
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            random_state=random_state,
            method=method,
            xtol=self.xtol,
            gtol=self.gtol,
            trust_radius=self.trust_radius,
            max_trust_radius=max_trust_radius,
            eta=eta,
            **kwargs
        )
        
        # SciPy optimization result
        self.scipy_result = None
    
    def _create_hessian_function(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """
        Create Hessian function for Newton methods.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Hessian function that takes weights as input
        """
        def hessian(weights: np.ndarray) -> np.ndarray:
            n_samples, n_features = X.shape
            
            if self.loss_type == 'ols':
                # For OLS: H = 2/n * X^T * X
                hess = (2.0 / n_samples) * X.T @ X
            elif self.loss_type == 'ridge':
                # For Ridge: H = 2/n * X^T * X + 2*lambda*I
                hess = (2.0 / n_samples) * X.T @ X + 2 * self.regularization * np.eye(n_features)
            elif self.loss_type == 'lasso':
                # For Lasso, Hessian is not well-defined, use Ridge approximation
                print("Warning: Using Ridge approximation for Lasso Hessian")
                hess = (2.0 / n_samples) * X.T @ X + 2 * self.regularization * np.eye(n_features)
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
            # Track matrix operation
            self.track_matrix_operation(hess.shape, "hessian_computation")
            
            return hess
        
        return hessian
    
    def _create_external_optimizer(self, n_features: int) -> str:
        """
        Create SciPy Newton method specification.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            Method name for SciPy minimize
        """
        valid_methods = ['Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact', 'trust-krylov']
        if self.method not in valid_methods:
            print(f"Warning: Unknown method {self.method}, using Newton-CG")
            return 'Newton-CG'
        return self.method
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run SciPy Newton optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Create objective, gradient, and Hessian functions
            objective_func = self._create_objective_function(X, y)
            gradient_func = self._create_gradient_function(X, y)
            hessian_func = self._create_hessian_function(X, y)
            
            # Create callback function for tracking progress
            callback_func = self._create_scipy_callback(X, y)
            
            # Set up minimize options based on method
            minimize_options = {
                'xtol': self.xtol,
                'gtol': self.gtol,
                'maxiter': self.max_iterations,
                'disp': False
            }
            
            # Add method-specific options
            if 'trust' in self.method.lower():
                minimize_options.update({
                    'initial_trust_radius': self.trust_radius,
                    'max_trust_radius': self.max_trust_radius,
                    'eta': self.eta
                })
            
            # Add any additional options
            minimize_options.update(self.scipy_kwargs.get('options', {}))
            
            # Run optimization
            self.scipy_result = minimize(
                fun=objective_func,
                x0=self.weights,
                method=self.external_optimizer,
                jac=gradient_func,
                hess=hessian_func,
                callback=callback_func,
                options=minimize_options
            )
            
            # Extract results
            final_weights = self.scipy_result.x
            converged = self.scipy_result.success
            iterations = self.scipy_result.nit if hasattr(self.scipy_result, 'nit') else self.max_iterations
            
            return {
                'final_weights': final_weights,
                'converged': converged,
                'iterations': int(iterations),
                'final_loss': float(self.scipy_result.fun),
                'scipy_message': self.scipy_result.message,
                'scipy_status': self.scipy_result.status if hasattr(self.scipy_result, 'status') else 0
            }
            
        except Exception as e:
            print(f"SciPy {self.method} optimization failed: {str(e)}")
            return {
                'final_weights': self.weights,
                'converged': False,
                'iterations': 0,
                'error': str(e)
            }
    
    def _create_scipy_callback(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """
        Create callback function for SciPy optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Callback function for SciPy minimize
        """
        def callback(xk: np.ndarray) -> None:
            # Compute current metrics
            loss_value = self.loss_func(X, y, xk)
            gradient_w, _ = self.grad_func(X, y, xk)
            gradient_norm = np.linalg.norm(gradient_w)
            
            # Store in callback history
            iteration = len(self.callback_history['losses'])
            self.callback_history['losses'].append(float(loss_value))
            self.callback_history['gradient_norms'].append(float(gradient_norm))
            self.callback_history['weights'].append(xk.copy())
            self.callback_history['iterations'].append(iteration)
            
        return callback
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get SciPy Newton-specific results.
        
        Returns:
            Dictionary containing SciPy Newton-specific results
        """
        base_results = super()._get_algorithm_specific_results()
        
        scipy_specific = {
            'scipy_newton_params': {
                'method': self.method,
                'xtol': self.xtol,
                'gtol': self.gtol,
                'trust_radius': self.trust_radius,
                'max_trust_radius': self.max_trust_radius,
                'eta': self.eta
            }
        }
        
        # Add SciPy result information if available
        if self.scipy_result is not None:
            scipy_specific['scipy_result'] = {
                'success': self.scipy_result.success,
                'status': getattr(self.scipy_result, 'status', None),
                'message': self.scipy_result.message,
                'nfev': getattr(self.scipy_result, 'nfev', None),  # Function evaluations
                'njev': getattr(self.scipy_result, 'njev', None),  # Jacobian evaluations
                'nhev': getattr(self.scipy_result, 'nhev', None),  # Hessian evaluations
                'nit': getattr(self.scipy_result, 'nit', None),   # Iterations
                'fun': float(self.scipy_result.fun) if hasattr(self.scipy_result, 'fun') else None
            }
        
        base_results['algorithm_specific'].update(scipy_specific)
        return base_results


def create_scipy_newton_optimizer(method: str = 'Newton-CG',
                                 loss_type: str = 'ols',
                                 regularization: float = 0.01,
                                 max_iterations: int = 100000,
                                 convergence_tolerance: float = 1e-3,
                                 random_state: Optional[int] = None,
                                 **kwargs) -> SciPyNewtonWrapper:
    """
    Factory function to create SciPy Newton optimizer.
    
    Args:
        method: Newton method ('Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact')
        loss_type: Type of loss function ('ols', 'ridge', 'lasso')
        regularization: Regularization parameter
        max_iterations: Maximum number of iterations
        convergence_tolerance: Convergence tolerance
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        SciPyNewtonWrapper instance
    """
    return SciPyNewtonWrapper(
        method=method,
        loss_type=loss_type,
        regularization=regularization,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        random_state=random_state,
        **kwargs
    )