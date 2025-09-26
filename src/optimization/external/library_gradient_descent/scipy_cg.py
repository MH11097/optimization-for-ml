"""
SciPy Conjugate Gradient wrapper for gradient descent comparison.
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


class SciPyCGWrapper(BaseLibraryWrapper):
    """
    Wrapper for SciPy Conjugate Gradient method.
    
    Provides gradient descent functionality using SciPy's Conjugate Gradient
    implementation, which is particularly effective for quadratic problems.
    """
    
    def __init__(self,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 max_iterations: int = 100000,
                 convergence_tolerance: float = 1e-3,
                 random_state: Optional[int] = None,
                 # SciPy CG-specific parameters
                 gtol: Optional[float] = None,
                 norm: float = float('inf'),
                 eps: float = 1.4901161193847656e-08,
                 **kwargs):
        """
        Initialize SciPy CG wrapper.
        
        Args:
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
            random_state: Random seed
            gtol: Gradient tolerance (None uses convergence_tolerance)
            norm: Order of norm (Infinity norm or Euclidean norm)
            eps: Step size for finite-difference approximation
            **kwargs: Additional parameters for scipy.optimize.minimize
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for SciPyCGWrapper")
        
        # Store SciPy CG-specific parameters
        self.gtol = gtol if gtol is not None else convergence_tolerance
        self.norm = norm
        self.eps = eps
        self.scipy_kwargs = kwargs
        
        super().__init__(
            library_name='scipy',
            algorithm_name='CG',
            loss_type=loss_type,
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            random_state=random_state,
            gtol=self.gtol,
            norm=norm,
            eps=eps,
            **kwargs
        )
        
        # SciPy optimization result
        self.scipy_result = None
    
    def _create_external_optimizer(self, n_features: int) -> str:
        """
        Create SciPy CG method specification.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            Method name for SciPy minimize
        """
        return 'CG'  # Conjugate Gradient method
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run SciPy CG optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Create objective and gradient functions
            objective_func = self._create_objective_function(X, y)
            gradient_func = self._create_gradient_function(X, y)
            
            # Create callback function for tracking progress
            callback_func = self._create_scipy_callback(X, y)
            
            # Set up minimize options
            minimize_options = {
                'gtol': self.gtol,
                'norm': self.norm,
                'eps': self.eps,
                'maxiter': self.max_iterations,
                'disp': False  # Suppress SciPy output
            }
            
            # Add any additional options
            minimize_options.update(self.scipy_kwargs.get('options', {}))
            
            # Run optimization
            self.scipy_result = minimize(
                fun=objective_func,
                x0=self.weights,
                method=self.external_optimizer,
                jac=gradient_func,
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
            print(f"SciPy CG optimization failed: {str(e)}")
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
            
            # Track complexity (already tracked in objective/gradient functions)
            
        return callback
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get SciPy CG-specific results.
        
        Returns:
            Dictionary containing SciPy-specific results
        """
        base_results = super()._get_algorithm_specific_results()
        
        scipy_specific = {
            'scipy_cg_params': {
                'gtol': self.gtol,
                'norm': self.norm,
                'eps': self.eps,
                'method': 'CG'
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
                'nit': getattr(self.scipy_result, 'nit', None),   # Iterations
                'fun': float(self.scipy_result.fun) if hasattr(self.scipy_result, 'fun') else None
            }
        
        base_results['algorithm_specific'].update(scipy_specific)
        return base_results


def create_scipy_cg_optimizer(loss_type: str = 'ols',
                             regularization: float = 0.01,
                             max_iterations: int = 100000,
                             convergence_tolerance: float = 1e-3,
                             random_state: Optional[int] = None,
                             **kwargs) -> SciPyCGWrapper:
    """
    Factory function to create SciPy CG optimizer.
    
    Args:
        loss_type: Type of loss function ('ols', 'ridge', 'lasso')
        regularization: Regularization parameter
        max_iterations: Maximum number of iterations
        convergence_tolerance: Convergence tolerance
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        SciPyCGWrapper instance
    """
    return SciPyCGWrapper(
        loss_type=loss_type,
        regularization=regularization,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        random_state=random_state,
        **kwargs
    )