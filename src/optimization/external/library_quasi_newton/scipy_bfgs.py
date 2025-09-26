"""
SciPy BFGS and L-BFGS wrapper for quasi-Newton optimization comparison.
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


class SciPyBFGSWrapper(BaseLibraryWrapper):
    """
    Wrapper for SciPy BFGS and L-BFGS methods.
    
    Provides quasi-Newton optimization functionality using SciPy's BFGS
    and L-BFGS implementations with various configuration options.
    """
    
    def __init__(self,
                 method: str = 'BFGS',
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 max_iterations: int = 100000,
                 convergence_tolerance: float = 1e-3,
                 random_state: Optional[int] = None,
                 # BFGS-specific parameters
                 gtol: Optional[float] = None,
                 norm: float = float('inf'),
                 eps: float = 1.4901161193847656e-08,
                 # L-BFGS-B specific parameters
                 m: int = 10,  # Memory limit for L-BFGS-B
                 factr: float = 1e7,  # Tolerance for L-BFGS-B
                 pgtol: float = 1e-5,  # Projected gradient tolerance
                 maxcor: int = 10,  # Maximum corrections for L-BFGS-B
                 **kwargs):
        """
        Initialize SciPy BFGS wrapper.
        
        Args:
            method: BFGS method ('BFGS', 'L-BFGS-B')
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
            random_state: Random seed
            gtol: Gradient tolerance (None uses convergence_tolerance)
            norm: Order of norm for gradient tolerance
            eps: Step size for finite-difference approximation
            m: Memory limit for L-BFGS-B
            factr: Tolerance for L-BFGS-B (convergence when gradient norm < factr * machine_epsilon)
            pgtol: Projected gradient tolerance for L-BFGS-B
            maxcor: Maximum number of corrections for L-BFGS-B
            **kwargs: Additional parameters for scipy.optimize.minimize
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for SciPyBFGSWrapper")
        
        # Store BFGS-specific parameters
        self.method = method
        self.gtol = gtol if gtol is not None else convergence_tolerance
        self.norm = norm
        self.eps = eps
        self.m = m
        self.factr = factr
        self.pgtol = pgtol
        self.maxcor = maxcor
        self.scipy_kwargs = kwargs
        
        super().__init__(
            library_name='scipy',
            algorithm_name=method,
            loss_type=loss_type,
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            random_state=random_state,
            method=method,
            gtol=self.gtol,
            norm=norm,
            eps=eps,
            m=m,
            factr=factr,
            pgtol=pgtol,
            maxcor=maxcor,
            **kwargs
        )
        
        # SciPy optimization result
        self.scipy_result = None
    
    def _create_external_optimizer(self, n_features: int) -> str:
        """
        Create SciPy BFGS method specification.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            Method name for SciPy minimize
        """
        valid_methods = ['BFGS', 'L-BFGS-B']
        if self.method not in valid_methods:
            print(f"Warning: Unknown method {self.method}, using BFGS")
            return 'BFGS'
        return self.method
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run SciPy BFGS optimization.
        
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
            
            # Set up minimize options based on method
            minimize_options = {
                'maxiter': self.max_iterations,
                'disp': False
            }
            
            if self.method == 'BFGS':
                minimize_options.update({
                    'gtol': self.gtol,
                    'norm': self.norm,
                    'eps': self.eps
                })
            elif self.method == 'L-BFGS-B':
                minimize_options.update({
                    'ftol': self.factr * np.finfo(float).eps,
                    'gtol': self.pgtol,
                    'maxcor': self.maxcor
                })
            
            # Add any additional options
            minimize_options.update(self.scipy_kwargs.get('options', {}))
            
            # Set bounds for L-BFGS-B (None means unbounded)
            bounds = None
            if self.method == 'L-BFGS-B':
                bounds = self.scipy_kwargs.get('bounds', None)
            
            # Run optimization
            minimize_args = {
                'fun': objective_func,
                'x0': self.weights,
                'method': self.external_optimizer,
                'jac': gradient_func,
                'callback': callback_func,
                'options': minimize_options
            }
            
            if bounds is not None:
                minimize_args['bounds'] = bounds
            
            self.scipy_result = minimize(**minimize_args)
            
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
        Get SciPy BFGS-specific results.
        
        Returns:
            Dictionary containing SciPy BFGS-specific results
        """
        base_results = super()._get_algorithm_specific_results()
        
        scipy_specific = {
            'scipy_bfgs_params': {
                'method': self.method,
                'gtol': self.gtol,
                'norm': self.norm,
                'eps': self.eps
            }
        }
        
        # Add method-specific parameters
        if self.method == 'L-BFGS-B':
            scipy_specific['scipy_bfgs_params'].update({
                'm': self.m,
                'factr': self.factr,
                'pgtol': self.pgtol,
                'maxcor': self.maxcor
            })
        
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
            
            # L-BFGS-B specific information
            if self.method == 'L-BFGS-B' and hasattr(self.scipy_result, 'task'):
                scipy_specific['scipy_result']['task'] = self.scipy_result.task
        
        base_results['algorithm_specific'].update(scipy_specific)
        return base_results


def create_scipy_bfgs_optimizer(method: str = 'BFGS',
                               loss_type: str = 'ols',
                               regularization: float = 0.01,
                               max_iterations: int = 100000,
                               convergence_tolerance: float = 1e-3,
                               random_state: Optional[int] = None,
                               **kwargs) -> SciPyBFGSWrapper:
    """
    Factory function to create SciPy BFGS optimizer.
    
    Args:
        method: BFGS method ('BFGS', 'L-BFGS-B')
        loss_type: Type of loss function ('ols', 'ridge', 'lasso')
        regularization: Regularization parameter
        max_iterations: Maximum number of iterations
        convergence_tolerance: Convergence tolerance
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        SciPyBFGSWrapper instance
    """
    return SciPyBFGSWrapper(
        method=method,
        loss_type=loss_type,
        regularization=regularization,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        random_state=random_state,
        **kwargs
    )