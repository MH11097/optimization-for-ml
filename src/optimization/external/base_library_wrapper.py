"""
Base wrapper for external optimization libraries.
Provides a standardized interface to external libraries while maintaining
compatibility with the existing BaseOptimizer structure.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Callable
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.base.base_optimizer import BaseOptimizer
from utils.optimization_utils import (
    add_bias_column, 
    tinh_gia_tri_ham_loss, 
    tinh_gradient_ham_loss, 
    du_doan, 
    danh_gia_mo_hinh,
    in_ket_qua_danh_gia,
    kiem_tra_dieu_kien_dung
)


class BaseLibraryWrapper(BaseOptimizer):
    """
    Base class for wrapping external optimization libraries.
    
    This class extends BaseOptimizer to provide a standardized interface
    for external libraries while maintaining the same API and metrics tracking.
    """
    
    def __init__(self, 
                 library_name: str,
                 algorithm_name: str,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 convergence_check_freq: int = 1,
                 random_state: Optional[int] = None,
                 # Legacy parameter support
                 ham_loss: Optional[str] = None,
                 diem_dung: Optional[float] = None,
                 **library_specific_params):
        """
        Initialize base library wrapper.
        
        Args:
            library_name: Name of the external library (e.g., 'sklearn', 'pytorch', 'scipy')
            algorithm_name: Name of the specific algorithm (e.g., 'SGD', 'BFGS', 'Adam')
            loss_type: Type of loss function
            regularization: Regularization parameter
            convergence_tolerance: Convergence threshold
            max_iterations: Maximum number of iterations
            convergence_check_freq: Frequency of convergence checking
            random_state: Seed for random number generator
            ham_loss: Legacy parameter name for loss_type
            diem_dung: Legacy parameter name for convergence_tolerance
            **library_specific_params: Additional parameters specific to the library
        """
        super().__init__(
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            convergence_check_freq=convergence_check_freq,
            random_state=random_state,
            ham_loss=ham_loss,
            diem_dung=diem_dung
        )
        
        self.library_name = library_name
        self.algorithm_name = algorithm_name
        self.library_specific_params = library_specific_params
        
        # External optimizer instance (will be created in subclasses)
        self.external_optimizer = None
        
        # Callback tracking for external optimizers that support callbacks
        self.callback_history = {
            'losses': [],
            'gradient_norms': [],
            'weights': [],
            'iterations': []
        }
    
    @abstractmethod
    def _create_external_optimizer(self, n_features: int) -> Any:
        """
        Create the external optimizer instance.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            External optimizer instance
        """
        pass
    
    @abstractmethod
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run the external optimization algorithm.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        pass
    
    def _create_objective_function(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """
        Create objective function for external optimizers.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Objective function that takes weights as input
        """
        def objective(weights: np.ndarray) -> float:
            loss_value = self.loss_func(X, y, weights)
            # Track function evaluation
            self.track_function_evaluation(X.shape)
            return float(loss_value)
        
        return objective
    
    def _create_gradient_function(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """
        Create gradient function for external optimizers.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Gradient function that takes weights as input
        """
        def gradient(weights: np.ndarray) -> np.ndarray:
            grad_w, _ = self.grad_func(X, y, weights)
            # Track gradient evaluation
            self.track_gradient_evaluation(X.shape)
            self.track_vector_operation(len(grad_w), \"gradient_computation\")
            return grad_w
        
        return gradient
    
    def _create_callback_function(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """
        Create callback function for tracking optimization progress.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Callback function for external optimizers
        """
        def callback(weights: np.ndarray) -> None:
            # Compute current metrics
            loss_value = self.loss_func(X, y, weights)
            gradient_w, _ = self.grad_func(X, y, weights)
            gradient_norm = np.linalg.norm(gradient_w)
            
            # Store in callback history
            iteration = len(self.callback_history['losses'])
            self.callback_history['losses'].append(float(loss_value))
            self.callback_history['gradient_norms'].append(float(gradient_norm))
            self.callback_history['weights'].append(weights.copy())
            self.callback_history['iterations'].append(iteration)
            
            # Track complexity
            self.track_function_evaluation(X.shape)
            self.track_gradient_evaluation(X.shape)
            self.track_vector_operation(len(gradient_w), \"norm\")
            
        return callback
    
    def _initialize_algorithm_specific_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize algorithm-specific parameters for external optimizer.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
        """
        # Create external optimizer instance
        self.external_optimizer = self._create_external_optimizer(X.shape[1])
        
        # Reset callback history
        self.callback_history = {
            'losses': [],
            'gradient_norms': [],
            'weights': [],
            'iterations': []
        }
    
    def _perform_single_iteration(self, X: np.ndarray, y: np.ndarray, iteration: int) -> None:
        """
        This method is not used for external optimizers as they handle iterations internally.
        External optimization is performed in fit() method.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            iteration: Current iteration number
        """
        # External optimizers handle iterations internally
        # This method is required by BaseOptimizer but not used
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        \"\"\"
        Template method for training with external optimizer.
        
        Args:
            X: Feature matrix (without bias)
            y: Target vector
            
        Returns:
            Dictionary containing training results
        \"\"\"\n        print(f\"Training {self.library_name} {self.algorithm_name} - {self.loss_type.upper()}\")\n        if self.loss_type in ['ridge', 'lasso']:\n            print(f\"   Regularization: {self.regularization}\")\n        \n        # Add bias column to X\n        X_with_bias = add_bias_column(X)\n        print(f\"   Feature: {X.shape[1]} (+1 bias)\")\n        \n        # Initialize complexity tracker\n        self.init_complexity_tracker(X, y)\n        \n        # Initialize weights with fixed point for consistent comparison\n        n_features_with_bias = X_with_bias.shape[1]\n        # Use fixed seed for reproducible initialization\n        np.random.seed(42)\n        self.weights = np.random.normal(0, 0.01, n_features_with_bias)\n        # Reset to original random state if it was set\n        if self.random_state is not None:\n            np.random.seed(self.random_state)\n        \n        # Algorithm-specific initialization\n        self._initialize_algorithm_specific_params(X_with_bias, y)\n        \n        # Reset state\n        self.loss_history = []\n        self.gradient_norms = []\n        self.weights_history = []\n        self.converged = False\n        self.final_iteration = 0\n        \n        start_time = time.time()\n        \n        # Run external optimization\n        try:\n            external_results = self._optimize_external(X_with_bias, y)\n            \n            # Extract results from external optimizer\n            if 'final_weights' in external_results:\n                self.weights = external_results['final_weights']\n            \n            # Use callback history if available, otherwise create minimal history\n            if self.callback_history['losses']:\n                self.loss_history = self.callback_history['losses']\n                self.gradient_norms = self.callback_history['gradient_norms']\n                self.weights_history = self.callback_history['weights']\n                self.final_iteration = len(self.loss_history)\n            else:\n                # Create minimal history with final results\n                final_loss = self.loss_func(X_with_bias, y, self.weights)\n                final_gradient_w, _ = self.grad_func(X_with_bias, y, self.weights)\n                final_gradient_norm = np.linalg.norm(final_gradient_w)\n                \n                self.loss_history = [float(final_loss)]\n                self.gradient_norms = [float(final_gradient_norm)]\n                self.weights_history = [self.weights.copy()]\n                self.final_iteration = external_results.get('iterations', 1)\n            \n            # Check convergence based on external optimizer results\n            self.converged = external_results.get('converged', False)\n            \n            # Mark convergence in complexity tracker if converged\n            if self.converged:\n                self.mark_convergence_tracking(self.final_iteration)\n                print(f\"[SUCCESS] {self.library_name} {self.algorithm_name} converged\")\n            else:\n                print(f\"[WARNING] {self.library_name} {self.algorithm_name} did not converge\")\n            \n        except Exception as e:\n            print(f\"[ERROR] External optimization failed: {str(e)}\")\n            # Use initial weights if optimization failed\n            if self.weights is None:\n                self.weights = np.random.normal(0, 0.01, n_features_with_bias)\n            \n            # Create minimal history\n            final_loss = self.loss_func(X_with_bias, y, self.weights)\n            final_gradient_w, _ = self.grad_func(X_with_bias, y, self.weights)\n            final_gradient_norm = np.linalg.norm(final_gradient_w)\n            \n            self.loss_history = [float(final_loss)]\n            self.gradient_norms = [float(final_gradient_norm)]\n            self.weights_history = [self.weights.copy()]\n            self.converged = False\n            self.final_iteration = 1\n        \n        self.training_time = time.time() - start_time\n        \n        print(f\"Training time: {self.training_time:.2f}s\")\n        print(f\"Final loss: {self.loss_history[-1]:.6f}\")\n        print(f\"Final gradient norm: {self.gradient_norms[-1]:.6f}\")\n        \n        # Print complexity summary\n        self.print_complexity_summary()\n        \n        # Get best results\n        best_results = self._get_best_results()\n        print(f\"[BEST] Best results (lowest gradient norm):\")\n        print(f\"   Best iteration: {best_results['best_iteration']}\")\n        print(f\"   Best loss: {best_results['best_loss']:.6f}\")\n        print(f\"   Best gradient norm: {best_results['best_gradient_norm']:.6f}\")\n        \n        # Create results dictionary\n        results = self.create_standard_results_dict(\n            algorithm_name=f\"{self.library_name} {self.algorithm_name}\",\n            loss_function=self.loss_type\n        )\n        \n        # Add library-specific information\n        results['library_info'] = {\n            'library_name': self.library_name,\n            'algorithm_name': self.algorithm_name,\n            'library_params': self.library_specific_params\n        }\n        \n        # Add best results information\n        results['best_results'] = {\n            'best_weights': best_results['best_weights'],\n            'best_loss': best_results['best_loss'],\n            'best_iteration': best_results['best_iteration'],\n            'best_gradient_norm': best_results['best_gradient_norm']\n        }\n        \n        # Add standard fields for compatibility\n        results.update({\n            'weights': best_results['best_weights'],\n            'bias': best_results['best_weights'][-1],\n            'loss_history': self.loss_history,\n            'gradient_norms': self.gradient_norms,\n            'weights_history': self.weights_history,\n            'training_time': self.training_time,\n            'converged': self.converged,\n            'final_iteration': self.final_iteration,\n            'best_iteration': best_results['best_iteration'],\n            'best_loss': best_results['best_loss'],\n            'best_gradient_norm': best_results['best_gradient_norm'],\n            'final_loss': self.loss_history[-1],\n            'final_gradient_norm': self.gradient_norms[-1]\n        })\n        \n        # Add algorithm-specific results\n        algorithm_specific = self._get_algorithm_specific_results()\n        results.update(algorithm_specific)\n        \n        return results\n    \n    def _get_algorithm_specific_results(self) -> Dict[str, Any]:\n        \"\"\"\n        Get library-specific results.\n        \n        Returns:\n            Dictionary containing library-specific results\n        \"\"\"\n        return {\n            'algorithm_specific': {\n                'library_name': self.library_name,\n                'algorithm_name': self.algorithm_name,\n                'external_optimizer_type': type(self.external_optimizer).__name__ if self.external_optimizer else 'Unknown',\n                'library_params': self.library_specific_params,\n                'callback_tracking_enabled': len(self.callback_history['losses']) > 0\n            }\n        }