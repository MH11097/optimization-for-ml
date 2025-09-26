"""
Base class for iterative optimization algorithms.
This module provides a common base for algorithms that use iterative
updates like gradient descent, SGD, Newton's method, etc.
"""
import numpy as np
from typing import Dict, Any, Optional
from abc import abstractmethod
from .base_optimizer import BaseOptimizer
from .optimizer_mixins import ValidationMixin, ConvergenceMixin, VisualizationMixin

class IterativeOptimizer(BaseOptimizer, ValidationMixin, ConvergenceMixin, VisualizationMixin):
    """
    Base class for iterative optimization algorithms.
    
    Inherits from BaseOptimizer and mixins to provide:
    - Input data validation
    - Advanced convergence checking  
    - Unified visualization
    
    Additional attributes:
        step_size_history: History of step sizes
        momentum_history: History of momentum (if applicable)
    """
    
    def __init__(self, 
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 convergence_check_freq: int = 1,
                 random_state: Optional[int] = None,
                 # Legacy parameter support
                 ham_loss: Optional[str] = None,
                 diem_dung: Optional[float] = None):
        """
        Initialize iterative optimizer.
        
        Args:
            loss_type: Type of loss function  
            regularization: Regularization parameter
            convergence_tolerance: Convergence threshold
            max_iterations: Maximum number of iterations
            convergence_check_freq: Frequency of convergence checking
            random_state: Seed for random number generator
            ham_loss: Legacy parameter name for loss_type
            diem_dung: Legacy parameter name for convergence_tolerance
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
        
        # Additional state for iterative algorithms
        self.step_size_history: list = []
        self.momentum_history: list = []
        self.learning_rate_history: list = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Template method with validation for iterative algorithms.
        
        Args:
            X: Feature matrix (without bias)
            y: Target vector
            
        Returns:
            Dictionary containing training results
        """
        # Validate input data
        self.validate_input_data(X, y)
        
        # Validate algorithm parameters
        self.validate_parameters()
        
        # Reset additional histories
        self.step_size_history = []
        self.momentum_history = []
        self.learning_rate_history = []
        
        # Call parent fit method
        return super().fit(X, y)
    
    @abstractmethod
    def _compute_update_direction(self, 
                                X: np.ndarray, 
                                y: np.ndarray, 
                                iteration: int) -> np.ndarray:
        """
        Compute update direction for weights.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            iteration: Current iteration number
            
        Returns:
            Update direction vector
        """
        pass
    
    @abstractmethod  
    def _compute_step_size(self,
                          X: np.ndarray,
                          y: np.ndarray, 
                          direction: np.ndarray,
                          iteration: int) -> float:
        """
        Compute step size for current iteration.
        
        Args:
            X: Feature matrix (with bias) 
            y: Target vector
            direction: Update direction
            iteration: Current iteration number
            
        Returns:
            Step size scalar or vector
        """
        pass
    
    def _perform_single_iteration(self, X: np.ndarray, y: np.ndarray, iteration: int) -> None:
        """
        Perform one iteration of the iterative algorithm.
        
        Template method:
        1. Compute update direction
        2. Compute step size  
        3. Update weights
        4. Save step size history
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            iteration: Current iteration number
        """
        # Step 1: Compute update direction
        direction = self._compute_update_direction(X, y, iteration)
        
        # Track vector operations
        self.track_vector_operation(len(direction), "basic")
        
        # Step 2: Compute step size
        step_size = self._compute_step_size(X, y, direction, iteration)
        
        # Step 3: Update weights
        if np.isscalar(step_size):
            # Scalar step size
            self.weights = self.weights + step_size * direction
            self.step_size_history.append(float(step_size))
            self.learning_rate_history.append(float(step_size))
        else:
            # Vector step size (e.g., adaptive methods)
            self.weights = self.weights + step_size * direction
            self.step_size_history.append(float(np.mean(step_size)))
            self.learning_rate_history.append(float(np.mean(step_size)))
        
        # Track weight update operations
        self.track_vector_operation(len(self.weights), "basic")
        self.track_memory_allocation(len(self.weights))
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get results specific to iterative algorithms.
        
        Returns:
            Dictionary containing step_size_history and momentum_history
        """
        results = {
            'step_sizes_history': self.step_size_history.copy(),
            'learning_rate_history': self.learning_rate_history.copy(),
        }

        # Add momentum history if available
        if hasattr(self, 'momentum_history') and self.momentum_history:
            results['momentum_history'] = self.momentum_history.copy()
        
        return results
        
    def _initialize_algorithm_specific_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize basic parameters for iterative algorithms.
        
        Can be overridden by subclasses for special initialization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
        """
        # Default: no additional initialization needed
        # Subclasses can override this for specific initialization
        pass