"""
Subgradient optimizer implementation for non-smooth optimization.
This module provides a unified Subgradient optimizer that supports various
step size strategies for non-smooth convex optimization problems.
"""
import numpy as np
from typing import Dict, Any, Optional, Union
from copy import deepcopy
from ..base import IterativeOptimizer
from ..components import StepSizeStrategy, create_step_size_strategy

class SubgradientStepSizeStrategy(StepSizeStrategy):
    """Base class for subgradient-specific step size strategies."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reset(self) -> None:
        """Reset strategy state."""
        pass

class ConstantSubgradientStepSize(SubgradientStepSizeStrategy):
    """Constant step size for subgradient method."""
    
    def __init__(self, step_size: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.step_size = step_size
    
    def compute_step_size(self, iteration: int, gradient: np.ndarray, 
                         X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Return constant step size."""
        return self.step_size
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'type': 'constant_subgradient',
            'step_size': self.step_size
        }

class SquareSummableStepSize(SubgradientStepSizeStrategy):
    """Square summable step size: alpha_k = a/(b + k)."""
    
    def __init__(self, a: float = 1.0, b: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
    
    def compute_step_size(self, iteration: int, gradient: np.ndarray,
                         X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Return square summable step size."""
        return self.a / (self.b + iteration)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'type': 'square_summable',
            'a': self.a,
            'b': self.b
        }

class NonSummableDiminishingStepSize(SubgradientStepSizeStrategy):
    """Non-summable diminishing step size: alpha_k = a/sqrt(k)."""
    
    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
    
    def compute_step_size(self, iteration: int, gradient: np.ndarray,
                         X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Return non-summable diminishing step size."""
        return self.a / np.sqrt(iteration)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'type': 'non_summable_diminishing',
            'a': self.a
        }

class ConstantStepLengthStepSize(SubgradientStepSizeStrategy):
    """Constant step length: alpha_k = step_length / ||g_k||."""
    
    def __init__(self, step_length: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.step_length = step_length
    
    def compute_step_size(self, iteration: int, gradient: np.ndarray,
                         X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Return constant step length normalized by gradient norm."""
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm == 0:
            return 0.0
        return self.step_length / gradient_norm
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'type': 'constant_step_length',
            'step_length': self.step_length
        }

def create_subgradient_step_size_strategy(method: str, **kwargs) -> SubgradientStepSizeStrategy:
    """
    Factory function for creating subgradient step size strategies.
    
    Args:
        method: Step size method ('constant', 'square_summable', 'non_summable', 'constant_length')
        **kwargs: Method-specific parameters
        
    Returns:
        SubgradientStepSizeStrategy instance
    """
    strategies = {
        'constant': ConstantSubgradientStepSize,
        'square_summable': SquareSummableStepSize,
        'non_summable_diminishing': NonSummableDiminishingStepSize,
        'constant_step_length': ConstantStepLengthStepSize
    }
    
    if method not in strategies:
        raise ValueError(f"Unknown subgradient step size method: {method}")
    
    return strategies[method](**kwargs)

class SubgradientOptimizer(IterativeOptimizer):
    """
    Subgradient optimizer for non-smooth convex optimization.
    
    Supports various step size strategies for problems with non-differentiable
    objective functions, particularly useful for L1-regularized problems.
    
    Parameters:
        ham_loss: Loss function type ('ols', 'ridge', 'lasso')
        lambda_penalty: L1 regularization parameter (for lasso)
        step_size_method: Step size strategy ('constant', 'square_summable', etc.)
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
    """
    
    def __init__(self,
                 ham_loss: str = 'lasso',
                 lambda_penalty: float = 0.1,
                 regularization: float = 0.1,  # For compatibility
                 diem_dung: float = 1e-8,
                 max_iterations: int = 10000,
                 convergence_check_freq: int = 50,
                 random_state: Optional[int] = None,
                 # Step size parameters
                 step_size_method: str = 'constant',
                 step_size: float = 0.02,
                 step_length: float = 0.1,
                 square_summable_a: float = 1.0,
                 square_summable_b: float = 0.0,
                 non_summable_a: float = 1.0):
        """
        Initialize SubgradientOptimizer.
        """
        super().__init__(
            ham_loss=ham_loss,
            regularization=max(lambda_penalty, regularization),  # Use the larger value
            diem_dung=diem_dung,
            max_iterations=max_iterations,
            convergence_check_freq=convergence_check_freq,
            random_state=random_state
        )
        
        # Store parameters
        self.lambda_penalty = max(lambda_penalty, regularization)
        self.step_size_method = step_size_method
        
        # Create step size strategy
        if step_size_method == 'constant':
            step_size_params = {'step_size': step_size}
        elif step_size_method == 'square_summable':
            step_size_params = {'a': square_summable_a, 'b': square_summable_b}
        elif step_size_method == 'non_summable_diminishing':
            step_size_params = {'a': non_summable_a}
        elif step_size_method == 'constant_step_length':
            step_size_params = {'step_length': step_length}
        else:
            step_size_params = {'step_size': step_size}
        
        self.step_size_strategy = create_subgradient_step_size_strategy(
            step_size_method, **step_size_params
        )
        
        # Track minimum loss (subgradient can have non-monotonic loss)
        self.min_loss_tracker = None
        self.min_loss_2_tracker = None
    
    def _initialize_algorithm_specific_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize subgradient-specific parameters."""
        self.step_size_strategy.reset()
        
        # Initialize minimum loss tracking
        BASE_LOSS_VALUE = 1000.0
        self.min_loss_tracker = {
            "iteration": 0,
            "loss_value": BASE_LOSS_VALUE,
            "weights": None
        }
        self.min_loss_2_tracker = {
            "iteration": 0,
            "loss_value": BASE_LOSS_VALUE,
            "weights": None
        }
    
    def _compute_update_direction(self, 
                                X: np.ndarray, 
                                y: np.ndarray, 
                                iteration: int) -> np.ndarray:
        """
        Compute subgradient direction.
        
        For non-smooth functions, we compute the subgradient instead of gradient.
        """
        # Get predictions and residuals
        predictions = X @ self.weights
        residuals = predictions - y
        
        # Gradient of squared loss term
        grad_loss = X.T @ residuals / X.shape[0]
        
        # Subgradient of L1 regularization term
        subgrad_reg = self.lambda_penalty * np.sign(self.weights)
        # Note: np.sign(0) = 0, which is valid since 0 is in the subgradient at 0
        
        # Full subgradient
        subgradient = grad_loss + subgrad_reg
        
        # Track gradient computation
        self.track_gradient_evaluation(X.shape)
        
        return subgradient
    
    def _compute_step_size(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          direction: np.ndarray,
                          iteration: int) -> Union[float, np.ndarray]:
        """Compute step size using subgradient strategy."""
        return self.step_size_strategy.compute_step_size(
            iteration, direction, X, y, self.weights
        )
    
    def _update_weights(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       direction: np.ndarray,
                       step_size: Union[float, np.ndarray],
                       iteration: int) -> None:
        """Update weights and track minimum loss."""
        # Standard weight update
        super()._update_weights(X, y, direction, step_size, iteration)
        
        # Update minimum loss tracking
        current_loss = self.loss_func(X, y, self.weights)
        
        if current_loss < self.min_loss_tracker["loss_value"]:
            self.min_loss_2_tracker = deepcopy(self.min_loss_tracker)
            self.min_loss_tracker = {
                "iteration": iteration,
                "loss_value": current_loss,
                "weights": deepcopy(self.weights)
            }
    
    def _check_convergence(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          iteration: int) -> tuple[bool, bool, str]:
        """Check convergence based on minimum loss stability."""
        # Use parent convergence check first
        should_stop, converged, reason = super()._check_convergence(X, y, iteration)
        
        if should_stop:
            return should_stop, converged, reason
        
        # Additional subgradient-specific convergence check
        if (iteration > 10 and 
            self.min_loss_tracker["loss_value"] < 1000.0 and
            abs(self.min_loss_tracker["loss_value"] - self.min_loss_2_tracker["loss_value"]) < self.diem_dung):
            
            # Set weights to minimum loss weights
            self.weights = deepcopy(self.min_loss_tracker["weights"])
            return True, True, "Minimum loss convergence"
        
        return False, False, ""
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """Get subgradient-specific results."""
        results = super()._get_algorithm_specific_results()
        
        # Set final weights to minimum loss weights if available
        if self.min_loss_tracker and self.min_loss_tracker["weights"] is not None:
            self.weights = deepcopy(self.min_loss_tracker["weights"])
            
            # Apply sparsity threshold (set small values to zero)
            self.weights[np.abs(self.weights) < 1e-3] = 0
        
        results.update({
            'algorithm_type': 'subgradient',
            'step_size_strategy': self.step_size_strategy.get_parameters(),
            'lambda_penalty': self.lambda_penalty,
            'uses_l1_regularization': True,
            'sparsity_level': float(np.sum(np.abs(self.weights) < 1e-3) / len(self.weights))
        })
        
        # Add minimum loss information
        if self.min_loss_tracker:
            results.update({
                'min_loss_iteration': self.min_loss_tracker["iteration"],
                'min_loss_value': self.min_loss_tracker["loss_value"]
            })
        
        return results
    
    def get_step_size_info(self) -> Dict[str, Any]:
        """Get step size strategy information."""
        return self.step_size_strategy.get_parameters()

def create_subgradient_optimizer(config: Dict[str, Any]) -> SubgradientOptimizer:
    """
    Factory function to create SubgradientOptimizer from config.
    
    Args:
        config: Dictionary containing all parameters
        
    Returns:
        Configured SubgradientOptimizer instance
    """
    return SubgradientOptimizer(**config)