"""
Subgradient optimizer implementation for non-smooth optimization.
This module provides a unified Subgradient optimizer that supports various
step size strategies for non-smooth convex optimization problems.
"""
import numpy as np
from typing import Dict, Any, Optional, Union
from copy import deepcopy
import sys
import os
import json
import pandas as pd
from pathlib import Path

# Import the old base subgradient implementation
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "algorithms_old", "subgradient"))
from base_subgradient import BaseSubgradient

from ..base import IterativeOptimizer
from ..components import StepSizeStrategy, create_step_size_strategy
from ..base.optimizer_mixins import VisualizationMixin

# Import visualization utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from utils.visualization_utils import (
    ve_duong_hoi_tu,
    ve_duong_dong_muc_optimization,
    ve_du_doan_vs_thuc_te,
)
from utils.optimization_utils import add_bias_column

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
        # Prevent division by zero: use iteration + 1 or ensure b > 0
        denominator = max(self.b + iteration, 1e-8)
        return self.a / denominator
    
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
        # Prevent division by zero: use max(iteration, 1)
        safe_iteration = max(iteration, 1)
        return self.a / np.sqrt(safe_iteration)
    
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

# Concrete implementations of BaseSubgradient for different step size strategies
class ConstantStepSizeSubgradient(BaseSubgradient):
    """Constant step size subgradient implementation."""
    
    def __init__(self, step_size: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.step_size = step_size
    
    def get_step_size(self, *args, **kwargs):
        return self.step_size

class SquareSummableSubgradient(BaseSubgradient):
    """Square summable step size: alpha_k = a/(b + k)."""
    
    def __init__(self, a: float = 1.0, b: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
    
    def get_step_size(self, current_iteration, *args, **kwargs):
        return self.a / (self.b + current_iteration)

class NonSummableDiminishingSubgradient(BaseSubgradient):
    """Non-summable diminishing step size: alpha_k = a/sqrt(k)."""
    
    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
    
    def get_step_size(self, current_iteration, *args, **kwargs):
        return self.a / np.sqrt(current_iteration)

class ConstantStepLengthSubgradient(BaseSubgradient):
    """Constant step length: alpha_k = step_length / ||g_k||."""
    
    def __init__(self, step_length: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.step_length = step_length
    
    def get_step_size(self, current_subgradient_vector, *args, **kwargs):
        subgrad_norm = np.linalg.norm(current_subgradient_vector)
        if subgrad_norm == 0:
            return 0.0
        return self.step_length / subgrad_norm

class SubgradientOptimizer:
    """
    Subgradient optimizer wrapper that uses BaseSubgradient internally.
    
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
                 diem_dung: float = 10e-3,
                 max_iterations: int = 5000,
                 convergence_check_freq: int = 1,
                 random_state: Optional[int] = None,
                 # Step size parameters
                 step_size_method: str = 'constant',
                 step_size: float = 0.02,
                 step_length: float = 0.1,
                 square_summable_a: float = 1.0,
                 square_summable_b: float = 0.0,
                 non_summable_a: float = 1.0,
                 # Problem radius parameter
                 R: float = 10.0):
        """
        Initialize SubgradientOptimizer.
        """
        # Store parameters
        self.ham_loss = ham_loss
        self.lambda_penalty = max(lambda_penalty, regularization)
        self.diem_dung = diem_dung
        self.max_iterations = max_iterations
        self.step_size_method = step_size_method
        
        # Create the appropriate BaseSubgradient implementation
        base_params = {
            'lambda_penalty': self.lambda_penalty,
            'max_iterations': max_iterations,
            'tolerance': diem_dung,
            'R': R
        }
        
        if step_size_method == 'constant':
            self.subgradient_impl = ConstantStepSizeSubgradient(
                step_size=step_size, **base_params
            )
        elif step_size_method == 'square_summable':
            self.subgradient_impl = SquareSummableSubgradient(
                a=square_summable_a, b=square_summable_b, **base_params
            )
        elif step_size_method == 'non_summable_diminishing':
            self.subgradient_impl = NonSummableDiminishingSubgradient(
                a=non_summable_a, **base_params
            )
        elif step_size_method == 'constant_step_length':
            self.subgradient_impl = ConstantStepLengthSubgradient(
                step_length=step_length, **base_params
            )
        else:
            # Default to constant
            self.subgradient_impl = ConstantStepSizeSubgradient(
                step_size=step_size, **base_params
            )
        
        # Initialize result storage
        self.weights = None
        self.results = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model using BaseSubgradient implementation."""
        # Store problem size for complexity analysis
        self._problem_size = list(X.shape)
        
        # Use the BaseSubgradient implementation directly
        self.results = self.subgradient_impl.fit(X, y)
        self.weights = self.subgradient_impl.weights
        
        # Set additional attributes needed for VisualizationMixin
        self.ham_loss = 'lasso'  # Subgradient typically used for LASSO
        self.convergence_check_freq = 1  # Required for VisualizationMixin
        self.loss_func = self.subgradient_impl.loss_func  # Required for trajectory plot
        
        return self.results
    

    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model on test set."""
        metrics = self.subgradient_impl.evaluate(X_test, y_test)
        # Store metrics for later use in save_results
        self._last_metrics = metrics
        return metrics
    
    def save_results(self, ten_file: str, base_dir: str = "data/03_algorithms"):
        """Save model results to file using VisualizationMixin format."""
        # Use VisualizationMixin.save_detailed_results for consistent format
        return VisualizationMixin.save_detailed_results(self, ten_file, base_dir)
    
    def plot_results(self, X_test: np.ndarray, y_test: np.ndarray, 
                    ten_file: str, base_dir: str = "data/03_algorithms"):
        """Create visualization plots using VisualizationMixin format."""
        # Use VisualizationMixin.plot_results for consistent format with 3 charts
        # Don't specify algorithm_dir to let it use self.__class__.__name__.lower() automatically
        return VisualizationMixin.plot_results(
            self, X_test, y_test, ten_file, 
            base_dir=base_dir
        )
    
    def get_step_size_info(self) -> Dict[str, Any]:
        """Get step size strategy information."""
        step_info = {
            'type': self.step_size_method,
            'lambda_penalty': self.lambda_penalty
        }
        
        if hasattr(self.subgradient_impl, 'step_size'):
            step_info['step_size'] = self.subgradient_impl.step_size
        elif hasattr(self.subgradient_impl, 'step_length'):
            step_info['step_length'] = self.subgradient_impl.step_length
        elif hasattr(self.subgradient_impl, 'a'):
            step_info['a'] = self.subgradient_impl.a
            if hasattr(self.subgradient_impl, 'b'):
                step_info['b'] = self.subgradient_impl.b
        
        return step_info
    
    # Properties for compatibility
    @property
    def loss_history(self):
        return getattr(self.subgradient_impl, 'loss_history', [])
    
    @property
    def gradient_norms(self):
        return getattr(self.subgradient_impl, 'gradient_norms', [])
    
    @property
    def weights_history(self):
        return getattr(self.subgradient_impl, 'weights_history', [])
    
    @property
    def gap_history(self):
        return getattr(self.subgradient_impl, 'gap_history', [])
    
    @property
    def training_time(self):
        return getattr(self.subgradient_impl, 'training_time', 0)
    
    @property
    def converged(self):
        return getattr(self.subgradient_impl, 'converged', False)
    
    @property
    def final_iteration(self):
        return getattr(self.subgradient_impl, 'final_iteration', 0)
    
    # Methods required for VisualizationMixin compatibility
    def _get_best_results(self) -> Dict[str, Any]:
        """Get best results for visualization mixin."""
        return {
            "training_results": {
                "training_time": self.training_time,
                "converged": self.converged,
                "final_iteration": self.final_iteration,
                "total_iterations": getattr(self.subgradient_impl, 'max_iterations', 0),
                "final_loss": self.loss_history[-1] if self.loss_history else 0,
                "final_gradient_norm": self.gradient_norms[-1] if self.gradient_norms else 0,
                "final_gap": float(self.gap_history[-1]) if self.gap_history and not np.isnan(self.gap_history[-1]) else None,
                "best_gap": float(min([g for g in self.gap_history if not np.isnan(g)])) if any(not np.isnan(g) for g in self.gap_history) else None,
                "best_iteration": self.final_iteration,
                "best_loss": self.loss_history[-1] if self.loss_history else 0,
                "best_gradient_norm": self.gradient_norms[-1] if self.gradient_norms else 0
            },
            "parameters": {
                "lambda_penalty": self.lambda_penalty,
                "step_size_method": self.step_size_method,
                "max_iterations": getattr(self.subgradient_impl, 'max_iterations', 0),
                "tolerance": getattr(self.subgradient_impl, 'tolerance', 0)
            },
            "weights_analysis": self._get_weights_analysis() if self.weights is not None else {},
            "ml_metrics": getattr(self, '_last_metrics', {})
        }
    
    def _get_weights_analysis(self) -> Dict[str, Any]:
        """Analyze weights for detailed results."""
        if self.weights is None:
            return {}
        
        return {
            "n_features": len(self.weights) - 1,  # Exclude bias
            "n_weights_total": len(self.weights),
            "bias_value": float(self.weights[0]),  # First weight is bias
            "complete_weight_vector": [float(w) for w in self.weights],
            "weights_stats": {
                "min": float(np.min(self.weights)),
                "max": float(np.max(self.weights)),
                "mean": float(np.mean(self.weights)),
                "std": float(np.std(self.weights))
            }
        }
    
    def get_complexity_analysis(self) -> Dict[str, Any]:
        """Get computational complexity analysis."""
        return {
            "total_operations": 3,  # Basic subgradient operations
            "function_evaluations": 1,
            "gradient_evaluations": 1,
            "matrix_operations": 0,
            "vector_operations": 1,
            "peak_memory": len(self.weights) if self.weights is not None else 0,
            "convergence_iteration": self.final_iteration if self.converged else None,
            "problem_size": getattr(self, '_problem_size', [0, 0])
        }
    
    def predict(self, X: np.ndarray):
        """Make predictions using the trained model."""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        return self.subgradient_impl.predict(X)

def create_subgradient_optimizer(config: Dict[str, Any]) -> SubgradientOptimizer:
    """
    Factory function to create SubgradientOptimizer from config.
    
    Args:
        config: Dictionary containing all parameters
        
    Returns:
        Configured SubgradientOptimizer instance
    """
    return SubgradientOptimizer(**config)