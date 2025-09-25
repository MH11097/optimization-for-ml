"""
Line search utilities for optimization algorithms.
"""
import numpy as np
from typing import Callable, Tuple

def backtracking_line_search(cost_func: Callable[[np.ndarray], float],
                             gradient: np.ndarray, weights: np.ndarray,
                             direction: np.ndarray, alpha: float = 1.0,
                             rho: float = 0.5, c1: float = 1e-4) -> float:
    """
    Backtracking line search implementation.
    Args:
        cost_func: Cost function
        gradient: Current gradient
        weights: Current weights
        direction: Search direction
        alpha: Initial step size
        rho: Backtracking factor
        c1: Armijo constant
    Returns:
        Optimal step size
    """
    current_cost = cost_func(weights)
    directional_derivative = np.dot(gradient, direction)
    while True:
        new_weights = weights + alpha * direction
        new_cost = cost_func(new_weights)
        # Armijo condition
        if new_cost <= current_cost + c1 * alpha * directional_derivative:
            return alpha
        alpha *= rho
        # Safety check
        if alpha < 1e-16:
            return alpha

class LineSearch:
    """Line search strategy interface."""
    def __init__(self, method: str = 'backtracking'):
        self.method = method
    def search(self, cost_func: Callable, gradient: np.ndarray,
               weights: np.ndarray, direction: np.ndarray) -> float:
        """Perform line search."""
        if self.method == 'backtracking':
            return backtracking_line_search(cost_func, gradient, weights, direction)
        else:
            return 1.0  # Default step size