"""
Convergence checking utilities for optimization algorithms.
"""
import numpy as np
from typing import Dict, Any, Optional

def check_convergence(gradient_norm: float, cost_change: float = None,
                     iteration: int = None, tolerance: float = 1e-6,
                     max_iterations: int = 100000) -> bool:
    """
    Check convergence conditions for optimization algorithms.
    Args:
        gradient_norm: Norm of the gradient
        cost_change: Change in cost function
        iteration: Current iteration number
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations allowed
    Returns:
        True if converged
    """
    # Check gradient norm convergence
    if gradient_norm < tolerance:
        return True
    # Check cost change convergence
    if cost_change is not None and abs(cost_change) < tolerance:
        return True
    # Check maximum iterations
    if iteration is not None and iteration >= max_iterations:
        return True
    return False

class ConvergenceChecker:
    """Enhanced convergence checker with multiple criteria."""
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    def check(self, gradient_norm: float, cost_change: float = None,
              iteration: int = None) -> bool:
        """Check convergence using configured criteria."""
        return check_convergence(
            gradient_norm, cost_change, iteration,
            self.tolerance, self.max_iterations
        )