"""
Optimization Utilities
This module provides utilities for optimization algorithms including
convergence checking, line search methods, and numerical stability functions.
"""
from .convergence import check_convergence, ConvergenceChecker
from .line_search import LineSearch, backtracking_line_search
from .numerical import check_numerical_stability, handle_numerical_issues
__all__ = [
    'check_convergence',
    'ConvergenceChecker',
    'LineSearch',
    'backtracking_line_search',
    'check_numerical_stability', 
    'handle_numerical_issues',
]