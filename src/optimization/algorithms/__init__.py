"""
Concrete optimizer implementations.
This module contains the specific implementations of optimization algorithms
that inherit from the base classes and mixins.
"""
from .gradient_descent import GradientDescentOptimizer
__all__ = [
    'GradientDescentOptimizer',
]