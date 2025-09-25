"""
Base classes for optimization algorithms.
This module provides abstract base classes and mixins that define
common interfaces and functionality for all optimization algorithms.
"""
from .base_optimizer import BaseOptimizer
from .iterative_optimizer import IterativeOptimizer
from .optimizer_mixins import ValidationMixin, ConvergenceMixin
__all__ = [
    'BaseOptimizer',
    'IterativeOptimizer',
    'ValidationMixin',
    'ConvergenceMixin',
]