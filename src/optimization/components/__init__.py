"""
Reusable components for optimization algorithms.
This module provides pluggable components that can be combined
to create different optimization strategies.
"""
from .step_size_strategies import (
    StepSizeStrategy, ConstantStepSize, AdaptiveStepSize, 
    create_step_size_strategy
)
from .line_search import (
    LineSearchStrategy, ArmijoLineSearch, WolfeLineSearch,
    create_line_search_strategy
)
from .momentum import (
    MomentumStrategy, StandardMomentum,
    create_momentum_strategy
)
from .convergence import ConvergenceChecker
__all__ = [
    'StepSizeStrategy',
    'ConstantStepSize', 
    'AdaptiveStepSize',
    'create_step_size_strategy',
    'LineSearchStrategy',
    'ArmijoLineSearch',
    'WolfeLineSearch',
    'create_line_search_strategy',
    'MomentumStrategy',
    'StandardMomentum',
    'create_momentum_strategy',
    'ConvergenceChecker',
]