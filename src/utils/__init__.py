"""
Refactored Utilities for Machine Learning Optimization
This package provides clean, scientific, and well-organized utilities for
machine learning optimization algorithms. It replaces the original utils
with a more maintainable and extensible architecture.
Key Features:
- Unified mathematical functions (no duplication)
- Clean English naming conventions
- Scientific documentation with mathematical formulations
- Type safety and comprehensive error handling
- Modular organization by functionality
Quick Start:
    from utils_refactored.core import LossFunction, GradientFunction
    from utils_refactored.data import add_bias_column
    from utils_refactored.optimization import check_convergence
    
    # Unified interface for loss functions
    loss_fn = LossFunction()
    loss = loss_fn.compute('ols', X, y, weights)
    
    # Clean gradient computation
    grad_fn = GradientFunction()
    gradient = grad_fn.compute('ridge', X, y, weights, alpha=0.01)
Modules:
    core: Mathematical functions (loss, gradient, hessian, metrics)
    data: Data preprocessing and validation
    optimization: Convergence, line search, numerical methods
    visualization: Plotting and visualization utilities
    mixins: Advanced mixins for complexity and results tracking
"""
__version__ = "1.0.0"
__author__ = "ML Optimization Team"
# Core mathematical functions
from .core import LossFunction, GradientFunction, HessianFunction, MetricsCalculator
# Data utilities
from .data import add_bias_column, validate_input_data, preprocess_data
# Optimization utilities  
from .optimization import check_convergence, LineSearch
# Backward compatibility - import all original functions
from .optimization_utils import *
from .data_process_utils import *
from .visualization_utils import *
from .model_mixins import *
# Commonly used functions for backward compatibility
from .core.loss_functions import compute_loss
from .core.gradients import compute_gradient
__all__ = [
    # Core classes
    'LossFunction',
    'GradientFunction', 
    'HessianFunction',
    'MetricsCalculator',
    # Data functions
    'add_bias_column',
    'validate_input_data',
    'preprocess_data',
    # Optimization functions
    'check_convergence',
    'LineSearch',
    # Backward compatibility
    'compute_loss',
    'compute_gradient',
]