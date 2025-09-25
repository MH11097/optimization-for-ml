"""
Optimization Algorithms Module
This module provides a unified, object-oriented interface for optimization algorithms
with clean architecture and mathematical rigor.
Architecture:
- algorithms/: Core optimization algorithm implementations
- base/: Base classes and abstract interfaces
- components/: Reusable optimization components (line search, momentum, etc.)
- tests/: Comprehensive test suite
Features:
- Factory pattern for algorithm creation
- Component-based architecture for extensibility
- Backward compatibility with legacy interfaces
- Mathematical rigor and numerical stability
- Comprehensive testing and validation
Example:
    from optimization import OptimizerFactory
    
    # Create optimizer with modern interface
    optimizer = OptimizerFactory.create_optimizer(
        'gradient_descent',
        loss_type='ols',
        learning_rate=0.01,
        step_size_method='constant'
    )
    
    # Train the model
    results = optimizer.fit(X_train, y_train)
    
    # Make predictions
    predictions = optimizer.predict(X_test)
"""
from .factory import OptimizerFactory
from .base import BaseOptimizer, IterativeOptimizer
from .algorithms import GradientDescentOptimizer
# Version info
__version__ = "2.0.0"
__author__ = "Optimization ML Team"
# Main exports
__all__ = [
    # Factory pattern
    'OptimizerFactory',
    
    # Base classes
    'BaseOptimizer',
    'IterativeOptimizer',
    
    # Algorithm implementations
    'GradientDescentOptimizer',
    
    # Convenience functions
    'create_optimizer',
    'list_available_optimizers',
    'get_optimizer_info',
]
# Convenience functions for easier access
def create_optimizer(optimizer_name: str, **kwargs) -> BaseOptimizer:
    """
    Create an optimizer instance.
    
    Args:
        optimizer_name: Name of the optimizer ('gradient_descent', etc.)
        **kwargs: Optimizer parameters
        
    Returns:
        Configured optimizer instance
    """
    return OptimizerFactory.create_optimizer(optimizer_name, **kwargs)

def list_available_optimizers() -> list[str]:
    """
    List all available optimizer names.
    
    Returns:
        List of available optimizer names
    """
    return OptimizerFactory.list_available_optimizers()

def get_optimizer_info(optimizer_name: str) -> dict:
    """
    Get detailed information about an optimizer.
    
    Args:
        optimizer_name: Name of the optimizer
        
    Returns:
        Dictionary with optimizer information
    """
    return OptimizerFactory.get_optimizer_info(optimizer_name)

# Backward compatibility aliases
create_gradient_descent = OptimizerFactory.create_gradient_descent