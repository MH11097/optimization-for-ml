"""
Newton Method wrappers for external libraries.
Contains implementations from scipy, pytorch, sklearn, and jax.

Available optimizers:
- SciPyNewtonWrapper: SciPy Newton methods (Newton-CG, trust-ncg, dogleg)
- PyTorchLBFGSWrapper: PyTorch L-BFGS optimizer
- SklearnNewtonWrapper: scikit-learn Newton solvers (newton-cg in Ridge)
"""

# Import all available wrappers
try:
    from .scipy_newton import SciPyNewtonWrapper, create_scipy_newton_optimizer
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    SciPyNewtonWrapper = None
    create_scipy_newton_optimizer = None

try:
    from .pytorch_lbfgs import PyTorchLBFGSWrapper, create_pytorch_lbfgs_optimizer
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    PyTorchLBFGSWrapper = None
    create_pytorch_lbfgs_optimizer = None

try:
    from .sklearn_newton import SklearnNewtonWrapper, create_sklearn_newton_optimizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SklearnNewtonWrapper = None
    create_sklearn_newton_optimizer = None

# Create __all__ list with available optimizers
__all__ = []

if SCIPY_AVAILABLE:
    __all__.extend(['SciPyNewtonWrapper', 'create_scipy_newton_optimizer'])

if PYTORCH_AVAILABLE:
    __all__.extend(['PyTorchLBFGSWrapper', 'create_pytorch_lbfgs_optimizer'])

if SKLEARN_AVAILABLE:
    __all__.extend(['SklearnNewtonWrapper', 'create_sklearn_newton_optimizer'])

# Availability flags
__all__.extend(['SCIPY_AVAILABLE', 'PYTORCH_AVAILABLE', 'SKLEARN_AVAILABLE'])


def get_available_optimizers():
    """
    Get list of available Newton method optimizers.
    
    Returns:
        Dict with optimizer names and availability status
    """
    return {
        'scipy_newton': SCIPY_AVAILABLE,
        'pytorch_lbfgs': PYTORCH_AVAILABLE,
        'sklearn_newton': SKLEARN_AVAILABLE
    }


def create_newton_optimizer(library: str, **kwargs):
    """
    Factory function to create any Newton method optimizer.
    
    Args:
        library: Library name ('scipy', 'pytorch', 'sklearn')
        **kwargs: Parameters for the specific optimizer
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If library is not available or unknown
    """
    if library.lower() == 'scipy':
        if not SCIPY_AVAILABLE:
            raise ValueError("scipy is not available")
        return create_scipy_newton_optimizer(**kwargs)
    elif library.lower() == 'pytorch':
        if not PYTORCH_AVAILABLE:
            raise ValueError("PyTorch is not available")
        return create_pytorch_lbfgs_optimizer(**kwargs)
    elif library.lower() == 'sklearn':
        if not SKLEARN_AVAILABLE:
            raise ValueError("sklearn is not available")
        return create_sklearn_newton_optimizer(**kwargs)
    else:
        available = [k for k, v in get_available_optimizers().items() if v]
        raise ValueError(f"Unknown library '{library}'. Available: {available}")