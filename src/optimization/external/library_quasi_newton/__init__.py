"""
Quasi-Newton Method wrappers for external libraries.
Contains BFGS and L-BFGS implementations from scipy, pytorch, and sklearn.

Available optimizers:
- SciPyBFGSWrapper: SciPy BFGS and L-BFGS-B methods
- PyTorchLBFGSWrapper: PyTorch L-BFGS optimizer
- SklearnLBFGSWrapper: scikit-learn L-BFGS solvers
"""

# Import all available wrappers
try:
    from .scipy_bfgs import SciPyBFGSWrapper, create_scipy_bfgs_optimizer
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    SciPyBFGSWrapper = None
    create_scipy_bfgs_optimizer = None

try:
    from .pytorch_lbfgs import PyTorchLBFGSWrapper, create_pytorch_lbfgs_optimizer
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    PyTorchLBFGSWrapper = None
    create_pytorch_lbfgs_optimizer = None

try:
    from .sklearn_lbfgs import SklearnLBFGSWrapper, create_sklearn_lbfgs_optimizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SklearnLBFGSWrapper = None
    create_sklearn_lbfgs_optimizer = None

# Create __all__ list with available optimizers
__all__ = []

if SCIPY_AVAILABLE:
    __all__.extend(['SciPyBFGSWrapper', 'create_scipy_bfgs_optimizer'])

if PYTORCH_AVAILABLE:
    __all__.extend(['PyTorchLBFGSWrapper', 'create_pytorch_lbfgs_optimizer'])

if SKLEARN_AVAILABLE:
    __all__.extend(['SklearnLBFGSWrapper', 'create_sklearn_lbfgs_optimizer'])

# Availability flags
__all__.extend(['SCIPY_AVAILABLE', 'PYTORCH_AVAILABLE', 'SKLEARN_AVAILABLE'])


def get_available_optimizers():
    """
    Get list of available quasi-Newton optimizers.
    
    Returns:
        Dict with optimizer names and availability status
    """
    return {
        'scipy_bfgs': SCIPY_AVAILABLE,
        'pytorch_lbfgs': PYTORCH_AVAILABLE,
        'sklearn_lbfgs': SKLEARN_AVAILABLE
    }


def create_quasi_newton_optimizer(library: str, **kwargs):
    """
    Factory function to create any quasi-Newton optimizer.
    
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
        return create_scipy_bfgs_optimizer(**kwargs)
    elif library.lower() == 'pytorch':
        if not PYTORCH_AVAILABLE:
            raise ValueError("PyTorch is not available")
        return create_pytorch_lbfgs_optimizer(**kwargs)
    elif library.lower() == 'sklearn':
        if not SKLEARN_AVAILABLE:
            raise ValueError("sklearn is not available")
        return create_sklearn_lbfgs_optimizer(**kwargs)
    else:
        available = [k for k, v in get_available_optimizers().items() if v]
        raise ValueError(f"Unknown library '{library}'. Available: {available}")