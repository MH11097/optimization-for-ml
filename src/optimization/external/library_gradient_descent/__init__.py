"""
Gradient Descent wrappers for external libraries.
Contains implementations from sklearn, pytorch, tensorflow, scipy, and jax.

Available optimizers:
- SklearnSGDWrapper: scikit-learn SGDRegressor
- PyTorchSGDWrapper: PyTorch SGD optimizer
- TensorFlowSGDWrapper: TensorFlow/Keras SGD optimizer
- SciPyCGWrapper: SciPy Conjugate Gradient
"""

# Import all available wrappers
try:
    from .sklearn_sgd import SklearnSGDWrapper, create_sklearn_sgd_optimizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SklearnSGDWrapper = None
    create_sklearn_sgd_optimizer = None

try:
    from .pytorch_sgd import PyTorchSGDWrapper, create_pytorch_sgd_optimizer
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    PyTorchSGDWrapper = None
    create_pytorch_sgd_optimizer = None

try:
    from .tensorflow_sgd import TensorFlowSGDWrapper, create_tensorflow_sgd_optimizer
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    TensorFlowSGDWrapper = None
    create_tensorflow_sgd_optimizer = None

try:
    from .scipy_cg import SciPyCGWrapper, create_scipy_cg_optimizer
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    SciPyCGWrapper = None
    create_scipy_cg_optimizer = None

# Create __all__ list with available optimizers
__all__ = []

if SKLEARN_AVAILABLE:
    __all__.extend(['SklearnSGDWrapper', 'create_sklearn_sgd_optimizer'])

if PYTORCH_AVAILABLE:
    __all__.extend(['PyTorchSGDWrapper', 'create_pytorch_sgd_optimizer'])

if TENSORFLOW_AVAILABLE:
    __all__.extend(['TensorFlowSGDWrapper', 'create_tensorflow_sgd_optimizer'])

if SCIPY_AVAILABLE:
    __all__.extend(['SciPyCGWrapper', 'create_scipy_cg_optimizer'])

# Availability flags
__all__.extend(['SKLEARN_AVAILABLE', 'PYTORCH_AVAILABLE', 'TENSORFLOW_AVAILABLE', 'SCIPY_AVAILABLE'])


def get_available_optimizers():
    """
    Get list of available gradient descent optimizers.
    
    Returns:
        Dict with optimizer names and availability status
    """
    return {
        'sklearn_sgd': SKLEARN_AVAILABLE,
        'pytorch_sgd': PYTORCH_AVAILABLE,
        'tensorflow_sgd': TENSORFLOW_AVAILABLE,
        'scipy_cg': SCIPY_AVAILABLE
    }


def create_gradient_descent_optimizer(library: str, **kwargs):
    """
    Factory function to create any gradient descent optimizer.
    
    Args:
        library: Library name ('sklearn', 'pytorch', 'tensorflow', 'scipy')
        **kwargs: Parameters for the specific optimizer
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If library is not available or unknown
    """
    if library.lower() == 'sklearn':
        if not SKLEARN_AVAILABLE:
            raise ValueError("sklearn is not available")
        return create_sklearn_sgd_optimizer(**kwargs)
    elif library.lower() == 'pytorch':
        if not PYTORCH_AVAILABLE:
            raise ValueError("PyTorch is not available")
        return create_pytorch_sgd_optimizer(**kwargs)
    elif library.lower() == 'tensorflow':
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow is not available")
        return create_tensorflow_sgd_optimizer(**kwargs)
    elif library.lower() == 'scipy':
        if not SCIPY_AVAILABLE:
            raise ValueError("SciPy is not available")
        return create_scipy_cg_optimizer(**kwargs)
    else:
        available = [k for k, v in get_available_optimizers().items() if v]
        raise ValueError(f"Unknown library '{library}'. Available: {available}")