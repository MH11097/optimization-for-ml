"""
Stochastic Gradient Descent wrappers for external libraries.
Contains SGD variants from sklearn, pytorch, tensorflow, and jax.
"""

# Import availability checks
SKLEARN_AVAILABLE = True
PYTORCH_AVAILABLE = True
TENSORFLOW_AVAILABLE = True
JAX_AVAILABLE = True

try:
    from sklearn.linear_model import SGDRegressor
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import jax
    import optax
except ImportError:
    JAX_AVAILABLE = False

# Sklearn SGD variants
if SKLEARN_AVAILABLE:
    from .sklearn_sgd_variants import (
        SklearnSGDVariants,
        create_sklearn_sgd_constant,
        create_sklearn_sgd_optimal,
        create_sklearn_sgd_adaptive,
        create_sklearn_sgd_momentum,
        create_sklearn_passive_aggressive
    )
    
    __all_sklearn__ = [
        'SklearnSGDVariants',
        'create_sklearn_sgd_constant',
        'create_sklearn_sgd_optimal', 
        'create_sklearn_sgd_adaptive',
        'create_sklearn_sgd_momentum',
        'create_sklearn_passive_aggressive'
    ]
else:
    __all_sklearn__ = []

# PyTorch SGD variants
if PYTORCH_AVAILABLE:
    from .pytorch_sgd_variants import (
        PyTorchSGDVariants,
        create_pytorch_sgd_vanilla,
        create_pytorch_sgd_momentum,
        create_pytorch_sgd_nesterov,
        create_pytorch_sgd_weight_decay,
        create_pytorch_sgd_full_batch
    )
    
    __all_pytorch__ = [
        'PyTorchSGDVariants',
        'create_pytorch_sgd_vanilla',
        'create_pytorch_sgd_momentum',
        'create_pytorch_sgd_nesterov',
        'create_pytorch_sgd_weight_decay',
        'create_pytorch_sgd_full_batch'
    ]
else:
    __all_pytorch__ = []

# TensorFlow SGD variants
if TENSORFLOW_AVAILABLE:
    from .tensorflow_sgd_variants import (
        TensorFlowSGDVariants,
        create_tensorflow_sgd_vanilla,
        create_tensorflow_sgd_momentum,
        create_tensorflow_sgd_nesterov,
        create_tensorflow_sgd_exponential_decay,
        create_tensorflow_sgd_polynomial_decay
    )
    
    __all_tensorflow__ = [
        'TensorFlowSGDVariants',
        'create_tensorflow_sgd_vanilla',
        'create_tensorflow_sgd_momentum',
        'create_tensorflow_sgd_nesterov',
        'create_tensorflow_sgd_exponential_decay',
        'create_tensorflow_sgd_polynomial_decay'
    ]
else:
    __all_tensorflow__ = []

# JAX SGD variants
if JAX_AVAILABLE:
    from .jax_sgd import (
        JAXSGDVariants,
        create_jax_sgd_vanilla,
        create_jax_sgd_momentum,
        create_jax_sgd_nesterov,
        create_jax_sgd_weight_decay
    )
    
    __all_jax__ = [
        'JAXSGDVariants',
        'create_jax_sgd_vanilla',
        'create_jax_sgd_momentum',
        'create_jax_sgd_nesterov',
        'create_jax_sgd_weight_decay'
    ]
else:
    __all_jax__ = []

# Combined exports
__all__ = (
    __all_sklearn__ + 
    __all_pytorch__ + 
    __all_tensorflow__ + 
    __all_jax__
)

# Availability information
LIBRARY_AVAILABILITY = {
    'sklearn': SKLEARN_AVAILABLE,
    'pytorch': PYTORCH_AVAILABLE,
    'tensorflow': TENSORFLOW_AVAILABLE,
    'jax': JAX_AVAILABLE
}

def get_available_libraries():
    """Get list of available libraries for stochastic optimization."""
    return [lib for lib, available in LIBRARY_AVAILABILITY.items() if available]

def print_availability_status():
    """Print the availability status of all libraries."""
    print("Library Stochastic - Availability Status:")
    for lib, available in LIBRARY_AVAILABILITY.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {lib.ljust(12)}: {status}")
    print()