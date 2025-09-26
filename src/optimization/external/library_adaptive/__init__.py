"""
Adaptive Optimization wrappers for external libraries.
Contains Adam, AdamW, RMSprop, and other adaptive methods from pytorch, tensorflow, and jax.
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

# PyTorch adaptive optimizers
if PYTORCH_AVAILABLE:
    from .pytorch_adaptive import (
        PyTorchAdaptive,
        create_pytorch_adam,
        create_pytorch_adamw,
        create_pytorch_rmsprop,
        create_pytorch_adagrad,
        create_pytorch_adadelta,
        create_pytorch_adamax
    )
    
    __all_pytorch__ = [
        'PyTorchAdaptive',
        'create_pytorch_adam',
        'create_pytorch_adamw',
        'create_pytorch_rmsprop',
        'create_pytorch_adagrad',
        'create_pytorch_adadelta',
        'create_pytorch_adamax'
    ]
else:
    __all_pytorch__ = []

# TensorFlow adaptive optimizers
if TENSORFLOW_AVAILABLE:
    from .tensorflow_adaptive import (
        TensorFlowAdaptive,
        create_tensorflow_adam,
        create_tensorflow_rmsprop,
        create_tensorflow_adagrad,
        create_tensorflow_adadelta,
        create_tensorflow_adamax,
        create_tensorflow_nadam,
        create_tensorflow_ftrl
    )
    
    __all_tensorflow__ = [
        'TensorFlowAdaptive',
        'create_tensorflow_adam',
        'create_tensorflow_rmsprop',
        'create_tensorflow_adagrad',
        'create_tensorflow_adadelta',
        'create_tensorflow_adamax',
        'create_tensorflow_nadam',
        'create_tensorflow_ftrl'
    ]
else:
    __all_tensorflow__ = []

# JAX adaptive optimizers
if JAX_AVAILABLE:
    from .jax_adaptive import (
        JAXAdaptive,
        create_jax_adam,
        create_jax_adamw,
        create_jax_rmsprop,
        create_jax_adagrad,
        create_jax_nadam,
        create_jax_yogi,
        create_jax_lamb
    )
    
    __all_jax__ = [
        'JAXAdaptive',
        'create_jax_adam',
        'create_jax_adamw',
        'create_jax_rmsprop',
        'create_jax_adagrad',
        'create_jax_nadam',
        'create_jax_yogi',
        'create_jax_lamb'
    ]
else:
    __all_jax__ = []

# Sklearn adaptive learning rate techniques
if SKLEARN_AVAILABLE:
    from .sklearn_adaptive import (
        SklearnAdaptive,
        create_sklearn_adaptive_standard,
        create_sklearn_optimal_lr,
        create_sklearn_invscaling_lr,
        create_sklearn_adaptive_momentum,
        create_sklearn_adaptive_aggressive
    )
    
    __all_sklearn__ = [
        'SklearnAdaptive',
        'create_sklearn_adaptive_standard',
        'create_sklearn_optimal_lr',
        'create_sklearn_invscaling_lr',
        'create_sklearn_adaptive_momentum',
        'create_sklearn_adaptive_aggressive'
    ]
else:
    __all_sklearn__ = []

# Combined exports
__all__ = (
    __all_pytorch__ + 
    __all_tensorflow__ + 
    __all_jax__ +
    __all_sklearn__
)

# Availability information
LIBRARY_AVAILABILITY = {
    'sklearn': SKLEARN_AVAILABLE,
    'pytorch': PYTORCH_AVAILABLE,
    'tensorflow': TENSORFLOW_AVAILABLE,
    'jax': JAX_AVAILABLE
}

def get_available_libraries():
    """Get list of available libraries for adaptive optimization."""
    return [lib for lib, available in LIBRARY_AVAILABILITY.items() if available]

def print_availability_status():
    """Print the availability status of all libraries."""
    print("Library Adaptive - Availability Status:")
    for lib, available in LIBRARY_AVAILABILITY.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {lib.ljust(12)}: {status}")
    print()

# Optimizer type mappings for easy access
OPTIMIZER_TYPES = {
    'adam': ['pytorch', 'tensorflow', 'jax'],
    'adamw': ['pytorch', 'jax'],
    'rmsprop': ['pytorch', 'tensorflow', 'jax'],
    'adagrad': ['pytorch', 'tensorflow', 'jax'],
    'adadelta': ['pytorch', 'tensorflow'],
    'adamax': ['pytorch', 'tensorflow'],
    'nadam': ['tensorflow', 'jax'],
    'ftrl': ['tensorflow'],
    'yogi': ['jax'],
    'lamb': ['jax'],
    'adaptive_lr': ['sklearn']
}

def get_available_optimizers():
    """Get mapping of available optimizers by library."""
    available = {}
    for optimizer, libraries in OPTIMIZER_TYPES.items():
        available_libs = [lib for lib in libraries if LIBRARY_AVAILABILITY[lib]]
        if available_libs:
            available[optimizer] = available_libs
    return available