"""
External optimization library wrappers.
Provides standardized interfaces to popular optimization libraries.
"""

# Base wrapper
from .base_library_wrapper import BaseLibraryWrapper

# Import submodules with availability checks
try:
    from . import library_gradient_descent
    LIBRARY_GRADIENT_DESCENT_AVAILABLE = True
except ImportError:
    LIBRARY_GRADIENT_DESCENT_AVAILABLE = False

try:
    from . import library_newton
    LIBRARY_NEWTON_AVAILABLE = True
except ImportError:
    LIBRARY_NEWTON_AVAILABLE = False

try:
    from . import library_quasi_newton
    LIBRARY_QUASI_NEWTON_AVAILABLE = True
except ImportError:
    LIBRARY_QUASI_NEWTON_AVAILABLE = False

try:
    from . import library_stochastic
    LIBRARY_STOCHASTIC_AVAILABLE = True
except ImportError:
    LIBRARY_STOCHASTIC_AVAILABLE = False

try:
    from . import library_adaptive
    LIBRARY_ADAPTIVE_AVAILABLE = True
except ImportError:
    LIBRARY_ADAPTIVE_AVAILABLE = False

# Availability mapping
ALGORITHM_FAMILY_AVAILABILITY = {
    'library_gradient_descent': LIBRARY_GRADIENT_DESCENT_AVAILABLE,
    'library_newton': LIBRARY_NEWTON_AVAILABLE,
    'library_quasi_newton': LIBRARY_QUASI_NEWTON_AVAILABLE,
    'library_stochastic': LIBRARY_STOCHASTIC_AVAILABLE,
    'library_adaptive': LIBRARY_ADAPTIVE_AVAILABLE
}

def get_available_algorithm_families():
    """Get list of available algorithm families."""
    return [family for family, available in ALGORITHM_FAMILY_AVAILABILITY.items() if available]

def print_external_library_status():
    """Print comprehensive status of all external library integrations."""
    print("=== External Library Integration Status ===")
    print()
    
    # Algorithm families
    print("Algorithm Families:")
    for family, available in ALGORITHM_FAMILY_AVAILABILITY.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {family.ljust(25)}: {status}")
    print()
    
    # Individual library availability
    if LIBRARY_STOCHASTIC_AVAILABLE:
        print("Stochastic Optimization Libraries:")
        available_libs = library_stochastic.get_available_libraries()
        all_libs = ['sklearn', 'pytorch', 'tensorflow', 'jax']
        for lib in all_libs:
            status = "✓ Available" if lib in available_libs else "✗ Not Available"
            print(f"  {lib.ljust(12)}: {status}")
        print()
    
    if LIBRARY_ADAPTIVE_AVAILABLE:
        print("Adaptive Optimization Libraries:")
        available_libs = library_adaptive.get_available_libraries()
        all_libs = ['sklearn', 'pytorch', 'tensorflow', 'jax']
        for lib in all_libs:
            status = "✓ Available" if lib in available_libs else "✗ Not Available"
            print(f"  {lib.ljust(12)}: {status}")
        print()
    
    # Available optimizers summary
    if LIBRARY_ADAPTIVE_AVAILABLE:
        print("Available Adaptive Optimizers:")
        optimizers = library_adaptive.get_available_optimizers()
        for optimizer, libs in optimizers.items():
            print(f"  {optimizer.ljust(15)}: {', '.join(libs)}")
        print()

# Export main components
__all__ = [
    'BaseLibraryWrapper',
    'get_available_algorithm_families',
    'print_external_library_status',
    'ALGORITHM_FAMILY_AVAILABILITY'
]

# Conditionally add available modules to __all__
if LIBRARY_GRADIENT_DESCENT_AVAILABLE:
    __all__.append('library_gradient_descent')

if LIBRARY_NEWTON_AVAILABLE:
    __all__.append('library_newton')

if LIBRARY_QUASI_NEWTON_AVAILABLE:
    __all__.append('library_quasi_newton')

if LIBRARY_STOCHASTIC_AVAILABLE:
    __all__.append('library_stochastic')

if LIBRARY_ADAPTIVE_AVAILABLE:
    __all__.append('library_adaptive')