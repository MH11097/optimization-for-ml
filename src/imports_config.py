"""
Import Configuration Module
This module provides centralized import management to eliminate sys.path manipulations
throughout the codebase. All modules can import from this to get clean access to
project modules.
Usage:
    # Instead of sys.path manipulations, use:
    from src.imports_config import *
    
    # Or specific imports:
    from src.imports_config import utils_refactored, optimization, legacy_utils
"""
import sys
import os
from pathlib import Path
# Get project root directory and src directory
_PROJECT_ROOT = Path(__file__).parent.parent
_SRC_ROOT = Path(__file__).parent
# Add both project root and src root to path
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
# Import refactored modules (now in utils)
try:
    import src.utils as utils_refactored  # Now utils contains the refactored code
    from src.utils.core import (
        LossFunction, GradientFunction, HessianFunction,
        MetricsCalculator, LinearAlgebraUtils
    )
    from src.utils.data import (
        FileLoader, DataCleaner, DataValidator, FeatureScaler
    )
    from src.utils.visualization import (
        setup_plot_style, plot_convergence, plot_algorithm_comparison
    )
except ImportError as e:
    print(f"Warning: Could not import refactored utilities: {e}")
    utils_refactored = None
# Import optimization modules  
try:
    import src.optimization as optimization
    from src.optimization import OptimizerFactory, create_optimizer
except ImportError as e:
    print(f"Warning: Could not import optimization modules: {e}")
    optimization = None
# Import legacy modules for backward compatibility (now using new utils)
try:
    import src.utils as legacy_utils
    from src.utils.optimization_utils import (
        add_bias_column, du_doan, danh_gia_mo_hinh,
        tinh_gia_tri_ham_loss, tinh_gradient_ham_loss
    )
    from src.utils.visualization_utils import (
        thiet_lap_style_bieu_do, ve_duong_hoi_tu, ve_so_sanh_algorithms
    )
    from src.utils.data_process_utils import (
        load_and_split_data, tach_du_lieu_train_test
    )
except ImportError as e:
    print(f"Warning: Could not import legacy utilities: {e}")
    legacy_utils = None
# Convenience functions for common operations
def get_project_root() -> Path:
    """Get the project root directory."""
    return _PROJECT_ROOT
def setup_imports():
    """
    Setup project imports without sys.path manipulation.
    Call this at the beginning of scripts that need project modules.
    """
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
# Legacy compatibility - functions that were commonly imported
def add_project_to_path():
    """Legacy function for backward compatibility."""
    setup_imports()
# Export commonly used items
__all__ = [
    # Modules
    'utils_refactored',
    'optimization', 
    'legacy_utils',
    
    # Core refactored utilities (if available)
    'LossFunction', 'GradientFunction', 'HessianFunction',
    'MetricsCalculator', 'LinearAlgebraUtils',
    'FileLoader', 'DataCleaner', 'DataValidator', 'FeatureScaler',
    'setup_plot_style', 'plot_convergence', 'plot_algorithm_comparison',
    
    # Optimization utilities (if available)
    'OptimizerFactory', 'create_optimizer',
    
    # Legacy utilities (if available)
    'add_bias_column', 'du_doan', 'danh_gia_mo_hinh',
    'tinh_gia_tri_ham_loss', 'tinh_gradient_ham_loss',
    'thiet_lap_style_bieu_do', 've_duong_hoi_tu', 've_so_sanh_algorithms',
    'load_and_split_data', 'tach_du_lieu_train_test',
    
    # Utility functions
    'get_project_root', 'setup_imports', 'add_project_to_path',
]