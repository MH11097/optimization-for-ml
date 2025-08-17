"""Utility functions for optimization and data processing"""

# Optimization utilities
from .optimization_utils import (
    predict,
    compute_mse,
    compute_mae,
    compute_r2_score,
    check_convergence,
    plot_convergence,
    backtracking_line_search,
    create_batches
)

# Mathematical utilities
from .math_utils import (
    # Data transformation
    convert_to_price_scale,
    log_transform_safe,
    inverse_log_transform,
    remove_outliers_iqr,
    normalize_features,
    standardize_features,
    
    # Optimization math
    compute_gradient,
    compute_hessian,
    soft_thresholding,
    proximal_l1,
    compute_subgradient_l1,
    armijo_condition,
    bfgs_update
)

# Data loading utilities
from .data_loader import (
    load_csv_safe,
    get_data_info,
    clean_column_names,
    reduce_memory_usage,
    load_data_chunked,
    optimize_dataframe_dtypes
)
from .plotting import *

__all__ = [
    # Optimization utilities
    'predict',
    'compute_mse',
    'compute_mae', 
    'compute_r2_score',
    'check_convergence',
    'plot_convergence',
    'backtracking_line_search',
    'create_batches',
    
    # Math utilities - data processing
    'convert_to_price_scale',
    'log_transform_safe',
    'inverse_log_transform',
    'remove_outliers_iqr',
    'normalize_features',
    'standardize_features',
    
    # Math utilities - optimization
    'compute_gradient',
    'compute_hessian',
    'soft_thresholding',
    'proximal_l1',
    'compute_subgradient_l1',
    'armijo_condition',
    'bfgs_update',
    
    # Data loading utilities
    'load_csv_safe',
    'get_data_info',
    'clean_column_names',
    'reduce_memory_usage',
    'load_data_chunked',
    'optimize_dataframe_dtypes'
]