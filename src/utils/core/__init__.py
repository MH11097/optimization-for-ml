"""
Core mathematical utilities for optimization and machine learning.
This module provides unified, scientific implementations of fundamental
mathematical operations used across the optimization algorithms.
Key Components:
- LossFunction: Unified loss computation (OLS, Ridge, Lasso, etc.)
- GradientFunction: Unified gradient computation with mathematical rigor
- HessianFunction: Unified Hessian computation with proper regularization
- MetricsCalculator: Comprehensive evaluation metrics
- LinearAlgebraUtils: Numerically stable linear algebra operations
"""
from .loss_functions import LossFunction, LossType
from .gradients import GradientFunction, LossType
from .hessians import HessianFunction, LossType as HessianLossType
from .metrics import MetricsCalculator, MetricType
from .linear_algebra import LinearAlgebraUtils, MatrixDecompositions, SolverMethod
# Backward compatibility imports
from .loss_functions import (
    tinh_gia_tri_ham_OLS, tinh_gia_tri_ham_Ridge_with_bias,
    tinh_gia_tri_ham_Lasso_with_bias, tinh_gia_tri_ham_ElasticNet_with_bias
)
from .gradients import tinh_gradient_OLS, tinh_gradient_Ridge_with_bias
from .hessians import tinh_Hessian_OLS, tinh_Hessian_Ridge_with_bias
from .metrics import tinh_MSE, tinh_RMSE, tinh_MAE, tinh_R_squared
from .linear_algebra import giai_he_phuong_trinh_tuyen_tinh, tinh_nghich_dao_ma_tran
__all__ = [
    # Modern unified interfaces
    'LossFunction', 'LossType',
    'GradientFunction', 'LossType', 
    'HessianFunction', 'HessianLossType',
    'MetricsCalculator', 'MetricType',
    'LinearAlgebraUtils', 'MatrixDecompositions', 'SolverMethod',
    
    # Backward compatibility functions
    'tinh_gia_tri_ham_OLS', 'tinh_gia_tri_ham_Ridge_with_bias',
    'tinh_gia_tri_ham_Lasso_with_bias', 'tinh_gia_tri_ham_ElasticNet_with_bias',
    'tinh_gradient_OLS', 'tinh_gradient_Ridge_with_bias',
    'tinh_Hessian_OLS', 'tinh_Hessian_Ridge_with_bias',
    'tinh_MSE', 'tinh_RMSE', 'tinh_MAE', 'tinh_R_squared',
    'giai_he_phuong_trinh_tuyen_tinh', 'tinh_nghich_dao_ma_tran'
]