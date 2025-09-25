"""
Linear algebra utilities for optimization algorithms.
This module provides efficient and numerically stable implementations of
common linear algebra operations used in machine learning optimization.
Mathematical Foundation:
- QR Decomposition: A = QR where Q is orthogonal, R is upper triangular
- SVD: A = UΣV^T for solving least squares and rank-deficient systems  
- Cholesky: A = L*L^T for positive definite matrices (faster than QR)
- Normal Equations: (X^T*X)*β = X^T*y for least squares
"""
import numpy as np
from typing import Tuple, Optional, Union
from enum import Enum
import warnings

class SolverMethod(Enum):
    """Enumeration of linear system solving methods."""
    QR = "qr"
    SVD = "svd"
    CHOLESKY = "cholesky"
    NORMAL_EQUATIONS = "normal"
    NUMPY_LSTSQ = "lstsq"

class LinearAlgebraUtils:
    """
    Unified utilities for linear algebra operations in optimization.
    
    This class provides numerically stable and efficient implementations
    of common linear algebra operations, replacing scattered utility functions.
    """
    
    @staticmethod
    def solve_linear_system(X: np.ndarray, 
                           y: np.ndarray,
                           method: SolverMethod = SolverMethod.QR,
                           regularization: float = 0.0) -> np.ndarray:
        """
        Solve linear system X*β = y for β using specified method.
        
        Args:
            X: Design matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            method: Solving method to use
            regularization: L2 regularization parameter
            
        Returns:
            Solution vector β
            
        Raises:
            ValueError: If inputs have incompatible shapes
            np.linalg.LinAlgError: If system is singular and method doesn't handle it
        """
        if X.shape[0] != len(y):
            raise ValueError("X and y must have compatible dimensions")
            
        # Add bias column if not present (assuming last column should be bias)
        if X.shape[1] == len(y) - 1:
            X = np.column_stack([X, np.ones(X.shape[0])])
            
        if method == SolverMethod.QR:
            return LinearAlgebraUtils._solve_qr(X, y, regularization)
        elif method == SolverMethod.SVD:
            return LinearAlgebraUtils._solve_svd(X, y, regularization)
        elif method == SolverMethod.CHOLESKY:
            return LinearAlgebraUtils._solve_cholesky(X, y, regularization)
        elif method == SolverMethod.NORMAL_EQUATIONS:
            return LinearAlgebraUtils._solve_normal(X, y, regularization)
        elif method == SolverMethod.NUMPY_LSTSQ:
            return LinearAlgebraUtils._solve_lstsq(X, y, regularization)
        else:
            raise ValueError(f"Unsupported solver method: {method}")
    
    @staticmethod
    def _solve_qr(X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """Solve using QR decomposition - numerically stable."""
        if reg > 0:
            # Add regularization by augmenting the system
            n_features = X.shape[1]
            X_aug = np.vstack([X, np.sqrt(reg) * np.eye(n_features)])
            y_aug = np.concatenate([y, np.zeros(n_features)])
            Q, R = np.linalg.qr(X_aug)
        else:
            Q, R = np.linalg.qr(X)
            
        # Solve R*β = Q^T*y
        if reg > 0:
            Qt_y = Q.T @ y_aug
        else:
            Qt_y = Q.T @ y
            
        return np.linalg.solve(R, Qt_y)
    
    @staticmethod
    def _solve_svd(X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """Solve using SVD - handles rank-deficient matrices."""
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Handle regularization and small singular values
        if reg > 0:
            s_reg = s / (s**2 + reg)
        else:
            # Use pseudo-inverse threshold
            threshold = np.finfo(s.dtype).eps * max(X.shape) * s[0]
            s_reg = np.where(s > threshold, 1/s, 0)
            
        return Vt.T @ (s_reg * (U.T @ y))
    
    @staticmethod 
    def _solve_cholesky(X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """Solve using Cholesky decomposition - fastest for well-conditioned systems."""
        XtX = X.T @ X
        Xty = X.T @ y
        
        if reg > 0:
            XtX += reg * np.eye(XtX.shape[0])
            
        try:
            L = np.linalg.cholesky(XtX)
            # Solve L*L^T*β = X^T*y
            z = np.linalg.solve(L, Xty)
            return np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            warnings.warn("Cholesky decomposition failed, falling back to SVD")
            return LinearAlgebraUtils._solve_svd(X, y, reg)
    
    @staticmethod
    def _solve_normal(X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """Solve normal equations - fastest but numerically unstable."""
        XtX = X.T @ X
        Xty = X.T @ y
        
        if reg > 0:
            XtX += reg * np.eye(XtX.shape[0])
            
        return np.linalg.solve(XtX, Xty)
    
    @staticmethod
    def _solve_lstsq(X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """Solve using numpy's least squares - general purpose."""
        if reg > 0:
            n_features = X.shape[1]
            X_aug = np.vstack([X, np.sqrt(reg) * np.eye(n_features)])
            y_aug = np.concatenate([y, np.zeros(n_features)])
            return np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]
        else:
            return np.linalg.lstsq(X, y, rcond=None)[0]
    
    @staticmethod
    def compute_condition_number(X: np.ndarray) -> float:
        """
        Compute condition number of matrix X.
        
        A high condition number indicates numerical instability.
        
        Args:
            X: Input matrix
            
        Returns:
            Condition number
        """
        return float(np.linalg.cond(X))
    
    @staticmethod
    def is_positive_definite(X: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Check if matrix is positive definite.
        
        Args:
            X: Square matrix to check
            tol: Tolerance for eigenvalue positivity
            
        Returns:
            True if positive definite
        """
        if X.shape[0] != X.shape[1]:
            return False
            
        try:
            eigenvals = np.linalg.eigvals(X)
            return np.all(eigenvals > tol)
        except np.linalg.LinAlgError:
            return False
    
    @staticmethod
    def safe_matrix_inverse(X: np.ndarray, 
                           regularization: float = 1e-8) -> np.ndarray:
        """
        Compute matrix inverse with regularization for numerical stability.
        
        Args:
            X: Square matrix to invert
            regularization: Regularization parameter to add to diagonal
            
        Returns:
            Inverse matrix
            
        Raises:
            ValueError: If matrix is not square
        """
        if X.shape[0] != X.shape[1]:
            raise ValueError("Matrix must be square for inversion")
            
        X_reg = X + regularization * np.eye(X.shape[0])
        return np.linalg.inv(X_reg)
    
    @staticmethod
    def compute_matrix_rank(X: np.ndarray, tol: float = None) -> int:
        """
        Compute numerical rank of matrix.
        
        Args:
            X: Input matrix
            tol: Tolerance for singular value cutoff
            
        Returns:
            Numerical rank
        """
        if tol is None:
            tol = np.finfo(X.dtype).eps * max(X.shape) * np.linalg.svd(X, compute_uv=False)[0]
            
        s = np.linalg.svd(X, compute_uv=False)
        return int(np.sum(s > tol))

class MatrixDecompositions:
    """Unified interface for matrix decompositions used in optimization."""
    
    @staticmethod
    def qr_decomposition(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute QR decomposition."""
        return np.linalg.qr(X)
    
    @staticmethod
    def svd_decomposition(X: np.ndarray, 
                         full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute SVD decomposition."""
        return np.linalg.svd(X, full_matrices=full_matrices)
    
    @staticmethod
    def cholesky_decomposition(X: np.ndarray) -> np.ndarray:
        """Compute Cholesky decomposition for positive definite matrices."""
        return np.linalg.cholesky(X)
    
    @staticmethod
    def eigenvalue_decomposition(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalue decomposition."""
        return np.linalg.eig(X)

# Backward compatibility functions
def giai_he_phuong_trinh_tuyen_tinh(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Backward compatibility wrapper for linear system solving."""
    return LinearAlgebraUtils.solve_linear_system(X, y, SolverMethod.QR)

def tinh_nghich_dao_ma_tran(X: np.ndarray) -> np.ndarray:
    """Backward compatibility wrapper for matrix inversion."""
    return LinearAlgebraUtils.safe_matrix_inverse(X)