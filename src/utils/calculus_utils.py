"""Module cơ bản cho tính toán gradient và Hessian trong optimization"""

import numpy as np
from typing import Callable, Tuple, Optional


def compute_gradient_linear_regression(X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                                     bias: float, regularization: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Tính gradient cho linear regression với regularization
    
    Cost function: f(w,b) = (1/2n) * ||Xw + b - y||² + (λ/2) * ||w||²
    
    Args:
        X: ma trận đặc trưng (n_samples, n_features)
        y: vector target (n_samples,)
        weights: trọng số hiện tại (n_features,)
        bias: bias hiện tại (scalar)
        regularization: hệ số regularization λ (mặc định 0.0)
    
    Returns:
        gradient_w: gradient theo weights (n_features,)
        gradient_b: gradient theo bias (scalar)
    """
    n_samples = X.shape[0]
    
    # Dự đoán hiện tại
    predictions = X @ weights + bias
    
    # Sai số
    errors = predictions - y
    
    # Gradient theo weights: (1/n) * X^T * errors + λ * weights
    gradient_w = (1/n_samples) * X.T @ errors + regularization * weights
    
    # Gradient theo bias: (1/n) * sum(errors)
    gradient_b = (1/n_samples) * np.sum(errors)
    
    return gradient_w, gradient_b


def compute_hessian_linear_regression(X: np.ndarray, regularization: float = 0.0) -> np.ndarray:
    """
    Tính Hessian cho linear regression với regularization
    
    Hessian: H = (1/n) * X^T * X + λ * I
    
    Args:
        X: ma trận đặc trưng (n_samples, n_features)
        regularization: hệ số regularization λ (mặc định 0.0)
    
    Returns:
        hessian: ma trận Hessian (n_features, n_features)
    """
    n_samples, n_features = X.shape
    
    # Hessian cơ bản: (1/n) * X^T * X
    hessian = (1/n_samples) * X.T @ X
    
    # Thêm regularization: λ * I
    if regularization > 0:
        hessian += regularization * np.eye(n_features)
    
    return hessian


def compute_gradient_numerical(func: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Tính gradient bằng phương pháp numerical differentiation (finite differences)
    
    Args:
        func: hàm cần tính gradient
        x: điểm tính gradient
        h: step size cho finite differences
    
    Returns:
        gradient: gradient tại x
    """
    n = len(x)
    gradient = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[i] += h
        x_minus[i] -= h
        
        # Central difference
        gradient[i] = (func(x_plus) - func(x_minus)) / (2 * h)
    
    return gradient


def compute_hessian_numerical(func: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Tính Hessian bằng phương pháp numerical differentiation
    
    Args:
        func: hàm cần tính Hessian
        x: điểm tính Hessian
        h: step size cho finite differences
    
    Returns:
        hessian: ma trận Hessian tại x
    """
    n = len(x)
    hessian = np.zeros((n, n))
    
    # Tính các phần tử trên đường chéo và trên tam giác trên
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal elements: ∂²f/∂x_i²
                x_plus = x.copy()
                x_minus = x.copy()
                
                x_plus[i] += h
                x_minus[i] -= h
                
                hessian[i, j] = (func(x_plus) - 2*func(x) + func(x_minus)) / (h**2)
            else:
                # Off-diagonal elements: ∂²f/∂x_i∂x_j
                x_pp = x.copy()  # x + h_i + h_j
                x_pm = x.copy()  # x + h_i - h_j
                x_mp = x.copy()  # x - h_i + h_j
                x_mm = x.copy()  # x - h_i - h_j
                
                x_pp[i] += h
                x_pp[j] += h
                
                x_pm[i] += h
                x_pm[j] -= h
                
                x_mp[i] -= h
                x_mp[j] += h
                
                x_mm[i] -= h
                x_mm[j] -= h
                
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
                hessian[j, i] = hessian[i, j]  # Symmetric
    
    return hessian


def check_positive_definite(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
    """
    Kiểm tra ma trận có positive definite hay không
    
    Args:
        matrix: ma trận cần kiểm tra
        tolerance: tolerance cho eigenvalues
    
    Returns:
        is_pd: True nếu positive definite
    """
    try:
        eigenvals = np.linalg.eigvals(matrix)
        return np.all(eigenvals > tolerance)
    except np.linalg.LinAlgError:
        return False


def regularize_hessian(hessian: np.ndarray, regularization: float = 1e-8) -> np.ndarray:
    """
    Regularize Hessian để đảm bảo positive definite
    
    Args:
        hessian: ma trận Hessian
        regularization: hệ số regularization
    
    Returns:
        regularized_hessian: Hessian đã regularize
    """
    n = hessian.shape[0]
    return hessian + regularization * np.eye(n)


def compute_condition_number(matrix: np.ndarray) -> float:
    """
    Tính condition number của ma trận
    
    Args:
        matrix: ma trận cần tính
    
    Returns:
        condition_number: condition number
    """
    try:
        return np.linalg.cond(matrix)
    except np.linalg.LinAlgError:
        return np.inf


def safe_matrix_inverse(matrix: np.ndarray, regularization: float = 1e-8) -> np.ndarray:
    """
    Tính nghịch đảo ma trận một cách an toàn
    
    Args:
        matrix: ma trận cần nghịch đảo
        regularization: regularization nếu singular
    
    Returns:
        inv_matrix: ma trận nghịch đảo
    """
    try:
        # Thử nghịch đảo trực tiếp
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Nếu singular, thêm regularization
        regularized_matrix = matrix + regularization * np.eye(matrix.shape[0])
        try:
            return np.linalg.inv(regularized_matrix)
        except np.linalg.LinAlgError:
            # Nếu vẫn không được, dùng pseudo-inverse
            return np.linalg.pinv(matrix)


def solve_linear_system(A: np.ndarray, b: np.ndarray, regularization: float = 1e-8) -> np.ndarray:
    """
    Giải hệ phương trình tuyến tính Ax = b một cách an toàn
    
    Args:
        A: ma trận hệ số
        b: vector vế phải
        regularization: regularization nếu ill-conditioned
    
    Returns:
        x: nghiệm của hệ phương trình
    """
    try:
        # Thử giải trực tiếp
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Nếu singular/ill-conditioned, thêm regularization
        regularized_A = A + regularization * np.eye(A.shape[0])
        try:
            return np.linalg.solve(regularized_A, b)
        except np.linalg.LinAlgError:
            # Dùng least squares nếu vẫn không được
            return np.linalg.lstsq(A, b, rcond=None)[0]


# Các hàm test để verify gradient và Hessian
def verify_gradient(func: Callable, gradient_func: Callable, x: np.ndarray, 
                   tolerance: float = 1e-5) -> bool:
    """
    Verify gradient calculation bằng cách so sánh với numerical gradient
    
    Args:
        func: hàm objective
        gradient_func: hàm tính gradient
        x: điểm test
        tolerance: tolerance cho sự khác biệt
    
    Returns:
        is_correct: True nếu gradient đúng
    """
    analytical_grad = gradient_func(x)
    numerical_grad = compute_gradient_numerical(func, x)
    
    max_diff = np.max(np.abs(analytical_grad - numerical_grad))
    return max_diff < tolerance


def verify_hessian(func: Callable, hessian_func: Callable, x: np.ndarray, 
                  tolerance: float = 1e-4) -> bool:
    """
    Verify Hessian calculation bằng cách so sánh với numerical Hessian
    
    Args:
        func: hàm objective
        hessian_func: hàm tính Hessian
        x: điểm test
        tolerance: tolerance cho sự khác biệt
    
    Returns:
        is_correct: True nếu Hessian đúng
    """
    analytical_hess = hessian_func(x)
    numerical_hess = compute_hessian_numerical(func, x)
    
    max_diff = np.max(np.abs(analytical_hess - numerical_hess))
    return max_diff < tolerance


# Helper functions cho debugging
def print_matrix_info(matrix: np.ndarray, name: str = "Matrix") -> None:
    """
    In thông tin về ma trận để debugging
    
    Args:
        matrix: ma trận cần in thông tin
        name: tên ma trận
    """
    print(f"\n{name} Information:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Condition number: {compute_condition_number(matrix):.2e}")
    print(f"  Determinant: {np.linalg.det(matrix):.2e}")
    
    eigenvals = np.linalg.eigvals(matrix)
    print(f"  Min eigenvalue: {np.min(eigenvals):.2e}")
    print(f"  Max eigenvalue: {np.max(eigenvals):.2e}")
    print(f"  Is positive definite: {check_positive_definite(matrix)}")


def print_gradient_info(gradient: np.ndarray, name: str = "Gradient") -> None:
    """
    In thông tin về gradient để debugging
    
    Args:
        gradient: vector gradient
        name: tên gradient
    """
    print(f"\n{name} Information:")
    print(f"  Shape: {gradient.shape}")
    print(f"  Norm: {np.linalg.norm(gradient):.6f}")
    print(f"  Max absolute value: {np.max(np.abs(gradient)):.6f}")
    print(f"  Min value: {np.min(gradient):.6f}")
    print(f"  Max value: {np.max(gradient):.6f}")