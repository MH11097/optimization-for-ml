"""
Tiện ích Tối ưu hóa - Các hàm cần thiết cho Newton Method

=== MỤC ĐÍCH: TỐI ƯU HÓA ===

Bao gồm tất cả các hàm cần thiết cho:
1. Tính toán Gradient và Hessian
2. Giải hệ phương trình tuyến tính  
3. Kiểm tra hội tụ và line search
4. Đánh giá mô hình (MSE, MAE, R²)
5. Dự đoán và tính toán loss

Code đơn giản, dễ hiểu, dễ sử dụng.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, Dict, List


# ==============================================================================
# 1. TÍNH TOÁN GRADIENT VÀ HESSIAN
# ==============================================================================

def tinh_gradient_hoi_quy_tuyen_tinh(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray, 
                                   he_so_tu_do: float, dieu_chinh: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Tính vector gradient cho hồi quy tuyến tính với điều chỉnh
    
    Hàm mục tiêu: f(w,b) = (1/2n) * ||Xw + b - y||² + (λ/2) * ||w||²
    
    Gradient:
    - ∂f/∂w = (1/n) * X^T(Xw + b - y) + λ * w  
    - ∂f/∂b = (1/n) * Σ(Xw + b - y)
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        y: vector mục tiêu (n_samples,)
        trong_so: trọng số hiện tại (n_features,)
        he_so_tu_do: hệ số tự do hiện tại (scalar)
        dieu_chinh: hệ số điều chỉnh λ (mặc định 0.0)
    
    Trả về:
        gradient_w: gradient theo trọng số (n_features,)
        gradient_b: gradient theo hệ số tự do (scalar)
    """
    so_mau = X.shape[0]
    
    # Dự đoán hiện tại: ŷ = Xw + b
    du_doan = X @ trong_so + he_so_tu_do
    
    # Sai số: e = ŷ - y
    sai_so = du_doan - y
    
    # Gradient theo trọng số: ∂f/∂w = (1/n) * X^T * e + λ * w
    gradient_w = (1/so_mau) * X.T @ sai_so + dieu_chinh * trong_so
    
    # Gradient theo hệ số tự do: ∂f/∂b = (1/n) * Σ(e)
    gradient_b = (1/so_mau) * np.sum(sai_so)
    
    return gradient_w, gradient_b


def tinh_ma_tran_hessian_hoi_quy_tuyen_tinh(X: np.ndarray, dieu_chinh: float = 0.0) -> np.ndarray:
    """
    Tính ma trận Hessian cho hồi quy tuyến tính với điều chỉnh
    
    Ma trận Hessian: H = (1/n) * X^T * X + λ * I
    
    Đây là ma trận đạo hàm bậc 2 của hàm mục tiêu.
    Đối với hồi quy tuyến tính, Hessian là hằng số (không phụ thuộc vào w, b).
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        dieu_chinh: hệ số điều chỉnh λ (mặc định 0.0)
    
    Trả về:
        H: ma trận Hessian (n_features, n_features)
    
    Lưu ý: 
    - Ma trận này là positive semi-definite khi λ ≥ 0
    - Khi λ > 0, ma trận trở thành positive definite và khả nghịch
    """
    so_mau, so_dac_trung = X.shape
    
    # Tính tích X^T @ X (ma trận Gram)
    XTX = X.T @ X
    
    # Chia cho số mẫu và thêm điều chỉnh
    H = (1/so_mau) * XTX + dieu_chinh * np.eye(so_dac_trung)
    
    return H


# ==============================================================================
# 2. GIẢI HỆ PHƯƠNG TRÌNH VÀ KIỂM TRA MA TRẬN
# ==============================================================================

def giai_he_phuong_trinh_tuyen_tinh(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Giải hệ phương trình tuyến tính Ax = b một cách an toàn
    
    Sử dụng LU decomposition với partial pivoting để ổn định số học.
    
    Tham số:
        A: ma trận hệ số (n x n)
        b: vector vế phải (n,)
    
    Trả về:
        x: nghiệm của hệ phương trình (n,)
    
    Lưu ý: Tự động kiểm tra điều kiện và thêm regularization nếu cần
    """
    try:
        # Thử giải trực tiếp
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Nếu ma trận singular, thêm regularization nhỏ
        regularization = 1e-10
        A_reg = A + regularization * np.eye(A.shape[0])
        return np.linalg.solve(A_reg, b)


def kiem_tra_positive_definite(matrix: np.ndarray) -> bool:
    """
    Kiểm tra ma trận có phải positive definite không
    
    Ma trận positive definite khi tất cả eigenvalues > 0.
    Điều này quan trọng cho Newton Method.
    
    Tham số:
        matrix: ma trận cần kiểm tra (n x n)
    
    Trả về:
        bool: True nếu positive definite, False nếu không
    """
    try:
        # Thử Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def tinh_condition_number(matrix: np.ndarray) -> float:
    """
    Tính condition number của ma trận
    
    Condition number cho biết mức độ ill-conditioned của ma trận.
    - Gần 1: ma trận well-conditioned
    - Rất lớn: ma trận ill-conditioned
    
    Tham số:
        matrix: ma trận cần tính (n x n)
    
    Trả về:
        float: condition number
    """
    return np.linalg.cond(matrix)


# ==============================================================================
# 3. ĐÁNH GIÁ MÔ HÌNH VÀ DỰ ĐOÁN
# ==============================================================================

def du_doan(X: np.ndarray, w: np.ndarray, he_so_tu_do: float) -> np.ndarray:
    """
    Thực hiện dự đoán với mô hình tuyến tính
    
    Công thức: ŷ = Xw + he_so_tu_do
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        w: trọng số đã học (n_features,)
        he_so_tu_do: hệ số tự do đã học (scalar)
    
    Trả về:
        predictions: dự đoán (n_samples,)
    """
    return X @ w + he_so_tu_do


def tinh_mse(y_that: np.ndarray, y_du_doan: np.ndarray) -> float:
    """
    Tính Mean Squared Error (Sai số bình phương trung bình)
    
    MSE = (1/n) * Σ(y_thật - y_dự_đoán)²
    
    Tham số:
        y_that: giá trị thật (n_samples,)
        y_du_doan: giá trị dự đoán (n_samples,)
    
    Trả về:
        float: MSE
    """
    return np.mean((y_that - y_du_doan) ** 2)


def tinh_mae(y_that: np.ndarray, y_du_doan: np.ndarray) -> float:
    """
    Tính Mean Absolute Error (Sai số tuyệt đối trung bình)
    
    MAE = (1/n) * Σ|y_thật - y_dự_đoán|
    
    Tham số:
        y_that: giá trị thật (n_samples,)
        y_du_doan: giá trị dự đoán (n_samples,)
    
    Trả về:
        float: MAE
    """
    return np.mean(np.abs(y_that - y_du_doan))


def tinh_r2_score(y_that: np.ndarray, y_du_doan: np.ndarray) -> float:
    """
    Tính R² score (Coefficient of determination)
    
    R² = 1 - (SS_res / SS_tot)
    Trong đó:
    - SS_res = Σ(y_thật - y_dự_đoán)² (residual sum of squares)
    - SS_tot = Σ(y_thật - ȳ)² (total sum of squares)
    
    Tham số:
        y_that: giá trị thật (n_samples,)
        y_du_doan: giá trị dự đoán (n_samples,)
    
    Trả về:
        float: R² score (1.0 = perfect, 0.0 = no better than mean)
    """
    ss_res = np.sum((y_that - y_du_doan) ** 2)
    ss_tot = np.sum((y_that - np.mean(y_that)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


# ==============================================================================
# 4. KIỂM TRA HỘI TỤ VÀ LINE SEARCH
# ==============================================================================

def kiem_tra_hoi_tu(gradient_norm: float, cost_change: float, iteration: int,
                   tolerance: float = 1e-6, max_iterations: int = 100) -> Tuple[bool, str]:
    """
    Kiểm tra điều kiện hội tụ cho thuật toán optimization
    
    Tham số:
        gradient_norm: chuẩn của gradient hiện tại
        cost_change: thay đổi cost từ iteration trước
        iteration: số iteration hiện tại
        tolerance: ngưỡng hội tụ
        max_iterations: số iteration tối đa
    
    Trả về:
        converged: có hội tụ hay không
        reason: lý do dừng
    """
    # Hội tụ theo gradient norm
    if gradient_norm < tolerance:
        return True, f"Hội tụ theo chuẩn gradient: {gradient_norm:.2e} < {tolerance:.2e}"
    
    # Hội tụ theo thay đổi cost
    if iteration > 0 and abs(cost_change) < tolerance:
        return True, f"Hội tụ theo thay đổi cost: {abs(cost_change):.2e} < {tolerance:.2e}"
    
    # Đạt giới hạn iteration
    if iteration >= max_iterations:
        return True, f"Đạt giới hạn iteration: {iteration}"
    
    return False, ""


def backtracking_line_search(cost_func: Callable, gradient: np.ndarray, 
                            current_point: np.ndarray, search_direction: np.ndarray,
                            current_cost: float, alpha_init: float = 1.0,
                            rho: float = 0.5, c1: float = 1e-4, max_iter: int = 50) -> float:
    """
    Backtracking line search để tìm learning rate tối ưu
    
    Sử dụng Armijo condition để đảm bảo sufficient decrease.
    
    Tham số:
        cost_func: hàm tính cost
        gradient: gradient tại điểm hiện tại
        current_point: điểm hiện tại
        search_direction: hướng tìm kiếm (thường là -H^{-1} * gradient)
        current_cost: cost tại điểm hiện tại
        alpha_init: learning rate ban đầu
        rho: hệ số giảm learning rate
        c1: hằng số cho Armijo condition
        max_iter: số iteration tối đa
    
    Trả về:
        alpha: learning rate tối ưu tìm được
    """
    alpha = alpha_init
    directional_derivative = gradient.T @ search_direction
    
    for i in range(max_iter):
        new_point = current_point + alpha * search_direction
        new_cost = cost_func(new_point)
        
        # Armijo condition: f(x + α*d) ≤ f(x) + c1*α*∇f(x)^T*d
        if new_cost <= current_cost + c1 * alpha * directional_derivative:
            return alpha
        
        alpha *= rho
    
    return alpha


# ==============================================================================
# 5. HÀM LOSS CHO CÁC BIẾN THỂ
# ==============================================================================

def tinh_loss_ols(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray, he_so_tu_do: float) -> float:
    """
    Tính loss cho OLS (Ordinary Least Squares)
    
    Loss = (1/2n) * ||Xw + b - y||²
    """
    predictions = du_doan(X, trong_so, he_so_tu_do)
    return 0.5 * tinh_mse(y, predictions)


def tinh_loss_ridge(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray, 
                   he_so_tu_do: float, dieu_chinh: float) -> float:
    """
    Tính loss cho Ridge Regression
    
    Loss = (1/2n) * ||Xw + b - y||² + (λ/2) * ||w||²
    """
    mse_loss = tinh_loss_ols(X, y, trong_so, he_so_tu_do)
    l2_penalty = 0.5 * dieu_chinh * np.sum(trong_so ** 2)
    return mse_loss + l2_penalty


def tinh_loss_lasso_smooth(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray,
                          he_so_tu_do: float, dieu_chinh: float, epsilon: float = 1e-8) -> float:
    """
    Tính loss cho Lasso Regression (smooth approximation)
    
    Loss = (1/2n) * ||Xw + b - y||² + λ * Σ√(w²+ ε)
    """
    mse_loss = tinh_loss_ols(X, y, trong_so, he_so_tu_do)
    smooth_l1 = dieu_chinh * np.sum(np.sqrt(trong_so ** 2 + epsilon))
    return mse_loss + smooth_l1


# ==============================================================================
# 6. TIỆN ÍCH DEBUG VÀ IN THÔNG TIN
# ==============================================================================

def in_thong_tin_ma_tran(matrix: np.ndarray, ten_ma_tran: str = "Matrix"):
    """
    In thông tin chi tiết về ma trận
    """
    print(f"\n=== {ten_ma_tran} ===")
    print(f"Kích thước: {matrix.shape}")
    print(f"Condition number: {tinh_condition_number(matrix):.2e}")
    print(f"Positive definite: {kiem_tra_positive_definite(matrix)}")
    print(f"Eigenvalues min/max: {np.min(np.linalg.eigvals(matrix)):.2e} / {np.max(np.linalg.eigvals(matrix)):.2e}")


def in_thong_tin_gradient(gradient_w: np.ndarray, gradient_b: float):
    """
    In thông tin về gradient
    """
    print(f"\n=== Gradient Info ===")
    print(f"||∇w||: {np.linalg.norm(gradient_w):.2e}")
    print(f"|∇b|: {abs(gradient_b):.2e}")
    print(f"||∇f||: {np.sqrt(np.sum(gradient_w**2) + gradient_b**2):.2e}")