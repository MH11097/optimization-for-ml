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
# 0. DATA PREPROCESSING UTILITIES
# ==============================================================================
def add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Thêm cột bias (cột toàn số 1) vào cuối ma trận X
    
    Chuyển từ format: Xw + b = y
    Sang format: X_new @ w_new = y (với w_new = [w; b])
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
    
    Trả về:
        X_with_bias: ma trận mở rộng (n_samples, n_features + 1)
                     với cột cuối cùng là cột bias (toàn số 1)
    """
    n_samples = X.shape[0]
    bias_column = np.ones((n_samples, 1))
    X_with_bias = np.hstack([X, bias_column])
    return X_with_bias

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
def du_doan(X: np.ndarray, w: np.ndarray, bias: float = None) -> np.ndarray:
    """
    Thực hiện dự đoán với mô hình tuyến tính
    
    Hỗ trợ cả 2 format:
    - Format cũ: ŷ = Xw + bias (khi bias != None)
    - Format mới: ŷ = Xw (khi X đã bao gồm cột bias và w bao gồm bias weight)
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        w: trọng số đã học (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
    
    Trả về:
        predictions: dự đoán trên log scale (n_samples,)
        
    Lưu ý: 
        - Model được train trên log-transformed targets
        - Predictions trả về ở log scale để consistency
        - Sử dụng np.expm1() để chuyển về original scale khi cần
    """
    if bias is not None:
        # Format cũ: Xw + bias
        predictions_log = X @ w + bias
    else:
        # Format mới: Xw (với X đã bao gồm cột bias)
        predictions_log = X @ w
    return predictions_log

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
# 4. NUMERICAL STABILITY CHECKS
# ==============================================================================
def check_for_numerical_issues(gradient_norm: float, loss_value: Optional[float] = None, 
                               weights: Optional[np.ndarray] = None, iteration: int = 0) -> Tuple[bool, str]:
    """
    Kiểm tra các vấn đề về tính ổn định số học trong quá trình optimization
    
    Tham số:
        gradient_norm: chuẩn của gradient hiện tại
        loss_value: giá trị loss hiện tại (tùy chọn)
        weights: vector trọng số hiện tại (tùy chọn)
        iteration: số iteration hiện tại
    
    Trả về:
        has_issues: True nếu phát hiện vấn đề số học
        issue_description: mô tả chi tiết vấn đề
    """
    issues = []
    
    # Kiểm tra gradient norm
    if np.isnan(gradient_norm):
        issues.append(f"Gradient norm = NaN")
    elif np.isinf(gradient_norm):
        issues.append(f"Gradient norm = ±Inf")
    
    # Kiểm tra loss value nếu có
    if loss_value is not None:
        if np.isnan(loss_value):
            issues.append(f"Loss = NaN")
        elif np.isinf(loss_value):
            issues.append(f"Loss = ±Inf")
    
    # Kiểm tra weights nếu có
    if weights is not None:
        if np.any(np.isnan(weights)):
            nan_count = np.sum(np.isnan(weights))
            issues.append(f"Weights contain {nan_count} NaN values")
        if np.any(np.isinf(weights)):
            inf_count = np.sum(np.isinf(weights))
            issues.append(f"Weights contain {inf_count} ±Inf values")
    
    if issues:
        issue_description = f"NUMERICAL INSTABILITY at iteration {iteration}: " + ", ".join(issues)
        return True, issue_description
    
    return False, "No numerical issues detected"

# ==============================================================================
# 5. KIỂM TRA HỘI TỤ VÀ LINE SEARCH
# ==============================================================================
def kiem_tra_dieu_kien_dung(gradient_norm: float, cost_change: float, iteration: int,
                           tolerance: float = 1e-6, max_iterations: int = 10000, 
                           loss_value: Optional[float] = None, weights: Optional[np.ndarray] = None) -> Tuple[bool, bool, str]:
    """
    Kiểm tra điều kiện dừng chung cho thuật toán optimization
    KIỂM TRA THỨ TỰ: 1) Numerical stability, 2) Max iterations, 3) Convergence conditions
    
    Tham số:
        gradient_norm: chuẩn của gradient hiện tại
        cost_change: thay đổi cost từ iteration trước
        iteration: số iteration hiện tại
        tolerance: ngưỡng hội tụ
        max_iterations: số iteration tối đa
        loss_value: giá trị loss hiện tại (tùy chọn, để kiểm tra numerical stability)
        weights: vector trọng số hiện tại (tùy chọn, để kiểm tra numerical stability)
    
    Trả về:
        should_stop: có nên dừng algorithm không
        converged: đã hội tụ thành công không  
        reason: lý do dừng
    """
    # 1. KIỂM TRA NUMERICAL STABILITY TRƯỚC TIÊN (ưu tiên cao nhất)
    has_issues, issue_description = check_for_numerical_issues(
        gradient_norm, loss_value, weights, iteration
    )
    if has_issues:
        return True, False, issue_description  # Dừng vì lỗi, không hội tụ
    
    # 2. Đạt giới hạn iteration
    if iteration >= max_iterations:
        return True, False, f"Đạt giới hạn iteration: {iteration}"  # Dừng vì hết iterations, không hội tụ
    
    # 3. Kiểm tra điều kiện gradient norm
    gradient_converged = gradient_norm < tolerance
    
    # 4. Kiểm tra điều kiện thay đổi cost (chỉ sau iteration đầu tiên)  
    cost_converged = iteration > 0 and abs(cost_change) < tolerance
    
    # 5. YÊU CẦU ĐỒNG THỜI CẢ HAI ĐIỀU KIỆN
    if gradient_converged and cost_converged:
        return True, True, f"Hội tụ đồng thời: gradient norm {gradient_norm:.2e} < {tolerance:.2e} VÀ cost change {abs(cost_change):.2e} < {tolerance:.2e}"
    
    # 6. Chưa hội tụ - hiển thị trạng thái hiện tại, tiếp tục chạy
    if iteration > 0:
        return False, False, f"Chưa hội tụ: gradient={gradient_norm:.2e} ({'✓' if gradient_converged else '✗'}), cost_change={abs(cost_change):.2e} ({'✓' if cost_converged else '✗'})"
    else:
        return False, False, f"Chưa hội tụ: gradient={gradient_norm:.2e} ({'✓' if gradient_converged else '✗'}), cost_change=N/A"

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
def tinh_loss_ols(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray, he_so_tu_do: float = 0) -> float:
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
# 5.1. CÁC HÀM CHỈ NHẬN WEIGHTS (KHÔNG CÓ BIAS) - CHO GRADIENT DESCENT
# ==============================================================================
def tinh_gia_tri_ham_OLS(X: np.ndarray, y: np.ndarray, w: np.ndarray, bias: float = None) -> float:
    """
    Tính giá trị hàm OLS tại điểm w
    
    Hỗ trợ cả 2 format:
    - Format cũ: L(w,b) = (1/2n) * ||Xw + b - y||² (khi bias != None)
    - Format mới: L(w) = (1/2n) * ||Xw - y||² (khi X đã bao gồm cột bias)
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        y: vector target (n_samples,)
        w: vector weights (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
    
    Trả về:
        float: giá trị hàm OLS
    """
    n_samples = X.shape[0]
    if bias is not None:
        # Format cũ: Xw + bias
        predictions = X @ w + bias
    else:
        # Format mới: Xw (với X đã bao gồm cột bias)
        predictions = X @ w
    residuals = predictions - y
    ols_value = (1 / (2 * n_samples)) * np.sum(residuals ** 2)
    return ols_value

def tinh_gradient_OLS(X: np.ndarray, y: np.ndarray, w: np.ndarray, bias: float = None) -> Tuple[np.ndarray, float]:
    """
    Tính gradient của hàm OLS theo weights
    
    Hỗ trợ cả 2 format:
    - Format cũ: ∇L(w,b) = ((1/n) * X^T(Xw + b - y), (1/n) * Σ(Xw + b - y)) (khi bias != None)
    - Format mới: ∇L(w) = (1/n) * X^T(Xw - y) (khi X đã bao gồm cột bias)
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        y: vector target (n_samples,)
        w: vector weights (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
    
    Trả về:
        gradient_w: gradient theo weights (n_features,) hoặc (n_features + 1,)
        gradient_b: gradient theo bias (scalar, hoặc 0.0 cho format mới)
    """
    n_samples = X.shape[0]
    
    if bias is not None:
        # Format cũ: tách riêng gradient cho w và b
        predictions = X @ w + bias
        errors = predictions - y
        gradient_w = (1 / n_samples) * X.T @ errors
        gradient_b = (1 / n_samples) * np.sum(errors)
        return gradient_w, gradient_b
    else:
        # Format mới: gradient thống nhất cho w (bao gồm bias)
        predictions = X @ w
        errors = predictions - y
        gradient_w = (1 / n_samples) * X.T @ errors
        return gradient_w, 0.0



def tinh_gia_tri_ham_Ridge_with_bias(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, regularization: float) -> float:
    """
    Tính giá trị hàm Ridge regression với bias term
    
    Hàm Ridge: L(w,b) = (1/2n) * ||Xw + b - y||² + (λ/2) * ||w||²
    Lưu ý: Không regularize bias term
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        y: vector target (n_samples,)
        w: vector weights (n_features,)
        b: bias term (scalar)
        regularization: hệ số regularization λ
    
    Trả về:
        float: giá trị hàm Ridge tại (w, b)
    """
    n_samples = X.shape[0]
    predictions = X @ w + b
    residuals = predictions - y
    data_loss = (1 / (2 * n_samples)) * np.sum(residuals ** 2)
    reg_loss = (regularization / 2) * np.sum(w ** 2)  # Không regularize bias
    return data_loss + reg_loss

def tinh_gradient_Ridge_with_bias(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, regularization: float) -> Tuple[np.ndarray, float]:
    """
    Tính gradient của hàm Ridge theo weights và bias
    
    ∇L(w,b) = ((1/n) * X^T(Xw + b - y) + λ*w, (1/n) * Σ(Xw + b - y))
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        y: vector target (n_samples,)
        w: vector weights (n_features,)
        b: bias term (scalar)
        regularization: hệ số regularization λ
    
    Trả về:
        gradient_w: gradient theo weights (n_features,)
        gradient_b: gradient theo bias (scalar)
    """
    n_samples = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    
    gradient_w = (1 / n_samples) * X.T @ errors + regularization * w  # Regularize weights
    gradient_b = (1 / n_samples) * np.sum(errors)  # Không regularize bias
    
    return gradient_w, gradient_b

def tinh_gia_tri_ham_Lasso_smooth_with_bias(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, regularization: float) -> float:
    """
    Tính giá trị hàm Lasso (smooth approximation) với bias term
    
    Hàm Lasso: L(w,b) = (1/2n) * ||Xw + b - y||² + λ * Σ|w_i|
    Sử dụng smooth approximation: |x| ≈ √(x² + ε²) với ε = 1e-8
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        y: vector target (n_samples,)
        w: vector weights (n_features,)
        b: bias term (scalar)
        regularization: hệ số regularization λ
    
    Trả về:
        float: giá trị hàm Lasso tại (w, b)
    """
    n_samples = X.shape[0]
    predictions = X @ w + b
    residuals = predictions - y
    data_loss = (1 / (2 * n_samples)) * np.sum(residuals ** 2)
    
    # Smooth approximation của |w|
    epsilon = 1e-8
    reg_loss = regularization * np.sum(np.sqrt(w ** 2 + epsilon))  # Không regularize bias
    
    return data_loss + reg_loss

def tinh_gradient_Lasso_smooth_with_bias(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, regularization: float) -> Tuple[np.ndarray, float]:
    """
    Tính gradient của hàm Lasso (smooth) theo weights và bias
    
    ∇L(w,b) = ((1/n) * X^T(Xw + b - y) + λ * w/√(w² + ε²), (1/n) * Σ(Xw + b - y))
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        y: vector target (n_samples,)
        w: vector weights (n_features,)
        b: bias term (scalar)
        regularization: hệ số regularization λ
    
    Trả về:
        gradient_w: gradient theo weights (n_features,)
        gradient_b: gradient theo bias (scalar)
    """
    n_samples = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    
    # Gradient của data term
    gradient_w_data = (1 / n_samples) * X.T @ errors
    gradient_b = (1 / n_samples) * np.sum(errors)
    
    # Gradient của regularization term (smooth approximation)
    epsilon = 1e-8
    gradient_w_reg = regularization * w / np.sqrt(w ** 2 + epsilon)
    
    gradient_w = gradient_w_data + gradient_w_reg
    
    return gradient_w, gradient_b

def tinh_hessian_OLS(X: np.ndarray) -> np.ndarray:
    """
    Tính ma trận Hessian của hàm OLS (không có bias)
    
    H = (1/n) * X^T * X
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
    
    Trả về:
        hessian: ma trận Hessian (n_features, n_features)
    """
    n_samples = X.shape[0]
    hessian = (1 / n_samples) * X.T @ X
    return hessian

def tinh_gia_tri_ham_Ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, bias: float = None, lambda_reg: float = 0.01) -> float:
    """
    Tính giá trị hàm Ridge
    
    Hỗ trợ cả 2 format:
    - Format cũ: L(w,b) = (1/2n) * ||Xw + b - y||² + (λ/2) * ||w||² (khi bias != None)
    - Format mới: L(w) = (1/2n) * ||Xw - y||² + (λ/2) * ||w[:-1]||² (khi X đã bao gồm cột bias)
    
    Lưu ý: Không regularize bias term trong cả 2 format
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        y: vector target (n_samples,)
        w: vector weights (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
        lambda_reg: hệ số regularization
    
    Trả về:
        float: giá trị hàm Ridge
    """
    ols_loss = tinh_gia_tri_ham_OLS(X, y, w, bias)
    
    if bias is not None:
        # Format cũ: chỉ regularize weights, không regularize bias
        l2_penalty = 0.5 * lambda_reg * np.sum(w ** 2)
    else:
        # Format mới: chỉ regularize weights (không bao gồm bias ở cuối)
        l2_penalty = 0.5 * lambda_reg * np.sum(w[:-1] ** 2)
    
    return ols_loss + l2_penalty

def tinh_gradient_Ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, bias: float = None, lambda_reg: float = 0.01) -> Tuple[np.ndarray, float]:
    """
    Tính gradient của hàm Ridge theo weights
    
    Hỗ trợ cả 2 format:
    - Format cũ: ∇L(w,b) = ((1/n) * X^T(Xw + b - y) + λ*w, (1/n) * Σ(Xw + b - y)) (khi bias != None)
    - Format mới: ∇L(w) = (1/n) * X^T(Xw - y) + λ*[w[:-1]; 0] (khi X đã bao gồm cột bias)
    
    Lưu ý: Không regularize bias term trong cả 2 format
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        y: vector target (n_samples,)
        w: vector weights (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
        lambda_reg: hệ số regularization
    
    Trả về:
        gradient_w: gradient theo weights (n_features,) hoặc (n_features + 1,)
        gradient_b: gradient theo bias (scalar, hoặc 0.0 cho format mới)
    """
    gradient_w, gradient_b = tinh_gradient_OLS(X, y, w, bias)
    
    if bias is not None:
        # Format cũ: chỉ regularize weights, không regularize bias
        l2_gradient = lambda_reg * w
        gradient_w = gradient_w + l2_gradient
        return gradient_w, gradient_b
    else:
        # Format mới: chỉ regularize weights (không bao gồm bias ở cuối)
        l2_gradient = np.zeros_like(w)
        l2_gradient[:-1] = lambda_reg * w[:-1]  # Không regularize bias (phần tử cuối)
        gradient_w = gradient_w + l2_gradient
        return gradient_w, 0.0

def tinh_hessian_Ridge(X: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    Tính ma trận Hessian của hàm Ridge (không có bias)
    
    H = (1/n) * X^T * X + λ * I
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        lambda_reg: hệ số regularization
    
    Trả về:
        hessian: ma trận Hessian (n_features, n_features)
    """
    ols_hessian = tinh_hessian_OLS(X)
    n_features = X.shape[1]
    l2_hessian = lambda_reg * np.eye(n_features)
    return ols_hessian + l2_hessian

def tinh_gia_tri_ham_Lasso_smooth(X: np.ndarray, y: np.ndarray, w: np.ndarray, 
                                  bias: float = None, lambda_reg: float = 0.01, epsilon: float = 1e-8) -> float:
    """
    Tính giá trị hàm Lasso với smooth approximation
    
    Hỗ trợ cả 2 format:
    - Format cũ: L(w,b) = (1/2n) * ||Xw + b - y||² + λ * Σ√(w²+ ε) (khi bias != None)
    - Format mới: L(w) = (1/2n) * ||Xw - y||² + λ * Σ√(w[:-1]²+ ε) (khi X đã bao gồm cột bias)
    
    Lưu ý: Không regularize bias term trong cả 2 format
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        y: vector target (n_samples,)
        w: vector weights (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
        lambda_reg: hệ số regularization
        epsilon: tham số smoothing
    
    Trả về:
        float: giá trị hàm Lasso
    """
    ols_loss = tinh_gia_tri_ham_OLS(X, y, w, bias)
    
    if bias is not None:
        # Format cũ: chỉ regularize weights, không regularize bias
        smooth_l1_penalty = lambda_reg * np.sum(np.sqrt(w ** 2 + epsilon))
    else:
        # Format mới: chỉ regularize weights (không bao gồm bias ở cuối)
        smooth_l1_penalty = lambda_reg * np.sum(np.sqrt(w[:-1] ** 2 + epsilon))
        
    return ols_loss + smooth_l1_penalty

def tinh_gradient_Lasso_smooth(X: np.ndarray, y: np.ndarray, w: np.ndarray, 
                              bias: float = None, lambda_reg: float = 0.01, epsilon: float = 1e-8) -> Tuple[np.ndarray, float]:
    """
    Tính gradient của hàm Lasso với smooth approximation theo weights
    
    Hỗ trợ cả 2 format:
    - Format cũ: ∇L(w,b) = ((1/n) * X^T(Xw + b - y) + λ * w / √(w² + ε), (1/n) * Σ(Xw + b - y)) (khi bias != None)
    - Format mới: ∇L(w) = (1/n) * X^T(Xw - y) + λ * [w[:-1] / √(w[:-1]² + ε); 0] (khi X đã bao gồm cột bias)
    
    Lưu ý: Không regularize bias term trong cả 2 format
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        y: vector target (n_samples,)
        w: vector weights (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
        lambda_reg: hệ số regularization
        epsilon: tham số smoothing
    
    Trả về:
        gradient_w: gradient theo weights (n_features,) hoặc (n_features + 1,)
        gradient_b: gradient theo bias (scalar, hoặc 0.0 cho format mới)
    """
    gradient_w, gradient_b = tinh_gradient_OLS(X, y, w, bias)
    
    if bias is not None:
        # Format cũ: chỉ regularize weights, không regularize bias
        smooth_l1_gradient = lambda_reg * w / np.sqrt(w ** 2 + epsilon)
        gradient_w = gradient_w + smooth_l1_gradient
        return gradient_w, gradient_b
    else:
        # Format mới: chỉ regularize weights (không bao gồm bias ở cuối)
        smooth_l1_gradient = np.zeros_like(w)
        smooth_l1_gradient[:-1] = lambda_reg * w[:-1] / np.sqrt(w[:-1] ** 2 + epsilon)
        gradient_w = gradient_w + smooth_l1_gradient
        return gradient_w, 0.0

def tinh_hessian_Lasso_smooth(X: np.ndarray, w: np.ndarray, lambda_reg: float, epsilon: float = 1e-8) -> np.ndarray:
    """
    Tính ma trận Hessian của hàm Lasso với smooth approximation (không có bias)
    
    H = (1/n) * X^T * X + λ * diag(ε / (w² + ε)^(3/2))
    
    Tham số:
        X: ma trận đặc trưng (n_samples, n_features)
        w: vector weights (n_features,)
        lambda_reg: hệ số regularization
        epsilon: tham số smoothing
    
    Trả về:
        hessian: ma trận Hessian (n_features, n_features)
    """
    ols_hessian = tinh_hessian_OLS(X)
    
    # Diagonal elements cho L1 smooth penalty
    denominator = (w ** 2 + epsilon) ** (3/2)
    l1_diagonal = lambda_reg * epsilon / denominator
    
    # Thêm diagonal elements
    l1_hessian = np.diag(l1_diagonal)
    
    return ols_hessian + l1_hessian

# ==============================================================================
# 6. ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
def danh_gia_mo_hinh(weights: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                      bias: float = 0.0) -> Dict[str, float]:
    """
    Đánh giá model trên test set với đầy đủ các metrics
    
    
    Tham số:
        weights: trọng số đã học (n_features,)
        X_test: ma trận đặc trưng test (n_samples, n_features)
        y_test: giá trị thật test (n_samples,)
        bias: bias term (mặc định 0.0)
        is_log_transformed: có phải target đã được log transform không
    
    Trả về:
        dict: dictionary chứa các metrics đánh giá trên scale gốc
    """
    # Dự đoán trên log scale (nếu model được train trên log)
    predictions_log = du_doan(X_test, weights, bias)
    
    # Convert cả predictions và test về original scale
    predictions_original = np.expm1(predictions_log)  # inverse of log1p
    y_test_original = np.expm1(y_test)                # inverse of log1p
    
    # Use original scale for evaluation
    predictions_eval = predictions_original
    y_test_eval = y_test_original
    
    # Tính các metrics cơ bản trên scale gốc
    mse = tinh_mse(y_test_eval, predictions_eval)
    rmse = np.sqrt(mse)
    mae = tinh_mae(y_test_eval, predictions_eval)
    r2 = tinh_r2_score(y_test_eval, predictions_eval)
    
    # MAPE (Mean Absolute Percentage Error) - cẩn thận với chia cho 0
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs((y_test_eval - predictions_eval) / y_test_eval)
        # Loại bỏ các giá trị inf và nan
        valid_errors = percentage_errors[np.isfinite(percentage_errors)]
        mape = np.mean(valid_errors) * 100 if len(valid_errors) > 0 else float('inf')
    
    # Thêm một số metrics bổ sung
    # Max Error
    max_error = np.max(np.abs(y_test_eval - predictions_eval))
    
    # Explained variance score
    var_y = np.var(y_test_eval)
    var_residual = np.var(y_test_eval - predictions_eval)
    explained_variance = 1 - (var_residual / var_y) if var_y != 0 else 0
    
    # Thêm metrics cho cả log scale (nếu có transform)
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'max_error': float(max_error),
        'explained_variance': float(explained_variance)
    }
    
    # Nếu có log transform, thêm metrics trên log scale để so sánh
    mse_log = tinh_mse(y_test, predictions_log)
    r2_log = tinh_r2_score(y_test, predictions_log)
    mae_log = tinh_mae(y_test, predictions_log)
    
    metrics.update({
        'mse_log_scale': float(mse_log),
        'r2_log_scale': float(r2_log),
        'mae_log_scale': float(mae_log)
    })
    
    return metrics
def danh_gia_mo_hinh_with_bias(weights: np.ndarray, bias: float, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Đánh giá model trên test set với đầy đủ các metrics (có bias term)
    
    Tham số:
        weights: trọng số đã học (n_features,)
        bias: bias term đã học (scalar)
        X_test: ma trận đặc trưng test (n_samples, n_features)
        y_test: giá trị thật test (n_samples,)
    
    Trả về:
        dict: dictionary chứa các metrics đánh giá trên scale gốc
    """
    # Dự đoán trên log scale (nếu model được train trên log)
    predictions_log = du_doan(X_test, weights, bias)
        
    # Convert cả predictions và test về original scale
    predictions_original = np.expm1(predictions_log)  # inverse of log1p
    y_test_original = np.expm1(y_test)                # inverse of log1p
    # Use original scale for evaluation
    predictions_eval = predictions_original
    y_test_eval = y_test_original
    
    # Tính các metrics cơ bản trên scale gốc
    mse = tinh_mse(y_test_eval, predictions_eval)
    rmse = np.sqrt(mse)
    mae = tinh_mae(y_test_eval, predictions_eval)
    r2 = tinh_r2_score(y_test_eval, predictions_eval)
    
    # MAPE (Mean Absolute Percentage Error) - cẩn thận với chia cho 0
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs((y_test_eval - predictions_eval) / y_test_eval)
        # Loại bỏ các giá trị inf và nan
        valid_errors = percentage_errors[np.isfinite(percentage_errors)]
        mape = np.mean(valid_errors) * 100 if len(valid_errors) > 0 else float('inf')
    
    # Thêm một số metrics bổ sung
    # Max Error
    max_error = np.max(np.abs(y_test_eval - predictions_eval))
    
    # Explained variance score
    var_y = np.var(y_test_eval)
    var_residual = np.var(y_test_eval - predictions_eval)
    explained_variance = 1 - (var_residual / var_y) if var_y != 0 else 0
    
    # Thêm metrics cho cả log scale (nếu có transform)
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'max_error': float(max_error),
        'explained_variance': float(explained_variance)
    }
    
    # Nếu có log transform, thêm metrics trên log scale để so sánh
    mse_log = tinh_mse(y_test, predictions_log)
    r2_log = tinh_r2_score(y_test, predictions_log)
    mae_log = tinh_mae(y_test, predictions_log)
    
    metrics.update({
        'mse_log_scale': float(mse_log),
        'r2_log_scale': float(r2_log),
        'mae_log_scale': float(mae_log)
    })
    
    return metrics

def in_ket_qua_danh_gia(metrics: Dict[str, float], training_time: float = None, 
                       algorithm_name: str = "Model"):
    """
    In kết quả đánh giá model một cách đẹp mắt
    
    Tham số:
        metrics: dictionary chứa các metrics từ evaluate_model
        training_time: thời gian training (tùy chọn)
        algorithm_name: tên thuật toán
    """
    print("\n" + "="*60)
    print(f"📊 {algorithm_name.upper()} - EVALUATION RESULTS")
    print("="*60)
    
    # Thông báo scale đánh giá    
    print(f"\n🎯 REGRESSION METRICS:")
    print(f"   MSE:      {metrics['mse']:.8f}")
    print(f"   RMSE:     {metrics['rmse']:.6f}")
    print(f"   MAE:      {metrics['mae']:.6f}")
    print(f"   R² Score: {metrics['r2']:.6f}")
    
    if metrics['mape'] != float('inf'):
        print(f"   MAPE:     {metrics['mape']:.2f}%")
    else:
        print(f"   MAPE:     N/A (division by zero)")
        
    print(f"   Max Error: {metrics['max_error']:.6f}")
    print(f"   Explained Variance: {metrics['explained_variance']:.6f}")
    
    if training_time is not None:
        print(f"   Training Time: {training_time:.4f}s")
    
# ==============================================================================
# 7. TIỆN ÍCH DEBUG VÀ IN THÔNG TIN
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

# ==============================================================================
# 9. BỔ SUNG LOSS FUNCTIONS KHÁC
# ==============================================================================
def tinh_loss_elastic_net(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray,
                         he_so_tu_do: float, alpha: float, l1_ratio: float) -> float:
    """
    Tính loss cho Elastic Net Regression
    
    Loss = MSE + alpha * (l1_ratio * L1 + (1-l1_ratio) * L2)
    """
    mse_loss = tinh_loss_ols(X, y, trong_so, he_so_tu_do)
    l1_penalty = alpha * l1_ratio * np.sum(np.abs(trong_so))
    l2_penalty = alpha * (1 - l1_ratio) * 0.5 * np.sum(trong_so ** 2)
    return mse_loss + l1_penalty + l2_penalty

def tinh_loss_huber(X: np.ndarray, y: np.ndarray, trong_so: np.ndarray,
                   he_so_tu_do: float, delta: float = 1.0) -> float:
    """
    Tính Huber loss - robust loss function
    
    Huber(e) = 0.5 * e² if |e| ≤ δ else δ * (|e| - 0.5 * δ)
    """
    predictions = du_doan(X, trong_so, he_so_tu_do)
    errors = predictions - y
    
    # Áp dụng Huber loss
    abs_errors = np.abs(errors)
    quadratic = 0.5 * errors ** 2
    linear = delta * (abs_errors - 0.5 * delta)
    
    # Chọn quadratic hoặc linear dựa trên threshold
    huber_losses = np.where(abs_errors <= delta, quadratic, linear)
    return np.mean(huber_losses)

def tinh_gradient_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """
    Tính gradient cho Ridge regression
    
    ∇f = X^T(Xw - y)/n + λw
    """
    n = X.shape[0]
    predictions = X @ w
    errors = predictions - y
    gradient = (X.T @ errors) / n + lam * w
    return gradient

def tinh_gradient_lasso_smooth(X: np.ndarray, y: np.ndarray, w: np.ndarray, 
                              lam: float, epsilon: float = 1e-8) -> np.ndarray:
    """
    Tính gradient cho Lasso regression (smooth approximation)
    
    ∇f = X^T(Xw - y)/n + λ * w/√(w² + ε)
    """
    n = X.shape[0]
    predictions = X @ w
    errors = predictions - y
    mse_gradient = (X.T @ errors) / n
    
    # Smooth L1 gradient
    l1_gradient = lam * w / np.sqrt(w ** 2 + epsilon)
    return mse_gradient + l1_gradient

def tinh_hessian_ridge(X: np.ndarray, lam: float) -> np.ndarray:
    """
    Tính Hessian cho Ridge regression
    
    H = X^TX/n + λI
    """
    n = X.shape[0]
    XTX = X.T @ X
    I = np.eye(X.shape[1])
    return XTX / n + lam * I

def tinh_gia_tri_ham_loss(ham_loss: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, 
                         bias: float = None, regularization: float = 0.01, **kwargs) -> float:
    """
    Hàm thống nhất để tính giá trị loss function
    
    Hỗ trợ cả 2 format:
    - Format cũ: loss với bias riêng (khi bias != None)
    - Format mới: loss với bias trong X (khi bias = None)
    
    Tham số:
        ham_loss: loại loss function ('ols', 'ridge', 'lasso')
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        y: vector target (n_samples,)
        w: vector weights (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
        regularization: hệ số regularization (mặc định 0.01)
        **kwargs: các tham số bổ sung (ví dụ epsilon cho Lasso)
    
    Trả về:
        float: giá trị loss function
    """
    ham_loss = ham_loss.lower()
    
    if ham_loss == 'ols':
        return tinh_gia_tri_ham_OLS(X, y, w, bias)
    elif ham_loss == 'ridge':
        return tinh_gia_tri_ham_Ridge(X, y, w, bias, regularization)
    elif ham_loss == 'lasso':
        epsilon = kwargs.get('epsilon', 1e-8)
        return tinh_gia_tri_ham_Lasso_smooth(X, y, w, bias, regularization, epsilon)
    else:
        raise ValueError(f"Không hỗ trợ loss function: {ham_loss}. Chỉ hỗ trợ: 'ols', 'ridge', 'lasso'")
def tinh_gradient_ham_loss(ham_loss: str, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                          bias: float = None, regularization: float = 0.01, **kwargs) -> Tuple[np.ndarray, float]:
    """
    Hàm thống nhất để tính gradient của loss function
    
    Hỗ trợ cả 2 format:
    - Format cũ: gradient với bias riêng (khi bias != None)
    - Format mới: gradient với bias trong X (khi bias = None)
    
    Tham số:
        ham_loss: loại loss function ('ols', 'ridge', 'lasso')
        X: ma trận đặc trưng (n_samples, n_features) hoặc (n_samples, n_features + 1) với bias
        y: vector target (n_samples,)
        w: vector weights (n_features,) hoặc (n_features + 1,) với bias
        bias: bias term (scalar, deprecated - sử dụng None cho format mới)
        regularization: hệ số regularization (mặc định 0.01)
        **kwargs: các tham số bổ sung (ví dụ epsilon cho Lasso)
    
    Trả về:
        gradient_w: gradient theo weights (n_features,) hoặc (n_features + 1,)
        gradient_b: gradient theo bias (scalar, hoặc 0.0 cho format mới)
    """
    ham_loss = ham_loss.lower()
    
    if ham_loss == 'ols':
        return tinh_gradient_OLS(X, y, w, bias)
    elif ham_loss == 'ridge':
        return tinh_gradient_Ridge(X, y, w, bias, regularization)
    elif ham_loss == 'lasso':
        epsilon = kwargs.get('epsilon', 1e-8)
        return tinh_gradient_Lasso_smooth(X, y, w, bias, regularization, epsilon)
    else:
        raise ValueError(f"Không hỗ trợ loss function: {ham_loss}. Chỉ hỗ trợ: 'ols', 'ridge', 'lasso'")
def tinh_hessian_ham_loss(ham_loss: str, X: np.ndarray, w: np.ndarray = None,
                         regularization: float = 0.01, **kwargs) -> np.ndarray:
    """
    Hàm thống nhất để tính Hessian matrix của loss function
    
    Tham số:
        ham_loss: loại loss function ('ols', 'ridge', 'lasso')
        X: ma trận đặc trưng (n_samples, n_features)
        w: vector weights (n_features,) - chỉ cần cho Lasso
        regularization: hệ số regularization (mặc định 0.01)
        **kwargs: các tham số bổ sung (ví dụ epsilon cho Lasso)
    
    Trả về:
        np.ndarray: Hessian matrix (n_features, n_features)
    """
    ham_loss = ham_loss.lower()
    
    if ham_loss == 'ols':
        return tinh_hessian_OLS(X)
    elif ham_loss == 'ridge':
        return tinh_hessian_Ridge(X, regularization)
    elif ham_loss == 'lasso':
        if w is None:
            raise ValueError("Vector weights w cần thiết để tính Hessian cho Lasso")
        epsilon = kwargs.get('epsilon', 1e-8)
        return tinh_hessian_Lasso_smooth(X, w, regularization, epsilon)
    else:
        raise ValueError(f"Không hỗ trợ loss function: {ham_loss}. Chỉ hỗ trợ: 'ols', 'ridge', 'lasso'")
