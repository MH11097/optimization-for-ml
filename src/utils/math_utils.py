"""Mathematical utility functions"""

import numpy as np


def convert_to_price_scale(log_predictions):
    """Chuyển đổi dự đoán log về thang giá gốc"""
    return np.exp(log_predictions)


def log_transform_safe(data, offset=1):
    """Apply log transformation safely by adding offset to handle zeros"""
    return np.log(data + offset)


def inverse_log_transform(log_data, offset=1):
    """Inverse of log transformation"""
    return np.exp(log_data) - offset


def remove_outliers_iqr(data, multiplier=1.5):
    """Remove outliers using IQR method"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data >= lower_bound) & (data <= upper_bound)


def normalize_features(X):
    """Normalize features to [0, 1] range"""
    return (X - X.min()) / (X.max() - X.min())


def standardize_features(X):
    """Standardize features to mean=0, std=1"""
    return (X - X.mean()) / X.std()


# Optimization-specific mathematical functions

def compute_gradient(X, y, weights, bias):
    """
    Tính gradient của Mean Squared Error
    
    Tham số:
    - X: ma trận đặc trưng (n_samples, n_features)
    - y: vector mục tiêu (n_samples,)
    - weights: trọng số hiện tại (n_features,)
    - bias: bias hiện tại (scalar)
    
    Trả về:
    - grad_w: gradient của weights (n_features,)
    - grad_b: gradient của bias (scalar)
    """
    n_samples = X.shape[0]
    y_pred = X.dot(weights) + bias
    error = y_pred - y
    
    grad_w = (2 / n_samples) * X.T.dot(error)
    grad_b = (2 / n_samples) * np.sum(error)
    
    return grad_w, grad_b


def compute_hessian(X):
    """
    Tính ma trận Hessian cho bài toán linear regression
    
    Tham số:
    - X: ma trận đặc trưng (n_samples, n_features)
    
    Trả về:
    - hessian: ma trận Hessian (n_features, n_features)
    """
    n_samples = X.shape[0]
    return (2 / n_samples) * X.T.dot(X)


def soft_thresholding(x, threshold):
    """
    Phép toán soft thresholding cho proximal gradient
    
    Tham số:
    - x: input vector hoặc scalar
    - threshold: ngưỡng thresholding
    
    Trả về:
    - Kết quả sau soft thresholding
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def proximal_l1(x, lambda_reg, step_size):
    """
    Proximal operator cho L1 regularization (Lasso)
    
    Tham số:
    - x: input vector
    - lambda_reg: hệ số regularization L1
    - step_size: step size
    
    Trả về:
    - Kết quả sau proximal operator
    """
    threshold = lambda_reg * step_size
    return soft_thresholding(x, threshold)


def compute_subgradient_l1(weights, lambda_reg):
    """
    Tính subgradient của L1 regularization
    
    Tham số:
    - weights: trọng số hiện tại
    - lambda_reg: hệ số regularization L1
    
    Trả về:
    - subgradient: subgradient của L1 term
    """
    subgrad = np.zeros_like(weights)
    subgrad[weights > 0] = lambda_reg
    subgrad[weights < 0] = -lambda_reg
    # Với weights = 0, subgradient có thể là bất kỳ giá trị nào trong [-lambda_reg, lambda_reg]
    # Chúng ta chọn 0 cho đơn giản
    return subgrad


def armijo_condition(current_cost, new_cost, step_size, gradient_norm_sq, c1=1e-4):
    """
    Kiểm tra điều kiện Armijo cho line search
    
    Tham số:
    - current_cost: cost hiện tại
    - new_cost: cost mới
    - step_size: step size đang kiểm tra
    - gradient_norm_sq: ||gradient||^2
    - c1: tham số Armijo (mặc định 1e-4)
    
    Trả về:
    - satisfied: True nếu thỏa mãn điều kiện Armijo
    """
    return new_cost <= current_cost - c1 * step_size * gradient_norm_sq


def bfgs_update(H_inv, s, y, rho_threshold=1e-6):
    """
    Cập nhật ma trận Hessian nghịch đảo trong BFGS
    
    Tham số:
    - H_inv: ma trận Hessian nghịch đảo hiện tại
    - s: sự thay đổi của weights (x_new - x_old)
    - y: sự thay đổi của gradient (grad_new - grad_old)
    - rho_threshold: ngưỡng để kiểm tra điều kiện curvature
    
    Trả về:
    - H_inv_new: ma trận Hessian nghịch đảo cập nhật
    """
    rho = 1.0 / np.dot(y, s)
    
    # Kiểm tra điều kiện curvature
    if abs(rho) < rho_threshold:
        return H_inv  # Không cập nhật nếu không thỏa mãn điều kiện
    
    I = np.eye(len(s))
    V = I - rho * np.outer(y, s)
    H_inv_new = V.T.dot(H_inv).dot(V) + rho * np.outer(s, s)
    
    return H_inv_new