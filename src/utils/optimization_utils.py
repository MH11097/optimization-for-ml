"""Các hàm tiện ích dùng chung cho tất cả thuật toán tối ưu"""

import numpy as np
import matplotlib.pyplot as plt


def predict(X, weights, bias):
    """
    Dự đoán sử dụng trọng số và bias đã học
    
    Tham số:
    - X: ma trận đặc trưng (n_samples, n_features)
    - weights: trọng số (n_features,)
    - bias: bias (scalar)
    
    Trả về:
    - y_pred: dự đoán (n_samples,)
    """
    return X.dot(weights) + bias


def compute_mse(y_true, y_pred):
    """
    Tính Mean Squared Error
    
    Tham số:
    - y_true: giá trị thực (n_samples,)
    - y_pred: giá trị dự đoán (n_samples,)
    
    Trả về:
    - mse: Mean Squared Error (scalar)
    """
    return np.mean((y_true - y_pred) ** 2)


def compute_mae(y_true, y_pred):
    """
    Tính Mean Absolute Error
    
    Tham số:
    - y_true: giá trị thực (n_samples,)
    - y_pred: giá trị dự đoán (n_samples,)
    
    Trả về:
    - mae: Mean Absolute Error (scalar)
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_r2_score(y_true, y_pred):
    """
    Tính R-squared score
    
    Tham số:
    - y_true: giá trị thực (n_samples,)
    - y_pred: giá trị dự đoán (n_samples,)
    
    Trả về:
    - r2: R-squared score (scalar)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def check_convergence(cost_history, tolerance=1e-6, min_iterations=10):
    """
    Kiểm tra điều kiện hội tụ
    
    Tham số:
    - cost_history: lịch sử cost (list)
    - tolerance: độ chính xác (mặc định 1e-6)
    - min_iterations: số vòng lặp tối thiểu (mặc định 10)
    
    Trả về:
    - converged: True nếu đã hội tụ (bool)
    """
    if len(cost_history) < min_iterations:
        return False
    
    recent_change = abs(cost_history[-1] - cost_history[-2])
    return recent_change < tolerance


def plot_convergence(cost_history, title="Convergence Plot"):
    """
    Vẽ biểu đồ hội tụ
    
    Tham số:
    - cost_history: lịch sử cost (list)
    - title: tiêu đề biểu đồ (str)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)
    plt.show()


def backtracking_line_search(X, y, weights, bias, gradient_w, gradient_b, 
                           alpha=0.3, beta=0.8, max_iterations=50):
    """
    Backtracking line search để tìm learning rate tối ưu
    
    Tham số:
    - X: ma trận đặc trưng
    - y: vector mục tiêu
    - weights: trọng số hiện tại
    - bias: bias hiện tại
    - gradient_w: gradient của weights
    - gradient_b: gradient của bias
    - alpha: tham số Armijo (mặc định 0.3)
    - beta: tham số giảm step size (mặc định 0.8)
    - max_iterations: số vòng lặp tối đa (mặc định 50)
    
    Trả về:
    - step_size: step size tìm được (scalar)
    """
    # Cost hiện tại
    y_pred_current = X.dot(weights) + bias
    current_cost = np.mean((y_pred_current - y) ** 2)
    
    # Gradient descent direction
    gradient_norm_sq = np.sum(gradient_w ** 2) + gradient_b ** 2
    
    step_size = 1.0
    for _ in range(max_iterations):
        # Thử weights mới với step_size hiện tại
        new_weights = weights - step_size * gradient_w
        new_bias = bias - step_size * gradient_b
        
        # Tính cost mới
        y_pred_new = X.dot(new_weights) + new_bias
        new_cost = np.mean((y_pred_new - y) ** 2)
        
        # Kiểm tra điều kiện Armijo
        if new_cost <= current_cost - alpha * step_size * gradient_norm_sq:
            break
            
        step_size *= beta
    
    return step_size


def create_batches(X, y, batch_size):
    """
    Tạo các batch cho mini-batch gradient descent
    
    Tham số:
    - X: ma trận đặc trưng (n_samples, n_features)
    - y: vector mục tiêu (n_samples,)
    - batch_size: kích thước batch
    
    Trả về:
    - batches: list các tuple (X_batch, y_batch)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches