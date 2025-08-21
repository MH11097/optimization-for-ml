#!/usr/bin/env python3
"""
Setup script for Fast OLS Gradient Descent
- Learning Rate: 0.1
- Max Iterations: 500  
- Tolerance: 1e-5
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem  # Lấy tên file không có extension

def main():
    """Chạy Fast OLS Gradient Descent với tham số như ols_01.py"""
    print("GRADIENT DESCENT - FAST OLS SETUP (lr=0.1)")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với tham số giống ols_01.py
    model = GradientDescentModel(
        ham_loss='ols',
        learning_rate=0.1,
        so_lan_thu=500,
        diem_dung=1e-5
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()  # Sẽ là "setup_ols_01"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()