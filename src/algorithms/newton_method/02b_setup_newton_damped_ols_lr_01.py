#!/usr/bin/env python3
"""
Setup script for Damped Newton Method - Fixed Step Size
Group 02b: Damped Newton với OLS và fixed learning rate 0.1
- Learning rate: 0.1 (fixed)
- Max Iterations: 100
- Tolerance: 1e-8
- Numerical regularization: 1e-8
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.newton_method.damped_newton_model import DampedNewtonModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    """Chạy Damped Newton Method với fixed step size cho OLS"""
    print("=" * 65)
    print("NEWTON METHOD - GROUP 02B: DAMPED NEWTON FIXED LR 0.1")
    print("=" * 65)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo Damped Newton model với fixed step size
    model = DampedNewtonModel(
        ham_loss='ols',
        diem_dung=1e-8,
        use_line_search=False,
        learning_rate=0.1
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nTraining and visualization completed!")
    
    return model, results, metrics


if __name__ == "__main__":
    main()