#!/usr/bin/env python3
"""
Setup script for Momentum Gradient Descent - OLS
- Learning Rate: 0.01
- Momentum: 0.9
- Max Iterations: 1000
- Tolerance: 1e-6
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.momentum_gd_model import MomentumGDModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    """Chạy Momentum Gradient Descent cho OLS"""
    print("MOMENTUM GRADIENT DESCENT - OLS SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với momentum
    model = MomentumGDModel(
        ham_loss='ols',
        learning_rate=0.01,
        momentum=0.9,
        so_lan_thu=1000,
        diem_dung=1e-6
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
    
    print(f"\\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()