#!/usr/bin/env python3
"""
Setup script for Pure Newton Method - Ridge Regression
- Regularization: 0.01 (Ridge)
- Numerical regularization: 1e-8 (stability)
- Max Iterations: 50
- Tolerance: 1e-10
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.newton_method.newton_model import NewtonModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    """Chạy Pure Newton Method cho Ridge Regression"""
    print("NEWTON METHOD - PURE NEWTON RIDGE SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với Ridge regularization
    model = NewtonModel(
        ham_loss='ridge',
        regularization=0.01,      # Ridge regularization
        so_lan_thu=50,
        diem_dung=1e-10,
        numerical_regularization=1e-8  # Cho numerical stability
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