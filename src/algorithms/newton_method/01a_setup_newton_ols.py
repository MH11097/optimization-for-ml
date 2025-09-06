#!/usr/bin/env python3
"""
Setup script for Pure Newton Method - OLS
Group 01a: Pure Newton với OLS loss function
- Max Iterations: 50  
- Tolerance: 1e-10
- Numerical regularization: 1e-8 (cho stability)
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
    """Chạy Pure Newton Method cho OLS"""
    print("=" * 60)
    print("NEWTON METHOD - GROUP 01A: PURE NEWTON OLS")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo Pure Newton model cho OLS
    model = NewtonModel(
        ham_loss='ols',
        diem_dung=1e-10,
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