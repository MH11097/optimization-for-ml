#!/usr/bin/env python3
"""
Setup script for Damped Newton Method - Backtracking Line Search
Group 03b: Damped Newton với OLS và backtracking line search
- Initial learning rate: 1.0
- Armijo constant c1: 0.001
- Backtrack factor rho: 0.5
- Max Iterations: 100
- Tolerance: 1e-8
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
    """Chạy Damped Newton Method với Backtracking Line Search cho OLS"""
    print("=" * 70)
    print("NEWTON METHOD - GROUP 03B: BACKTRACKING LR 1.0 C1 0.001 RHO 0.5")
    print("=" * 70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo Damped Newton model với backtracking line search
    model = DampedNewtonModel(
        ham_loss='ols',
        diem_dung=1e-8,
        armijo_c1=0.001,
        backtrack_rho=0.5,
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