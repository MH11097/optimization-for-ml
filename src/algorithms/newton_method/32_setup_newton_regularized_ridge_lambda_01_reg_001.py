#!/usr/bin/env python3
"""
Setup script for Dual Regularized Newton Method - Ridge
- Hessian regularization: lambda = 0.1 for stability  
- Ridge penalty: regularization = 0.01
- Max Iterations: 100
- Tolerance: 1e-8
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
    return Path(filename).stem  # Lấy tên file không có extension

def main():
    """Chạy Dual Regularized Newton Method cho Ridge"""
    print("NEWTON METHOD - DUAL REGULARIZED RIDGE SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với cả Ridge penalty và Hessian regularization
    model = NewtonModel(
        ham_loss='ridge',
        regularization=0.01,  # Ridge penalty parameter
        so_lan_thu=10000,
        diem_dung=1e-8,
        numerical_regularization=0.1  # Hessian regularization: H + λI (stronger)
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()  # Sẽ là "setup_newton_regularized_ridge_lambda_01_reg_001"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nDual Regularized Newton training completed!")
    print(f"Ridge penalty: {model.regularization}")
    print(f"Hessian regularization: {model.numerical_regularization}")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()