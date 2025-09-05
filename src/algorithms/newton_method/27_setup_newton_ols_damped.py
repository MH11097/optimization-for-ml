#!/usr/bin/env python3
"""
Setup script for Damped Newton Method - OLS
- Regularization: 1e-8 (numerical stability)
- Max Iterations: 100
- Tolerance: 1e-8
- Armijo constant: 1e-4
- Backtrack factor: 0.8
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.newton_method.damped_newton_model import DampedNewtonModel
from utils.data_process_utils import load_du_lieu


def main():
    """Chạy Damped Newton Method cho OLS"""
    print("NEWTON METHOD - DAMPED NEWTON OLS SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với tham số giống damped_newton_ols.py
    model = DampedNewtonModel(
        ham_loss='ols',
        regularization=0.0,  # Không regularization cho OLS
        diem_dung=1e-8,
        numerical_regularization=1e-8,  # Cho numerical stability
        armijo_c1=1e-4,
        backtrack_rho=0.8,
        max_line_search_iter=50
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả
    ten_file = "damped_newton_ols"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()