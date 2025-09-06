#!/usr/bin/env python3
"""
Setup script for Pure Newton Method - OLS
- Regularization: 1e-8 (numerical stability)
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


def main():
    """Chạy Pure Newton Method cho OLS"""
    print("NEWTON METHOD - PURE NEWTON OLS SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với tham số giống pure_newton_ols.py
    model = NewtonModel(
        ham_loss='ols',
        diem_dung=1e-10,
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả
    ten_file = "pure_newton_ols"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nTraining and visualization completed!")
    
    return model, results, metrics


if __name__ == "__main__":
    main()