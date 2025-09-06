#!/usr/bin/env python3
"""
Setup script for BFGS với Enhanced Backtracking Line Search - OLS
- BFGS với stronger line search control
- Max Iterations: 100
- Tolerance: 1e-6
- Armijo c1: 1e-4 (sufficient decrease)
- Enhanced backtracking parameters
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.quasi_newton.quasi_newton_model import QuasiNewtonModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    """Chạy BFGS với Enhanced Backtracking cho OLS"""
    print("QUASI-NEWTON METHOD - BFGS BACKTRACKING OLS SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với enhanced backtracking
    model = QuasiNewtonModel(
        ham_loss='ols',
        diem_dung=1e-5,
        method='bfgs',       # Full BFGS
        armijo_c1=1e-4,      # Sufficient decrease parameter
        wolfe_c2=0.1,        # Lower curvature condition (more restrictive)
        backtrack_rho=0.5,   # More aggressive backtracking
        max_line_search_iter=100,  # More line search iterations
        damping=1e-8
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()  # Sẽ là "setup_bfgs_backtracking_ols_c1_0001"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nBFGS + Enhanced Backtracking training completed!")
    print(f"Armijo c1: {model.armijo_c1}")
    print(f"Wolfe c2: {model.wolfe_c2}")
    print(f"Backtrack factor: {model.backtrack_rho}")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()