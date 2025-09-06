#!/usr/bin/env python3
"""
Setup script for L-BFGS Quasi-Newton Method - OLS
- Memory size: m = 10 (store 10 recent (s,y) pairs)
- Max Iterations: 100
- Tolerance: 1e-6
- Armijo c1: 1e-4
- Wolfe c2: 0.9
- Backtrack factor: 0.8
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
    """Chạy L-BFGS Quasi-Newton Method cho OLS với memory=10"""
    print("QUASI-NEWTON METHOD - L-BFGS OLS (m=10) SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với L-BFGS memory=10
    model = QuasiNewtonModel(
        ham_loss='ols',
        diem_dung=1e-6,
        method='lbfgs',      # L-BFGS method
        memory_size=10,      # Store 10 recent (s,y) pairs
        armijo_c1=1e-4,
        wolfe_c2=0.9,
        backtrack_rho=0.8,
        max_line_search_iter=50,
        damping=1e-8
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()  # Sẽ là "setup_lbfgs_ols_m_10"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nL-BFGS (m=10) training completed!")
    print(f"Memory size: {model.memory_size}")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()