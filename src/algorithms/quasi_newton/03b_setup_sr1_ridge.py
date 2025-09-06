#!/usr/bin/env python3
"""
Setup script for SR1 Quasi-Newton Method - Ridge
- Skip threshold: 1e-8 (numerical stability)
- Regularization: 0.01 (Ridge penalty)
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
    """Chạy SR1 Quasi-Newton Method cho Ridge"""
    print("QUASI-NEWTON METHOD - SR1 RIDGE SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với SR1 method và Ridge regularization
    model = QuasiNewtonModel(
        ham_loss='ridge',
        regularization=0.01,    # Ridge penalty parameter
        method='sr1',           # SR1 method
        sr1_skip_threshold=1e-8, # Skip condition for numerical stability
        diem_dung=1e-5,
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
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nSR1 Ridge training completed!")
    print(f"Skip threshold: {model.sr1_skip_threshold}")
    print(f"Regularization: {model.regularization}")
    if results.get('skipped_updates'):
        skipped_count = sum(results['skipped_updates'])
        total_updates = len(results['skipped_updates'])
        skip_rate = skipped_count / total_updates if total_updates > 0 else 0
        print(f"Updates skipped: {skipped_count}/{total_updates} ({skip_rate:.1%})")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()