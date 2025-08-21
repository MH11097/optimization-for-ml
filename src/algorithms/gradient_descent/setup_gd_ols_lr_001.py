#!/usr/bin/env python3
"""
Setup script for Slow OLS Gradient Descent  
- Learning Rate: 0.01
- Max Iterations: 500
- Tolerance: 1e-5
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
from utils.data_process_utils import load_du_lieu


def main():
    """Chạy Slow OLS Gradient Descent với learning rate 0.01"""
    print("GRADIENT DESCENT - SLOW OLS SETUP (lr=0.01)")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model với learning rate nhỏ hơn
    model = GradientDescentModel(
        ham_loss='ols',
        learning_rate=0.01,
        so_lan_thu=500,
        diem_dung=1e-5
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả
    ten_file = "slow_ols_gd_lr_001"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()