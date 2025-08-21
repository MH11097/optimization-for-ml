#!/usr/bin/env python3
"""
Setup script for Ridge Regression với Gradient Descent
- Learning Rate: 0.01
- Regularization: 0.01
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
    """Chạy Ridge Regression với Gradient Descent"""
    print("GRADIENT DESCENT - RIDGE REGRESSION SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model cho Ridge regression
    model = GradientDescentModel(
        ham_loss='ridge',
        learning_rate=0.01,
        so_lan_thu=500,
        diem_dung=1e-5,
        regularization=0.01
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả
    ten_file = "ridge_gd_reg_001"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()