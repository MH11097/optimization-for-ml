#!/usr/bin/env python3
"""
Setup script for Original Stochastic Gradient Descent
- Learning Rate: 0.01
- Epochs: 100
- Batch Size: 1 (pure SGD)
- Random State: 42
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.stochastic_gd.sgd_model import SGDModel


def load_sampled_data():
    """Load sampled training data từ 02.1_sampled"""
    import pandas as pd
    
    data_dir = Path("data/02.1_sampled")
    
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Sampled data not found: {data_dir / file}")
    
    print("📂 Loading sampled data...")
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    print(f"✅ Loaded: Train {X_train.shape}, Test {X_test.shape}")
    
    return X_train.values, X_test.values, y_train, y_test


def main():
    """Chạy Original Stochastic Gradient Descent"""
    print("STOCHASTIC GRADIENT DESCENT - ORIGINAL SETUP")
    
    # Load sampled data
    X_train, X_test, y_train, y_test = load_sampled_data()
    
    # Khởi tạo model với tham số giống original_stochastic_gd.py
    model = SGDModel(
        learning_rate=0.01,
        so_epochs=100,
        random_state=42,
        batch_size=1000,  # Pure SGD
        ham_loss='ols'
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả
    ten_file = "original_sgd"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()