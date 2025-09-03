#!/usr/bin/env python3
"""
Setup script for SGD with Square Root Learning Rate Decay
- Learning Rate Schedule: sqrt_decay (α / sqrt(epoch + 1))
- Base Learning Rate: 0.1
- Epochs: 100
- Batch Size: 1000
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


def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem  # Lấy tên file không có extension


def main():
    """Chạy SGD với Square Root Learning Rate Decay"""
    print("STOCHASTIC GRADIENT DESCENT - SQUARE ROOT LEARNING RATE DECAY")
    
    # Load sampled data
    X_train, X_test, y_train, y_test = load_sampled_data()
    
    # Khởi tạo model với sqrt decay schedule
    model = SGDModel(
        learning_rate=0.1,  # Base learning rate
        so_epochs=100,
        random_state=42,
        batch_size=1000,
        ham_loss='ols',
        learning_rate_schedule='sqrt_decay'  # α / sqrt(epoch + 1)
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
    
    print(f"\nSGD Square Root Decay training completed!")
    print(f"Initial learning rate: {results['learning_rates_history'][0]:.6f}")
    print(f"Final learning rate: {results['learning_rates_history'][-1]:.6f}")
    print(f"Results saved to: {results_dir.absolute()}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()