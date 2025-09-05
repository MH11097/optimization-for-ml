#!/usr/bin/env python3
"""
Setup script for SGD with Full Randomization Each Epoch
- Full Randomization: Sample with replacement each epoch using different random seed
- Base Learning Rate: 0.01
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
    return Path(filename).stem


def main():
    """Chạy SGD với Full Randomization Each Epoch"""
    print("STOCHASTIC GRADIENT DESCENT - FULL RANDOMIZATION EACH EPOCH")
    
    # Load sampled data
    X_train, X_test, y_train, y_test = load_sampled_data()
    
    # Khởi tạo model với full randomization enabled
    model = SGDModel(
        learning_rate=0.01,
        so_epochs=100,
        random_state=42,
        batch_size=1000,
        ham_loss='ols',
        tolerance=1e-6,
        convergence_check_freq=10,
        learning_rate_schedule='constant',
        shuffle_each_epoch=False,  # Standard shuffling disabled
        randomize_each_epoch=True  # Full randomization: sample with replacement
    )
    
    print(f"\n🎲 Full Randomization Strategy:")
    print(f"   - Sample with replacement each epoch")
    print(f"   - Different random seed each epoch")
    print(f"   - Some samples may appear multiple times per epoch")
    print(f"   - Some samples may not appear in certain epochs")
    print(f"   - Maximum training diversity and stochasticity")
    
    # Training
    results = model.fit(X_train, y_train)
    
    # Evaluation
    metrics = model.evaluate(X_test, y_test)
    
    # Get experiment name and save results
    experiment_name = get_experiment_name()
    results_dir = model.save_results(experiment_name)
    
    # Generate plots
    model.plot_results(X_test, y_test, experiment_name)
    
    print(f"\n🎯 Experiment Complete: {experiment_name}")
    print(f"📁 Results saved to: {results_dir}")
    
    return {
        'model': model,
        'results': results,
        'metrics': metrics,
        'experiment_name': experiment_name
    }


if __name__ == "__main__":
    main()