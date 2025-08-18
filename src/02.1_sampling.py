"""
Step 2.1: Data Sampling cho test optimization algorithms
Tạo sample data từ processed data để test code trước khi chạy full dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('.')
from utils.data_loader import load_data_chunked

class DataSampler:
    """Tạo sample data cho test optimization algorithms"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def create_samples(self, 
                      data_path: str = "../data/02_processed/X_train.csv",
                      target_path: str = "../data/02_processed/y_train.csv",
                      sample_sizes: list = [1000, 10000, 100000],
                      output_dir: str = "../data/02.1_sampled") -> dict:
        """Tạo sample data cho test algorithms"""
        
        print("🎯 Tạo sample data...")
        
        # Load data using chunking
        X_df = load_data_chunked(data_path, max_rows=200000)
        y_df = load_data_chunked(target_path, max_rows=200000)
        df = pd.concat([X_df, y_df], axis=1)
        target_column = y_df.columns[0]
        
        print(f"📄 Data: {df.shape[0]:,} × {df.shape[1]}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        sample_results = {}
        
        for size in sample_sizes:
            if size >= len(df):
                size = len(df) - 1
                
            # Random sampling
            sample_df = df.sample(n=size, random_state=self.random_state)
            
            # Tách X và y
            X_sample = sample_df.drop(columns=[target_column])
            y_sample = sample_df[[target_column]]
            
            # Lưu files
            X_path = Path(output_dir) / f"X_sample_{size}.csv"
            y_path = Path(output_dir) / f"y_sample_{size}.csv"
            
            X_sample.to_csv(X_path, index=False)
            y_sample.to_csv(y_path, index=False)
            
            sample_results[f"sample_{size}"] = {
                'size': size,
                'X_path': str(X_path),
                'y_path': str(y_path)
            }
            
            print(f"✅ Sample {size}: {sample_df.shape}")

        # Lưu report
        report = {
            'original_shape': df.shape,
            'samples': sample_results,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        report_path = Path(output_dir) / 'report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Report: {report_path}")
        return report


def load_sample_data(size: int, sample_dir: str = "../data/02.1_sampled"):
    """Load sample data để test algorithms"""
    X_path = Path(sample_dir) / f"X_sample_{size}.csv"
    y_path = Path(sample_dir) / f"y_sample_{size}.csv"
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    
    print(f"📂 Loaded sample {size}: X{X.shape}, y{y.shape}")
    return X, y


if __name__ == "__main__":
    # Tạo samples
    sampler = DataSampler()
    report = sampler.create_samples()
    
    print("\n🎯 Samples created!")
    print("📋 Usage:")
    print("  X, y = load_sample_data(1000)")
    print("  X, y = load_sample_data(10000)")
    print("  X, y = load_sample_data(100000)")