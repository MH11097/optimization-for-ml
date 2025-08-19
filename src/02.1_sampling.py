"""
Step 2.1: Data Sampling cho test optimization algorithms
Tạo stratified sample để giữ đặc trưng dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_process_utils import tai_du_lieu_chunked

# Khai báo số dòng sample trực tiếp
SAMPLE_SIZE = 100000


def create_stratified_samples():
    """Tạo random sample từ processed data (đã scale)"""
    
    print(f"🎯 Tạo random sample với {SAMPLE_SIZE:,} dòng...")
    
    output_dir = Path("data/02.1_sampled")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data (đã scale)
    print("📂 Loading processed data...")
    X_train = tai_du_lieu_chunked("data/02_processed/X_train.csv", max_rows=SAMPLE_SIZE*2)
    y_train = tai_du_lieu_chunked("data/02_processed/y_train.csv", max_rows=SAMPLE_SIZE*2)
    X_test = tai_du_lieu_chunked("data/02_processed/X_test.csv", max_rows=SAMPLE_SIZE//2)
    y_test = tai_du_lieu_chunked("data/02_processed/y_test.csv", max_rows=SAMPLE_SIZE//2)
    
    print(f"   Loaded train: {X_train.shape}, test: {X_test.shape}")
    
    # Random sampling train data
    print("🔄 Random sampling train...")
    train_sample_size = min(SAMPLE_SIZE, len(X_train))
    train_indices = np.random.RandomState(42).choice(len(X_train), size=train_sample_size, replace=False)
    X_train_sample = X_train.iloc[train_indices]
    y_train_sample = y_train.iloc[train_indices]
    
    # Random sampling test data  
    print("🔄 Random sampling test...")
    test_sample_size = min(SAMPLE_SIZE//4, len(X_test))
    test_indices = np.random.RandomState(42).choice(len(X_test), size=test_sample_size, replace=False)
    X_test_sample = X_test.iloc[test_indices]
    y_test_sample = y_test.iloc[test_indices]
    
    # Save files
    X_train_sample.to_csv(output_dir / "X_train.csv", index=False)
    y_train_sample.to_csv(output_dir / "y_train.csv", index=False)
    X_test_sample.to_csv(output_dir / "X_test.csv", index=False)
    y_test_sample.to_csv(output_dir / "y_test.csv", index=False)
    
    # Validation
    print("\n📊 Sample quality check:")
    print(f"   Original train target: mean={y_train.iloc[:,0].mean():.2f}, std={y_train.iloc[:,0].std():.2f}")
    print(f"   Sample train target:   mean={y_train_sample.iloc[:,0].mean():.2f}, std={y_train_sample.iloc[:,0].std():.2f}")
    print(f"   Original test target:  mean={y_test.iloc[:,0].mean():.2f}, std={y_test.iloc[:,0].std():.2f}")
    print(f"   Sample test target:    mean={y_test_sample.iloc[:,0].mean():.2f}, std={y_test_sample.iloc[:,0].std():.2f}")
    
    print(f"\n✅ Train sample: {X_train_sample.shape}")
    print(f"✅ Test sample: {X_test_sample.shape}")
    print(f"📁 Files saved in {output_dir}")

if __name__ == "__main__":
    create_stratified_samples()
    print("\n🎯 Stratified sampling completed!")
    print("📋 Files created:")
    print("   data/02.1_sampled/X_train.csv")
    print("   data/02.1_sampled/y_train.csv") 
    print("   data/02.1_sampled/X_test.csv")
    print("   data/02.1_sampled/y_test.csv")