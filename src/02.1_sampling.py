"""
Step 2.1: Data Sampling cho test optimization algorithms
Tạo stratified sample để giữ đặc trưng dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')
from utils.data_loader import load_data_chunked

# Khai báo số dòng sample trực tiếp
SAMPLE_SIZE = 100000


def create_stratified_samples():
    """Tạo stratified sample giữ đặc trưng price distribution"""
    
    print(f"🎯 Tạo stratified sample với {SAMPLE_SIZE:,} dòng...")
    
    output_dir = Path("data/02.1_sampled")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train data
    print("📂 Loading train data...")
    X_train = load_data_chunked("data/02_processed/X_train.csv", max_rows=SAMPLE_SIZE*2)
    y_train = load_data_chunked("data/02_processed/y_train.csv", max_rows=SAMPLE_SIZE*2)
    
    # Load test data
    print("📂 Loading test data...")
    X_test = load_data_chunked("data/02_processed/X_test.csv", max_rows=SAMPLE_SIZE//2)
    y_test = load_data_chunked("data/02_processed/y_test.csv", max_rows=SAMPLE_SIZE//2)
    
    # Stratified sampling train data
    print("🔄 Stratified sampling train...")
    train_sample_size = min(SAMPLE_SIZE, len(X_train))
    X_train_sample, y_train_sample = _stratified_sample_by_price(
        X_train, y_train, train_sample_size
    )
    
    # Stratified sampling test data
    print("🔄 Stratified sampling test...")
    test_sample_size = min(SAMPLE_SIZE//4, len(X_test))
    X_test_sample, y_test_sample = _stratified_sample_by_price(
        X_test, y_test, test_sample_size
    )
    
    # Save files
    X_train_sample.to_csv(output_dir / "X_train.csv", index=False)
    y_train_sample.to_csv(output_dir / "y_train.csv", index=False)
    X_test_sample.to_csv(output_dir / "X_test.csv", index=False)
    y_test_sample.to_csv(output_dir / "y_test.csv", index=False)
    
    # Validation
    print("\n📊 Sample quality check:")
    print(f"   Original train price: mean={y_train.iloc[:,0].mean():.0f}, std={y_train.iloc[:,0].std():.0f}")
    print(f"   Sample train price:   mean={y_train_sample.iloc[:,0].mean():.0f}, std={y_train_sample.iloc[:,0].std():.0f}")
    print(f"   Original test price:  mean={y_test.iloc[:,0].mean():.0f}, std={y_test.iloc[:,0].std():.0f}")
    print(f"   Sample test price:    mean={y_test_sample.iloc[:,0].mean():.0f}, std={y_test_sample.iloc[:,0].std():.0f}")
    
    print(f"\n✅ Train sample: {X_train_sample.shape}")
    print(f"✅ Test sample: {X_test_sample.shape}")
    print(f"📁 Files saved in {output_dir}")


def _stratified_sample_by_price(X, y, sample_size, n_bins=8, random_state=42):
    """Stratified sampling theo price bins để giữ distribution"""
    
    # Tạo price bins
    try:
        price_col = y.iloc[:, 0]
        bins = pd.qcut(price_col, q=n_bins, duplicates='drop', precision=0)
        
        sample_indices = []
        
        # Sample từ mỗi bin proportionally
        for bin_label in bins.cat.categories:
            bin_mask = bins == bin_label
            bin_size = bin_mask.sum()
            
            if bin_size == 0:
                continue
            
            # Proportional allocation
            n_from_bin = max(1, int(sample_size * bin_size / len(X)))
            n_from_bin = min(n_from_bin, bin_size)
            
            # Random sample trong bin
            bin_indices = X[bin_mask].index.tolist()
            if len(bin_indices) > 0:
                sampled = np.random.RandomState(random_state).choice(
                    bin_indices, size=min(n_from_bin, len(bin_indices)), replace=False
                )
                sample_indices.extend(sampled)
        
        # Adjust về đúng sample size
        if len(sample_indices) > sample_size:
            sample_indices = np.random.RandomState(random_state).choice(
                sample_indices, size=sample_size, replace=False
            )
        elif len(sample_indices) < sample_size:
            # Thêm random samples nếu thiếu
            remaining = list(set(X.index) - set(sample_indices))
            additional_needed = sample_size - len(sample_indices)
            if len(remaining) > 0:
                additional = np.random.RandomState(random_state).choice(
                    remaining, size=min(additional_needed, len(remaining)), replace=False
                )
                sample_indices.extend(additional)
        
        return X.loc[sample_indices], y.loc[sample_indices]
        
    except Exception as e:
        print(f"⚠️  Stratified sampling failed ({e}), using random sampling")
        # Fallback to random sampling
        indices = np.random.RandomState(random_state).choice(
            len(X), size=min(sample_size, len(X)), replace=False
        )
        return X.iloc[indices], y.iloc[indices]


if __name__ == "__main__":
    create_stratified_samples()
    print("\n🎯 Stratified sampling completed!")
    print("📋 Files created:")
    print("   data/02.1_sampled/X_train.csv")
    print("   data/02.1_sampled/y_train.csv") 
    print("   data/02.1_sampled/X_test.csv")
    print("   data/02.1_sampled/y_test.csv")