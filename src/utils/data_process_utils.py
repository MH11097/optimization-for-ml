"""
Data Processing Utilities - Backward Compatibility Module
This module provides backward compatibility with the original data_process_utils.py
by re-exporting functions with their original names and signatures.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
from pathlib import Path
# Import from refactored modules
from .data.loaders import FileLoader, ChunkedLoader
from .data.cleaners import DataCleaner, ColumnCleaner
from .data.splitters import DataSplitter, BatchGenerator
from .data.validators import DataValidator, InputValidator
from .data.transformers import DataTransformer, FeatureScaler

# Initialize objects for backward compatibility
_loader = FileLoader()
_cleaner = DataCleaner()
_splitter = DataSplitter()
_validator = DataValidator()
_transformer = DataTransformer()

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================
def load_du_lieu(dataset_path: str = None) -> tuple:
    """Load processed dataset (backward compatibility)"""
    from .data.loaders import load_du_lieu as _load_du_lieu
    return _load_du_lieu(dataset_path)

def load_data_file(file_path: str) -> pd.DataFrame:
    """Load data from file (backward compatibility)"""
    return _loader.load_csv(file_path)

def tai_du_lieu_chunked(file_path: str, chunk_size: int = 10000,
                       max_chunks: Optional[int] = None) -> pd.DataFrame:
    """Load data in chunks (backward compatibility)"""
    chunked_loader = ChunkedLoader(chunk_size=chunk_size)
    return chunked_loader.load_csv(file_path, max_chunks=max_chunks)

def lam_sach_ten_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names (backward compatibility)"""
    return _cleaner.clean_column_names(df)

def xu_ly_gia_tri_null(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """Handle null values (backward compatibility)"""
    return _cleaner.handle_missing_values(df, strategy=strategy)

def tach_dac_trung_va_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target (backward compatibility)"""
    return _cleaner.separate_features_target(df, target_col)

def chuan_hoa_du_lieu(X: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize data (backward compatibility)"""
    scaler = FeatureScaler(method=method)
    X_scaled = scaler.fit_transform(X)
    scaler_params = scaler.get_parameters()
    return X_scaled, scaler_params

def toi_uu_memory_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage (backward compatibility)"""
    return _transformer.optimize_memory_usage(df)

def lay_thong_tin_du_lieu(df: pd.DataFrame) -> Dict[str, Any]:
    """Get data information (backward compatibility)"""
    return _validator.get_data_info(df)

def tao_batches(X: np.ndarray, y: np.ndarray, batch_size: int = 32,
               shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create batches (backward compatibility)"""
    generator = BatchGenerator(batch_size=batch_size, shuffle=shuffle)
    return list(generator.create_batches(X, y))

def chia_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test (backward compatibility)"""
    return _splitter.train_test_split(X, y, test_size=test_size, random_state=random_state)

def kiem_tra_du_lieu_dau_vao(X: np.ndarray, y: np.ndarray) -> bool:
    """Validate input data (backward compatibility)"""
    return _validator.validate_arrays(X, y)

def chuyen_pandas_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """Convert pandas DataFrame to numpy array (backward compatibility)"""
    return df.values

def in_thong_tin_du_lieu(df: pd.DataFrame) -> None:
    """Print data information (backward compatibility)"""
    info = lay_thong_tin_du_lieu(df)
    print(f"\nDataFrame Information:")
    print(f"  Shape: {info['shape']}")
    print(f"  Memory usage: {info['memory_usage_mb']:.2f} MB")
    print(f"  Null values: {info['null_count']}")
    print(f"  Data types: {len(info['dtypes'])} unique types")
    if 'numeric_columns' in info:
        print(f"  Numeric columns: {len(info['numeric_columns'])}")
    if 'categorical_columns' in info:
        print(f"  Categorical columns: {len(info['categorical_columns'])}")

# Additional utility functions for experimental setups
def tach_du_lieu_train_test(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
                           random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into train/test sets (backward compatibility)"""
    X, y = tach_dac_trung_va_target(df, target_col)
    X_train, X_test, y_train, y_test = chia_train_test(
        X.values, y.values, test_size=test_size, random_state=random_state
    )
    # Convert back to pandas
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.Series(y_train, name=target_col)
    y_test = pd.Series(y_test, name=target_col)
    return X_train, X_test, y_train, y_test

def load_and_split_data(file_path: str, target_col: str, test_size: float = 0.2,
                       clean_data: bool = True, normalize: bool = True,
                       random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split data in one step (backward compatibility)"""
    # Load data
    df = load_du_lieu(file_path)
    # Clean data if requested
    if clean_data:
        df = lam_sach_ten_cot(df)
        df = xu_ly_gia_tri_null(df)
    # Separate features and target
    X, y = tach_dac_trung_va_target(df, target_col)
    # Normalize if requested
    if normalize:
        X, _ = chuan_hoa_du_lieu(X)
    # Convert to numpy and split
    X_train, X_test, y_train, y_test = chia_train_test(
        X.values, y.values, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def create_validation_split(X: np.ndarray, y: np.ndarray, val_size: float = 0.2,
                          random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create validation split from training data (backward compatibility)"""
    return chia_train_test(X, y, test_size=val_size, random_state=random_state)

def preprocess_data_pipeline(df: pd.DataFrame, target_col: str,
                           clean_columns: bool = True,
                           handle_missing: str = 'auto',
                           normalize_method: str = 'standard') -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Complete data preprocessing pipeline (backward compatibility)"""
    preprocessing_info = {}
    # Clean column names
    if clean_columns:
        df = lam_sach_ten_cot(df)
        preprocessing_info['columns_cleaned'] = True
    # Handle missing values
    original_nulls = df.isnull().sum().sum()
    df = xu_ly_gia_tri_null(df, strategy=handle_missing)
    final_nulls = df.isnull().sum().sum()
    preprocessing_info['nulls_handled'] = {
        'original': int(original_nulls),
        'final': int(final_nulls),
        'strategy': handle_missing
    }
    # Separate features and target
    X, y = tach_dac_trung_va_target(df, target_col)
    # Normalize features
    X_normalized, scaler_params = chuan_hoa_du_lieu(X, method=normalize_method)
    preprocessing_info['normalization'] = {
        'method': normalize_method,
        'scaler_params': scaler_params
    }
    return X_normalized, y, preprocessing_info

# Export all backward compatibility functions
__all__ = [
    # Core data loading
    'load_du_lieu',
    'tai_du_lieu_chunked',
    # Data cleaning
    'lam_sach_ten_cot',
    'xu_ly_gia_tri_null',
    'tach_dac_trung_va_target',
    # Data transformation
    'chuan_hoa_du_lieu',
    'toi_uu_memory_dataframe',
    'chuyen_pandas_to_numpy',
    # Data analysis
    'lay_thong_tin_du_lieu',
    'in_thong_tin_du_lieu',
    # Data splitting
    'tao_batches',
    'chia_train_test',
    'tach_du_lieu_train_test',
    # Validation
    'kiem_tra_du_lieu_dau_vao',
    # High-level functions
    'load_and_split_data',
    'create_validation_split',
    'preprocess_data_pipeline',
]