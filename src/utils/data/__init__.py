"""
Data processing utilities for machine learning pipelines.
This module provides clean, scientific implementations of data processing
functions including loading, cleaning, preprocessing, and validation.
Key Components:
- FileLoader: Efficient data loading with chunking support
- DataCleaner: Column cleaning and preprocessing  
- DataSplitter: Train/test/validation splitting utilities
- DataValidator: Input validation and quality checks
- DataTransformer: Normalization and feature scaling
"""
from .loaders import FileLoader, ChunkedLoader
from .cleaners import DataCleaner, ColumnCleaner
from .splitters import DataSplitter, BatchGenerator
from .validators import DataValidator, InputValidator
from .transformers import DataTransformer, FeatureScaler
# Backward compatibility imports
from .loaders import tai_du_lieu_chunked, load_du_lieu, add_bias_column
from .cleaners import lam_sach_ten_cot, xu_ly_gia_tri_null
from .splitters import tao_batches, chia_train_test
from .validators import kiem_tra_du_lieu_dau_vao, chuyen_pandas_to_numpy, validate_input_data
from .transformers import chuan_hoa_du_lieu, toi_uu_memory_dataframe, preprocess_data
__all__ = [
    # Modern unified interfaces
    'FileLoader', 'ChunkedLoader',
    'DataCleaner', 'ColumnCleaner', 
    'DataSplitter', 'BatchGenerator',
    'DataValidator', 'InputValidator',
    'DataTransformer', 'FeatureScaler',
    
    # Backward compatibility functions
    'tai_du_lieu_chunked', 'load_du_lieu', 'add_bias_column',
    'lam_sach_ten_cot', 'xu_ly_gia_tri_null',
    'tao_batches', 'chia_train_test',
    'kiem_tra_du_lieu_dau_vao', 'chuyen_pandas_to_numpy', 'validate_input_data',
    'chuan_hoa_du_lieu', 'toi_uu_memory_dataframe', 'preprocess_data'
]