"""
Scientific data loading utilities with memory optimization.
This module provides efficient and robust data loading capabilities
for machine learning workflows, with support for large datasets
through chunked loading and memory optimization.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import warnings

class FileLoader:
    """
    Scientific data loading with automatic optimization.
    
    Provides a clean interface for loading various data formats
    with automatic memory optimization and error handling.
    """
    
    @staticmethod
    def load_csv(file_path: Union[str, Path], 
                 columns: Optional[List[str]] = None,
                 max_rows: Optional[int] = None,
                 optimize_memory: bool = True,
                 clean_columns: bool = True) -> pd.DataFrame:
        """
        Load CSV file with optimization.
        
        Args:
            file_path: Path to the CSV file
            columns: Specific columns to load (None for all)
            max_rows: Maximum rows to load
            optimize_memory: Apply memory optimization
            clean_columns: Clean column names
            
        Returns:
            Loaded and optimized DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or corrupted
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # Load data
            df = pd.read_csv(file_path, usecols=columns, nrows=max_rows)
            
            if df.empty:
                raise ValueError(f"Empty DataFrame loaded from {file_path}")
                
            # Apply optimizations
            if clean_columns:
                from .cleaners import ColumnCleaner
                df = ColumnCleaner.clean_column_names(df)
                
            if optimize_memory:
                df = FileLoader._optimize_memory(df)
                
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")
    
    @staticmethod
    def _optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        df = df.copy()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try numeric conversion
                numeric_converted = pd.to_numeric(df[col], errors='ignore')
                if numeric_converted.dtype != 'object':
                    df[col] = numeric_converted
                    col_type = df[col].dtype
            
            # Optimize integer columns
            if col_type in ['int64', 'int32']:
                c_min, c_max = df[col].min(), df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            # Optimize float columns
            elif col_type in ['float64', 'float32']:
                c_min, c_max = df[col].min(), df[col].max()
                
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
            
            # Convert to category if low cardinality
            elif col_type == 'object':
                num_unique = df[col].nunique()
                num_total = len(df[col])
                
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
                    
        return df

class ChunkedLoader:
    """
    Memory-efficient chunked data loading for large datasets.
    
    Provides capabilities to load datasets that don't fit in memory
    by processing them in chunks with automatic optimization.
    """
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize chunked loader.
        
        Args:
            chunk_size: Number of rows per chunk
        """
        self.chunk_size = chunk_size
        
    def load_csv_chunked(self, 
                        file_path: Union[str, Path],
                        columns: Optional[List[str]] = None,
                        max_rows: Optional[int] = None,
                        optimize_each_chunk: bool = True,
                        progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Load large CSV files in chunks.
        
        Args:
            file_path: Path to the CSV file
            columns: Specific columns to load
            max_rows: Maximum total rows to load
            optimize_each_chunk: Optimize memory for each chunk
            progress_callback: Function to call with progress updates
            
        Returns:
            Combined DataFrame from all chunks
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If error during processing
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        chunks = []
        total_rows = 0
        
        try:
            chunk_reader = pd.read_csv(file_path, 
                                     chunksize=self.chunk_size,
                                     usecols=columns)
            
            for i, chunk in enumerate(chunk_reader):
                # Clean and optimize chunk
                if optimize_each_chunk:
                    from .cleaners import ColumnCleaner
                    chunk = ColumnCleaner.clean_column_names(chunk)
                    chunk = FileLoader._optimize_memory(chunk)
                
                chunks.append(chunk)
                total_rows += len(chunk)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total_rows)
                
                # Check row limit
                if max_rows and total_rows >= max_rows:
                    break
                    
        except Exception as e:
            if chunks:
                warnings.warn(f"Error after loading {len(chunks)} chunks: {e}")
            else:
                raise RuntimeError(f"Error loading chunks from {file_path}: {e}")
        
        if not chunks:
            raise ValueError("No data chunks loaded successfully")
            
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Apply final row limit
        if max_rows:
            df = df.head(max_rows)
            
        return df

class DatasetLoader:
    """
    High-level dataset loading for machine learning workflows.
    
    Provides convenient methods for loading common ML dataset formats
    with automatic preprocessing and validation.
    """
    
    @staticmethod
    def load_ml_dataset(data_dir: Union[str, Path], 
                       dataset_name: str = "processed") -> tuple:
        """
        Load preprocessed ML dataset (X_train, X_test, y_train, y_test).
        
        Args:
            data_dir: Directory containing dataset files
            dataset_name: Name of dataset subdirectory
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) as numpy arrays
            
        Raises:
            FileNotFoundError: If dataset files not found
        """
        data_path = Path(data_dir) / dataset_name
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_path}")
            
        try:
            # Load training data
            X_train_path = data_path / "X_train.csv"
            y_train_path = data_path / "y_train.csv"
            X_test_path = data_path / "X_test.csv" 
            y_test_path = data_path / "y_test.csv"
            
            # Use chunked loader for potentially large files
            chunked_loader = ChunkedLoader(chunk_size=50000)
            
            X_train = chunked_loader.load_csv_chunked(X_train_path).values
            X_test = chunked_loader.load_csv_chunked(X_test_path).values
            y_train = chunked_loader.load_csv_chunked(y_train_path).values.ravel()
            y_test = chunked_loader.load_csv_chunked(y_test_path).values.ravel()
            
            print(f"Loaded dataset: Train {X_train.shape}, Test {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise RuntimeError(f"Error loading ML dataset from {data_path}: {e}")
    
    @staticmethod
    def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about a dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset statistics and information
        """
        info = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'null_counts': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['category', 'object']).columns),
            'duplicate_rows': df.duplicated().sum(),
        }
        
        # Add statistics for numeric columns
        if info['numeric_columns']:
            info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
            
        return info

# Backward compatibility functions
def tai_du_lieu_chunked(file_path: str,
                       chunk_size: int = 10000,
                       max_rows: Optional[int] = None,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Backward compatibility wrapper for chunked CSV loading."""
    loader = ChunkedLoader(chunk_size=chunk_size)
    return loader.load_csv_chunked(file_path, columns=columns, max_rows=max_rows)

def load_du_lieu(dataset_path: str = None) -> tuple:
    """Backward compatibility wrapper for loading processed dataset."""
    if dataset_path is None:
        # Default path for backward compatibility
        return DatasetLoader.load_ml_dataset("data", "02.1_sampled")
    else:
        # Use provided path
        return DatasetLoader.load_ml_dataset(dataset_path, "processed")

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """
    Add bias column (column of ones) to feature matrix.
    Args:
        X: Feature matrix (n_samples, n_features)
    Returns:
        Feature matrix with bias column added as first column (n_samples, n_features+1)
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim != 2:
        raise ValueError("Input must be 2D array")
    bias_column = np.ones((X.shape[0], 1))
    return np.hstack([bias_column, X])