"""Data loading utilities for numbered workflow"""

import pandas as pd
import numpy as np


def load_csv_safe(file_path, **kwargs):
    """Safely load CSV file with error handling"""
    try:
        df = pd.read_csv(file_path, low_memory=False, **kwargs)
        print(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def get_data_info(df):
    """Get comprehensive information about the dataset"""
    info = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    return info


def clean_column_names(df):
    """Clean column names by removing spaces and special characters"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


def reduce_memory_usage(df):
    """Reduce memory usage by optimizing data types"""
    return optimize_dataframe_dtypes(df)


def load_data_chunked(file_path: str, chunk_size: int = 10000, max_rows: int = None, 
                      columns: list = None, dtype_dict: dict = None):
    """Memory-efficient chunked data loading - always returns DataFrame
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    chunk_size : int, default 10000
        Number of rows per chunk
    max_rows : int, optional
        Maximum number of rows to load
    columns : list, optional
        List of columns to load (None = all columns)
    dtype_dict : dict, optional
        Custom dtype mappings for columns
    
    Returns:
    --------
    pandas.DataFrame
        Loaded data as DataFrame (user converts to .values if needed)
    """
    import pandas as pd
    import gc
    from pathlib import Path
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"📂 Loading: {file_path}")
    
    try:
        # Get available columns if columns filter specified
        if columns:
            available_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
            columns = [col for col in columns if col in available_cols]
            print(f"  Loading {len(columns)} columns out of {len(available_cols)} available")
        
        # Sample for dtype optimization if not provided
        if dtype_dict is None:
            sample = pd.read_csv(file_path, nrows=1000, usecols=columns)
            print(f"📋 Sample: {sample.shape}, Columns: {len(sample.columns)}")
            
            # Optimize dtypes
            dtype_dict = {}
            for col in sample.columns:
                if sample[col].dtype == 'object':
                    if sample[col].nunique() / len(sample) < 0.5:
                        dtype_dict[col] = 'category'
                elif 'int' in str(sample[col].dtype):
                    col_min, col_max = sample[col].min(), sample[col].max()
                    if col_min >= 0:
                        if col_max < 255: dtype_dict[col] = 'uint8'
                        elif col_max < 65535: dtype_dict[col] = 'uint16'
                        elif col_max < 4294967295: dtype_dict[col] = 'uint32'
                    else:
                        if -128 <= col_min <= 127: dtype_dict[col] = 'int8'
                        elif -32768 <= col_min <= 32767: dtype_dict[col] = 'int16'
                        elif -2147483648 <= col_min <= 2147483647: dtype_dict[col] = 'int32'
                elif sample[col].dtype == 'float64':
                    dtype_dict[col] = 'float32'
            
            print(f"🔧 Optimized {len(dtype_dict)} dtypes")
        
        # Load in chunks
        chunks = []
        total_rows = 0
        
        reader = pd.read_csv(file_path, chunksize=chunk_size, dtype=dtype_dict, 
                           usecols=columns, low_memory=False)
        
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if i % 10 == 0:  # Progress every 10 chunks
                print(f"✅ Chunk {i+1}: {len(chunk):,} rows (Total: {total_rows:,})")
            
            if max_rows and total_rows >= max_rows:
                print(f"🛑 Stopped at {max_rows:,} rows")
                break
                
            # Memory management - combine chunks periodically
            if len(chunks) > 20:
                print("🔄 Combining chunks...")
                temp_df = pd.concat(chunks, ignore_index=True)
                chunks = [temp_df]
                gc.collect()
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            
            # Apply sample size if specified
            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
            
            # Handle boolean columns
            bool_columns = ['is_new', 'has_accidents', 'frame_damaged', 'fleet']
            for col in bool_columns:
                if col in df.columns:
                    try:
                        df[col] = df[col].astype('boolean')
                    except Exception as e:
                        print(f"  Warning: Could not convert {col} to boolean: {e}")
            
            # Final optimization
            df = optimize_dataframe_dtypes(df)
            
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"🎉 Loaded: {df.shape[0]:,} × {df.shape[1]} ({memory_mb:.1f}MB)")
            return df
        
        return pd.DataFrame()  # Return empty DataFrame instead of None
        
    except Exception as e:
        print(f"❌ Loading error: {e}")
        raise


def optimize_dataframe_dtypes(df):
    """Optimize pandas DataFrame dtypes for memory efficiency"""
    import numpy as np
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category if beneficial
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif 'int' in str(df[col].dtype):
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f'Memory optimized: {start_mem:.1f}MB → {end_mem:.1f}MB '
          f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df
