"""
Data cleaning utilities for preprocessing pipelines.
This module provides systematic data cleaning capabilities including
column name standardization, null value handling, and data quality
improvements with scientific rigor.
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional, List
from enum import Enum

class NullStrategy(Enum):
    """Enumeration of null handling strategies."""
    AUTO = "auto"
    DROP_ROWS = "drop"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    DROP_COLUMNS = "drop_columns"

class ColumnCleaner:
    """
    Scientific column name cleaning and standardization.
    
    Provides systematic methods to clean and standardize column names
    following consistent naming conventions for data science workflows.
    """
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame, 
                          convention: str = 'snake_case',
                          remove_special: bool = True,
                          max_length: Optional[int] = None) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Args:
            df: DataFrame with columns to clean
            convention: Naming convention ('snake_case', 'camelCase', 'PascalCase')
            remove_special: Remove special characters
            max_length: Maximum column name length
            
        Returns:
            DataFrame with cleaned column names
        """
        df = df.copy()
        new_columns = []
        
        for col in df.columns:
            # Convert to string and strip whitespace
            clean_col = str(col).strip()
            
            if convention == 'snake_case':
                clean_col = ColumnCleaner._to_snake_case(clean_col, remove_special)
            elif convention == 'camelCase':
                clean_col = ColumnCleaner._to_camel_case(clean_col, remove_special)
            elif convention == 'PascalCase':
                clean_col = ColumnCleaner._to_pascal_case(clean_col, remove_special)
            
            # Apply length limit
            if max_length and len(clean_col) > max_length:
                clean_col = clean_col[:max_length]
                
            # Ensure uniqueness
            original_col = clean_col
            counter = 1
            while clean_col in new_columns:
                clean_col = f"{original_col}_{counter}"
                counter += 1
                
            new_columns.append(clean_col)
            
        df.columns = new_columns
        return df
    
    @staticmethod
    def _to_snake_case(text: str, remove_special: bool = True) -> str:
        """Convert text to snake_case."""
        # Convert to lowercase
        text = text.lower()
        
        if remove_special:
            # Replace special characters with underscore
            text = re.sub(r'[^a-zA-Z0-9_]', '_', text)
            # Remove consecutive underscores
            text = re.sub(r'_+', '_', text)
            # Remove leading/trailing underscores
            text = text.strip('_')
        
        return text
    
    @staticmethod
    def _to_camel_case(text: str, remove_special: bool = True) -> str:
        """Convert text to camelCase."""
        if remove_special:
            # Split on special characters and spaces
            parts = re.split(r'[^a-zA-Z0-9]+', text)
        else:
            parts = text.split()
            
        # First part lowercase, rest title case
        if parts:
            camel = parts[0].lower()
            camel += ''.join(word.capitalize() for word in parts[1:])
            return camel
        return text
    
    @staticmethod
    def _to_pascal_case(text: str, remove_special: bool = True) -> str:
        """Convert text to PascalCase."""
        if remove_special:
            parts = re.split(r'[^a-zA-Z0-9]+', text)
        else:
            parts = text.split()
            
        return ''.join(word.capitalize() for word in parts if word)

class DataCleaner:
    """
    Comprehensive data cleaning for machine learning pipelines.
    
    Provides systematic approaches to handle missing values, outliers,
    and data quality issues with scientific methodology.
    """
    
    @staticmethod
    def handle_null_values(df: pd.DataFrame, 
                          strategy: NullStrategy = NullStrategy.AUTO,
                          threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle null values using specified strategy.
        
        Args:
            df: DataFrame to clean
            strategy: Null handling strategy
            threshold: Threshold for column dropping (when strategy is AUTO)
            
        Returns:
            DataFrame with null values handled
        """
        df = df.copy()
        
        if strategy == NullStrategy.AUTO:
            return DataCleaner._handle_nulls_auto(df, threshold)
        elif strategy == NullStrategy.DROP_ROWS:
            return df.dropna()
        elif strategy == NullStrategy.DROP_COLUMNS:
            return DataCleaner._drop_null_columns(df, threshold)
        elif strategy == NullStrategy.FILL_MEAN:
            return DataCleaner._fill_numeric_mean(df)
        elif strategy == NullStrategy.FILL_MEDIAN:
            return DataCleaner._fill_numeric_median(df)
        elif strategy == NullStrategy.FILL_MODE:
            return DataCleaner._fill_mode(df)
        else:
            raise ValueError(f"Unsupported null strategy: {strategy}")
    
    @staticmethod
    def _handle_nulls_auto(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Automatically handle nulls based on data characteristics."""
        df = df.copy()
        
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            
            if null_ratio > threshold:
                # Too many nulls - drop column
                df = df.drop(columns=[col])
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Numeric column - fill with median (robust to outliers)
                df[col] = df[col].fillna(df[col].median())
            else:
                # Categorical column - fill with mode
                mode_values = df[col].mode()
                if len(mode_values) > 0:
                    df[col] = df[col].fillna(mode_values[0])
                    
        return df
    
    @staticmethod
    def _drop_null_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Drop columns with null ratio above threshold."""
        columns_to_keep = []
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio <= threshold:
                columns_to_keep.append(col)
        return df[columns_to_keep]
    
    @staticmethod
    def _fill_numeric_mean(df: pd.DataFrame) -> pd.DataFrame:
        """Fill numeric columns with mean values."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    
    @staticmethod
    def _fill_numeric_median(df: pd.DataFrame) -> pd.DataFrame:
        """Fill numeric columns with median values."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    
    @staticmethod
    def _fill_mode(df: pd.DataFrame) -> pd.DataFrame:
        """Fill all columns with their mode values."""
        df = df.copy()
        for col in df.columns:
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                df[col] = df[col].fillna(mode_values[0])
        return df
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, List[int]]:
        """
        Detect outliers in numeric columns.
        
        Args:
            df: DataFrame to analyze
            method: Outlier detection method ('iqr', 'z_score')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary mapping column names to lists of outlier indices
        """
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                outlier_indices = DataCleaner._detect_iqr_outliers(df[col], threshold)
            elif method == 'z_score':
                outlier_indices = DataCleaner._detect_zscore_outliers(df[col], threshold)
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
                
            if outlier_indices:
                outliers[col] = outlier_indices
                
        return outliers
    
    @staticmethod
    def _detect_iqr_outliers(series: pd.Series, threshold: float = 1.5) -> List[int]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        return series[outlier_mask].index.tolist()
    
    @staticmethod
    def _detect_zscore_outliers(series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_mask = z_scores > threshold
        return series[outlier_mask].index.tolist()
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, 
                         subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows with scientific approach.
        
        Args:
            df: DataFrame to deduplicate
            subset: Columns to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_shape = df.shape
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        
        removed_count = initial_shape[0] - df_clean.shape[0]
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate rows ({removed_count/initial_shape[0]*100:.2f}%)")
            
        return df_clean

# Backward compatibility functions
def lam_sach_ten_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility wrapper for column name cleaning."""
    return ColumnCleaner.clean_column_names(df, convention='snake_case')

def xu_ly_gia_tri_null(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """Backward compatibility wrapper for null value handling."""
    strategy_map = {
        'auto': NullStrategy.AUTO,
        'drop': NullStrategy.DROP_ROWS,
        'fill_mean': NullStrategy.FILL_MEAN,
        'fill_median': NullStrategy.FILL_MEDIAN,
        'fill_mode': NullStrategy.FILL_MODE
    }
    
    null_strategy = strategy_map.get(strategy, NullStrategy.AUTO)
    return DataCleaner.handle_null_values(df, null_strategy)