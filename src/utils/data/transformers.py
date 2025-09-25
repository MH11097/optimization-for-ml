"""
Data transformation utilities for machine learning preprocessing.
This module provides scientific data transformation capabilities including
feature scaling, normalization, encoding, and feature engineering with
mathematical rigor and reproducibility.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from enum import Enum
import warnings

class ScalingMethod(Enum):
    """Enumeration of feature scaling methods."""
    STANDARD = "standard"      # Z-score normalization
    MINMAX = "minmax"         # Min-max scaling to [0,1]
    ROBUST = "robust"         # Robust scaling using IQR
    MAXABS = "maxabs"         # Maximum absolute scaling
    UNIT_VECTOR = "unit"      # Unit vector scaling

class FeatureScaler:
    """
    Scientific feature scaling with mathematical rigor.
    
    Provides various scaling methods with proper statistical foundations
    and support for inverse transformations and reproducibility.
    """
    
    def __init__(self, method: ScalingMethod = ScalingMethod.STANDARD):
        """
        Initialize feature scaler.
        
        Args:
            method: Scaling method to use
        """
        self.method = method
        self.is_fitted = False
        self.scaling_params_ = {}
        
    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """
        Fit scaler to data.
        
        Args:
            X: Feature matrix to fit on
            
        Returns:
            Self for method chaining
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
            
        self.n_features_ = X.shape[1]
        self.scaling_params_ = {}
        
        for feature_idx in range(self.n_features_):
            feature_data = X[:, feature_idx]
            params = self._compute_scaling_params(feature_data, self.method)
            self.scaling_params_[feature_idx] = params
            
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        X = np.asarray(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
            
        X_scaled = X.copy().astype(float)
        
        for feature_idx in range(self.n_features_):
            params = self.scaling_params_[feature_idx]
            X_scaled[:, feature_idx] = self._apply_scaling(
                X_scaled[:, feature_idx], params, self.method)
                
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data in one step.
        
        Args:
            X: Feature matrix to fit and transform
            
        Returns:
            Scaled feature matrix
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Original scale feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
            
        X_scaled = np.asarray(X_scaled)
        if X_scaled.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X_scaled.shape[1]}")
            
        X_original = X_scaled.copy().astype(float)
        
        for feature_idx in range(self.n_features_):
            params = self.scaling_params_[feature_idx]
            X_original[:, feature_idx] = self._apply_inverse_scaling(
                X_original[:, feature_idx], params, self.method)
                
        return X_original
    
    @staticmethod
    def _compute_scaling_params(data: np.ndarray, method: ScalingMethod) -> Dict[str, float]:
        """Compute scaling parameters for given method."""
        if method == ScalingMethod.STANDARD:
            return {
                'mean': float(np.mean(data)),
                'std': float(np.std(data, ddof=0)),
                'method': 'standard'
            }
        elif method == ScalingMethod.MINMAX:
            return {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'method': 'minmax'
            }
        elif method == ScalingMethod.ROBUST:
            return {
                'median': float(np.median(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'method': 'robust'
            }
        elif method == ScalingMethod.MAXABS:
            return {
                'max_abs': float(np.max(np.abs(data))),
                'method': 'maxabs'
            }
        elif method == ScalingMethod.UNIT_VECTOR:
            return {
                'norm': float(np.linalg.norm(data)),
                'method': 'unit'
            }
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
    
    @staticmethod
    def _apply_scaling(data: np.ndarray, params: Dict[str, Any], method: ScalingMethod) -> np.ndarray:
        """Apply scaling transformation."""
        if method == ScalingMethod.STANDARD:
            if params['std'] > 0:
                return (data - params['mean']) / params['std']
            else:
                return data - params['mean']  # Constant feature
                
        elif method == ScalingMethod.MINMAX:
            data_range = params['max'] - params['min']
            if data_range > 0:
                return (data - params['min']) / data_range
            else:
                return np.zeros_like(data)  # Constant feature
                
        elif method == ScalingMethod.ROBUST:
            iqr = params['q75'] - params['q25']
            if iqr > 0:
                return (data - params['median']) / iqr
            else:
                return data - params['median']  # No variability
                
        elif method == ScalingMethod.MAXABS:
            if params['max_abs'] > 0:
                return data / params['max_abs']
            else:
                return data  # All zeros
                
        elif method == ScalingMethod.UNIT_VECTOR:
            if params['norm'] > 0:
                return data / params['norm']
            else:
                return data  # Zero vector
                
        return data
    
    @staticmethod
    def _apply_inverse_scaling(data: np.ndarray, params: Dict[str, Any], method: ScalingMethod) -> np.ndarray:
        """Apply inverse scaling transformation."""
        if method == ScalingMethod.STANDARD:
            if params['std'] > 0:
                return data * params['std'] + params['mean']
            else:
                return data + params['mean']
                
        elif method == ScalingMethod.MINMAX:
            data_range = params['max'] - params['min']
            if data_range > 0:
                return data * data_range + params['min']
            else:
                return np.full_like(data, params['min'])
                
        elif method == ScalingMethod.ROBUST:
            iqr = params['q75'] - params['q25']
            if iqr > 0:
                return data * iqr + params['median']
            else:
                return data + params['median']
                
        elif method == ScalingMethod.MAXABS:
            if params['max_abs'] > 0:
                return data * params['max_abs']
            else:
                return data
                
        elif method == ScalingMethod.UNIT_VECTOR:
            if params['norm'] > 0:
                return data * params['norm']
            else:
                return data
                
        return data

class DataTransformer:
    """
    Comprehensive data transformation pipeline.
    
    Provides end-to-end data transformation capabilities including
    feature scaling, encoding, and preprocessing with scientific methodology.
    """
    
    def __init__(self):
        self.transformations = {}
        self.is_fitted = False
        
    def add_scaler(self, 
                   columns: Optional[List[str]] = None,
                   method: ScalingMethod = ScalingMethod.STANDARD) -> 'DataTransformer':
        """
        Add feature scaling transformation.
        
        Args:
            columns: Columns to scale (None for all numeric columns)
            method: Scaling method to use
            
        Returns:
            Self for method chaining
        """
        self.transformations['scaler'] = {
            'columns': columns,
            'method': method,
            'transformer': FeatureScaler(method)
        }
        return self
        
    def fit_dataframe(self, df: pd.DataFrame) -> 'DataTransformer':
        """
        Fit transformations to DataFrame.
        
        Args:
            df: DataFrame to fit on
            
        Returns:
            Self for method chaining
        """
        self.feature_names_ = list(df.columns)
        self.original_dtypes_ = df.dtypes.to_dict()
        
        # Fit scaler if configured
        if 'scaler' in self.transformations:
            scaler_config = self.transformations['scaler']
            columns = scaler_config['columns']
            
            if columns is None:
                # Use all numeric columns
                columns = list(df.select_dtypes(include=[np.number]).columns)
                scaler_config['columns'] = columns
            
            if columns:
                X_numeric = df[columns].values
                scaler_config['transformer'].fit(X_numeric)
                
        self.is_fitted = True
        return self
    
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted transformations.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
            
        df_transformed = df.copy()
        
        # Apply scaling if configured
        if 'scaler' in self.transformations:
            scaler_config = self.transformations['scaler']
            columns = scaler_config['columns']
            
            if columns:
                X_numeric = df_transformed[columns].values
                X_scaled = scaler_config['transformer'].transform(X_numeric)
                
                for i, col in enumerate(columns):
                    df_transformed[col] = X_scaled[:, i]
                    
        return df_transformed
    
    def fit_transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform DataFrame in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit_dataframe(df).transform_dataframe(df)

class FeatureEncoder:
    """
    Scientific feature encoding utilities.
    
    Provides encoding methods for categorical variables with proper
    handling of unseen categories and mathematical foundations.
    """
    
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, 
                      columns: List[str],
                      drop_first: bool = False,
                      handle_unknown: str = 'error') -> pd.DataFrame:
        """
        One-hot encode categorical columns.
        
        Args:
            df: DataFrame with categorical columns
            columns: Columns to encode
            drop_first: Drop first category to avoid collinearity
            handle_unknown: How to handle unknown categories ('error', 'ignore')
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        df_encoded = df.copy()
        
        for col in columns:
            if col not in df.columns:
                if handle_unknown == 'error':
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                else:
                    continue
                    
            # Create one-hot encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
            
            # Drop original column and add dummy columns
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
        return df_encoded
    
    @staticmethod
    def target_encode(df: pd.DataFrame,
                     categorical_columns: List[str],
                     target_column: str,
                     smoothing: float = 1.0) -> pd.DataFrame:
        """
        Target encode categorical variables.
        
        Args:
            df: DataFrame with categorical columns
            categorical_columns: Columns to encode
            target_column: Target column for encoding
            smoothing: Smoothing factor for regularization
            
        Returns:
            DataFrame with target encoded columns
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame")
                continue
                
            # Calculate category means
            category_means = df.groupby(col)[target_column].mean()
            global_mean = df[target_column].mean()
            
            # Apply smoothing
            category_counts = df.groupby(col)[target_column].count()
            smoothed_means = (category_counts * category_means + smoothing * global_mean) / (category_counts + smoothing)
            
            # Map encoded values
            df_encoded[f'{col}_encoded'] = df_encoded[col].map(smoothed_means)
            
            # Handle missing mappings with global mean
            df_encoded[f'{col}_encoded'].fillna(global_mean, inplace=True)
            
        return df_encoded

# Backward compatibility functions
def chuan_hoa_du_lieu(X: pd.DataFrame, 
                     method: str = 'standard') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Backward compatibility wrapper for feature scaling."""
    method_map = {
        'standard': ScalingMethod.STANDARD,
        'minmax': ScalingMethod.MINMAX,
        'robust': ScalingMethod.ROBUST
    }
    
    scaling_method = method_map.get(method, ScalingMethod.STANDARD)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return X.copy(), {}
    
    # Apply scaling
    scaler = FeatureScaler(scaling_method)
    X_scaled = X.copy()
    X_numeric_scaled = scaler.fit_transform(X[numeric_cols].values)
    
    for i, col in enumerate(numeric_cols):
        X_scaled[col] = X_numeric_scaled[:, i]
    
    # Extract parameters for backward compatibility
    scaler_params = {}
    for i, col in enumerate(numeric_cols):
        params = scaler.scaling_params_[i]
        scaler_params[col] = params
        
    return X_scaled, scaler_params

def toi_uu_memory_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility wrapper for memory optimization."""
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type == 'object':
            # Try numeric conversion
            numeric_converted = pd.to_numeric(df_optimized[col], errors='ignore')
            if numeric_converted.dtype != 'object':
                df_optimized[col] = numeric_converted
                col_type = df_optimized[col].dtype
        
        # Optimize integer columns
        if col_type in ['int64', 'int32']:
            c_min, c_max = df_optimized[col].min(), df_optimized[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
        
        # Optimize float columns
        elif col_type in ['float64', 'float32']:
            c_min, c_max = df_optimized[col].min(), df_optimized[col].max()
            
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # Convert to category for low cardinality
        elif col_type == 'object':
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            
            if num_unique / num_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
                
    return df_optimized

def preprocess_data(df: pd.DataFrame, target_column: str = None,
                   clean_columns: bool = True, handle_missing: bool = True,
                   scale_features: bool = True) -> pd.DataFrame:
    """
    Comprehensive data preprocessing pipeline (backward compatibility).
    Args:
        df: DataFrame to preprocess
        target_column: Name of target column (optional)
        clean_columns: Whether to clean column names
        handle_missing: Whether to handle missing values
        scale_features: Whether to scale numeric features
    Returns:
        Preprocessed DataFrame
    """
    from .cleaners import lam_sach_ten_cot, xu_ly_gia_tri_null
    df_processed = df.copy()
    # Clean column names
    if clean_columns:
        df_processed = lam_sach_ten_cot(df_processed)
    # Handle missing values
    if handle_missing:
        df_processed = xu_ly_gia_tri_null(df_processed)
    # Scale numeric features (excluding target)
    if scale_features:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_cols:
            feature_cols = [col for col in numeric_cols if col != target_column]
        else:
            feature_cols = list(numeric_cols)
        if feature_cols:
            df_processed[feature_cols], _ = chuan_hoa_du_lieu(df_processed[feature_cols])
    return df_processed