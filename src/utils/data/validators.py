"""
Data validation utilities for machine learning pipelines.
This module provides systematic data validation capabilities to ensure
data quality, consistency, and suitability for machine learning workflows
with comprehensive error reporting and scientific rigor.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

class ValidationResult:
    """
    Container for validation results with detailed reporting.
    
    Provides structured information about data validation outcomes
    including errors, warnings, and quality metrics.
    """
    
    def __init__(self, is_valid: bool = True):
        self.is_valid = is_valid
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}
        
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
        
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        
    def add_info(self, key: str, value: Any):
        """Add information."""
        self.info[key] = value
        
    def summary(self) -> str:
        """Get validation summary."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        summary = f"{status}\n"
        
        if self.errors:
            summary += f"Errors ({len(self.errors)}):\n"
            for error in self.errors:
                summary += f"  - {error}\n"
                
        if self.warnings:
            summary += f"Warnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                summary += f"  - {warning}\n"
                
        if self.info:
            summary += f"Info:\n"
            for key, value in self.info.items():
                summary += f"  - {key}: {value}\n"
                
        return summary

class InputValidator:
    """
    Scientific input validation for machine learning data.
    
    Provides comprehensive validation of input data including shape,
    type, range, and quality checks with detailed reporting.
    """
    
    @staticmethod
    def validate_ml_inputs(X: np.ndarray, 
                          y: np.ndarray,
                          require_finite: bool = True,
                          min_samples: int = 10,
                          max_features: Optional[int] = None) -> ValidationResult:
        """
        Validate machine learning input data.
        
        Args:
            X: Feature matrix
            y: Target vector
            require_finite: Require all values to be finite
            min_samples: Minimum number of samples required
            max_features: Maximum number of features allowed
            
        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult()
        
        # Basic existence checks
        if X is None:
            result.add_error("Feature matrix X is None")
            return result
            
        if y is None:
            result.add_error("Target vector y is None")
            return result
        
        # Convert to numpy arrays if needed
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X)
                result.add_info("X_conversion", "Converted to numpy array")
            except Exception as e:
                result.add_error(f"Cannot convert X to numpy array: {e}")
                return result
                
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
                result.add_info("y_conversion", "Converted to numpy array")
            except Exception as e:
                result.add_error(f"Cannot convert y to numpy array: {e}")
                return result
        
        # Shape validation
        InputValidator._validate_shapes(X, y, result, min_samples, max_features)
        
        # Data quality validation  
        if require_finite:
            InputValidator._validate_finite_values(X, y, result)
            
        # Statistical validation
        InputValidator._validate_statistics(X, y, result)
        
        return result
    
    @staticmethod
    def _validate_shapes(X: np.ndarray, 
                        y: np.ndarray, 
                        result: ValidationResult,
                        min_samples: int,
                        max_features: Optional[int]):
        """Validate array shapes and dimensions."""
        # Check X dimensionality
        if len(X.shape) != 2:
            result.add_error(f"X must be 2D array, got {len(X.shape)}D")
            return
            
        # Check y dimensionality
        if len(y.shape) not in [1, 2]:
            result.add_error(f"y must be 1D or 2D array, got {len(y.shape)}D")
            return
            
        # For 2D y, check if it's a single column
        if len(y.shape) == 2:
            if y.shape[1] != 1:
                result.add_warning(f"y has {y.shape[1]} columns, expected 1")
            y = y.ravel()  # Flatten for consistency
            
        # Check sample consistency
        if X.shape[0] != len(y):
            result.add_error(f"Sample mismatch: X has {X.shape[0]} samples, y has {len(y)} samples")
            return
        
        n_samples, n_features = X.shape
        
        # Check minimum samples
        if n_samples < min_samples:
            result.add_error(f"Insufficient samples: {n_samples} < {min_samples}")
            
        # Check maximum features
        if max_features and n_features > max_features:
            result.add_warning(f"Many features: {n_features} > {max_features}")
            
        # Add shape info
        result.add_info("n_samples", n_samples)
        result.add_info("n_features", n_features)
        result.add_info("X_shape", X.shape)
        result.add_info("y_shape", y.shape)
    
    @staticmethod
    def _validate_finite_values(X: np.ndarray, 
                               y: np.ndarray, 
                               result: ValidationResult):
        """Validate that all values are finite."""
        # Check X for non-finite values
        if not np.all(np.isfinite(X)):
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            
            if nan_count > 0:
                result.add_error(f"X contains {nan_count} NaN values")
            if inf_count > 0:
                result.add_error(f"X contains {inf_count} infinite values")
                
        # Check y for non-finite values
        if not np.all(np.isfinite(y)):
            nan_count = np.sum(np.isnan(y))
            inf_count = np.sum(np.isinf(y))
            
            if nan_count > 0:
                result.add_error(f"y contains {nan_count} NaN values")
            if inf_count > 0:
                result.add_error(f"y contains {inf_count} infinite values")
    
    @staticmethod
    def _validate_statistics(X: np.ndarray, 
                           y: np.ndarray, 
                           result: ValidationResult):
        """Validate statistical properties of the data."""
        try:
            # X statistics
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            
            # Check for constant features
            constant_features = np.sum(X_std < 1e-12)
            if constant_features > 0:
                result.add_warning(f"X has {constant_features} constant features")
                
            # Check for extreme values
            X_min, X_max = np.min(X), np.max(X)
            if X_max - X_min > 1e6:
                result.add_warning(f"X has large dynamic range: [{X_min:.2e}, {X_max:.2e}]")
            
            # y statistics
            y_mean = np.mean(y)
            y_std = np.std(y)
            y_min, y_max = np.min(y), np.max(y)
            
            # Check for constant target
            if y_std < 1e-12:
                result.add_warning("Target y is approximately constant")
                
            # Add statistical info
            result.add_info("X_range", [float(X_min), float(X_max)])
            result.add_info("y_range", [float(y_min), float(y_max)])
            result.add_info("y_mean", float(y_mean))
            result.add_info("y_std", float(y_std))
            result.add_info("constant_features", int(constant_features))
            
        except Exception as e:
            result.add_warning(f"Statistical validation failed: {e}")

class DataValidator:
    """
    Comprehensive data validation for machine learning workflows.
    
    Provides validation for DataFrames, data quality checks,
    and preprocessing validation with scientific methodology.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          numeric_columns: Optional[List[str]] = None,
                          max_null_ratio: float = 0.5) -> ValidationResult:
        """
        Validate pandas DataFrame for ML workflows.
        
        Args:
            df: DataFrame to validate
            required_columns: Columns that must be present
            numeric_columns: Columns that must be numeric
            max_null_ratio: Maximum allowed null ratio per column
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()
        
        if df is None or df.empty:
            result.add_error("DataFrame is None or empty")
            return result
            
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                result.add_error(f"Missing required columns: {list(missing_cols)}")
        
        # Check numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        result.add_error(f"Column '{col}' is not numeric: {df[col].dtype}")
                        
        # Check null ratios
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio > max_null_ratio:
                result.add_warning(f"Column '{col}' has high null ratio: {null_ratio:.3f}")
                
        # Add DataFrame info
        result.add_info("shape", df.shape)
        result.add_info("columns", list(df.columns))
        result.add_info("dtypes", df.dtypes.to_dict())
        result.add_info("memory_mb", df.memory_usage(deep=True).sum() / (1024**2))
        
        return result
    
    @staticmethod
    def validate_target_feature_split(X: pd.DataFrame, 
                                    y: pd.Series,
                                    target_column: str) -> ValidationResult:
        """
        Validate target-feature split from DataFrame.
        
        Args:
            X: Feature DataFrame
            y: Target Series  
            target_column: Name of target column
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()
        
        # Check that target column is not in features
        if target_column in X.columns:
            result.add_error(f"Target column '{target_column}' found in features")
            
        # Check sample consistency
        if len(X) != len(y):
            result.add_error(f"Sample mismatch: X has {len(X)} samples, y has {len(y)} samples")
            
        # Check target column name consistency
        if hasattr(y, 'name') and y.name != target_column:
            result.add_warning(f"Target series name '{y.name}' != expected '{target_column}'")
            
        # Validate as ML inputs
        try:
            X_array = X.values
            y_array = y.values
            ml_validation = InputValidator.validate_ml_inputs(X_array, y_array)
            
            # Merge results
            result.errors.extend(ml_validation.errors)
            result.warnings.extend(ml_validation.warnings)
            result.info.update(ml_validation.info)
            
            if not ml_validation.is_valid:
                result.is_valid = False
                
        except Exception as e:
            result.add_error(f"Array conversion failed: {e}")
            
        return result

class QualityChecker:
    """
    Data quality assessment utilities.
    
    Provides comprehensive data quality metrics and checks
    for assessing dataset suitability for machine learning.
    """
    
    @staticmethod
    def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment.
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        quality_report = {
            'completeness': QualityChecker._assess_completeness(df),
            'consistency': QualityChecker._assess_consistency(df),
            'uniqueness': QualityChecker._assess_uniqueness(df),
            'validity': QualityChecker._assess_validity(df),
            'recommendations': []
        }
        
        # Generate recommendations
        QualityChecker._generate_recommendations(quality_report, df)
        
        return quality_report
    
    @staticmethod
    def _assess_completeness(df: pd.DataFrame) -> Dict[str, float]:
        """Assess data completeness (non-null ratios)."""
        completeness = {}
        for col in df.columns:
            non_null_ratio = 1 - (df[col].isnull().sum() / len(df))
            completeness[col] = round(non_null_ratio, 4)
            
        completeness['overall'] = round(np.mean(list(completeness.values())), 4)
        return completeness
    
    @staticmethod
    def _assess_consistency(df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency (data types, formats)."""
        consistency = {
            'mixed_types': [],
            'inconsistent_formats': [],
            'dtype_stability': {}
        }
        
        for col in df.columns:
            # Check for mixed types in object columns
            if df[col].dtype == 'object':
                types_in_col = df[col].apply(type).unique()
                if len(types_in_col) > 1:
                    consistency['mixed_types'].append({
                        'column': col,
                        'types': [str(t) for t in types_in_col]
                    })
                    
            # Check dtype stability (can be converted to more specific type)
            try:
                if df[col].dtype == 'object':
                    numeric_converted = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_converted.isnull().all():
                        consistency['dtype_stability'][col] = 'can_be_numeric'
            except:
                pass
                
        return consistency
    
    @staticmethod
    def _assess_uniqueness(df: pd.DataFrame) -> Dict[str, float]:
        """Assess data uniqueness (duplicate ratios)."""
        uniqueness = {}
        
        # Overall duplicate ratio
        duplicate_rows = df.duplicated().sum()
        uniqueness['duplicate_rows'] = duplicate_rows
        uniqueness['duplicate_ratio'] = round(duplicate_rows / len(df), 4)
        
        # Per-column uniqueness
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df[col].dropna())
            uniqueness[f'{col}_uniqueness'] = round(unique_ratio, 4)
            
        return uniqueness
    
    @staticmethod
    def _assess_validity(df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity (outliers, ranges)."""
        validity = {
            'outliers': {},
            'range_issues': {},
            'format_issues': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Detect outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                validity['outliers'][col] = {
                    'count': outlier_count,
                    'percentage': round(outlier_count / len(df) * 100, 2)
                }
                
            # Check for unreasonable ranges
            col_min, col_max = df[col].min(), df[col].max()
            if col_max == col_min:
                validity['range_issues'][col] = 'constant_values'
            elif np.isinf(col_max) or np.isinf(col_min):
                validity['range_issues'][col] = 'infinite_values'
                
        return validity
    
    @staticmethod
    def _generate_recommendations(quality_report: Dict[str, Any], df: pd.DataFrame):
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations
        completeness = quality_report['completeness']
        low_completeness_cols = [col for col, ratio in completeness.items() 
                               if col != 'overall' and ratio < 0.7]
        if low_completeness_cols:
            recommendations.append(f"Consider imputing or dropping columns with low completeness: {low_completeness_cols}")
            
        # Uniqueness recommendations  
        uniqueness = quality_report['uniqueness']
        if uniqueness['duplicate_ratio'] > 0.05:
            recommendations.append(f"Remove {uniqueness['duplicate_rows']} duplicate rows ({uniqueness['duplicate_ratio']:.1%})")
            
        # Consistency recommendations
        consistency = quality_report['consistency']
        if consistency['mixed_types']:
            recommendations.append("Fix mixed data types in object columns")
            
        if consistency['dtype_stability']:
            recommendations.append("Convert object columns to appropriate numeric types")
            
        # Validity recommendations
        validity = quality_report['validity']
        if validity['outliers']:
            outlier_cols = list(validity['outliers'].keys())
            recommendations.append(f"Investigate outliers in columns: {outlier_cols}")
            
        quality_report['recommendations'] = recommendations

# Backward compatibility functions
def kiem_tra_du_lieu_dau_vao(X: np.ndarray, y: np.ndarray) -> bool:
    """Backward compatibility wrapper for input validation."""
    result = InputValidator.validate_ml_inputs(X, y)
    
    if not result.is_valid:
        for error in result.errors:
            print(f"Lỗi: {error}")
        return False
    
    for warning in result.warnings:
        print(f"Cảnh báo: {warning}")
        
    if 'n_samples' in result.info and 'n_features' in result.info:
        print(f"Dữ liệu hợp lệ: {result.info['n_samples']} samples, {result.info['n_features']} features")
        
    return True

def chuyen_pandas_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """Backward compatibility wrapper for DataFrame to numpy conversion."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("Không có cột nào là numeric để chuyển đổi")
        
    return numeric_df.values.astype(np.float64)

def validate_input_data(X: np.ndarray, y: np.ndarray) -> bool:
    """Backward compatibility wrapper for input validation (alias)."""
    return kiem_tra_du_lieu_dau_vao(X, y)