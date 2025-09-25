"""
Unified metrics calculator for machine learning evaluation.
This module provides a scientific and unified interface for computing various
evaluation metrics used in regression and classification tasks.
Mathematical Foundation:
- Mean Squared Error (MSE): (1/n) * Σ(yi - ŷi)²
- Root Mean Squared Error (RMSE): √MSE
- Mean Absolute Error (MAE): (1/n) * Σ|yi - ŷi|
- R² Score: 1 - SS_res/SS_tot where SS_res = Σ(yi - ŷi)², SS_tot = Σ(yi - ȳ)²
- Adjusted R²: 1 - (1-R²)(n-1)/(n-k-1) where k is number of predictors
"""
import numpy as np
from typing import Union, Tuple, Dict, Any
from enum import Enum

class MetricType(Enum):
    """Enumeration of supported evaluation metrics."""
    MSE = "mse"
    RMSE = "rmse" 
    MAE = "mae"
    R_SQUARED = "r2"
    ADJUSTED_R_SQUARED = "adj_r2"
    MAPE = "mape"
    SMAPE = "smape"

class MetricsCalculator:
    """
    Unified calculator for regression evaluation metrics.
    
    This class provides a clean, scientific interface for computing various
    evaluation metrics, replacing multiple scattered utility functions.
    """
    
    @staticmethod
    def compute(metric_type: MetricType, 
                y_true: np.ndarray, 
                y_pred: np.ndarray,
                **kwargs) -> float:
        """
        Compute the specified metric.
        
        Args:
            metric_type: Type of metric to compute
            y_true: True target values
            y_pred: Predicted values
            **kwargs: Additional parameters (e.g., n_features for adjusted R²)
            
        Returns:
            Computed metric value
            
        Raises:
            ValueError: If inputs have incompatible shapes or invalid parameters
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
            
        if metric_type == MetricType.MSE:
            return MetricsCalculator._compute_mse(y_true, y_pred)
        elif metric_type == MetricType.RMSE:
            return MetricsCalculator._compute_rmse(y_true, y_pred)
        elif metric_type == MetricType.MAE:
            return MetricsCalculator._compute_mae(y_true, y_pred)
        elif metric_type == MetricType.R_SQUARED:
            return MetricsCalculator._compute_r_squared(y_true, y_pred)
        elif metric_type == MetricType.ADJUSTED_R_SQUARED:
            n_features = kwargs.get('n_features')
            if n_features is None:
                raise ValueError("n_features required for adjusted R²")
            return MetricsCalculator._compute_adjusted_r_squared(y_true, y_pred, n_features)
        elif metric_type == MetricType.MAPE:
            return MetricsCalculator._compute_mape(y_true, y_pred)
        elif metric_type == MetricType.SMAPE:
            return MetricsCalculator._compute_smape(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
    
    @staticmethod
    def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² coefficient of determination."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
            
        return float(1 - (ss_res / ss_tot))
    
    @staticmethod
    def _compute_adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """Compute adjusted R² accounting for number of features."""
        r2 = MetricsCalculator._compute_r_squared(y_true, y_pred)
        n_samples = len(y_true)
        
        if n_samples <= n_features + 1:
            return float('-inf')  # Undefined for this case
            
        return float(1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    
    @staticmethod
    def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            raise ValueError("Cannot compute MAPE when all true values are zero")
            
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Symmetric Mean Absolute Percentage Error."""
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        
        if not np.any(mask):
            return 0.0
            
        return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)
    
    @staticmethod
    def compute_multiple(metrics: list, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        **kwargs) -> Dict[str, float]:
        """
        Compute multiple metrics at once.
        
        Args:
            metrics: List of MetricType values to compute
            y_true: True target values
            y_pred: Predicted values  
            **kwargs: Additional parameters for specific metrics
            
        Returns:
            Dictionary mapping metric names to computed values
        """
        results = {}
        for metric in metrics:
            try:
                results[metric.value] = MetricsCalculator.compute(metric, y_true, y_pred, **kwargs)
            except Exception as e:
                results[metric.value] = f"Error: {str(e)}"
                
        return results
    
    @staticmethod
    def get_regression_report(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            n_features: int = None) -> Dict[str, float]:
        """
        Generate a comprehensive regression evaluation report.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            n_features: Number of features (for adjusted R²)
            
        Returns:
            Dictionary with all standard regression metrics
        """
        standard_metrics = [MetricType.MSE, MetricType.RMSE, MetricType.MAE, MetricType.R_SQUARED]
        
        kwargs = {}
        if n_features is not None:
            standard_metrics.append(MetricType.ADJUSTED_R_SQUARED)
            kwargs['n_features'] = n_features
            
        return MetricsCalculator.compute_multiple(standard_metrics, y_true, y_pred, **kwargs)

# Backward compatibility functions
def tinh_MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Backward compatibility wrapper for MSE calculation."""
    return MetricsCalculator.compute(MetricType.MSE, y_true, y_pred)

def tinh_RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Backward compatibility wrapper for RMSE calculation."""
    return MetricsCalculator.compute(MetricType.RMSE, y_true, y_pred)

def tinh_MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Backward compatibility wrapper for MAE calculation."""
    return MetricsCalculator.compute(MetricType.MAE, y_true, y_pred)

def tinh_R_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Backward compatibility wrapper for R² calculation."""
    return MetricsCalculator.compute(MetricType.R_SQUARED, y_true, y_pred)