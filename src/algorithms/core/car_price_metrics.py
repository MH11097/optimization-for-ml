"""
Car price specific metrics for optimization algorithms
Domain-specific evaluation metrics for used car price prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class CarPriceMetrics:
    """Calculate car price domain-specific metrics"""
    
    @staticmethod
    def calculate_price_metrics(y_actual: np.ndarray, 
                               y_predicted: np.ndarray,
                               price_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive car price prediction metrics
        
        Parameters:
        -----------
        y_actual : np.ndarray
            Actual car prices
        y_predicted : np.ndarray  
            Predicted car prices
        price_ranges : dict, optional
            Custom price ranges for analysis
            
        Returns:
        --------
        Dict with car price specific metrics
        """
        if price_ranges is None:
            price_ranges = {
                'budget': (0, 15000),
                'mid_range': (15000, 40000),
                'premium': (40000, 80000),
                'luxury': (80000, float('inf'))
            }
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = float(mean_squared_error(y_actual, y_predicted))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_actual, y_predicted))
        metrics['r2'] = float(r2_score(y_actual, y_predicted))
        
        # Car price specific metrics
        abs_errors = np.abs(y_predicted - y_actual)
        relative_errors = abs_errors / (y_actual + 1e-8)  # Avoid division by zero
        
        # Dollar-based metrics
        metrics['mean_absolute_error_dollars'] = float(np.mean(abs_errors))
        metrics['median_absolute_error_dollars'] = float(np.median(abs_errors))
        metrics['max_absolute_error_dollars'] = float(np.max(abs_errors))
        
        # Percentage-based metrics
        metrics['mean_absolute_percentage_error'] = float(np.mean(relative_errors) * 100)
        metrics['median_absolute_percentage_error'] = float(np.median(relative_errors) * 100)
        
        # Accuracy within thresholds
        metrics['predictions_within_5pct'] = float(np.mean(relative_errors <= 0.05))
        metrics['predictions_within_10pct'] = float(np.mean(relative_errors <= 0.10))
        metrics['predictions_within_15pct'] = float(np.mean(relative_errors <= 0.15))
        
        # Dollar threshold accuracy
        metrics['predictions_within_1000_dollars'] = float(np.mean(abs_errors <= 1000))
        metrics['predictions_within_2500_dollars'] = float(np.mean(abs_errors <= 2500))
        metrics['predictions_within_5000_dollars'] = float(np.mean(abs_errors <= 5000))
        
        # Price range specific analysis
        for range_name, (min_price, max_price) in price_ranges.items():
            mask = (y_actual >= min_price) & (y_actual < max_price)
            if np.sum(mask) > 0:
                range_mae = np.mean(abs_errors[mask])
                range_mape = np.mean(relative_errors[mask]) * 100
                range_r2 = r2_score(y_actual[mask], y_predicted[mask])
                
                metrics[f'{range_name}_mae_dollars'] = float(range_mae)
                metrics[f'{range_name}_mape_percent'] = float(range_mape)
                metrics[f'{range_name}_r2'] = float(range_r2)
                metrics[f'{range_name}_count'] = int(np.sum(mask))
        
        # Outlier analysis
        q75 = np.percentile(abs_errors, 75)
        q95 = np.percentile(abs_errors, 95)
        metrics['outlier_threshold_q75_dollars'] = float(q75)
        metrics['outlier_threshold_q95_dollars'] = float(q95)
        metrics['outliers_above_q95_count'] = int(np.sum(abs_errors > q95))
        
        # Business value metrics
        total_actual_value = np.sum(y_actual)
        total_predicted_value = np.sum(y_predicted)
        metrics['total_value_error_dollars'] = float(abs(total_predicted_value - total_actual_value))
        metrics['total_value_error_percent'] = float(abs(total_predicted_value - total_actual_value) / total_actual_value * 100)
        
        return metrics
    
    @staticmethod
    def analyze_prediction_errors(y_actual: np.ndarray, 
                                 y_predicted: np.ndarray) -> pd.DataFrame:
        """
        Create detailed error analysis DataFrame
        
        Returns:
        --------
        DataFrame with per-prediction error analysis
        """
        abs_errors = np.abs(y_predicted - y_actual)
        relative_errors = abs_errors / (y_actual + 1e-8)
        
        error_df = pd.DataFrame({
            'actual_price': y_actual,
            'predicted_price': y_predicted,
            'absolute_error': abs_errors,
            'relative_error': relative_errors,
            'error_dollars': y_predicted - y_actual,  # Signed error
            'price_category': pd.cut(y_actual, 
                                   bins=[0, 15000, 40000, 80000, float('inf')],
                                   labels=['budget', 'mid_range', 'premium', 'luxury'])
        })
        
        return error_df
    
    @staticmethod
    def get_worst_predictions(y_actual: np.ndarray, 
                             y_predicted: np.ndarray, 
                             n_worst: int = 10) -> pd.DataFrame:
        """Get the worst predictions for analysis"""
        abs_errors = np.abs(y_predicted - y_actual)
        worst_indices = np.argsort(abs_errors)[-n_worst:][::-1]
        
        worst_df = pd.DataFrame({
            'index': worst_indices,
            'actual_price': y_actual[worst_indices],
            'predicted_price': y_predicted[worst_indices],
            'absolute_error': abs_errors[worst_indices],
            'relative_error': abs_errors[worst_indices] / (y_actual[worst_indices] + 1e-8)
        })
        
        return worst_df

# Convenience function
def calculate_price_metrics(y_actual: np.ndarray, y_predicted: np.ndarray, **kwargs) -> Dict[str, float]:
    """Convenience function for calculating car price metrics"""
    return CarPriceMetrics.calculate_price_metrics(y_actual, y_predicted, **kwargs)