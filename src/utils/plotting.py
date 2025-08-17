"""Plotting utility functions"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_curve(cost_history, title="Training Curve"):
    """Plot training cost/loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost/Loss')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
    """Plot predictions vs actual values with perfect prediction line"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=30)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """Plot residuals to check for patterns"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 4))
    
    # Residual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Residual histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importance_values, title="Feature Importance", top_n=15):
    """Plot feature importance"""
    # Sort features by importance
    indices = np.argsort(np.abs(importance_values))[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = [importance_values[i] for i in indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(sorted_features)), sorted_importance)
    
    # Color bars based on positive/negative values
    for i, bar in enumerate(bars):
        if sorted_importance[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_learning_curves(train_scores, val_scores, train_sizes, title="Learning Curves"):
    """Plot learning curves for training and validation"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def setup_plotting_style():
    """Setup consistent plotting style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_comparison_chart(metrics_data: dict, title: str = "Algorithm Comparison"):
    """Plot comprehensive comparison chart for multiple algorithms"""
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_data).T
    
    # Separate error metrics (lower is better) and score metrics (higher is better)
    error_metrics = [col for col in df.columns if any(x in col.lower() for x in ['mse', 'rmse', 'mae'])]
    score_metrics = [col for col in df.columns if any(x in col.lower() for x in ['r2', 'score'])]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error metrics (lower is better)
    if error_metrics:
        df[error_metrics].plot(kind='bar', ax=axes[0], alpha=0.8)
        axes[0].set_title('Error Metrics (Lower is Better)')
        axes[0].set_ylabel('Error Value')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Score metrics (higher is better)
    if score_metrics:
        df[score_metrics].plot(kind='bar', ax=axes[1], alpha=0.8, color=['green', 'orange'])
        axes[1].set_title('Score Metrics (Higher is Better)')
        axes[1].set_ylabel('Score Value')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual", log_scale=False):
    """Enhanced predictions vs actual plot with optional log scale"""
    plt.figure(figsize=(10, 8))
    
    if log_scale:
        plt.loglog(y_true, y_pred, 'o', alpha=0.6, markersize=4)
        plt.xlabel('Actual Values (log scale)')
        plt.ylabel('Predicted Values (log scale)')
    else:
        plt.scatter(y_true, y_pred, alpha=0.6, s=30)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not log_scale:
        plt.axis('equal')
    
    plt.show()


def plot_algorithm_performance_summary(results: dict, save_path: str = None):
    """Create comprehensive performance summary visualization"""
    import pandas as pd
    
    # Convert results to DataFrame
    df = pd.DataFrame(results).T
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. RMSE Comparison
    if 'test_rmse' in df.columns:
        df['test_rmse'].plot(kind='bar', ax=axes[0, 0], color='lightcoral', alpha=0.8)
        axes[0, 0].set_title('Test RMSE (Lower is Better)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. R² Score Comparison
    if 'test_r2' in df.columns:
        df['test_r2'].plot(kind='bar', ax=axes[0, 1], color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Test R² Score (Higher is Better)')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Training Time Comparison
    if 'training_time' in df.columns:
        df['training_time'].plot(kind='bar', ax=axes[1, 0], color='lightblue', alpha=0.8)
        axes[1, 0].set_title('Training Time (Lower is Better)')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Overall Performance Radar Chart
    metrics_for_radar = ['test_rmse', 'test_r2', 'test_mae']
    available_metrics = [m for m in metrics_for_radar if m in df.columns]
    
    if len(available_metrics) >= 2:
        # Normalize metrics for radar chart
        normalized_df = df[available_metrics].copy()
        for col in available_metrics:
            if 'rmse' in col or 'mae' in col:
                # Invert error metrics (lower is better)
                normalized_df[col] = 1 / (1 + normalized_df[col])
            # R² is already higher is better
        
        # Plot normalized scores
        normalized_df.plot(kind='bar', ax=axes[1, 1], alpha=0.8)
        axes[1, 1].set_title('Normalized Performance Scores')
        axes[1, 1].set_ylabel('Normalized Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Algorithm Performance Summary', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
