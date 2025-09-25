"""
Basic Plotting Utilities and Styling
Provides fundamental plotting utilities including style setup,
color palettes, and basic chart functions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional, Union
import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_plot_style(style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Setup default plotting style and configuration.
    
    Args:
        style: matplotlib style ('seaborn-v0_8', 'ggplot', 'classic')
        figsize: default figure size
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def create_color_palette(n_colors: int) -> List[str]:
    """
    Create a beautiful color palette for plots.
    
    Args:
        n_colors: number of colors needed
        
    Returns:
        List[str]: list of hex color codes
    """
    if n_colors <= 10:
        return sns.color_palette("husl", n_colors).as_hex()
    else:
        return sns.color_palette("viridis", n_colors).as_hex()

def plot_multi_series(data_dict: dict, 
                     metric: str = 'loss',
                     title: str = "Multi-Series Comparison",
                     x_label: str = "Iteration",
                     y_label: str = None,
                     log_scale: bool = False,
                     save_path: str = None) -> None:
    """
    Plot multiple data series for comparison.
    
    Args:
        data_dict: dictionary with series names as keys and data arrays as values
        metric: metric name for y-axis label
        title: plot title
        x_label: x-axis label
        y_label: y-axis label (auto-generated if None)
        log_scale: whether to use log scale for y-axis
        save_path: path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    colors = create_color_palette(len(data_dict))
    
    for i, (name, data) in enumerate(data_dict.items()):
        if len(data) > 0:
            plt.plot(data, color=colors[i], linewidth=2, 
                    marker='o', markersize=4, label=name, alpha=0.8)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label or metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()

def plot_predictions_vs_actual(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              title: str = "Predictions vs Actual Values",
                              save_path: str = None) -> None:
    """
    Create scatter plot comparing predictions vs actual values.
    
    Args:
        y_true: actual values
        y_pred: predicted values  
        title: plot title
        save_path: path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    ax1.scatter(y_true, y_pred, alpha=0.6, s=30)
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions vs Actual')
    ax1.grid(True, alpha=0.3)
    
    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}', 
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Residuals plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def create_subplot_grid(nrows: int, ncols: int, 
                       figsize: Tuple[int, int] = None,
                       titles: List[str] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a subplot grid with optional titles.
    
    Args:
        nrows: number of rows
        ncols: number of columns
        figsize: figure size (auto-calculated if None)
        titles: list of subplot titles
        
    Returns:
        Tuple[plt.Figure, np.ndarray]: figure and axes array
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Ensure axes is always 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(-1, 1) if ncols == 1 else axes.reshape(1, -1)
    
    # Set titles if provided
    if titles:
        flat_axes = axes.flatten()
        for i, title in enumerate(titles):
            if i < len(flat_axes):
                flat_axes[i].set_title(title)
    
    return fig, axes

def save_figure(fig: plt.Figure, path: str, dpi: int = 300, 
               bbox_inches: str = 'tight', close_fig: bool = True) -> None:
    """
    Save figure with consistent settings.
    
    Args:
        fig: matplotlib figure
        path: save path
        dpi: resolution
        bbox_inches: bounding box setting
        close_fig: whether to close figure after saving
    """
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    if close_fig:
        plt.close(fig)

def add_value_labels(ax: plt.Axes, bars, format_str: str = '.2f') -> None:
    """
    Add value labels on top of bars in bar chart.
    
    Args:
        ax: matplotlib axes
        bars: bar container from bar plot
        format_str: format string for values
    """
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:{format_str}}',
                   ha='center', va='bottom', fontsize=10)