"""
Algorithm Comparison Visualizations
Functions for creating comparison charts, tables, and analysis
between different optimization algorithms and their performance.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import seaborn as sns
try:
    from .plots import create_color_palette, add_value_labels
except ImportError:
    from plots import create_color_palette, add_value_labels

def plot_algorithm_comparison(performance_data: Dict[str, Dict[str, float]],
                            metrics: List[str],
                            title: str = "Algorithm Performance Comparison",
                            save_path: Optional[str] = None) -> None:
    """
    Create comparison charts for multiple algorithms across different metrics.
    
    Args:
        performance_data: dict with algorithm names as keys and metric dicts as values
        metrics: list of metric names to compare
        title: plot title
        save_path: path to save the figure
    """
    if not performance_data or not metrics:
        print("No data to plot")
        return
    
    n_metrics = len(metrics)
    colors = create_color_palette(len(performance_data))
    
    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract data for this metric
        algorithm_names = list(performance_data.keys())
        values = [performance_data[alg].get(metric, 0) for alg in algorithm_names]
        
        # Create bar chart
        bars = ax.bar(algorithm_names, values, color=colors, alpha=0.8)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        add_value_labels(ax, bars)
        
        # Rotate x-axis labels if they're long
        if any(len(name) > 10 for name in algorithm_names):
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def create_comparison_table(performance_data: Dict[str, Dict[str, float]],
                          save_path: Optional[str] = None,
                          format_precision: int = 4) -> pd.DataFrame:
    """
    Create a formatted comparison table for algorithm performance.
    
    Args:
        performance_data: dict with algorithm names as keys and metric dicts as values
        save_path: path to save as CSV (optional)
        format_precision: decimal precision for formatting
        
    Returns:
        pd.DataFrame: comparison table
    """
    if not performance_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_data).T
    
    # Round numeric values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(format_precision)
    
    # Sort by first metric (assumed to be most important)
    if not df.empty:
        first_metric = df.columns[0]
        df = df.sort_values(first_metric, ascending=True)
    
    if save_path:
        df.to_csv(save_path, index=True)
    
    return df

def plot_radar_chart(performance_data: Dict[str, Dict[str, float]],
                    metrics: List[str],
                    title: str = "Algorithm Performance Radar Chart",
                    save_path: Optional[str] = None,
                    normalize: bool = True) -> None:
    """
    Create radar chart for algorithm comparison.
    
    Args:
        performance_data: dict with algorithm names as keys and metric dicts as values
        metrics: list of metric names
        title: plot title
        save_path: path to save the figure
        normalize: whether to normalize values to 0-1 range
    """
    if not performance_data or not metrics:
        print("No data for radar chart")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Set up angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = create_color_palette(len(performance_data))
    
    for i, (name, data) in enumerate(performance_data.items()):
        values = [data.get(metric, 0) for metric in metrics]
        
        if normalize:
            # Normalize to 0-1 range per metric
            max_values = [max(performance_data[alg].get(metric, 0) 
                            for alg in performance_data.keys()) 
                         for metric in metrics]
            values = [v/max_v if max_v > 0 else 0 for v, max_v in zip(values, max_values)]
        
        values = values + [values[0]]  # Close the plot
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
    ax.set_ylim(0, 1 if normalize else None)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_performance_matrix(performance_data: Dict[str, Dict[str, float]],
                          title: str = "Performance Heatmap",
                          save_path: Optional[str] = None,
                          cmap: str = 'viridis') -> None:
    """
    Create a heatmap matrix of algorithm performance.
    
    Args:
        performance_data: dict with algorithm names as keys and metric dicts as values
        title: plot title
        save_path: path to save the figure
        cmap: colormap name
    """
    if not performance_data:
        print("No data for heatmap")
        return
    
    # Convert to DataFrame and transpose
    df = pd.DataFrame(performance_data).T
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(df, annot=True, cmap=cmap, center=0,
                fmt='.4f', square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_convergence_comparison(algorithms_data: Dict[str, List[float]],
                              title: str = "Convergence Speed Comparison",
                              save_path: Optional[str] = None,
                              log_scale: bool = True) -> None:
    """
    Plot convergence curves for multiple algorithms.
    
    Args:
        algorithms_data: dict with algorithm names as keys and loss histories as values
        title: plot title
        save_path: path to save the figure
        log_scale: whether to use log scale for y-axis
    """
    plt.figure(figsize=(12, 8))
    
    colors = create_color_palette(len(algorithms_data))
    
    for i, (name, loss_history) in enumerate(algorithms_data.items()):
        if len(loss_history) > 0:
            plt.plot(loss_history, color=colors[i], linewidth=2, 
                    marker='o', markersize=4, label=name, alpha=0.8)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()

def create_performance_summary(results: Dict[str, Any],
                             metrics: List[str] = None,
                             title: str = "Performance Summary") -> str:
    """
    Create a text summary of algorithm performance.
    
    Args:
        results: results dictionary
        metrics: metrics to include in summary
        title: summary title
        
    Returns:
        str: formatted performance summary
    """
    if not results:
        return "No results to summarize."
    
    summary = [f"\n{title}", "=" * len(title)]
    
    if metrics is None:
        metrics = ['final_loss', 'iterations', 'training_time', 'converged']
    
    for algorithm, data in results.items():
        summary.append(f"\n{algorithm}:")
        summary.append("-" * (len(algorithm) + 1))
        
        for metric in metrics:
            if metric in data:
                value = data[metric]
                if isinstance(value, float):
                    summary.append(f"  {metric.replace('_', ' ').title()}: {value:.6f}")
                elif isinstance(value, bool):
                    summary.append(f"  {metric.replace('_', ' ').title()}: {'✓' if value else '✗'}")
                else:
                    summary.append(f"  {metric.replace('_', ' ').title()}: {value}")
    
    return "\n".join(summary)

def plot_efficiency_scatter(performance_data: Dict[str, Dict[str, float]],
                          x_metric: str = 'training_time',
                          y_metric: str = 'final_loss',
                          size_metric: Optional[str] = None,
                          title: str = "Algorithm Efficiency Analysis",
                          save_path: Optional[str] = None) -> None:
    """
    Create scatter plot showing algorithm efficiency trade-offs.
    
    Args:
        performance_data: performance data dictionary
        x_metric: metric for x-axis
        y_metric: metric for y-axis
        size_metric: metric for bubble size (optional)
        title: plot title
        save_path: path to save the figure
    """
    if not performance_data:
        print("No data for efficiency plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Extract data
    algorithms = list(performance_data.keys())
    x_data = [performance_data[alg].get(x_metric, 0) for alg in algorithms]
    y_data = [performance_data[alg].get(y_metric, 0) for alg in algorithms]
    
    if size_metric:
        size_data = [performance_data[alg].get(size_metric, 1) for alg in algorithms]
        # Normalize size data for better visualization
        max_size = max(size_data) if size_data else 1
        sizes = [100 + 400 * (s / max_size) for s in size_data]
    else:
        sizes = [100] * len(algorithms)
    
    colors = create_color_palette(len(algorithms))
    
    # Create scatter plot
    scatter = plt.scatter(x_data, y_data, s=sizes, c=colors, alpha=0.7, edgecolors='black')
    
    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        plt.annotate(alg, (x_data[i], y_data[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, alpha=0.8)
    
    plt.xlabel(x_metric.replace('_', ' ').title())
    plt.ylabel(y_metric.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if size_metric:
        # Add size legend
        plt.text(0.02, 0.98, f'Bubble size: {size_metric.replace("_", " ").title()}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')