"""
Computational Complexity Visualizations
Visualization functions for computational complexity analysis,
operation distribution, and scalability analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
try:
    from .plots import create_color_palette, add_value_labels
except ImportError:
    from plots import create_color_palette, add_value_labels

def plot_operation_distribution(complexity_data: Dict[str, Any], 
                              title: str = "Computational Complexity - Operation Distribution",
                              save_path: Optional[str] = None) -> None:
    """
    Visualize the distribution of different operations in the algorithm.
    
    Args:
        complexity_data: dictionary containing operation counts and details
        title: plot title
        save_path: path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    operations = complexity_data.get('operations', {})
    
    if not operations:
        print("⚠️ No operation data available for visualization")
        ax.text(0.5, 0.5, 'No Operation Data Available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        return
    
    # Prepare data
    operation_names = list(operations.keys())
    counts = list(operations.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(operations)))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(counts, labels=operation_names, autopct='%1.1f%%',
                                     colors=colors, startangle=90, textprops={'fontsize': 10})
    
    # Enhance text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add summary statistics
    total_ops = sum(counts)
    ax.text(0.02, 0.02, f'Total Operations: {total_ops:,}', 
           transform=ax.transAxes, fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Operation distribution plot saved to: {save_path}")
    
    plt.show()

def plot_scalability_analysis(complexity_results: List[Dict[str, Any]],
                            title: str = "Algorithm Scalability Analysis",
                            save_path: Optional[str] = None,
                            fit_curve: bool = True) -> None:
    """
    Plot scalability analysis showing how complexity grows with problem size.
    
    Args:
        complexity_results: list of complexity results for different problem sizes
        title: plot title  
        save_path: path to save the plot
        fit_curve: whether to fit polynomial curve to data
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    complexity_factors = [result.get('complexity_factor', 0) for result in complexity_results]
    total_ops = [result.get('total_operations', 0) for result in complexity_results]
    execution_times = [result.get('execution_time', 0) for result in complexity_results]
    
    # Plot 1: Operations vs Problem Size
    ax1 = axes[0]
    ax1.scatter(complexity_factors, total_ops, color='blue', s=60, alpha=0.7, label='Actual')
    ax1.plot(complexity_factors, total_ops, '--', alpha=0.5, color='blue')
    
    if fit_curve and len(complexity_factors) > 2:
        # Fit polynomial curve
        coeffs = np.polyfit(complexity_factors, total_ops, 2)
        p = np.poly1d(coeffs)
        x_smooth = np.linspace(min(complexity_factors), max(complexity_factors), 100)
        ax1.plot(x_smooth, p(x_smooth), 'r-', alpha=0.8, 
                label=f'Fitted: {coeffs[0]:.2e}x² + {coeffs[1]:.2e}x + {coeffs[2]:.2e}')
    
    ax1.set_xlabel('Problem Size Factor')
    ax1.set_ylabel('Total Operations')
    ax1.set_title('Operations Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Execution Time vs Problem Size
    ax2 = axes[1]
    if execution_times and any(t > 0 for t in execution_times):
        ax2.scatter(complexity_factors, execution_times, color='green', s=60, alpha=0.7, label='Actual')
        ax2.plot(complexity_factors, execution_times, '--', alpha=0.5, color='green')
        
        if fit_curve and len(complexity_factors) > 2:
            coeffs_time = np.polyfit(complexity_factors, execution_times, 2)
            p_time = np.poly1d(coeffs_time)
            ax2.plot(x_smooth, p_time(x_smooth), 'r-', alpha=0.8,
                    label=f'Fitted: {coeffs_time[0]:.2e}x² + {coeffs_time[1]:.2e}x + {coeffs_time[2]:.2e}')
        
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No timing data available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
    
    ax2.set_xlabel('Problem Size Factor')
    ax2.set_title('Time Complexity Growth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Scalability analysis plot saved to: {save_path}")
    
    plt.show()

def create_complexity_summary_table(complexity_data: Dict[str, Any],
                                  save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a summary table of computational complexity metrics.
    
    Args:
        complexity_data: complexity analysis data
        save_path: path to save table as CSV
        
    Returns:
        pd.DataFrame: summary table
    """
    # Extract key metrics
    summary_data = {
        'Metric': [
            'Total Operations',
            'Matrix Operations', 
            'Vector Operations',
            'Scalar Operations',
            'Memory Allocations',
            'Execution Time (ms)',
            'Complexity Factor'
        ],
        'Value': [
            complexity_data.get('total_operations', 0),
            complexity_data.get('operations', {}).get('matrix_ops', 0),
            complexity_data.get('operations', {}).get('vector_ops', 0), 
            complexity_data.get('operations', {}).get('scalar_ops', 0),
            complexity_data.get('memory_allocations', 0),
            complexity_data.get('execution_time', 0) * 1000,  # Convert to ms
            complexity_data.get('complexity_factor', 0)
        ]
    }
    
    df = pd.DataFrame(summary_data)
    
    # Add percentage column for operations
    total_ops = complexity_data.get('total_operations', 1)
    percentages = []
    for i, metric in enumerate(df['Metric']):
        if 'Operations' in metric and metric != 'Total Operations':
            pct = (df.iloc[i]['Value'] / total_ops) * 100
            percentages.append(f"{pct:.1f}%")
        else:
            percentages.append("-")
    
    df['Percentage'] = percentages
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df

def plot_complexity_comparison(algorithms_complexity: Dict[str, Dict[str, Any]],
                             metric: str = 'total_operations',
                             title: str = "Algorithm Complexity Comparison",
                             save_path: Optional[str] = None) -> None:
    """
    Compare computational complexity across different algorithms.
    
    Args:
        algorithms_complexity: dict with algorithm names and their complexity data
        metric: complexity metric to compare
        title: plot title
        save_path: path to save the plot
    """
    if not algorithms_complexity:
        print("No complexity data to plot")
        return
    
    # Extract data
    algorithms = list(algorithms_complexity.keys())
    values = [data.get(metric, 0) for data in algorithms_complexity.values()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = create_color_palette(len(algorithms))
    bars = ax.bar(algorithms, values, color=colors, alpha=0.8)
    
    # Add value labels
    add_value_labels(ax, bars, '.0f')
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if needed
    if any(len(name) > 10 for name in algorithms):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_memory_usage_analysis(memory_data: Dict[str, Any],
                              title: str = "Memory Usage Analysis", 
                              save_path: Optional[str] = None) -> None:
    """
    Visualize memory usage patterns throughout algorithm execution.
    
    Args:
        memory_data: dictionary containing memory usage information
        title: plot title
        save_path: path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Memory usage over time
    if 'memory_timeline' in memory_data:
        timeline = memory_data['memory_timeline']
        times = list(range(len(timeline)))
        
        ax1.plot(times, timeline, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.fill_between(times, timeline, alpha=0.3)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Add peak memory annotation
        peak_idx = np.argmax(timeline)
        peak_value = timeline[peak_idx]
        ax1.annotate(f'Peak: {peak_value:.1f} MB', 
                    xy=(peak_idx, peak_value),
                    xytext=(peak_idx + len(times)*0.1, peak_value),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    else:
        ax1.text(0.5, 0.5, 'No memory timeline data', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Memory allocation by type
    if 'allocations_by_type' in memory_data:
        alloc_data = memory_data['allocations_by_type']
        types = list(alloc_data.keys())
        sizes = list(alloc_data.values())
        
        colors = create_color_palette(len(types))
        wedges, texts, autotexts = ax2.pie(sizes, labels=types, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax2.set_title('Memory Allocation by Type')
    else:
        ax2.text(0.5, 0.5, 'No allocation data', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_big_o_comparison(algorithms_data: Dict[str, Dict[str, Any]],
                         problem_sizes: List[int],
                         title: str = "Big-O Complexity Comparison",
                         save_path: Optional[str] = None) -> None:
    """
    Compare theoretical Big-O complexity curves with actual performance.
    
    Args:
        algorithms_data: dict with algorithm performance at different sizes
        problem_sizes: list of problem sizes tested
        title: plot title
        save_path: path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = create_color_palette(len(algorithms_data) + 3)  # Extra colors for reference curves
    
    # Plot actual algorithm performance
    for i, (name, data) in enumerate(algorithms_data.items()):
        if 'execution_times' in data:
            ax.plot(problem_sizes, data['execution_times'], 
                   'o-', color=colors[i], linewidth=2, label=f'{name} (Actual)')
    
    # Add theoretical complexity reference curves
    n_max = max(problem_sizes)
    n_range = np.linspace(min(problem_sizes), n_max, 100)
    
    # O(n)
    linear_ref = (n_range / n_max) * max([max(data.get('execution_times', [1])) 
                                        for data in algorithms_data.values()])
    ax.plot(n_range, linear_ref, '--', color=colors[-3], alpha=0.7, label='O(n)')
    
    # O(n²)  
    quadratic_ref = (n_range / n_max) ** 2 * max([max(data.get('execution_times', [1])) 
                                                 for data in algorithms_data.values()]) * 0.5
    ax.plot(n_range, quadratic_ref, '--', color=colors[-2], alpha=0.7, label='O(n²)')
    
    # O(n³)
    cubic_ref = (n_range / n_max) ** 3 * max([max(data.get('execution_times', [1])) 
                                             for data in algorithms_data.values()]) * 0.1
    ax.plot(n_range, cubic_ref, '--', color=colors[-1], alpha=0.7, label='O(n³)')
    
    ax.set_xlabel('Problem Size (n)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')