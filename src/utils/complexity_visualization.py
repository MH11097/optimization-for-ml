"""
Computational Complexity Visualization Module

Provides visualization functions for computational complexity analysis
of optimization algorithms, enabling comparison of hardware-independent metrics.

Author: Claude Code Assistant
Date: 2025-01-09
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


def plot_operation_distribution(complexity_data: Dict[str, Any], 
                                save_path: Optional[str] = None,
                                title: str = "Operation Distribution"):
    """
    Create pie chart showing distribution of different operation types
    
    Args:
        complexity_data: Single complexity analysis dictionary
        save_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract operation counts
    basic_metrics = complexity_data.get('basic_metrics', {})
    
    operations = {
        'Function Evaluations': basic_metrics.get('function_evaluations', 0),
        'Gradient Evaluations': basic_metrics.get('gradient_evaluations', 0),
        'Matrix Operations': basic_metrics.get('matrix_vector_multiplications', 0),
        'Vector Operations': basic_metrics.get('vector_operations', 0),
        'Memory Operations': basic_metrics.get('memory_allocations', 0)
    }
    
    # Filter out zero operations
    operations = {k: v for k, v in operations.items() if v > 0}
    
    if not operations:
        print("⚠️ No operation data available for visualization")
        return
    
    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(operations)))
    wedges, texts, autotexts = ax.pie(operations.values(), 
                                     labels=operations.keys(),
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     startangle=90)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend with actual counts
    legend_labels = [f'{k}: {v:,}' for k, v in operations.items()]
    ax.legend(wedges, legend_labels, title="Operation Counts", 
             loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Operation distribution plot saved to: {save_path}")
    
    plt.show()


def plot_scalability_analysis(complexity_results: List[Dict[str, Any]],
                             problem_sizes: List[tuple],
                             save_path: Optional[str] = None,
                             title: str = "Scalability Analysis"):
    """
    Plot how computational complexity scales with problem size
    
    Args:
        complexity_results: List of complexity analysis dictionaries for different problem sizes
        problem_sizes: List of (n_samples, n_features) tuples corresponding to each result
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Calculate problem complexity factors (n * d)
    complexity_factors = [n * d for n, d in problem_sizes]
    
    # Extract total operations
    total_ops = [result.get('basic_metrics', {}).get('total_operations', 0) 
                for result in complexity_results]
    
    # Plot 1: Operations vs Problem Size
    ax1 = axes[0]
    ax1.scatter(complexity_factors, total_ops, s=100, alpha=0.7, color='blue')
    ax1.plot(complexity_factors, total_ops, '--', alpha=0.5, color='blue')
    
    # Fit polynomial to show scaling trend
    if len(complexity_factors) > 2:
        z = np.polyfit(complexity_factors, total_ops, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(complexity_factors), max(complexity_factors), 100)
        ax1.plot(x_smooth, p(x_smooth), 'r-', alpha=0.8, 
                label=f'Trend (degree 2)')
        ax1.legend()
    
    ax1.set_xlabel('Problem Complexity (n × d)')
    ax1.set_ylabel('Total Operations')
    ax1.set_title('Operations vs Problem Size')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Operations per Problem Unit
    ax2 = axes[1]
    ops_per_unit = [ops / factor if factor > 0 else 0 
                   for ops, factor in zip(total_ops, complexity_factors)]
    
    ax2.bar(range(len(ops_per_unit)), ops_per_unit, alpha=0.7, color='green')
    ax2.set_xlabel('Experiment Index')
    ax2.set_ylabel('Operations per Problem Unit')
    ax2.set_title('Computational Efficiency')
    
    # Add problem size labels
    size_labels = [f'n={n}, d={d}' for n, d in problem_sizes]
    ax2.set_xticks(range(len(size_labels)))
    ax2.set_xticklabels(size_labels, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Scalability analysis plot saved to: {save_path}")
    
    plt.show()


def create_complexity_summary_table(complexity_results: List[Dict[str, Any]], 
                                   algorithm_names: List[str],
                                   save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a comprehensive summary table of complexity metrics
    
    Args:
        complexity_results: List of complexity analysis dictionaries
        algorithm_names: List of algorithm names
        save_path: Path to save the table as CSV
        
    Returns:
        DataFrame with complexity summary
    """
    summary_data = []
    
    for i, (result, name) in enumerate(zip(complexity_results, algorithm_names)):
        basic_metrics = result.get('basic_metrics', {})
        per_iter = result.get('per_iteration_averages', {})
        efficiency = result.get('efficiency_metrics', {})
        scalability = result.get('scalability_metrics', {})
        
        row = {
            'Algorithm': name,
            'Total Operations': basic_metrics.get('total_operations', 0),
            'Function Evaluations': basic_metrics.get('function_evaluations', 0),
            'Gradient Evaluations': basic_metrics.get('gradient_evaluations', 0),
            'Matrix Operations': basic_metrics.get('matrix_vector_multiplications', 0),
            'Vector Operations': basic_metrics.get('vector_operations', 0),
            'Peak Memory': basic_metrics.get('peak_memory_size', 0),
            'Operations per Iteration': per_iter.get('operations_per_iter', 0),
            'Convergence Efficiency': efficiency.get('convergence_efficiency', 0),
            'Operations to Convergence': efficiency.get('operations_to_convergence', 0),
            'Memory Efficiency': scalability.get('memory_efficiency', 0),
            'Ops per Problem Unit': scalability.get('operations_per_problem_unit', 0)
        }
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"📊 Complexity summary table saved to: {save_path}")
    
    return df

