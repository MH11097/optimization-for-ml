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


def plot_complexity_comparison(complexity_data: List[Dict[str, Any]], 
                               save_path: Optional[str] = None,
                               title: str = "Computational Complexity Comparison"):
    """
    Create comprehensive complexity comparison plots
    
    Args:
        complexity_data: List of complexity analysis dictionaries
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract algorithm names and metrics
    algorithms = [data.get('algorithm', f'Algorithm {i}') for i, data in enumerate(complexity_data)]
    
    # 1. Total Operations Comparison (Bar chart)
    ax1 = axes[0, 0]
    total_ops = [data.get('basic_metrics', {}).get('total_operations', 0) for data in complexity_data]
    bars = ax1.bar(algorithms, total_ops, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Total Operations')
    ax1.set_title('Total Operations Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, ops in zip(bars, total_ops):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ops:,}', ha='center', va='bottom', fontsize=9)
    
    # 2. Function vs Gradient Evaluations
    ax2 = axes[0, 1]
    func_evals = [data.get('basic_metrics', {}).get('function_evaluations', 0) for data in complexity_data]
    grad_evals = [data.get('basic_metrics', {}).get('gradient_evaluations', 0) for data in complexity_data]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax2.bar(x - width/2, func_evals, width, label='Function Evaluations', alpha=0.7)
    ax2.bar(x + width/2, grad_evals, width, label='Gradient Evaluations', alpha=0.7)
    ax2.set_ylabel('Evaluation Count')
    ax2.set_title('Function vs Gradient Evaluations')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45)
    ax2.legend()
    
    # 3. Operations per Iteration
    ax3 = axes[0, 2]
    ops_per_iter = [data.get('per_iteration_averages', {}).get('operations_per_iter', 0) for data in complexity_data]
    bars = ax3.bar(algorithms, ops_per_iter, color='lightgreen', alpha=0.7)
    ax3.set_ylabel('Operations per Iteration')
    ax3.set_title('Computational Intensity')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, ops in zip(bars, ops_per_iter):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ops:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Memory Efficiency
    ax4 = axes[1, 0]
    peak_memory = [data.get('basic_metrics', {}).get('peak_memory_size', 0) for data in complexity_data]
    bars = ax4.bar(algorithms, peak_memory, color='coral', alpha=0.7)
    ax4.set_ylabel('Peak Memory (elements)')
    ax4.set_title('Memory Usage')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mem in zip(bars, peak_memory):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:,}', ha='center', va='bottom', fontsize=9)
    
    # 5. Convergence Efficiency
    ax5 = axes[1, 1]
    conv_efficiency = []
    for data in complexity_data:
        efficiency = data.get('efficiency_metrics', {}).get('convergence_efficiency', 0)
        if efficiency == 0:  # If not converged, use a low value
            efficiency = 0.1
        conv_efficiency.append(efficiency)
    
    bars = ax5.bar(algorithms, conv_efficiency, color='gold', alpha=0.7)
    ax5.set_ylabel('Convergence Efficiency')
    ax5.set_title('Convergence Efficiency (higher = better)')
    ax5.set_ylim(0, 1.1)
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, eff in zip(bars, conv_efficiency):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Operations to Convergence
    ax6 = axes[1, 2]
    ops_to_conv = []
    for data in complexity_data:
        ops = data.get('efficiency_metrics', {}).get('operations_to_convergence', 
                      data.get('basic_metrics', {}).get('total_operations', 0))
        ops_to_conv.append(ops)
    
    bars = ax6.bar(algorithms, ops_to_conv, color='mediumpurple', alpha=0.7)
    ax6.set_ylabel('Operations to Convergence')
    ax6.set_title('Efficiency to Convergence')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, ops in zip(bars, ops_to_conv):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{ops:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Complexity comparison plot saved to: {save_path}")
    
    plt.show()


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


def load_and_compare_complexity_from_results(results_dirs: List[Path],
                                            save_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load complexity analysis from multiple results directories and create comparison
    
    Args:
        results_dirs: List of paths to results directories
        save_dir: Directory to save comparison plots and tables
        
    Returns:
        Dictionary with loaded complexity data and comparison results
    """
    complexity_data = []
    algorithm_names = []
    
    for results_dir in results_dirs:
        # Try to load complexity analysis
        complexity_path = results_dir / "complexity_analysis.json"
        results_path = results_dir / "results.json"
        
        if complexity_path.exists():
            with open(complexity_path, 'r') as f:
                complexity = json.load(f)
            complexity_data.append(complexity)
        elif results_path.exists():
            # Fallback to results.json if complexity_analysis.json doesn't exist
            with open(results_path, 'r') as f:
                results = json.load(f)
            complexity = results.get('computational_complexity', {})
            complexity_data.append(complexity)
        else:
            print(f"⚠️ No complexity data found in {results_dir}")
            continue
            
        # Extract algorithm name
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            algorithm_names.append(results.get('algorithm', results_dir.name))
        else:
            algorithm_names.append(results_dir.name)
    
    if not complexity_data:
        print("❌ No complexity data found in any results directory")
        return {}
    
    # Create comparison plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Comparison plot
        plot_complexity_comparison(
            complexity_data, 
            save_path=str(save_dir / "complexity_comparison.png"),
            title="Computational Complexity Comparison"
        )
        
        # Summary table
        summary_df = create_complexity_summary_table(
            complexity_data, 
            algorithm_names,
            save_path=str(save_dir / "complexity_summary.csv")
        )
        
        return {
            'complexity_data': complexity_data,
            'algorithm_names': algorithm_names,
            'summary_table': summary_df,
            'save_directory': save_dir
        }
    else:
        # Just create plots without saving
        plot_complexity_comparison(complexity_data, title="Computational Complexity Comparison")
        summary_df = create_complexity_summary_table(complexity_data, algorithm_names)
        
        return {
            'complexity_data': complexity_data,
            'algorithm_names': algorithm_names,
            'summary_table': summary_df
        }