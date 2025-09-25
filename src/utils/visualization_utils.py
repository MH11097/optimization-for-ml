"""
Visualization Utilities - Backward Compatibility Module
This module provides backward compatibility with the original visualization_utils.py
by re-exporting functions with their original names and signatures.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
# Import from refactored modules
from .visualization.plots import (
    setup_plot_style, create_color_palette, plot_multi_series,
    plot_predictions_vs_actual, create_subplot_grid, save_figure
)
from .visualization.optimization_viz import (
    plot_convergence, plot_optimization_path, plot_residual_analysis,
    plot_gradient_vector, plot_multi_algorithm_convergence
)
from .visualization.comparison import (
    plot_algorithm_comparison, create_comparison_table, plot_radar_chart,
    plot_performance_matrix
)

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================
def thiet_lap_style_bieu_do(style: str = 'whitegrid', context: str = 'notebook',
                          font_scale: float = 1.1, color_palette: str = 'deep') -> None:
    """Setup plot style (backward compatibility)"""
    setup_plot_style(style=style, context=context, font_scale=font_scale, palette=color_palette)

def tao_color_palette(n_colors: int = 8, palette_name: str = 'deep') -> List[str]:
    """Create color palette (backward compatibility)"""
    return create_color_palette(n_colors=n_colors)

def ve_duong_hoi_tu(cost_history: List[float], gradient_norms: List[float] = None,
                   iterations: List[int] = None, title: str = "Convergence Curve",
                   algorithm_name: str = "Algorithm", save_path: Optional[str] = None,
                   show_plot: bool = True) -> plt.Figure:
    """Plot convergence curve (backward compatibility)"""
    return plot_convergence(
        loss_history=cost_history,
        gradient_norms=gradient_norms,
        iterations=iterations,
        title=title,
        save_path=save_path
    )

def ve_so_sanh_algorithms(results_dict: Dict[str, Dict[str, Any]],
                         metric: str = 'final_cost',
                         title: str = "Algorithm Comparison",
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> plt.Figure:
    """Plot algorithm comparison (backward compatibility)"""
    return plot_algorithm_comparison(
        results_dict=results_dict,
        metric=metric,
        title=title,
        save_path=save_path,
        show_plot=show_plot
    )

def ve_du_doan_vs_thuc_te(y_true: np.ndarray, y_pred: np.ndarray,
                         title: str = "Predictions vs Actual",
                         algorithm_name: str = "Model",
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> plt.Figure:
    """Plot predictions vs actual (backward compatibility)"""
    return plot_predictions_vs_actual(
        y_true=y_true,
        y_pred=y_pred,
        title=title,
        save_path=save_path
    )

def ve_phan_tich_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Residual Analysis",
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
    """Plot residual analysis (backward compatibility)"""
    return plot_residual_analysis(
        y_true=y_true,
        y_pred=y_pred,
        title=title,
        save_path=save_path,
        show_plot=show_plot
    )

def ve_bang_so_sanh_performance(results_dict: Dict[str, Dict[str, Any]],
                               metrics: List[str] = ['mse', 'r2_score', 'training_time'],
                               title: str = "Performance Comparison",
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> plt.Figure:
    """Create performance comparison table (backward compatibility)"""
    return create_comparison_table(
        results_dict=results_dict,
        metrics=metrics,
        title=title,
        save_path=save_path,
        show_plot=show_plot
    )

def ve_radar_chart_algorithms(results_dict: Dict[str, Dict[str, Any]],
                             metrics: List[str] = ['accuracy', 'speed', 'stability'],
                             title: str = "Algorithm Radar Chart",
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> plt.Figure:
    """Plot radar chart for algorithms (backward compatibility)"""
    return plot_radar_chart(
        results_dict=results_dict,
        metrics=metrics,
        title=title,
        save_path=save_path,
        show_plot=show_plot
    )

def ve_ma_tran_heatmap(matrix: np.ndarray, title: str = "Matrix Heatmap",
                      labels: Optional[List[str]] = None,
                      save_path: Optional[str] = None,
                      show_plot: bool = True) -> plt.Figure:
    """Plot matrix as heatmap (backward compatibility)"""
    return plot_performance_matrix(
        matrix=matrix,
        title=title,
        labels=labels,
        save_path=save_path,
        show_plot=show_plot
    )

def ve_gradient_vector(gradient: np.ndarray, title: str = "Gradient Vector",
                      save_path: Optional[str] = None,
                      show_plot: bool = True) -> plt.Figure:
    """Plot gradient vector (backward compatibility)"""
    return plot_gradient_vector(
        gradient=gradient,
        title=title,
        save_path=save_path,
        show_plot=show_plot
    )

def ve_duong_dong_muc_optimization(X: np.ndarray, y: np.ndarray, weights_history: List[np.ndarray],
                                  title: str = "Optimization Path",
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> plt.Figure:
    """Plot optimization path with contours (backward compatibility)"""
    return plot_optimization_path(
        X=X,
        y=y,
        weights_history=weights_history,
        title=title,
        save_path=save_path,
        show_plot=show_plot
    )

def ve_duong_hoi_tu_so_sanh(cost_histories: Dict[str, List[float]],
                           title: str = "Convergence Comparison",
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> plt.Figure:
    """Plot multiple convergence curves (backward compatibility)"""
    return plot_multi_algorithm_convergence(
        cost_histories=cost_histories,
        title=title,
        save_path=save_path,
        show_plot=show_plot
    )

# Additional visualization functions for comprehensive reporting
def tao_bao_cao_visual_tong_hop(results_dict: Dict[str, Dict[str, Any]],
                               X_test: np.ndarray, y_test: np.ndarray,
                               output_dir: str = "visualization_report") -> None:
    """Create comprehensive visual report (backward compatibility)"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    # Setup style
    thiet_lap_style_bieu_do()
    # 1. Algorithm comparison
    ve_so_sanh_algorithms(
        results_dict,
        save_path=str(output_path / "algorithm_comparison.png"),
        show_plot=False
    )
    # 2. Performance table
    ve_bang_so_sanh_performance(
        results_dict,
        save_path=str(output_path / "performance_table.png"),
        show_plot=False
    )
    # 3. Convergence curves
    cost_histories = {}
    for alg_name, results in results_dict.items():
        if 'cost_history' in results:
            cost_histories[alg_name] = results['cost_history']
    if cost_histories:
        ve_duong_hoi_tu_so_sanh(
            cost_histories,
            save_path=str(output_path / "convergence_comparison.png"),
            show_plot=False
        )
    # 4. Individual predictions plots (for best performing algorithm)
    if results_dict:
        best_alg = min(results_dict.keys(),
                      key=lambda k: results_dict[k].get('mse', float('inf')))
        best_results = results_dict[best_alg]
        if 'predict_func' in best_results:
            y_pred = best_results['predict_func'](X_test)
            ve_du_doan_vs_thuc_te(
                y_test, y_pred,
                title=f"Best Algorithm: {best_alg}",
                save_path=str(output_path / "best_predictions.png"),
                show_plot=False
            )
    print(f"📊 Comprehensive visual report saved to: {output_path}")

def luu_bieu_do_theo_batch(figures: List[plt.Figure], filenames: List[str],
                          output_dir: str = "plots") -> None:
    """Save multiple figures in batch (backward compatibility)"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    for fig, filename in zip(figures, filenames):
        if not filename.endswith('.png'):
            filename += '.png'
        save_path = output_path / filename
        save_figure(fig, str(save_path))
    print(f"📊 {len(figures)} plots saved to: {output_path}")

def tao_dashboard_optimization(results_dict: Dict[str, Dict[str, Any]],
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> plt.Figure:
    """Create optimization dashboard (backward compatibility)"""
    # Create a comprehensive dashboard with multiple subplots
    fig, axes = create_subplot_grid(2, 2, figsize=(15, 12))
    # 1. Cost comparison
    algorithms = list(results_dict.keys())
    final_costs = [results_dict[alg].get('final_cost', 0) for alg in algorithms]
    axes[0, 0].bar(algorithms, final_costs)
    axes[0, 0].set_title('Final Cost Comparison')
    axes[0, 0].set_ylabel('Final Cost')
    axes[0, 0].tick_params(axis='x', rotation=45)
    # 2. Training time comparison
    training_times = [results_dict[alg].get('training_time', 0) for alg in algorithms]
    axes[0, 1].bar(algorithms, training_times)
    axes[0, 1].set_title('Training Time Comparison')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    # 3. Convergence curves
    for alg_name, results in results_dict.items():
        if 'cost_history' in results:
            axes[1, 0].plot(results['cost_history'], label=alg_name)
    axes[1, 0].set_title('Convergence Curves')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Cost')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    # 4. Iterations comparison
    iterations = [results_dict[alg].get('iterations', 0) for alg in algorithms]
    axes[1, 1].bar(algorithms, iterations)
    axes[1, 1].set_title('Iterations to Convergence')
    axes[1, 1].set_ylabel('Iterations')
    axes[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show_plot:
        plt.show()
    return fig

def tao_bang_so_sanh_markdown(results_data: List[Dict[str, Any]], save_path: Optional[str] = None) -> str:
    """Create markdown comparison table (backward compatibility function)"""
    # This function was missing but imported in algorithm_comparator.py
    # For now, just print a summary table to console
    try:
        from tabulate import tabulate
    except ImportError:
        print("tabulate not available, using simple print")
        tabulate = None

    if not results_data:
        print("No results data available for comparison")
        return ""

    # Prepare table data
    table_data = []
    headers = ["Setup", "Algorithm", "Loss", "Gradient", "Time", "Converged", "Iterations"]

    for result in results_data:
        row = [
            result.get('setup_name', 'Unknown'),
            result.get('algorithm_name', 'Unknown'),
            f"{result.get('final_loss', float('inf')):.6f}",
            f"{result.get('final_gradient_norm', 0):.6f}",
            f"{result.get('training_time', 0):.4f}",
            "YES" if result.get('converged', False) else "NO",
            result.get('iterations', 0)
        ]
        table_data.append(row)

    # Print table
    if tabulate:
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
    else:
        # Fallback simple table
        table_str = "\n".join([" | ".join(map(str, row)) for row in [headers] + table_data])

    print("=" * 80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 80)
    print(table_str)
    print("=" * 80)

    return table_str

# Export all backward compatibility functions
__all__ = [
    # Setup and styling
    'thiet_lap_style_bieu_do',
    'tao_color_palette',
    # Basic plots
    've_duong_hoi_tu',
    've_so_sanh_algorithms',
    've_du_doan_vs_thuc_te',
    've_phan_tich_residuals',
    # Comparison plots
    've_bang_so_sanh_performance',
    've_radar_chart_algorithms',
    've_ma_tran_heatmap',
    # Specialized plots
    've_gradient_vector',
    've_duong_dong_muc_optimization',
    've_duong_hoi_tu_so_sanh',
    # High-level reporting
    'tao_bao_cao_visual_tong_hop',
    'luu_bieu_do_theo_batch',
    'tao_dashboard_optimization',
    # Missing function fix
    'tao_bang_so_sanh_markdown',
]