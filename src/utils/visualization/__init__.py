"""
Visualization Utilities
This module provides plotting and visualization functions for
optimization algorithms and machine learning models.
Modules:
    plots: Basic plotting utilities and styling
    optimization_viz: Optimization-specific visualizations
    comparison: Algorithm comparison charts and tables
    complexity: Computational complexity visualizations
"""
# Basic plotting utilities
from .plots import (
    setup_plot_style, create_color_palette, plot_multi_series,
    plot_predictions_vs_actual, create_subplot_grid, save_figure, add_value_labels
)
# Optimization visualizations
from .optimization_viz import (
    plot_convergence, plot_optimization_path, plot_contour, plot_residual_analysis,
    plot_gradient_vector, plot_multi_algorithm_convergence
)
# Comparison and analysis
from .comparison import (
    plot_algorithm_comparison, create_comparison_table, plot_radar_chart,
    plot_performance_matrix, plot_convergence_comparison, create_performance_summary,
    plot_efficiency_scatter
)
# Complexity analysis
from .complexity import (
    plot_operation_distribution, plot_scalability_analysis, create_complexity_summary_table,
    plot_complexity_comparison, plot_memory_usage_analysis, plot_big_o_comparison
)
# Interactive visualization tools
from .gd_fixed_stepsize_slider import GDFixedStepSizeSlider
# Backwards compatibility imports (from original visualization_utils.py)
from .plots import setup_plot_style as thiet_lap_style_bieu_do
from .plots import create_color_palette as tao_color_palette
from .optimization_viz import plot_convergence as ve_duong_hoi_tu
from .optimization_viz import plot_multi_algorithm_convergence as ve_duong_hoi_tu_so_sanh
from .optimization_viz import plot_optimization_path as ve_duong_dong_muc_optimization
from .comparison import plot_algorithm_comparison as ve_so_sanh_algorithms
from .plots import plot_predictions_vs_actual as ve_du_doan_vs_thuc_te
from .optimization_viz import plot_residual_analysis as ve_phan_tich_residuals
from .comparison import create_comparison_table as ve_bang_so_sanh_performance
from .comparison import plot_radar_chart as ve_radar_chart_algorithms
from .comparison import plot_performance_matrix as ve_ma_tran_heatmap
from .optimization_viz import plot_gradient_vector as ve_gradient_vector
__all__ = [
    # Basic plotting
    'setup_plot_style', 'create_color_palette', 'plot_multi_series',
    'plot_predictions_vs_actual', 'create_subplot_grid', 'save_figure', 'add_value_labels',
    
    # Optimization visualizations
    'plot_convergence', 'plot_optimization_path', 'plot_contour', 'plot_residual_analysis',
    'plot_gradient_vector', 'plot_multi_algorithm_convergence',
    
    # Comparison and analysis
    'plot_algorithm_comparison', 'create_comparison_table', 'plot_radar_chart',
    'plot_performance_matrix', 'plot_convergence_comparison', 'create_performance_summary',
    'plot_efficiency_scatter',
    
    # Complexity analysis
    'plot_operation_distribution', 'plot_scalability_analysis', 'create_complexity_summary_table',
    'plot_complexity_comparison', 'plot_memory_usage_analysis', 'plot_big_o_comparison',
    # Interactive tools
    'GDFixedStepSizeSlider',
    
    # Backwards compatibility (Vietnamese function names)
    'thiet_lap_style_bieu_do', 'tao_color_palette', 've_duong_hoi_tu', 've_duong_hoi_tu_so_sanh',
    've_duong_dong_muc_optimization', 've_so_sanh_algorithms', 've_du_doan_vs_thuc_te',
    've_phan_tich_residuals', 've_bang_so_sanh_performance', 've_radar_chart_algorithms',
    've_ma_tran_heatmap', 've_gradient_vector',
]