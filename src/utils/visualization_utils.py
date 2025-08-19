"""
Tiện ích Trực quan hóa - Các hàm vẽ biểu đồ và visualization

=== MỤC ĐÍCH: TRỰC QUAN HÓA ===

Bao gồm tất cả các hàm cần thiết cho:
1. Vẽ biểu đồ training curves (convergence, loss)
2. So sánh predictions vs actual values
3. Phân tích residuals và errors
4. So sánh performance các algorithms
5. Visualize ma trận và gradient

Code đơn giản, dễ hiểu, dễ sử dụng.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# 1. THIẾT LẬP STYLE VÀ CẤU HÌNH
# ==============================================================================

def thiet_lap_style_bieu_do(style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
    """
    Thiết lập style mặc định cho các biểu đồ
    
    Tham số:
        style: style matplotlib ('seaborn-v0_8', 'ggplot', 'classic')
        figsize: kích thước figure mặc định
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


def tao_color_palette(n_colors: int) -> List[str]:
    """
    Tạo palette màu đẹp cho biểu đồ
    
    Tham số:
        n_colors: số màu cần tạo
    
    Trả về:
        List[str]: danh sách mã màu hex
    """
    if n_colors <= 10:
        return sns.color_palette("husl", n_colors).as_hex()
    else:
        return sns.color_palette("viridis", n_colors).as_hex()


# ==============================================================================
# 2. BIỂU ĐỒ TRAINING CURVES
# ==============================================================================

def ve_duong_hoi_tu(cost_history: List[float], gradient_norms: List[float] = None,
                   title: str = "Quá trình Hội tụ", save_path: str = None):
    """
    Vẽ biểu đồ quá trình hội tụ (cost và gradient norm)
    
    Tham số:
        cost_history: lịch sử cost qua các iterations
        gradient_norms: lịch sử gradient norm (optional)
        title: tiêu đề biểu đồ
        save_path: đường dẫn lưu file (optional)
    """
    if gradient_norms:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Vẽ cost history
    ax1.plot(cost_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost Function')
    ax1.set_title('Quá trình Giảm Cost')
    ax1.grid(True, alpha=0.3)
    
    # Log scale nếu cần
    if len(cost_history) > 1 and cost_history[0] / cost_history[-1] > 100:
        ax1.set_yscale('log')
    
    # Vẽ gradient norm nếu có
    if gradient_norms:
        ax2.plot(gradient_norms, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('||Gradient||')
        ax2.set_title('Chuẩn Gradient')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def ve_so_sanh_algorithms(results_dict: Dict[str, Dict], metric: str = 'cost_history',
                         title: str = "So sánh Algorithms", save_path: str = None):
    """
    So sánh quá trình hội tụ của nhiều algorithms
    
    Tham số:
        results_dict: dictionary {algorithm_name: results}
        metric: metric để so sánh ('cost_history', 'gradient_norms')
        title: tiêu đề biểu đồ
        save_path: đường dẫn lưu file (optional)
    """
    plt.figure(figsize=(12, 8))
    
    colors = tao_color_palette(len(results_dict))
    
    for i, (name, results) in enumerate(results_dict.items()):
        if metric in results and results[metric] is not None:
            data = results[metric]
            plt.plot(data, color=colors[i], linewidth=2, 
                    marker='o', markersize=4, label=name, alpha=0.8)
    
    plt.xlabel('Iteration')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale nếu cần thiết
    if metric == 'gradient_norms':
        plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ==============================================================================
# 3. PHÂN TÍCH DỰ ĐOÁN VÀ SAI SỐ
# ==============================================================================

def ve_du_doan_vs_thuc_te(y_true: np.ndarray, y_pred: np.ndarray,
                         title: str = "Dự đoán vs Thực tế", save_path: str = None):
    """
    Vẽ biểu đồ scatter plot comparing predictions vs actual values
    
    Tham số:
        y_true: giá trị thực tế
        y_pred: giá trị dự đoán
        title: tiêu đề biểu đồ
        save_path: đường dẫn lưu file (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Giá trị Thực tế')
    ax1.set_ylabel('Giá trị Dự đoán')
    ax1.set_title('Predictions vs Actual')
    ax1.grid(True, alpha=0.3)
    
    # Calculate metrics
    from .toi_uu_hoa_utils import tinh_mse, tinh_mae, tinh_r2_score
    mse = tinh_mse(y_true, y_pred)
    mae = tinh_mae(y_true, y_pred)
    r2 = tinh_r2_score(y_true, y_pred)
    
    # Add text box with metrics
    textstr = f'MSE: {mse:.4f}\\nMAE: {mae:.4f}\\nR²: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Residuals plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Giá trị Dự đoán')
    ax2.set_ylabel('Residuals (Thực tế - Dự đoán)')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def ve_phan_tich_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Phân tích Residuals", save_path: str = None):
    """
    Vẽ phân tích chi tiết về residuals
    
    Tham số:
        y_true: giá trị thực tế
        y_pred: giá trị dự đoán
        title: tiêu đề biểu đồ
        save_path: đường dẫn lưu file (optional)
    """
    residuals = y_true - y_pred
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Residuals vs Fitted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.3)
    
    # 3. QQ plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Normal Q-Q Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals vs Order
    ax4.plot(residuals, 'o-', alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Observation Order')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals vs Order')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ==============================================================================
# 4. SO SÁNH PERFORMANCE ALGORITHMS
# ==============================================================================

def ve_bang_so_sanh_performance(results_dict: Dict[str, Dict], 
                               metrics: List[str] = ['final_mse', 'optimization_time', 'iterations'],
                               title: str = "So sánh Performance", save_path: str = None):
    """
    Vẽ bảng so sánh performance của các algorithms
    
    Tham số:
        results_dict: dictionary {algorithm_name: results}
        metrics: danh sách metrics để so sánh
        title: tiêu đề biểu đồ
        save_path: đường dẫn lưu file (optional)
    """
    # Chuẩn bị dữ liệu
    data = []
    for name, results in results_dict.items():
        row = {'Algorithm': name}
        for metric in metrics:
            if metric in results:
                value = results[metric]
                if isinstance(value, (int, float)):
                    row[metric] = value
                else:
                    row[metric] = len(value) if hasattr(value, '__len__') else 0
            else:
                row[metric] = np.nan
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Tạo subplot cho mỗi metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = tao_color_palette(len(results_dict))
    
    for i, metric in enumerate(metrics):
        values = df[metric].values
        names = df['Algorithm'].values
        
        bars = axes[i].bar(names, values, color=colors, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}' if value < 1 else f'{value:.2f}',
                           ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # In bảng số liệu
    print(f"\n{title}")
    print("="*50)
    print(df.to_string(index=False, float_format='%.6f'))


def ve_radar_chart_algorithms(results_dict: Dict[str, Dict],
                             metrics: List[str] = ['accuracy', 'speed', 'stability'],
                             title: str = "Radar Chart Comparison", save_path: str = None):
    """
    Vẽ radar chart so sánh tổng quan các algorithms
    
    Tham số:
        results_dict: dictionary {algorithm_name: results}
        metrics: danh sách metrics để so sánh
        title: tiêu đề biểu đồ
        save_path: đường dẫn lưu file (optional)
    """
    # Chuẩn bị dữ liệu (normalize về 0-1)
    data = {}
    for name, results in results_dict.items():
        data[name] = []
        for metric in metrics:
            if metric in results:
                value = results[metric]
                # Normalize value (ví dụ đơn giản)
                if metric in ['final_mse', 'optimization_time']:
                    # Smaller is better - invert
                    normalized = 1 / (1 + value) if value > 0 else 1
                else:
                    # Assume larger is better
                    normalized = min(value, 1.0) if value <= 1 else 1.0
                data[name].append(normalized)
            else:
                data[name].append(0.5)  # Default value
    
    # Thiết lập radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = tao_color_palette(len(results_dict))
    
    for i, (name, values) in enumerate(data.items()):
        values = values + [values[0]]  # Close the plot
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Customize chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title(title, size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ==============================================================================
# 5. VISUALIZE MA TRẬN VÀ GRADIENT
# ==============================================================================

def ve_ma_tran_heatmap(matrix: np.ndarray, title: str = "Matrix Heatmap",
                      labels: List[str] = None, save_path: str = None):
    """
    Vẽ heatmap cho ma trận (Hessian, correlation matrix, etc.)
    
    Tham số:
        matrix: ma trận cần vẽ
        title: tiêu đề biểu đồ
        labels: nhãn cho axes (optional)
        save_path: đường dẫn lưu file (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Vẽ heatmap
    sns.heatmap(matrix, annot=True, cmap='viridis', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                xticklabels=labels, yticklabels=labels, fmt='.2f')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def ve_gradient_vector(gradient: np.ndarray, title: str = "Gradient Vector",
                      feature_names: List[str] = None, save_path: str = None):
    """
    Vẽ gradient vector dưới dạng bar chart
    
    Tham số:
        gradient: vector gradient
        title: tiêu đề biểu đồ
        feature_names: tên các features (optional)
        save_path: đường dẫn lưu file (optional)
    """
    plt.figure(figsize=(12, 6))
    
    n_features = len(gradient)
    x_pos = np.arange(n_features)
    
    # Tạo màu dựa trên giá trị (positive/negative)
    colors = ['red' if g < 0 else 'blue' for g in gradient]
    
    bars = plt.bar(x_pos, gradient, color=colors, alpha=0.7)
    
    # Customize
    plt.xlabel('Features')
    plt.ylabel('Gradient Value')
    plt.title(title)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    if feature_names:
        plt.xticks(x_pos, feature_names, rotation=45)
    else:
        plt.xticks(x_pos, [f'Feature {i}' for i in range(n_features)])
    
    # Add value labels
    for bar, value in zip(bars, gradient):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ==============================================================================
# 6. TIỆN ÍCH VÀ HELPER FUNCTIONS
# ==============================================================================

def tao_bao_cao_visual_tong_hop(results_dict: Dict[str, Dict], 
                               y_true: np.ndarray = None, 
                               output_dir: str = "visualization_output"):
    """
    Tạo báo cáo visual tổng hợp cho tất cả algorithms
    
    Tham số:
        results_dict: dictionary {algorithm_name: results}
        y_true: giá trị thực tế (cho predictions plots)
        output_dir: thư mục lưu output
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Đang tạo báo cáo visual tổng hợp...")
    
    # 1. So sánh convergence
    ve_so_sanh_algorithms(results_dict, 'cost_history', 
                         "So sánh Quá trình Hội tụ",
                         f"{output_dir}/convergence_comparison.png")
    
    # 2. So sánh performance metrics
    ve_bang_so_sanh_performance(results_dict,
                               title="So sánh Performance Metrics",
                               save_path=f"{output_dir}/performance_comparison.png")
    
    # 3. Vẽ predictions cho từng algorithm (nếu có y_true)
    if y_true is not None:
        for name, results in results_dict.items():
            if 'predictions' in results:
                ve_du_doan_vs_thuc_te(y_true, results['predictions'],
                                     f"Predictions vs Actual - {name}",
                                     f"{output_dir}/predictions_{name.lower()}.png")
    
    print(f"Báo cáo visual đã được lưu trong thư mục: {output_dir}")


def luu_bieu_do_theo_batch(figures_list: List[plt.Figure], 
                          output_dir: str = "batch_plots",
                          prefix: str = "plot"):
    """
    Lưu nhiều biểu đồ theo batch
    
    Tham số:
        figures_list: danh sách các figure objects
        output_dir: thư mục lưu
        prefix: prefix cho tên file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, fig in enumerate(figures_list):
        filename = f"{output_dir}/{prefix}_{i:03d}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close để tiết kiệm memory
    
    print(f"Đã lưu {len(figures_list)} biểu đồ trong {output_dir}")


# Thiết lập style mặc định khi import module
thiet_lap_style_bieu_do()