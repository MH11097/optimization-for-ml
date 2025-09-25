"""
Optimization-Specific Visualizations
Specialized plotting functions for optimization algorithms including
convergence plots, optimization paths, and contour visualizations.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings
from scipy import stats
try:
    from .plots import create_color_palette, save_figure
except ImportError:
    from plots import create_color_palette, save_figure
warnings.filterwarnings('ignore')

def plot_convergence(loss_history: List[float], 
                    gradient_norms: Optional[List[float]] = None,
                    iterations: Optional[List[int]] = None, 
                    title: str = "Convergence Analysis",
                    save_path: Optional[str] = None) -> None:
    """
    Plot convergence analysis with loss and gradient norms (both linear and log scales).
    
    Args:
        loss_history: loss values over iterations
        gradient_norms: gradient norm values (optional)
        iterations: iteration numbers (optional, uses indices if None)
        title: plot title
        save_path: path to save the figure
    """
    # Use actual iteration numbers if provided, otherwise use indices
    if iterations is None:
        x_values = list(range(len(loss_history)))
        x_label = 'Iteration (Index)'
    else:
        x_values = iterations[:len(loss_history)]
        x_label = 'Iteration'
    
    # Create subplots based on available data
    if gradient_norms:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    else:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10))
        ax2 = ax4 = None
    
    # Linear Loss plot
    ax1.plot(x_values, loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Linear Gradient Norm plot (if available)
    if gradient_norms and ax2:
        grad_x_values = x_values[:len(gradient_norms)] if len(gradient_norms) < len(x_values) else x_values
        ax2.plot(grad_x_values, gradient_norms, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norm (Linear Scale)')
        ax2.grid(True, alpha=0.3)
    
    # Log Loss plot
    ax3.plot(x_values, loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel('Loss (Log Scale)')
    ax3.set_title('Loss (Log Scale)')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Log Gradient Norm plot (if available)
    if gradient_norms and ax4:
        ax4.plot(grad_x_values, gradient_norms, 'r-', linewidth=2, marker='s', markersize=4)
        ax4.set_xlabel(x_label)
        ax4.set_ylabel('Gradient Norm (Log Scale)')
        ax4.set_title('Gradient Norm (Log Scale)')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_optimization_path(w1_path: np.ndarray, w2_path: np.ndarray,
                          loss_func: callable = None,
                          title: str = "Optimization Path",
                          save_path: Optional[str] = None,
                          grid_size: int = 100,
                          zoom_factor: float = 1.5) -> None:
    """
    Plot 2D optimization path with contour background.
    
    Args:
        w1_path: weight 1 trajectory
        w2_path: weight 2 trajectory  
        loss_func: loss function for contour (optional)
        title: plot title
        save_path: path to save the figure
        grid_size: grid resolution for contours
        zoom_factor: zoom factor around the path
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if loss_func is not None:
        # Create contour background
        w1_min, w1_max = np.min(w1_path), np.max(w1_path)
        w2_min, w2_max = np.min(w2_path), np.max(w2_path)
        
        # Expand range for better visualization
        w1_range = w1_max - w1_min
        w2_range = w2_max - w2_min
        expansion = max(w1_range, w2_range) * (zoom_factor - 1) / 2
        
        w1_grid = np.linspace(w1_min - expansion, w1_max + expansion, grid_size)
        w2_grid = np.linspace(w2_min - expansion, w2_max + expansion, grid_size)
        W1, W2 = np.meshgrid(w1_grid, w2_grid)
        
        # Calculate loss values for contour
        Z = np.zeros_like(W1)
        for i in range(grid_size):
            for j in range(grid_size):
                try:
                    Z[i, j] = loss_func(np.array([W1[i, j], W2[i, j]]))
                except:
                    Z[i, j] = np.nan
        
        # Create contour plot
        levels = np.logspace(np.log10(np.nanmin(Z)), np.log10(np.nanmax(Z)), 20)
        contourf = ax.contourf(W1, W2, Z, levels=levels, cmap='viridis', alpha=0.7)
        ax.contour(W1, W2, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
        cbar.set_label('Loss Value', fontsize=12)
    
    # Plot optimization path
    ax.plot(w1_path, w2_path, 'r-', linewidth=3, alpha=0.9, label='Optimization Path', zorder=5)
    
    # Mark start and end points
    ax.plot(w1_path[0], w2_path[0], 'go', markersize=10, label='Start Point', 
           markeredgecolor='black', zorder=6)
    ax.plot(w1_path[-1], w2_path[-1], 'r*', markersize=15, label='Final Point', 
           markeredgecolor='black', zorder=6)
    
    ax.set_xlabel('Weight 1', fontsize=12)
    ax.set_ylabel('Weight 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_contour(loss_func: callable, 
                w_range: Tuple[Tuple[float, float], Tuple[float, float]],
                optimal_point: Optional[np.ndarray] = None,
                title: str = "Loss Function Contour",
                save_path: Optional[str] = None,
                grid_size: int = 100) -> None:
    """
    Plot contour of a 2D loss function.
    
    Args:
        loss_func: loss function that takes 2D array input
        w_range: ((w1_min, w1_max), (w2_min, w2_max))
        optimal_point: optimal point to mark (optional)
        title: plot title
        save_path: path to save the figure
        grid_size: grid resolution
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    w1_grid = np.linspace(w_range[0][0], w_range[0][1], grid_size)
    w2_grid = np.linspace(w_range[1][0], w_range[1][1], grid_size)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    
    # Calculate loss values
    Z = np.zeros_like(W1)
    for i in range(grid_size):
        for j in range(grid_size):
            try:
                Z[i, j] = loss_func(np.array([W1[i, j], W2[i, j]]))
            except:
                Z[i, j] = np.nan
    
    # Create contour plot
    levels = np.logspace(np.log10(np.nanmin(Z)), np.log10(np.nanmax(Z)), 20)
    contourf = ax.contourf(W1, W2, Z, levels=levels, cmap='viridis')
    ax.contour(W1, W2, Z, levels=levels, colors='black', alpha=0.4, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Loss Value', fontsize=12)
    
    # Mark optimal point if provided
    if optimal_point is not None:
        ax.plot(optimal_point[0], optimal_point[1], 'r*', markersize=15, 
               label='Optimal Point', markeredgecolor='black')
        ax.legend()
    
    ax.set_xlabel('Weight 1', fontsize=12)
    ax.set_ylabel('Weight 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_residual_analysis(residuals: np.ndarray,
                          predictions: np.ndarray,
                          title: str = "Residual Analysis",
                          save_path: Optional[str] = None) -> None:
    """
    Create comprehensive residual analysis plots.
    
    Args:
        residuals: residual values
        predictions: predicted values
        title: plot title  
        save_path: path to save the figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Residuals vs Predictions
    ax1.scatter(predictions, residuals, alpha=0.6, s=30)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.3)
    
    # 3. QQ plot
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Sequential residuals
    ax4.plot(residuals, 'o-', alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Observation Order')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Sequential Residual Plot')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_gradient_vector(gradient: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        title: str = "Gradient Vector Analysis",
                        save_path: Optional[str] = None) -> None:
    """
    Visualize gradient vector as bar chart.
    
    Args:
        gradient: gradient vector
        feature_names: feature names (optional)
        title: plot title
        save_path: path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    n_features = len(gradient)
    x_pos = np.arange(n_features)
    colors = create_color_palette(n_features)
    
    bars = plt.bar(x_pos, gradient, color=colors, alpha=0.7)
    
    plt.xlabel('Features')
    plt.ylabel('Gradient Value')
    plt.title(title)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    if feature_names:
        plt.xticks(x_pos, feature_names, rotation=45)
    else:
        plt.xticks(x_pos, [f'Feature {i}' for i in range(n_features)])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_multi_algorithm_convergence(algorithms_data: Dict[str, Dict[str, Any]],
                                    title: str = "Algorithm Convergence Comparison",
                                    save_path: Optional[str] = None) -> None:
    """
    Plot convergence comparison for multiple algorithms.
    
    Args:
        algorithms_data: dict with algorithm names as keys and data dicts as values
        title: plot title
        save_path: path to save the figure
    """
    if not algorithms_data:
        print("No data to plot")
        return
    
    n_algorithms = len(algorithms_data)
    colors = create_color_palette(n_algorithms)
    
    # Create subplots
    if any('gradient_norms' in data for data in algorithms_data.values()):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    else:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 10))
        ax2 = ax4 = None
    
    # Plot each algorithm
    for i, (name, data) in enumerate(algorithms_data.items()):
        loss_history = data.get('loss_history', [])
        gradient_norms = data.get('gradient_norms', [])
        iterations = data.get('iterations', list(range(len(loss_history))))
        
        if not loss_history:
            continue
        
        x_values = iterations[:len(loss_history)]
        
        # Linear Loss plot
        if loss_history:
            ax1.plot(x_values, loss_history, color=colors[i], linewidth=2, 
                    marker='o', markersize=4, label=name, alpha=0.8)
        
        # Linear Gradient plot
        if gradient_norms and ax2:
            grad_x_values = x_values[:len(gradient_norms)]
            ax2.plot(grad_x_values, gradient_norms, color=colors[i], linewidth=2,
                    marker='s', markersize=4, label=name, alpha=0.8)
        
        # Log Loss plot
        if loss_history:
            ax3.plot(x_values, loss_history, color=colors[i], linewidth=2, 
                    marker='o', markersize=4, label=name, alpha=0.8)
        
        # Log Gradient plot
        if gradient_norms and ax4:
            ax4.plot(grad_x_values, gradient_norms, color=colors[i], linewidth=2,
                    marker='s', markersize=4, label=name, alpha=0.8)
    
    # Configure axes
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if ax2:
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norm (Linear Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss (Log Scale)')
    ax3.set_title('Loss (Log Scale)')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    if ax4:
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Gradient Norm (Log Scale)')
        ax4.set_title('Gradient Norm (Log Scale)')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()