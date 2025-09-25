#!/usr/bin/env python3
"""
Simple Step Size Slider - Standalone Interactive Visualization
A self-contained slider to demonstrate the effect of fixed step sizes
on gradient descent convergence behavior. No complex dependencies.
Key Educational Messages:
- Small LR: Slow but steady convergence
- Optimal LR: Fast and stable convergence
- Large LR: Oscillation and divergence
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class SimpleStepSizeSlider:
    """Simple Interactive Step Size Slider for Gradient Descent Visualization"""
    def __init__(self, results_dir: str = None):
        """
        Initialize the step size slider visualization
        Args:
            results_dir: Path to lr_evaluation results directory
        """
        # Setup paths
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent.parent.parent / "lr_evaluation" / "results"
        else:
            self.results_dir = Path(results_dir)
        # Data storage
        self.lr_data = {}
        self.available_lrs = []
        self.current_lr = 0.01
        # UI elements
        self.fig = None
        self.axes = {}
        self.slider = None
        # Styling
        self.setup_styling()
        # Load data
        self.load_lr_results()
    def setup_styling(self):
        """Setup beautiful matplotlib styling"""
        plt.style.use('default')
        # Modern color palette
        self.colors = {
            'good': '#2E8B57',      # Sea Green
            'slow': '#DAA520',      # Goldenrod
            'bad': '#DC143C',       # Crimson
            'neutral': '#4682B4',   # Steel Blue
            'background': '#F8F9FA', # Light Gray
            'text': '#2C3E50'       # Dark Blue Gray
        }
        # Font settings
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    def load_lr_results(self):
        """Load learning rate evaluation results from JSON files"""
        print("Loading lr_evaluation results...")
        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            # Create dummy data for demo
            self.create_dummy_data()
            return
        # Find the latest results file
        json_files = list(self.results_dir.glob("*.json"))
        if not json_files:
            print("No JSON results files found, creating dummy data...")
            self.create_dummy_data()
            return
        latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        print(f"Loading: {latest_file}")
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            # Process both successful and failed results
            all_results = []
            if 'successful' in results:
                all_results.extend(results['successful'])
            if 'failed' in results:
                all_results.extend(results['failed'])
            # Extract learning rate data
            for result in all_results:
                lr = result.get('learning_rate')
                if lr is not None:
                    self.lr_data[lr] = result
            self.available_lrs = sorted(self.lr_data.keys())
            print(f"Loaded {len(self.available_lrs)} learning rate results")
            # Set default LR to a good middle value
            if self.available_lrs:
                mid_idx = len(self.available_lrs) // 2
                self.current_lr = self.available_lrs[mid_idx]
        except Exception as e:
            print(f"Error loading results: {e}")
            self.create_dummy_data()
    def create_dummy_data(self):
        """Create dummy data for demonstration purposes"""
        print("Creating dummy data for demonstration...")
        # Learning rates to simulate
        lrs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        for lr in lrs:
            # Simulate different convergence behaviors
            if lr < 0.005:
                # Too slow
                loss_history = self.simulate_slow_convergence(lr)
                status = 'successful'
                mse = 0.01 + lr * 10
                r2 = 0.9 - lr * 50
            elif lr <= 0.1:
                # Good convergence
                loss_history = self.simulate_good_convergence(lr)
                status = 'successful'
                mse = 0.005
                r2 = 0.95
            else:
                # Divergence
                loss_history = self.simulate_divergence(lr)
                status = 'failed'
                mse = float('inf')
                r2 = -float('inf')
            self.lr_data[lr] = {
                'learning_rate': lr,
                'status': status,
                'loss_history': loss_history,
                'metrics': {
                    'mse': mse,
                    'r2_score': r2,
                    'training_time': 10.0 / lr  # Slower for smaller LR
                }
            }
        self.available_lrs = sorted(self.lr_data.keys())
        self.current_lr = 0.05  # Good default
    def simulate_slow_convergence(self, lr):
        """Simulate slow but steady convergence"""
        iterations = 500
        loss_history = []
        loss = 100.0
        for i in range(iterations):
            loss = loss * (1 - lr * 0.1)  # Slow decay
            loss_history.append(loss)
            if loss < 0.01:
                break
        return loss_history
    def simulate_good_convergence(self, lr):
        """Simulate good convergence"""
        iterations = 200
        loss_history = []
        loss = 100.0
        for i in range(iterations):
            loss = loss * (1 - lr * 0.5)  # Good decay rate
            loss_history.append(loss)
            if loss < 0.001:
                break
        return loss_history
    def simulate_divergence(self, lr):
        """Simulate divergence behavior"""
        iterations = 50
        loss_history = []
        loss = 100.0
        for i in range(iterations):
            # Oscillating and growing
            oscillation = 1 + 0.1 * np.sin(i * 0.5)
            loss = loss * (1 + lr * 0.3) * oscillation
            loss_history.append(loss)
            if loss > 1e10:  # Cap at very large number
                loss_history.extend([float('inf')] * (iterations - i - 1))
                break
        return loss_history
    def get_lr_zone_color(self, lr: float) -> str:
        """Determine color zone for learning rate"""
        if lr < 0.005:
            return self.colors['slow']   # Too slow
        elif lr <= 0.1:
            return self.colors['good']   # Good range
        else:
            return self.colors['bad']    # Too large, likely diverges
    def get_convergence_status(self, lr: float) -> Tuple[str, str]:
        """Get convergence status and description for learning rate"""
        if lr not in self.lr_data:
            return "No Data", "neutral"
        result = self.lr_data[lr]
        # Check if result failed
        if result.get('status') == 'failed':
            return "Diverged", "bad"
        # For successful results, check convergence quality
        metrics = result.get('metrics', {})
        if not metrics:
            return "Unknown", "neutral"
        mse = metrics.get('mse', float('inf'))
        r2 = metrics.get('r2_score', -float('inf'))
        if mse == float('inf') or mse > 1e10:
            return "Diverged", "bad"
        elif r2 > 0.8:
            return "Converged Well", "good"
        elif r2 > 0.5:
            return "Converged Slowly", "slow"
        else:
            return "Poor Convergence", "bad"
    def create_interactive_visualization(self):
        """Create the main interactive visualization"""
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Gradient Descent: Fixed Step Size Effect Demonstration',
                         fontsize=16, fontweight='bold', y=0.95)
        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, height_ratios=[2, 2, 0.5], width_ratios=[2, 1, 1],
                                  hspace=0.3, wspace=0.3)
        # Panel 1: Loss Evolution (main chart)
        self.axes['loss'] = self.fig.add_subplot(gs[0:2, 0])
        # Panel 2: Convergence Status
        self.axes['status'] = self.fig.add_subplot(gs[0, 1])
        # Panel 3: Step Size Zones
        self.axes['zones'] = self.fig.add_subplot(gs[1, 1])
        # Panel 4: Learning Rate Info
        self.axes['info'] = self.fig.add_subplot(gs[0:2, 2])
        # Slider area
        slider_ax = self.fig.add_subplot(gs[2, :])
        # Setup slider
        self.setup_slider(slider_ax)
        # Initial plot
        self.update_visualization()
        return self.fig
    def setup_slider(self, ax):
        """Setup the learning rate slider"""
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if not self.available_lrs:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return
        # Create slider
        slider_ax = plt.axes([0.15, 0.02, 0.7, 0.03], facecolor='lightgray')
        min_lr = min(self.available_lrs)
        max_lr = max(self.available_lrs)
        self.slider = widgets.Slider(
            slider_ax, 'Learning Rate',
            min_lr, max_lr,
            valinit=self.current_lr,
            valfmt='%.3f'
        )
        # Connect slider to update function
        self.slider.on_changed(self.on_slider_change)
        # Add zone labels
        ax.text(0.1, 0.7, 'Too Slow', ha='center', color=self.colors['slow'], fontweight='bold')
        ax.text(0.5, 0.7, 'Optimal Zone', ha='center', color=self.colors['good'], fontweight='bold')
        ax.text(0.9, 0.7, 'Diverges', ha='center', color=self.colors['bad'], fontweight='bold')
    def on_slider_change(self, val):
        """Handle slider value changes"""
        # Find closest available learning rate
        closest_lr = min(self.available_lrs, key=lambda x: abs(x - val))
        self.current_lr = closest_lr
        self.update_visualization()
    def update_visualization(self):
        """Update all visualization panels"""
        if not self.available_lrs:
            return
        # Update loss evolution plot
        self.update_loss_plot()
        # Update convergence status
        self.update_status_panel()
        # Update step size zones
        self.update_zones_panel()
        # Update info panel
        self.update_info_panel()
        # Refresh the display
        self.fig.canvas.draw()
    def update_loss_plot(self):
        """Update the main loss evolution plot"""
        ax = self.axes['loss']
        ax.clear()
        if self.current_lr not in self.lr_data:
            ax.text(0.5, 0.5, f'No data for LR = {self.current_lr}',
                   ha='center', va='center', transform=ax.transAxes)
            return
        result = self.lr_data[self.current_lr]
        color = self.get_lr_zone_color(self.current_lr)
        # Get loss history
        loss_history = result.get('loss_history', [])
        if loss_history and len(loss_history) > 0:
            # Plot loss evolution
            iterations = range(len(loss_history))
            # Handle infinite values for plotting
            plot_losses = []
            for loss in loss_history:
                if np.isfinite(loss):
                    plot_losses.append(loss)
                else:
                    plot_losses.append(1e10)  # Cap for visualization
            ax.plot(iterations, plot_losses, color=color, linewidth=2, alpha=0.8)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss Value')
            ax.set_title(f'Loss Evolution (LR = {self.current_lr})')
            ax.grid(True, alpha=0.3)
            # Set y-scale based on data range
            valid_losses = [l for l in plot_losses if l > 0]
            if valid_losses and max(valid_losses) / min(valid_losses) > 100:
                ax.set_yscale('log')
        else:
            # Show status for failed cases
            status, _ = self.get_convergence_status(self.current_lr)
            ax.text(0.5, 0.5, f'Status: {status}\n(LR = {self.current_lr})',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color=color)
            ax.set_title('Loss Evolution')
    def update_status_panel(self):
        """Update convergence status panel"""
        ax = self.axes['status']
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        status, zone = self.get_convergence_status(self.current_lr)
        color = self.colors.get(zone, self.colors['neutral'])
        # Large status indicator
        circle = plt.Circle((0.5, 0.7), 0.2, color=color, alpha=0.7)
        ax.add_patch(circle)
        # Status text
        ax.text(0.5, 0.7, '✓' if 'Converged' in status else '✗',
               ha='center', va='center', fontsize=20, color='white', fontweight='bold')
        ax.text(0.5, 0.4, status, ha='center', va='center',
               fontsize=11, fontweight='bold', color=color)
        ax.text(0.5, 0.2, f'LR = {self.current_lr}', ha='center', va='center',
               fontsize=10, color=self.colors['text'])
        ax.set_title('Convergence Status', fontweight='bold')
    def update_zones_panel(self):
        """Update step size zones visualization"""
        ax = self.axes['zones']
        ax.clear()
        # Create zone visualization
        zones = [
            (0, 0.005, self.colors['slow'], 'Too Slow'),
            (0.005, 0.1, self.colors['good'], 'Optimal'),
            (0.1, 0.5, self.colors['bad'], 'Too Large')
        ]
        y_pos = 0.5
        bar_height = 0.3
        for start, end, color, label in zones:
            width = end - start
            rect = plt.Rectangle((start, y_pos - bar_height/2), width, bar_height,
                               facecolor=color, alpha=0.6, edgecolor='black')
            ax.add_patch(rect)
            # Add zone label
            ax.text(start + width/2, y_pos + 0.3, label, ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
        # Mark current position
        ax.axvline(x=self.current_lr, color='red', linewidth=3, alpha=0.8)
        ax.plot(self.current_lr, y_pos, 'ro', markersize=8, markeredgecolor='darkred')
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Learning Rate')
        ax.set_title('Step Size Zones', fontweight='bold')
        ax.grid(True, alpha=0.3)
    def update_info_panel(self):
        """Update information panel"""
        ax = self.axes['info']
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        # Get result data
        if self.current_lr in self.lr_data:
            result = self.lr_data[self.current_lr]
            info_text = f"Learning Rate: {self.current_lr}\n\n"
            if result.get('status') == 'failed':
                info_text += "Status: Failed\n"
                info_text += "Reason: Likely divergence\n"
            else:
                metrics = result.get('metrics', {})
                if metrics:
                    mse = metrics.get('mse', 'N/A')
                    r2 = metrics.get('r2_score', 'N/A')
                    time = metrics.get('training_time', 'N/A')
                    if isinstance(mse, (int, float)) and np.isfinite(mse):
                        info_text += f"MSE: {mse:.2e}\n"
                    else:
                        info_text += "MSE: Diverged\n"
                    if isinstance(r2, (int, float)) and np.isfinite(r2):
                        info_text += f"R² Score: {r2:.4f}\n"
                    else:
                        info_text += "R² Score: N/A\n"
                    if isinstance(time, (int, float)):
                        info_text += f"Training Time: {time:.2f}s\n"
            # Add educational note
            info_text += "\n" + "="*20 + "\n"
            if self.current_lr < 0.005:
                info_text += "📚 Learning:\nStep size too small\n→ Slow convergence\n→ Many iterations needed"
            elif self.current_lr <= 0.1:
                info_text += "📚 Learning:\nGood step size\n→ Fast convergence\n→ Stable training"
            else:
                info_text += "📚 Learning:\nStep size too large\n→ Overshooting\n→ Divergence risk"
        else:
            info_text = f"Learning Rate: {self.current_lr}\n\nNo data available"
        ax.text(0.05, 0.95, info_text, ha='left', va='top', fontsize=9,
               transform=ax.transAxes, family='monospace')
        ax.set_title('Information', fontweight='bold')
    def show(self):
        """Display the interactive visualization"""
        self.create_interactive_visualization()
        plt.show()
    def save_current_view(self, filename: str = "step_size_demo.png"):
        """Save current visualization to file"""
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {filename}")

def main():
    """Demo function"""
    print("Initializing Simple Gradient Descent Step Size Demonstration...")
    # Create and show the visualization
    slider_demo = SimpleStepSizeSlider()
    slider_demo.show()

if __name__ == "__main__":
    main()