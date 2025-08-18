#!/usr/bin/env python3
"""
Algorithm Comparator - Unified Comparison Tool
Tổng hợp tất cả tính năng comparison từ 4 files cũ:
- unified_comparator.py
- multi_algorithm_comparator.py  
- library_implementations_comparison.py
- setup_comparison_visualizer.py

Chỉ sử dụng kết quả đã lưu trong /data/03_algorithms/ - không rerun algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsLoader:
    """Class để load và parse results từ /data/03_algorithms/"""
    
    def __init__(self, base_dir: str = "data/03_algorithms"):
        self.base_dir = Path(base_dir)
        self.available_results = {}
        
    def scan_available_results(self) -> Dict[str, Dict[str, Any]]:
        """Scan tất cả results có sẵn"""
        print("Scanning available algorithm results...")
        
        if not self.base_dir.exists():
            print(f"Results directory not found: {self.base_dir}")
            print("Run some algorithms first to generate results!")
            return {}
        
        self.available_results = {}
        
        # Scan algorithm directories
        for algo_dir in self.base_dir.iterdir():
            if not algo_dir.is_dir():
                continue
                
            algorithm_name = algo_dir.name
            setups = {}
            
            # Scan setup directories
            for setup_dir in algo_dir.iterdir():
                if not setup_dir.is_dir():
                    continue
                    
                setup_name = setup_dir.name
                results_file = setup_dir / "results.json"
                history_file = setup_dir / "training_history.csv"
                
                if results_file.exists():
                    setups[setup_name] = {
                        'results_path': results_file,
                        'history_path': history_file if history_file.exists() else None,
                        'full_name': f"{algorithm_name}/{setup_name}",
                        'directory': setup_dir
                    }
            
            if setups:
                self.available_results[algorithm_name] = setups
        
        total_setups = sum(len(setups) for setups in self.available_results.values())
        print(f"Found {len(self.available_results)} algorithms with {total_setups} total setups")
        
        return self.available_results
    
    def load_setup_results(self, algorithm: str, setup: str) -> Optional[Dict[str, Any]]:
        """Load results cho 1 algorithm/setup"""
        try:
            if algorithm not in self.available_results:
                return None
            
            if setup not in self.available_results[algorithm]:
                return None
            
            setup_info = self.available_results[algorithm][setup]
            
            # Load main results
            with open(setup_info['results_path'], 'r') as f:
                results = json.load(f)
            
            # Load training history if available
            history = None
            if setup_info['history_path'] and setup_info['history_path'].exists():
                try:
                    history = pd.read_csv(setup_info['history_path'])
                except Exception as e:
                    print(f"Could not load history for {algorithm}/{setup}: {e}")
            
            return {
                'algorithm': algorithm,
                'setup': setup,
                'full_name': setup_info['full_name'],
                'results': results,
                'history': history,
                'directory': setup_info['directory']
            }
            
        except Exception as e:
            print(f"Failed to load {algorithm}/{setup}: {e}")
            return None
    
    def load_multiple(self, selections: List[str]) -> Dict[str, Any]:
        """Load multiple algorithm/setup combinations"""
        loaded_data = {}
        
        for selection in selections:
            if '/' not in selection:
                print(f"Invalid format: {selection}. Use 'algorithm/setup'")
                continue
            
            algorithm, setup = selection.split('/', 1)
            data = self.load_setup_results(algorithm, setup)
            
            if data:
                loaded_data[selection] = data
                print(f"Loaded {selection}")
            else:
                print(f"Failed to load {selection}")
        
        return loaded_data
    
    def list_available(self) -> None:
        """List tất cả results có sẵn"""
        if not self.available_results:
            self.scan_available_results()
        
        print("\n" + "="*70)
        print("AVAILABLE ALGORITHM RESULTS")
        print("="*70)
        
        if not self.available_results:
            print("No results found!")
            print("Run some algorithms first:")
            print("   python src/algorithms/gradient_descent/standard_setup.py")
            return
        
        for algo_name, setups in self.available_results.items():
            print(f"\n{algo_name.upper().replace('_', ' ')}:")
            for setup_name, setup_info in setups.items():
                print(f"   {setup_name} ({setup_info['full_name']})")
        
        total_setups = sum(len(setups) for setups in self.available_results.values())
        print(f"\nTotal: {len(self.available_results)} algorithms, {total_setups} setups")
        
        print(f"\nUsage examples:")
        print(f"   python {__file__} compare gradient_descent/standard newton_method/standard")
        print(f"   python {__file__} analyze gradient_descent")
        print(f"   python {__file__} --interactive")

class PerformanceComparator:
    """Class để so sánh performance metrics"""
    
    def __init__(self, loaded_data: Dict[str, Any]):
        self.data = loaded_data
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Tạo bảng so sánh performance"""
        if not self.data:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, data in self.data.items():
            results = data['results']
            metrics = results.get('metrics', {})
            convergence = results.get('convergence', {})
            params = results.get('parameters', {})
            
            row = {
                'Algorithm/Setup': name,
                'Algorithm': data['algorithm'],
                'Setup': data['setup'],
                'Test_MSE': metrics.get('mse', np.nan),
                'Test_RMSE': metrics.get('rmse', np.nan),
                'Test_MAE': metrics.get('mae', np.nan),
                'R2_Score': metrics.get('r2', np.nan),
                'MAPE_%': metrics.get('mape', np.nan),
                'Training_Time_s': results.get('training_time', np.nan),
                'Iterations': convergence.get('iterations', np.nan),
                'Final_Cost': convergence.get('final_cost', np.nan),
                'Learning_Rate': params.get('learning_rate', np.nan),
                'Max_Iterations': params.get('max_iterations', np.nan)
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def find_best_performers(self, df: pd.DataFrame) -> Dict[str, str]:
        """Tìm best performers theo các metrics khác nhau"""
        if df.empty:
            return {}
        
        best = {}
        
        # Best accuracy (lowest MSE)
        if 'Test_MSE' in df.columns and not df['Test_MSE'].isna().all():
            best_mse_idx = df['Test_MSE'].idxmin()
            best['best_accuracy'] = df.loc[best_mse_idx, 'Algorithm/Setup']
            best['best_mse'] = df.loc[best_mse_idx, 'Test_MSE']
        
        # Best R²
        if 'R2_Score' in df.columns and not df['R2_Score'].isna().all():
            best_r2_idx = df['R2_Score'].idxmax()
            best['best_r2'] = df.loc[best_r2_idx, 'Algorithm/Setup']
            best['best_r2_value'] = df.loc[best_r2_idx, 'R2_Score']
        
        # Fastest training
        if 'Training_Time_s' in df.columns and not df['Training_Time_s'].isna().all():
            fastest_idx = df['Training_Time_s'].idxmin()
            best['fastest'] = df.loc[fastest_idx, 'Algorithm/Setup']
            best['fastest_time'] = df.loc[fastest_idx, 'Training_Time_s']
        
        # Fastest convergence
        if 'Iterations' in df.columns and not df['Iterations'].isna().all():
            fastest_conv_idx = df['Iterations'].idxmin()
            best['fastest_convergence'] = df.loc[fastest_conv_idx, 'Algorithm/Setup']
            best['convergence_iterations'] = df.loc[fastest_conv_idx, 'Iterations']
        
        return best
    
    def print_comparison_summary(self) -> None:
        """In summary comparison"""
        df = self.create_comparison_table()
        
        if df.empty:
            print(" No data to compare")
            return
        
        print("n" + "="*80)
        print(" ALGORITHM PERFORMANCE COMPARISON")
        print("="*80)
        
        # Main comparison table
        display_columns = ['Algorithm/Setup', 'Test_MSE', 'R2_Score', 'Training_Time_s', 'Iterations']
        display_df = df[display_columns].copy()
        
        # Format numbers
        if 'Test_MSE' in display_df.columns:
            display_df['Test_MSE'] = display_df['Test_MSE'].apply(lambda x: f"{x:.6f}" if not pd.isna(x) else "N/A")
        if 'R2_Score' in display_df.columns:
            display_df['R2_Score'] = display_df['R2_Score'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        if 'Training_Time_s' in display_df.columns:
            display_df['Training_Time_s'] = display_df['Training_Time_s'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
        
        print(display_df.to_string(index=False))
        
        # Best performers
        best = self.find_best_performers(df)
        if best:
            print(f"n BEST PERFORMERS:")
            if 'best_accuracy' in best:
                print(f"    Best Accuracy (MSE): {best['best_accuracy']} ({best['best_mse']:.6f})")
            if 'best_r2' in best:
                print(f"    Best R² Score: {best['best_r2']} ({best['best_r2_value']:.4f})")
            if 'fastest' in best:
                print(f"    Fastest Training: {best['fastest']} ({best['fastest_time']:.3f}s)")
            if 'fastest_convergence' in best:
                print(f"    Fastest Convergence: {best['fastest_convergence']} ({best['convergence_iterations']} iterations)")

class VisualizationEngine:
    """Class để tạo visualizations"""
    
    def __init__(self, loaded_data: Dict[str, Any]):
        self.data = loaded_data
        self.comparator = PerformanceComparator(loaded_data)
    
    def plot_performance_comparison(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (16, 12)) -> None:
        """Plot performance comparison charts"""
        if not self.data:
            print(" No data to plot")
            return
        
        df = self.comparator.create_comparison_table()
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        
        # 1. Test MSE
        ax1 = axes[0, 0]
        if not df['Test_MSE'].isna().all():
            bars = ax1.bar(range(len(df)), df['Test_MSE'], color=colors)
            ax1.set_title('Test MSE (Lower = Better)')
            ax1.set_ylabel('MSE')
            ax1.set_xticks(range(len(df)))
            ax1.set_xticklabels([name.split('/')[-1] for name in df['Algorithm/Setup']], rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            
            # Highlight best
            best_idx = df['Test_MSE'].idxmin()
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
        
        # 2. R² Score
        ax2 = axes[0, 1]
        if not df['R2_Score'].isna().all():
            bars = ax2.bar(range(len(df)), df['R2_Score'], color=colors)
            ax2.set_title('R² Score (Higher = Better)')
            ax2.set_ylabel('R²')
            ax2.set_xticks(range(len(df)))
            ax2.set_xticklabels([name.split('/')[-1] for name in df['Algorithm/Setup']], rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            
            # Highlight best
            best_idx = df['R2_Score'].idxmax()
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)
        
        # 3. Training Time
        ax3 = axes[0, 2]
        if not df['Training_Time_s'].isna().all():
            bars = ax3.bar(range(len(df)), df['Training_Time_s'], color=colors)
            ax3.set_title('Training Time (Lower = Better)')
            ax3.set_ylabel('Seconds')
            ax3.set_xticks(range(len(df)))
            ax3.set_xticklabels([name.split('/')[-1] for name in df['Algorithm/Setup']], rotation=45)
            ax3.set_yscale('log')
            ax3.grid(axis='y', alpha=0.3)
            
            # Highlight best
            best_idx = df['Training_Time_s'].idxmin()
            bars[best_idx].set_color('lightgreen')
            bars[best_idx].set_edgecolor('blue')
            bars[best_idx].set_linewidth(3)
        
        # 4. MAPE
        ax4 = axes[1, 0]
        if not df['MAPE_%'].isna().all():
            bars = ax4.bar(range(len(df)), df['MAPE_%'], color=colors)
            ax4.set_title('MAPE % (Lower = Better)')
            ax4.set_ylabel('MAPE %')
            ax4.set_xticks(range(len(df)))
            ax4.set_xticklabels([name.split('/')[-1] for name in df['Algorithm/Setup']], rotation=45)
            ax4.grid(axis='y', alpha=0.3)
        
        # 5. Convergence Speed
        ax5 = axes[1, 1]
        if not df['Iterations'].isna().all():
            bars = ax5.bar(range(len(df)), df['Iterations'], color=colors)
            ax5.set_title('Convergence Iterations (Lower = Better)')
            ax5.set_ylabel('Iterations')
            ax5.set_xticks(range(len(df)))
            ax5.set_xticklabels([name.split('/')[-1] for name in df['Algorithm/Setup']], rotation=45)
            ax5.grid(axis='y', alpha=0.3)
        
        # 6. Speed vs Accuracy scatter
        ax6 = axes[1, 2]
        if not df['Training_Time_s'].isna().all() and not df['R2_Score'].isna().all():
            scatter = ax6.scatter(df['Training_Time_s'], df['R2_Score'], 
                                c=range(len(df)), cmap='viridis', s=100, alpha=0.7)
            ax6.set_xlabel('Training Time (s)')
            ax6.set_ylabel('R² Score')
            ax6.set_title('Speed vs Accuracy Trade-off')
            ax6.set_xscale('log')
            ax6.grid(True, alpha=0.3)
            
            # Add labels
            for i, name in enumerate(df['Algorithm/Setup']):
                ax6.annotate(name.split('/')[-1], 
                           (df['Training_Time_s'].iloc[i], df['R2_Score'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Performance comparison saved: {save_path}")
        
        plt.show()
    
    def plot_convergence_curves(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 8)) -> None:
        """Plot convergence curves comparison"""
        histories = []
        names = []
        
        for name, data in self.data.items():
            if data['history'] is not None and not data['history'].empty:
                histories.append(data['history'])
                names.append(name)
        
        if not histories:
            print(" No training history data available for convergence plots")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Convergence Analysis Comparison', fontsize=16, fontweight='bold')
        
        colors = sns.color_palette("husl", len(histories))
        
        # 1. Cost convergence
        ax1 = axes[0]
        for i, (history, name) in enumerate(zip(histories, names)):
            cost_col = None
            for col in ['cost', 'total_cost', 'loss']:
                if col in history.columns:
                    cost_col = col
                    break
            
            if cost_col:
                ax1.plot(history[cost_col], label=name.split('/')[-1], 
                        color=colors[i], linewidth=2)
        
        ax1.set_title('Cost/Loss Convergence')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Cost')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Gradient norms
        ax2 = axes[1]
        for i, (history, name) in enumerate(zip(histories, names)):
            grad_col = None
            for col in ['gradient_norm', 'grad_norm']:
                if col in history.columns:
                    grad_col = col
                    break
            
            if grad_col:
                ax2.plot(history[grad_col], label=name.split('/')[-1],
                        color=colors[i], linewidth=2)
        
        ax2.set_title('Gradient Norm Decay')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Convergence comparison saved: {save_path}")
        
        plt.show()
    
    def create_radar_chart(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Create radar chart comparison"""
        if not self.data:
            print(" No data for radar chart")
            return
        
        # Prepare normalized data
        radar_data = []
        for name, data in self.data.items():
            results = data['results']
            metrics = results.get('metrics', {})
            
            # Normalize metrics to 0-1 scale
            accuracy = min(metrics.get('r2', 0), 1.0) if metrics.get('r2', 0) >= 0 else 0
            speed = 1 / (1 + results.get('training_time', 1))  # Inverse of time
            efficiency = 1 / (1 + results.get('convergence', {}).get('iterations', 1000))  # Inverse of iterations
            stability = 1 - min(metrics.get('mse', 1), 1) if metrics.get('mse', 1) <= 1 else 0
            
            radar_data.append({
                'Name': name.split('/')[-1],
                'Accuracy': accuracy,
                'Speed': speed,
                'Efficiency': efficiency,
                'Stability': stability
            })
        
        if not radar_data:
            return
        
        # Setup radar chart
        categories = ['Accuracy', 'Speed', 'Efficiency', 'Stability']
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]  # Complete the circle
        
        colors = sns.color_palette("husl", len(radar_data))
        
        # Plot each algorithm
        for i, data_point in enumerate(radar_data):
            values = [data_point[cat] for cat in categories]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=data_point['Name'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.title('Algorithm Characteristics Comparison', size=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Radar chart saved: {save_path}")
        
        plt.show()

class AlgorithmComparator:
    """Main class orchestrating all comparison functionality"""
    
    def __init__(self, base_dir: str = "data/03_algorithms"):
        self.loader = ResultsLoader(base_dir)
        self.loaded_data = {}
        
    def list_available(self) -> None:
        """List all available results"""
        self.loader.scan_available_results()
        self.loader.list_available()
    
    def load_selections(self, selections: List[str]) -> bool:
        """Load selected algorithm/setup combinations"""
        if not self.loader.available_results:
            self.loader.scan_available_results()
        
        self.loaded_data = self.loader.load_multiple(selections)
        return len(self.loaded_data) > 0
    
    def interactive_selection(self) -> bool:
        """Interactive mode for selecting algorithms"""
        if not self.loader.available_results:
            self.loader.scan_available_results()
        
        if not self.loader.available_results:
            print(" No results available for selection")
            return False
        
        print("n INTERACTIVE ALGORITHM SELECTION")
        print("="*50)
        
        # Create options list
        options = []
        for algo_name, setups in self.loader.available_results.items():
            for setup_name in setups.keys():
                options.append(f"{algo_name}/{setup_name}")
        
        print("nAvailable options:")
        for i, option in enumerate(options, 1):
            print(f"{i:2d}. {option}")
        
        # Get user selection
        while True:
            try:
                selection = input(f"nEnter numbers to select (e.g., 1,3,5) or 'q' to quit: ")
                if selection.lower() == 'q':
                    return False
                
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_options = [options[i] for i in indices if 0 <= i < len(options)]
                
                if len(selected_options) == 0:
                    print(" No valid selections")
                    continue
                
                print(f"nSelected: {selected_options}")
                confirm = input("Confirm? (y/n): ")
                if confirm.lower() == 'y':
                    return self.load_selections(selected_options)
                    
            except (ValueError, IndexError):
                print(" Invalid input. Please enter valid numbers separated by commas.")
    
    def analyze_algorithm(self, algorithm_name: str) -> bool:
        """Analyze all setups of a specific algorithm"""
        if not self.loader.available_results:
            self.loader.scan_available_results()
        
        if algorithm_name not in self.loader.available_results:
            print(f" Algorithm '{algorithm_name}' not found")
            return False
        
        setups = self.loader.available_results[algorithm_name]
        selections = [f"{algorithm_name}/{setup}" for setup in setups.keys()]
        
        return self.load_selections(selections)
    
    def compare_performance(self) -> None:
        """Compare performance of loaded algorithms"""
        if not self.loaded_data:
            print(" No data loaded for comparison")
            return
        
        comparator = PerformanceComparator(self.loaded_data)
        comparator.print_comparison_summary()
    
    def create_visualizations(self, output_dir: Optional[str] = None) -> None:
        """Create all visualization plots"""
        if not self.loaded_data:
            print(" No data loaded for visualization")
            return
        
        viz = VisualizationEngine(self.loaded_data)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            perf_path = output_path / "performance_comparison.png"
            conv_path = output_path / "convergence_comparison.png"
            radar_path = output_path / "radar_comparison.png"
        else:
            perf_path = conv_path = radar_path = None
        
        print(" Creating performance comparison...")
        viz.plot_performance_comparison(perf_path)
        
        print(" Creating convergence comparison...")
        viz.plot_convergence_curves(conv_path)
        
        print(" Creating radar chart...")
        viz.create_radar_chart(radar_path)
    
    def generate_report(self, output_file: Optional[str] = None) -> None:
        """Generate comprehensive text report"""
        if not self.loaded_data:
            print(" No data loaded for report")
            return
        
        comparator = PerformanceComparator(self.loaded_data)
        df = comparator.create_comparison_table()
        best = comparator.find_best_performers(df)
        
        report_lines = [
            "="*80,
            "ALGORITHM COMPARISON COMPREHENSIVE REPORT",
            "="*80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total algorithms compared: {len(self.loaded_data)}",
            "",
            "ALGORITHMS INCLUDED:",
            "-" * 20
        ]
        
        for name, data in self.loaded_data.items():
            report_lines.append(f" {name} - {data['results'].get('setup_name', 'Unknown Setup')}")
        
        report_lines.extend([
            "",
            "PERFORMANCE COMPARISON TABLE:",
            "-" * 40,
            df.to_string(index=False),
            "",
            "BEST PERFORMERS:",
            "-" * 20
        ])
        
        if best:
            for key, value in best.items():
                if isinstance(value, str):
                    report_lines.append(f" {key.replace('_', ' ').title()}: {value}")
        
        report_lines.extend([
            "",
            "ALGORITHM INSIGHTS:",
            "-" * 20
        ])
        
        # Add algorithm-specific insights
        algorithms = set(data['algorithm'] for data in self.loaded_data.values())
        for algo in algorithms:
            algo_setups = [name for name, data in self.loaded_data.items() if data['algorithm'] == algo]
            report_lines.append(f" {algo}: {len(algo_setups)} setup(s) - {', '.join([name.split('/')[-1] for name in algo_setups])}")
        
        report_content = 'n'.join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f" Report saved: {output_file}")
        else:
            print("n" + report_content)
    
    def run_complete_analysis(self, output_dir: str = "data/04_comparison") -> None:
        """Run complete analysis with all features"""
        if not self.loaded_data:
            print(" No data loaded for analysis")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(" Running complete algorithm analysis...")
        
        # 1. Performance comparison
        print("1 Performance comparison...")
        self.compare_performance()
        
        # 2. Create visualizations
        print("2 Creating visualizations...")
        self.create_visualizations(output_dir)
        
        # 3. Generate report
        print("3 Generating report...")
        report_file = output_path / "comparison_report.txt"
        self.generate_report(str(report_file))
        
        print(f"n Complete analysis finished! Results in: {output_path}")

def main():
    """Main function với CLI interface"""
    parser = argparse.ArgumentParser(
        description='Algorithm Comparator - Unified comparison tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                              # List available results
  %(prog)s compare gd/standard newton/standard # Compare specific setups
  %(prog)s analyze gradient_descent            # Analyze all setups of one algorithm
  %(prog)s --interactive                       # Interactive selection mode
  %(prog)s report --all                        # Generate report for all available
        """
    )
    
    parser.add_argument('command', nargs='?', choices=['compare', 'analyze', 'report'],
                       help='Command to run')
    parser.add_argument('selections', nargs='*',
                       help='Algorithm/setup selections (e.g., gradient_descent/standard)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available algorithms and setups')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive selection mode')
    parser.add_argument('--all', action='store_true',
                       help='Use all available results')
    parser.add_argument('--output', '-o', default='data/04_comparison',
                       help='Output directory for results (default: data/04_comparison)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = AlgorithmComparator()
    
    # Handle --list
    if args.list:
        comparator.list_available()
        return
    
    # Handle --interactive
    if args.interactive:
        if comparator.interactive_selection():
            comparator.run_complete_analysis(args.output)
        return
    
    # Handle commands
    if args.command == 'compare':
        if not args.selections:
            print(" Please specify algorithm/setup selections to compare")
            return
        
        if comparator.load_selections(args.selections):
            comparator.compare_performance()
            if args.visualize:
                comparator.create_visualizations(args.output)
    
    elif args.command == 'analyze':
        if not args.selections:
            print(" Please specify an algorithm name to analyze")
            return
        
        algorithm_name = args.selections[0]
        if comparator.analyze_algorithm(algorithm_name):
            comparator.run_complete_analysis(args.output)
    
    elif args.command == 'report':
        if args.all:
            # Load all available results
            comparator.loader.scan_available_results()
            all_selections = []
            for algo, setups in comparator.loader.available_results.items():
                for setup in setups.keys():
                    all_selections.append(f"{algo}/{setup}")
            
            if comparator.load_selections(all_selections):
                comparator.run_complete_analysis(args.output)
        elif args.selections:
            if comparator.load_selections(args.selections):
                comparator.generate_report()
        else:
            print(" Please specify selections or use --all")
    
    else:
        # Default: show available options
        comparator.list_available()
        print(f"n Use --help for usage examples")

if __name__ == "__main__":
    main()