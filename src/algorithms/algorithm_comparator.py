#!/usr/bin/env python3
"""
Algorithm Comparator - Thu thập, phân tích và so sánh kết quả tất cả optimization algorithms

Chức năng chính:
1. Thu thập kết quả từ data/03_algorithms/
2. Phân tích performance metrics
3. Tạo bảng so sánh chi tiết  
4. Tạo visualization so sánh
5. Lưu báo cáo tổng hợp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class AlgorithmComparator:
    """
    Class chính để so sánh và phân tích các optimization algorithms
    """
    
    def __init__(self, data_dir="data/03_algorithms", output_dir="data/04_comparison"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.results_data = []
        self.algorithm_groups = {}
        self.comparison_df = None
        
        # Plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def collect_results(self):
        """Thu thap tat ca ket qua tu cac algorithm folders"""
        print("Thu thap ket qua tu tat ca algorithms...")
        
        if not self.data_dir.exists():
            print(f"Data directory khong ton tai: {self.data_dir}")
            return
        
        algorithm_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for alg_dir in algorithm_dirs:
            alg_name = alg_dir.name
            print(f"   Scanning {alg_name}...")
            
            # Find experiment folders (look for any folder that contains results.json)
            exp_folders = [f for f in alg_dir.iterdir() if f.is_dir()]
            
            for exp_folder in exp_folders:
                results_file = exp_folder / "results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Extract key information
                        result_info = {
                            'algorithm_family': alg_name,
                            'algorithm_name': data.get('algorithm', 'Unknown'),
                            'loss_function': data.get('loss_function', 'Unknown'),
                            'experiment_name': exp_folder.name,
                            'training_time': data.get('training_time', 0),
                            'converged': data.get('convergence', {}).get('converged', False),
                            'iterations': data.get('convergence', {}).get('iterations', 0),
                            'final_loss': data.get('convergence', {}).get('final_loss', float('inf')),
                            'final_gradient_norm': data.get('convergence', {}).get('final_gradient_norm', float('inf')),
                        }
                        
                        # Add algorithm-specific metrics
                        if 'sparsity_analysis' in data:
                            result_info.update({
                                'sparsity_ratio': data['sparsity_analysis'].get('sparsity_ratio', 0),
                                'non_zero_weights': data['sparsity_analysis'].get('non_zero_weights', 0)
                            })
                        
                        if 'numerical_analysis' in data:
                            numerical = data['numerical_analysis']
                            result_info.update({
                                'avg_step_size': numerical.get('average_step_size', 0),
                                'condition_number': numerical.get('final_condition_number', 0),
                                'line_search_iters': numerical.get('average_line_search_iterations', 0)
                            })
                        
                        # Add parameter information
                        params = data.get('parameters', {})
                        for key, value in params.items():
                            result_info[f'param_{key}'] = value
                        
                        self.results_data.append(result_info)
                        print(f"      Found: {exp_folder.name}")
                        
                    except Exception as e:
                        print(f"      Khong doc duoc {results_file}: {e}")
        
        print(f"Thu thap duoc {len(self.results_data)} ket qua experiments")
        
        # Create DataFrame
        if self.results_data:
            self.comparison_df = pd.DataFrame(self.results_data)
            self._group_algorithms()
        else:
            print("Khong tim thay ket qua nao!")
    
    def _group_algorithms(self):
        """Nhóm algorithms theo family"""
        if self.comparison_df is not None:
            for family in self.comparison_df['algorithm_family'].unique():
                self.algorithm_groups[family] = self.comparison_df[
                    self.comparison_df['algorithm_family'] == family
                ].copy()
    
    def create_summary_table(self):
        """Tao bang tom tat so sanh"""
        if self.comparison_df is None:
            print("Khong co du lieu de tao bang tom tat")
            return None
        
        print("Tao bang tom tat so sanh...")
        
        # Create summary with key metrics
        summary_cols = [
            'algorithm_name', 'loss_function', 'training_time', 
            'converged', 'iterations', 'final_loss', 'final_gradient_norm'
        ]
        
        # Add optional columns if they exist
        optional_cols = ['sparsity_ratio', 'avg_step_size', 'condition_number']
        for col in optional_cols:
            if col in self.comparison_df.columns:
                summary_cols.append(col)
        
        summary_df = self.comparison_df[summary_cols].copy()
        
        # Format for better readability
        summary_df['training_time'] = summary_df['training_time'].round(4)
        if 'final_loss' in summary_df.columns:
            summary_df['final_loss'] = summary_df['final_loss'].round(6)
        if 'final_gradient_norm' in summary_df.columns:
            summary_df['final_gradient_norm'] = summary_df['final_gradient_norm'].apply(lambda x: f"{x:.2e}")
        
        # Sort by performance (training time and final loss)
        summary_df = summary_df.sort_values(['training_time', 'final_loss'])
        
        # Save to CSV
        summary_file = self.output_dir / "algorithm_comparison_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Bang tom tat da luu: {summary_file}")
        
        return summary_df
    
    def create_detailed_analysis(self):
        """Tao phan tich chi tiet theo nhom algorithms"""
        if not self.algorithm_groups:
            print("Khong co du lieu nhom algorithms")
            return
        
        print("Tao phan tich chi tiet theo tung nhom...")
        
        detailed_results = {}
        
        for family, group_df in self.algorithm_groups.items():
            print(f"   Phan tich {family}...")
            
            analysis = {
                'family': family,
                'total_experiments': len(group_df),
                'converged_experiments': group_df['converged'].sum(),
                'convergence_rate': group_df['converged'].mean(),
                'avg_training_time': group_df['training_time'].mean(),
                'min_training_time': group_df['training_time'].min(),
                'max_training_time': group_df['training_time'].max(),
                'avg_iterations': group_df['iterations'].mean(),
                'best_final_loss': group_df['final_loss'].min(),
                'worst_final_loss': group_df['final_loss'].max(),
            }
            
            # Add algorithm-specific metrics
            if 'sparsity_ratio' in group_df.columns:
                analysis['avg_sparsity'] = group_df['sparsity_ratio'].mean()
            
            if 'condition_number' in group_df.columns:
                analysis['avg_condition_number'] = group_df['condition_number'].mean()
            
            detailed_results[family] = analysis
        
        # Convert to DataFrame and save
        detailed_df = pd.DataFrame(detailed_results).T
        detailed_file = self.output_dir / "algorithm_detailed_analysis.csv"
        detailed_df.to_csv(detailed_file)
        print(f"Phan tich chi tiet da luu: {detailed_file}")
        
        return detailed_results
    
    def create_performance_plots(self):
        """Tao cac bieu do so sanh performance"""
        if self.comparison_df is None:
            print("Khong co du lieu de ve bieu do")
            return
        
        print("Tao bieu do so sanh performance...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Training Time Comparison
        self._plot_training_time(axes[0, 0])
        
        # 2. Convergence Analysis
        self._plot_convergence_analysis(axes[0, 1])
        
        # 3. Final Loss Comparison
        self._plot_final_loss(axes[0, 2])
        
        # 4. Iterations vs Training Time
        self._plot_iterations_vs_time(axes[1, 0])
        
        # 5. Algorithm Family Performance
        self._plot_family_performance(axes[1, 1])
        
        # 6. Loss Function Comparison
        self._plot_loss_function_performance(axes[1, 2])
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "algorithm_performance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bieu do so sanh da luu: {plot_file}")    
    def _plot_training_time(self, ax):
        """Plot training time comparison"""
        data = self.comparison_df.sort_values('training_time')
        
        ax.barh(range(len(data)), data['training_time'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(data))))
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                           for name in data['algorithm_name']], fontsize=8)
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax):
        """Plot convergence analysis"""
        converged = self.comparison_df['converged'].value_counts()
        
        # Handle case where only one convergence status exists
        if len(converged) == 1:
            if converged.index[0]:  # All converged
                colors = ['#66b3ff']
                labels = ['Converged']
            else:  # None converged
                colors = ['#ff9999']
                labels = ['Not Converged']
            values = [100.0]
        else:
            # Both converged and not converged exist
            colors = ['#ff9999', '#66b3ff']
            labels = [f'Not Converged ({converged.get(False, 0)})', 
                     f'Converged ({converged.get(True, 0)})']
            values = [converged.get(False, 0), converged.get(True, 0)]
        
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Convergence Rate Analysis')
    
    def _plot_final_loss(self, ax):
        """Plot final loss comparison"""
        # Filter out infinite losses for better visualization
        data = self.comparison_df[self.comparison_df['final_loss'] < float('inf')].copy()
        data = data.sort_values('final_loss')
        
        ax.semilogy(range(len(data)), data['final_loss'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Algorithm (ranked by performance)')
        ax.set_ylabel('Final Loss (log scale)')
        ax.set_title('Final Loss Comparison')
        ax.grid(True, alpha=0.3)
    
    def _plot_iterations_vs_time(self, ax):
        """Plot iterations vs training time"""
        ax.scatter(self.comparison_df['iterations'], self.comparison_df['training_time'],
                  c=self.comparison_df['converged'], cmap='RdYlGn', alpha=0.7, s=100)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Iterations vs Training Time')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Converged')
    
    def _plot_family_performance(self, ax):
        """Plot algorithm family performance"""
        family_stats = self.comparison_df.groupby('algorithm_family').agg({
            'training_time': 'mean',
            'final_loss': 'mean',
            'converged': 'mean'
        }).round(4)
        
        x = np.arange(len(family_stats))
        width = 0.25
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - width, family_stats['training_time'], width, 
                      label='Avg Training Time', alpha=0.8, color='skyblue')
        bars2 = ax2.bar(x + width, family_stats['converged'] * 100, width, 
                       label='Convergence Rate (%)', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Algorithm Family')
        ax.set_ylabel('Average Training Time (s)', color='skyblue')
        ax2.set_ylabel('Convergence Rate (%)', color='lightcoral')
        ax.set_title('Algorithm Family Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(family_stats.index, rotation=45)
        
        # Legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _plot_loss_function_performance(self, ax):
        """Plot performance by loss function"""
        loss_stats = self.comparison_df.groupby('loss_function').agg({
            'training_time': 'mean',
            'converged': 'mean'
        }).round(4)
        
        ax.scatter(loss_stats['training_time'], loss_stats['converged'] * 100, 
                  s=[200] * len(loss_stats), alpha=0.7, c=range(len(loss_stats)), cmap='viridis')
        
        for i, txt in enumerate(loss_stats.index):
            ax.annotate(txt, (loss_stats['training_time'].iloc[i], loss_stats['converged'].iloc[i] * 100),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Average Training Time (s)')
        ax.set_ylabel('Convergence Rate (%)')
        ax.set_title('Performance by Loss Function')
        ax.grid(True, alpha=0.3)
    
    def create_convergence_plots(self):
        """Tao bieu do so sanh convergence curves"""
        print("Tao bieu do so sanh convergence curves...")
        
        # Read training histories and plot convergence curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Convergence Curves Comparison', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results_data)))
        
        for i, result in enumerate(self.results_data[:10]):  # Limit to first 10 for readability
            # Try to read training history
            exp_dir = self.data_dir / result['algorithm_family'] / result['experiment_name']
            history_file = exp_dir / "training_history.csv"
            
            if history_file.exists():
                try:
                    history = pd.read_csv(history_file)
                    
                    # Plot loss
                    if 'loss' in history.columns:
                        axes[0, 0].semilogy(history['loss'], color=colors[i], 
                                          label=result['algorithm_name'][:15], alpha=0.8)
                    
                    # Plot gradient norm
                    if 'gradient_norm' in history.columns:
                        axes[0, 1].semilogy(history['gradient_norm'], color=colors[i], 
                                          label=result['algorithm_name'][:15], alpha=0.8)
                    
                    # Plot additional metrics if available
                    if 'sparsity' in history.columns:
                        axes[1, 0].plot(history['sparsity'], color=colors[i], 
                                       label=result['algorithm_name'][:15], alpha=0.8)
                    
                    if 'step_size' in history.columns:
                        axes[1, 1].plot(history['step_size'], color=colors[i], 
                                       label=result['algorithm_name'][:15], alpha=0.8)
                    
                except Exception as e:
                    print(f"      Khong doc duoc history: {history_file}: {e}")
        
        # Set labels and titles
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss (log scale)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        axes[0, 1].set_title('Gradient Norm Convergence')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Gradient Norm (log scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Sparsity Evolution')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Number of Zero Weights')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Step Size Evolution')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Step Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        convergence_file = self.output_dir / "convergence_curves_comparison.png"
        plt.savefig(convergence_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bieu do convergence da luu: {convergence_file}")    
    def generate_report(self):
        """Tao bao cao tong hop HTML"""
        print("Tao bao cao tong hop...")
        
        if self.comparison_df is None:
            print("Khong co du lieu de tao bao cao")
            return
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Algorithms Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .summary {{ background-color: #e7f3ff; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>Optimization Algorithms Comparison Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metric"><strong>Total Experiments:</strong> {len(self.comparison_df)}</div>
        <div class="metric"><strong>Algorithm Families:</strong> {len(self.algorithm_groups)}</div>
        <div class="metric"><strong>Converged Experiments:</strong> {self.comparison_df['converged'].sum()}</div>
        <div class="metric"><strong>Overall Convergence Rate:</strong> {self.comparison_df['converged'].mean()*100:.1f}%</div>
    </div>
    
    <h2>Best Performing Algorithms</h2>
    <p>Based on training time and final loss:</p>
"""
        
        # Add best performers
        best_time = self.comparison_df.nsmallest(3, 'training_time')
        best_loss = self.comparison_df.nsmallest(3, 'final_loss')
        
        html_content += "<h3>Fastest Training Time</h3><ul>"
        for _, row in best_time.iterrows():
            html_content += f"<li><strong>{row['algorithm_name']}</strong>: {row['training_time']:.4f}s</li>"
        html_content += "</ul>"
        
        html_content += "<h3>Best Final Loss</h3><ul>"
        for _, row in best_loss.iterrows():
            html_content += f"<li><strong>{row['algorithm_name']}</strong>: {row['final_loss']:.6f}</li>"
        html_content += "</ul>"
        
        # Add detailed table
        html_content += """
    <h2>Detailed Results</h2>
    <table>
        <thead>
            <tr>
                <th>Algorithm</th>
                <th>Loss Function</th>
                <th>Training Time (s)</th>
                <th>Converged</th>
                <th>Iterations</th>
                <th>Final Loss</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for _, row in self.comparison_df.iterrows():
            converged_icon = "Yes" if row['converged'] else "No"
            html_content += f"""
            <tr>
                <td>{row['algorithm_name']}</td>
                <td>{row['loss_function']}</td>
                <td>{row['training_time']:.4f}</td>
                <td>{converged_icon}</td>
                <td>{row['iterations']}</td>
                <td>{row['final_loss']:.6f}</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
    
    <h2>Performance Visualizations</h2>
    <img src="algorithm_performance_comparison.png" alt="Algorithm Performance Comparison">
    <img src="convergence_curves_comparison.png" alt="Convergence Curves Comparison">
    
    <h2>Notes</h2>
    <ul>
        <li>All results are based on the same dataset split for fair comparison</li>
        <li>Training times may vary depending on system resources</li>
        <li>Convergence criteria may differ between algorithms</li>
        <li>Some algorithms may be better suited for specific types of problems</li>
    </ul>
    
    <p><em>Report generated by Algorithm Comparator - Master Optimization Suite</em></p>
</body>
</html>
"""
        
        # Save HTML report
        report_file = self.output_dir / "algorithm_comparison_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Bao cao HTML da luu: {report_file}")    
    def run_full_comparison(self):
        """Chay toan bo qua trinh so sanh"""
        print("ALGORITHM COMPARATOR - FULL ANALYSIS")
        print("=" * 60)
        
        # Step 1: Collect results
        self.collect_results()
        
        if len(self.results_data) == 0:
            print("Khong tim thay ket qua nao de so sanh!")
            return
        
        # Step 2: Create summary table
        summary_df = self.create_summary_table()
        
        # Step 3: Create detailed analysis
        detailed_analysis = self.create_detailed_analysis()
        
        # Step 4: Create visualizations
        self.create_performance_plots()
        self.create_convergence_plots()
        
        # Step 5: Generate comprehensive report
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("ALGORITHM COMPARISON COMPLETED!")
        print(f"Results saved to: {self.output_dir.absolute()}")
        print("=" * 60)
        
        return {
            'summary': summary_df,
            'detailed_analysis': detailed_analysis,
            'total_experiments': len(self.results_data),
            'algorithm_families': len(self.algorithm_groups)
        }


def main():
    """Main function to run algorithm comparison"""
    comparator = AlgorithmComparator()
    results = comparator.run_full_comparison()
    
    if results:
        print(f"\nSummary Statistics:")
        print(f"   Total Experiments: {results['total_experiments']}")
        print(f"   Algorithm Families: {results['algorithm_families']}")
        print(f"\nCheck the generated HTML report for detailed analysis!")

if __name__ == "__main__":
    main()