"""
Comparison Engine for Flask App
Tích hợp algorithm comparator functionality vào web app
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class WebComparisonEngine:
    """
    Engine để compare algorithms cho web interface
    """
    
    def __init__(self, data_dir: str = "data/03_algorithms"):
        self.data_dir = Path(data_dir)
        self.results_cache = {}
        self.last_scan_time = None
        
        # Plotting style cho web
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def collect_all_results(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Thu thập tất cả kết quả experiments
        
        Args:
            force_refresh: Force re-scan thay vì dùng cache
            
        Returns:
            List of experiment results
        """
        # Check cache
        if not force_refresh and self.results_cache and self.last_scan_time:
            return self.results_cache.get('results', [])
        
        print("Scanning for experiment results...")
        results = []
        
        if not self.data_dir.exists():
            return results
        
        # Scan each algorithm directory
        for algorithm_dir in self.data_dir.iterdir():
            if not algorithm_dir.is_dir():
                continue
                
            algorithm_name = algorithm_dir.name
            
            # Scan experiments in algorithm directory
            for experiment_dir in algorithm_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue
                
                result = self._extract_experiment_result(experiment_dir, algorithm_name)
                if result:
                    results.append(result)
        
        # Cache results
        self.results_cache = {'results': results}
        self.last_scan_time = datetime.now()
        
        print(f"Found {len(results)} experiment results across {len(set(r['algorithm_family'] for r in results))} algorithms")
        return results
    
    def _extract_experiment_result(self, experiment_dir: Path, algorithm_name: str) -> Optional[Dict[str, Any]]:
        """Extract result data from experiment directory"""
        # Check for results files
        results_json = experiment_dir / "results.json"
        model_state = experiment_dir / "model_state.json"
        training_history = experiment_dir / "training_history.csv"
        
        if not results_json.exists():
            return None
        
        try:
            # Load basic results
            with open(results_json, 'r') as f:
                data = json.load(f)
            
            result = {
                'algorithm_family': algorithm_name,
                'experiment_name': experiment_dir.name,
                'experiment_path': str(experiment_dir),
                'algorithm_full_name': data.get('algorithm', 'Unknown'),
                'loss_function': data.get('loss_function', 'Unknown'),
                'parameters': data.get('parameters', {}),
                'training_time': data.get('training_time', 0),
                'timestamp': experiment_dir.stat().st_mtime
            }
            
            # Extract convergence info
            convergence = data.get('convergence', {})
            result.update({
                'converged': convergence.get('converged', False),
                'iterations': convergence.get('iterations', 0),
                'final_loss': convergence.get('final_loss', float('inf')),
                'final_gradient_norm': convergence.get('final_gradient_norm', float('inf'))
            })
            
            # Algorithm-specific metrics
            if 'sparsity_analysis' in data:
                sparsity = data['sparsity_analysis']
                result['sparsity_ratio'] = sparsity.get('sparsity_ratio', 0)
                result['active_features'] = sparsity.get('active_features', 0)
            
            if 'numerical_analysis' in data:
                numerical = data['numerical_analysis']
                result['condition_number'] = numerical.get('hessian_condition_number', 1.0)
                result['avg_step_size'] = numerical.get('average_step_size', 0)
            
            # Load enhanced state if available
            if model_state.exists():
                try:
                    with open(model_state, 'r') as f:
                        state = json.load(f)
                    
                    result['prediction_ready'] = state.get('training_completed', False)
                    result['n_features'] = state.get('feature_info', {}).get('n_features')
                    result['has_preprocessing'] = 'preprocessing_info' in state
                    
                except Exception as e:
                    print(f"Warning: Could not load enhanced state for {experiment_dir}: {e}")
            
            # Load training history if available
            if training_history.exists():
                try:
                    history_df = pd.read_csv(training_history)
                    result['history_length'] = len(history_df)
                    if 'loss' in history_df.columns:
                        result['loss_improvement'] = float(history_df['loss'].iloc[0] - history_df['loss'].iloc[-1])
                except Exception as e:
                    print(f"Warning: Could not load training history for {experiment_dir}: {e}")
            
            return result
            
        except Exception as e:
            print(f"Warning: Could not process {experiment_dir}: {e}")
            return None
    
    def create_comparison_dataframe(self, results: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """Create comprehensive comparison DataFrame"""
        if results is None:
            results = self.collect_all_results()
        
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add derived columns
        df['algorithm_category'] = df['algorithm_family'].map(self._get_algorithm_category)
        df['performance_score'] = self._calculate_performance_score(df)
        df['efficiency_score'] = self._calculate_efficiency_score(df)
        df['convergence_rate'] = df.apply(self._calculate_convergence_rate, axis=1)
        df['timestamp_readable'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by performance
        df = df.sort_values(['algorithm_category', 'performance_score'], ascending=[True, False])
        
        return df
    
    def _get_algorithm_category(self, algorithm_family: str) -> str:
        """Categorize algorithms"""
        categories = {
            'gradient_descent': 'First-Order Methods',
            'stochastic_gd': 'Stochastic Methods', 
            'newton_method': 'Second-Order Methods',
            'quasi_newton': 'Quasi-Newton Methods',
            'proximal_gd': 'Proximal Methods',
            'advanced_methods': 'Advanced Methods'
        }
        return categories.get(algorithm_family, 'Other')
    
    def _calculate_performance_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate normalized performance score"""
        # Combine multiple metrics
        scores = pd.Series(index=df.index, dtype=float)
        
        for idx, row in df.iterrows():
            score = 0.0
            
            # Convergence (40%)
            if row['converged']:
                score += 40
            
            # Final loss (30%) - lower is better
            if pd.notna(row['final_loss']) and row['final_loss'] < float('inf'):
                # Normalize final loss (relative to others)
                valid_losses = df[df['final_loss'] < float('inf')]['final_loss']
                if len(valid_losses) > 1:
                    min_loss, max_loss = valid_losses.min(), valid_losses.max()
                    if max_loss > min_loss:
                        normalized_loss = 1 - (row['final_loss'] - min_loss) / (max_loss - min_loss)
                        score += normalized_loss * 30
            
            # Training efficiency (20%) - faster is better  
            if row['training_time'] > 0:
                valid_times = df[df['training_time'] > 0]['training_time']
                if len(valid_times) > 1:
                    min_time, max_time = valid_times.min(), valid_times.max()
                    if max_time > min_time:
                        normalized_time = 1 - (row['training_time'] - min_time) / (max_time - min_time)
                        score += normalized_time * 20
            
            # Iterations efficiency (10%) - fewer iterations is better
            if row['iterations'] > 0:
                valid_iters = df[df['iterations'] > 0]['iterations'] 
                if len(valid_iters) > 1:
                    min_iter, max_iter = valid_iters.min(), valid_iters.max()
                    if max_iter > min_iter:
                        normalized_iter = 1 - (row['iterations'] - min_iter) / (max_iter - min_iter)
                        score += normalized_iter * 10
            
            scores[idx] = score
        
        return scores
    
    def _calculate_efficiency_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate efficiency score (performance / time)"""
        scores = pd.Series(index=df.index, dtype=float)
        
        for idx, row in df.iterrows():
            if row['training_time'] > 0 and pd.notna(row['final_loss']) and row['final_loss'] < float('inf'):
                # Simple efficiency: inverse of (loss * time)
                efficiency = 1 / (row['final_loss'] * row['training_time'])
                scores[idx] = efficiency
            else:
                scores[idx] = 0.0
        
        return scores
    
    def _calculate_convergence_rate(self, row: pd.Series) -> float:
        """Calculate convergence rate"""
        if row['iterations'] > 0 and pd.notna(row.get('loss_improvement')):
            return row['loss_improvement'] / row['iterations']
        return 0.0
    
    def generate_comparison_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate comparison plots for web display
        
        Returns:
            Dict with plot names as keys and base64-encoded images as values
        """
        plots = {}
        
        if df.empty:
            return plots
        
        # 1. Performance Score Comparison
        plots['performance_comparison'] = self._plot_performance_comparison(df)
        
        # 2. Training Time vs Final Loss Scatter  
        plots['efficiency_scatter'] = self._plot_efficiency_scatter(df)
        
        # 3. Algorithm Category Summary
        plots['category_summary'] = self._plot_category_summary(df)
        
        # 4. Convergence Analysis
        plots['convergence_analysis'] = self._plot_convergence_analysis(df)
        
        # 5. Parameter Impact Analysis (if enough data)
        if len(df) > 5:
            plots['parameter_analysis'] = self._plot_parameter_analysis(df)
        
        return plots
    
    def _plot_performance_comparison(self, df: pd.DataFrame) -> str:
        """Plot performance score comparison"""
        plt.figure(figsize=(12, 8))
        
        # Group by algorithm family
        families = df['algorithm_family'].unique()
        colors = sns.color_palette("husl", len(families))
        
        for i, family in enumerate(families):
            family_data = df[df['algorithm_family'] == family]
            plt.scatter(family_data['performance_score'], family_data['training_time'], 
                       label=family, alpha=0.7, s=100, color=colors[i])
        
        plt.xlabel('Performance Score')
        plt.ylabel('Training Time (seconds)')
        plt.title('Algorithm Performance vs Training Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add top performer annotations
        top_performers = df.nlargest(3, 'performance_score')
        for idx, row in top_performers.iterrows():
            plt.annotate(row['experiment_name'][:20], 
                        (row['performance_score'], row['training_time']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        return self._figure_to_base64()
    
    def _plot_efficiency_scatter(self, df: pd.DataFrame) -> str:
        """Plot efficiency scatter"""
        plt.figure(figsize=(10, 6))
        
        valid_data = df[(df['final_loss'] < float('inf')) & (df['training_time'] > 0)]
        
        if len(valid_data) == 0:
            plt.text(0.5, 0.5, 'No valid data for efficiency plot', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        else:
            scatter = plt.scatter(valid_data['final_loss'], valid_data['training_time'], 
                                c=valid_data['performance_score'], cmap='viridis', 
                                s=100, alpha=0.7)
            plt.colorbar(scatter, label='Performance Score')
            
            plt.xlabel('Final Loss')
            plt.ylabel('Training Time (seconds)')
            plt.title('Training Efficiency: Loss vs Time (colored by Performance)')
            plt.yscale('log')
            
            if valid_data['final_loss'].min() > 0:
                plt.xscale('log')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._figure_to_base64()
    
    def _plot_category_summary(self, df: pd.DataFrame) -> str:
        """Plot algorithm category summary"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Count by category
        category_counts = df['algorithm_category'].value_counts()
        axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Experiments by Algorithm Category')
        
        # 2. Average performance by category
        avg_performance = df.groupby('algorithm_category')['performance_score'].mean()
        avg_performance.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average Performance Score by Category')
        axes[0, 1].set_ylabel('Performance Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training time distribution
        df['training_time'].hist(bins=20, ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('Training Time Distribution')
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Convergence rate by category
        convergence_rates = df.groupby('algorithm_category')['converged'].mean() * 100
        convergence_rates.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Convergence Rate by Category')
        axes[1, 1].set_ylabel('Convergence Rate (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return self._figure_to_base64()
    
    def _plot_convergence_analysis(self, df: pd.DataFrame) -> str:
        """Plot convergence analysis"""
        plt.figure(figsize=(12, 6))
        
        # Filter converged experiments
        converged_df = df[df['converged'] == True].copy()
        
        if len(converged_df) == 0:
            plt.text(0.5, 0.5, 'No converged experiments found', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        else:
            # Plot iterations vs final loss for converged experiments
            for category in converged_df['algorithm_category'].unique():
                cat_data = converged_df[converged_df['algorithm_category'] == category]
                plt.scatter(cat_data['iterations'], cat_data['final_loss'], 
                           label=category, alpha=0.7, s=60)
            
            plt.xlabel('Iterations to Convergence')
            plt.ylabel('Final Loss')
            plt.title('Convergence Analysis: Iterations vs Final Loss')
            plt.legend()
            
            if converged_df['final_loss'].min() > 0:
                plt.yscale('log')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._figure_to_base64()
    
    def _plot_parameter_analysis(self, df: pd.DataFrame) -> str:
        """Plot parameter impact analysis"""
        plt.figure(figsize=(12, 8))
        
        # Extract learning rate information
        learning_rates = []
        performance_scores = []
        
        for idx, row in df.iterrows():
            params = row['parameters']
            if 'learning_rate' in params:
                learning_rates.append(float(params['learning_rate']))
                performance_scores.append(row['performance_score'])
        
        if len(learning_rates) > 1:
            plt.subplot(2, 1, 1)
            plt.scatter(learning_rates, performance_scores, alpha=0.7)
            plt.xlabel('Learning Rate')
            plt.ylabel('Performance Score')
            plt.title('Learning Rate Impact on Performance')
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
        
        # Iterations analysis
        valid_iters = df[df['iterations'] > 0]
        if len(valid_iters) > 1:
            plt.subplot(2, 1, 2)
            plt.scatter(valid_iters['iterations'], valid_iters['performance_score'], alpha=0.7)
            plt.xlabel('Iterations')
            plt.ylabel('Performance Score') 
            plt.title('Iterations vs Performance')
            plt.grid(True, alpha=0.3)
        
        if len(learning_rates) <= 1 and len(valid_iters) <= 1:
            plt.text(0.5, 0.5, 'Insufficient data for parameter analysis', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        return self._figure_to_base64()
    
    def _figure_to_base64(self) -> str:
        """Convert current matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
    
    def get_top_performers(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get top N performing experiments"""
        return df.nlargest(n, 'performance_score')
    
    def get_algorithm_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall algorithm performance summary"""
        if df.empty:
            return {}
        
        summary = {
            'total_experiments': len(df),
            'algorithm_families': len(df['algorithm_family'].unique()),
            'convergence_rate': (df['converged'].sum() / len(df)) * 100,
            'avg_training_time': df['training_time'].mean(),
            'best_performance': df['performance_score'].max(),
            'top_algorithm': df.loc[df['performance_score'].idxmax(), 'algorithm_full_name'] if len(df) > 0 else 'N/A',
            'fastest_algorithm': df.loc[df['training_time'].idxmin(), 'algorithm_full_name'] if df['training_time'].min() > 0 else 'N/A'
        }
        
        # Category-wise summary
        category_summary = df.groupby('algorithm_category').agg({
            'performance_score': ['mean', 'max'],
            'training_time': 'mean',
            'converged': lambda x: (x.sum() / len(x)) * 100
        }).round(2)
        
        summary['category_performance'] = category_summary.to_dict()
        
        return summary
    
    def export_comparison_report(self, df: pd.DataFrame, output_file: str = None) -> str:
        """Export comprehensive comparison report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"algorithm_comparison_report_{timestamp}.json"
        
        # Generate comprehensive report
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_experiments': len(df),
                'data_source': str(self.data_dir)
            },
            'summary': self.get_algorithm_summary(df),
            'top_performers': self.get_top_performers(df, 10).to_dict('records'),
            'detailed_results': df.to_dict('records')
        }
        
        # Save report
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(output_path)

# Convenience functions for Flask app
def get_quick_comparison():
    """Quick comparison for dashboard"""
    engine = WebComparisonEngine()
    results = engine.collect_all_results()
    df = engine.create_comparison_dataframe(results)
    return engine.get_algorithm_summary(df)

def get_full_comparison_data():
    """Full comparison data for detailed analysis"""
    engine = WebComparisonEngine()
    results = engine.collect_all_results()
    df = engine.create_comparison_dataframe(results)
    plots = engine.generate_comparison_plots(df)
    summary = engine.get_algorithm_summary(df)
    
    return {
        'dataframe': df,
        'plots': plots,
        'summary': summary,
        'top_performers': engine.get_top_performers(df)
    }