#!/usr/bin/env python3
"""
Script để chạy và so sánh tất cả optimization methods trên real data
Sử dụng data từ 02.1_sampled consistent với workflow hiện tại
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all optimization methods
from newton_method.pure_newton import newton_standard_setup, newton_robust_setup, newton_fast_setup
from newton_method.damped_newton import damped_newton_standard_setup, damped_newton_robust_setup, damped_newton_fast_setup
from quasi_newton.bfgs import bfgs_standard_setup, bfgs_robust_setup, bfgs_fast_setup
from quasi_newton.lbfgs import lbfgs_standard_setup, lbfgs_memory_efficient_setup, lbfgs_high_memory_setup
from quasi_newton.sr1 import sr1_standard_setup, sr1_robust_setup, sr1_aggressive_setup

from utils.optimization_utils import compute_mse, compute_r2_score, predict


def setup_output_dir():
    """Tạo thư mục output cho comparison results"""
    output_dir = Path("data/03_algorithms/methods_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_sampled_data():
    """Load dữ liệu từ 02.1_sampled"""
    data_dir = Path("data/02.1_sampled")
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Sampled data not found: {data_dir / file}")
    
    print("📂 Loading sampled data...")
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test = pd.read_csv(data_dir / "X_test.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    print(f"✅ Loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run_method_with_timing(method_name, method_func, X_train, y_train, X_test, y_test):
    """
    Run a single optimization method và đo performance
    
    Returns:
        results dict với timing và performance metrics
    """
    print(f"  🔄 Running {method_name}...")
    
    start_time = time.time()
    
    try:
        # Run optimization
        result = method_func(X_train, y_train, verbose=False)
        
        # Extract weights và bias
        weights = result['weights']
        bias = result['bias']
        
        # Compute predictions
        train_predictions = predict(X_train, weights, bias)
        test_predictions = predict(X_test, weights, bias)
        
        # Compute metrics
        train_mse = compute_mse(y_train, train_predictions)
        test_mse = compute_mse(y_test, test_predictions)
        train_r2 = compute_r2_score(y_train, train_predictions)
        test_r2 = compute_r2_score(y_test, test_predictions)
        
        optimization_time = time.time() - start_time
        
        # Extract convergence info
        convergence_info = result.get('convergence_info', {})
        iterations = convergence_info.get('iterations', len(result.get('cost_history', [])))
        converged = convergence_info.get('converged', True)
        
        # Method-specific metrics
        method_metrics = {}
        if 'average_step_size' in result:
            method_metrics['average_step_size'] = result['average_step_size']
        if 'average_line_search_iterations' in result:
            method_metrics['average_line_search_iterations'] = result['average_line_search_iterations']
        if 'curvature_success_rate' in result:
            method_metrics['curvature_success_rate'] = result['curvature_success_rate']
        if 'sr1_success_rate' in result:
            method_metrics['sr1_success_rate'] = result['sr1_success_rate']
        if 'condition_number' in result:
            method_metrics['condition_number'] = result['condition_number']
        if 'hessian_condition_number' in result:
            method_metrics['hessian_condition_number'] = result['hessian_condition_number']
        
        success = True
        error_msg = None
        
    except Exception as e:
        # Handle errors
        optimization_time = time.time() - start_time
        success = False
        error_msg = str(e)
        
        # Default values for failed cases
        train_mse = test_mse = np.inf
        train_r2 = test_r2 = -np.inf
        iterations = 0
        converged = False
        method_metrics = {}
    
    return {
        'method': method_name,
        'success': success,
        'error': error_msg,
        'optimization_time': optimization_time,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'iterations': iterations,
        'converged': converged,
        **method_metrics
    }


def run_comprehensive_comparison(X_train, y_train, X_test, y_test):
    """
    Run comprehensive comparison của tất cả methods và setups
    """
    print("🚀 Starting comprehensive optimization methods comparison...")
    print(f"Dataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples, {X_train.shape[1]} features")
    print()
    
    # Define all methods và setups
    methods = {
        # Newton Methods
        'Newton (Standard)': newton_standard_setup,
        'Newton (Robust)': newton_robust_setup,
        'Newton (Fast)': newton_fast_setup,
        
        # Damped Newton Methods
        'Damped Newton (Standard)': damped_newton_standard_setup,
        'Damped Newton (Robust)': damped_newton_robust_setup,
        'Damped Newton (Fast)': damped_newton_fast_setup,
        
        # BFGS Methods
        'BFGS (Standard)': bfgs_standard_setup,
        'BFGS (Robust)': bfgs_robust_setup,
        'BFGS (Fast)': bfgs_fast_setup,
        
        # L-BFGS Methods
        'L-BFGS (Standard)': lbfgs_standard_setup,
        'L-BFGS (Memory Efficient)': lbfgs_memory_efficient_setup,
        'L-BFGS (High Memory)': lbfgs_high_memory_setup,
        
        # SR1 Methods
        'SR1 (Standard)': sr1_standard_setup,
        'SR1 (Robust)': sr1_robust_setup,
        'SR1 (Aggressive)': sr1_aggressive_setup
    }
    
    # Run all methods
    results = []
    for method_name, method_func in methods.items():
        result = run_method_with_timing(method_name, method_func, X_train, y_train, X_test, y_test)
        results.append(result)
        
        # Print immediate feedback
        if result['success']:
            print(f"    ✅ {method_name}: Test MSE = {result['test_mse']:.6f}, "
                  f"R² = {result['test_r2']:.4f}, Time = {result['optimization_time']:.3f}s, "
                  f"Iterations = {result['iterations']}")
        else:
            print(f"    ❌ {method_name}: FAILED - {result['error']}")
    
    print(f"\n✅ Completed comparison of {len(methods)} methods!")
    return results


def create_comprehensive_report(results, output_dir):
    """
    Tạo comprehensive report với tables và visualizations
    """
    print("\n📊 Creating comprehensive analysis report...")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Filter successful results for analysis
    successful_df = df[df['success']].copy()
    failed_df = df[~df['success']].copy()
    
    print(f"Successful methods: {len(successful_df)}")
    print(f"Failed methods: {len(failed_df)}")
    
    if len(failed_df) > 0:
        print("Failed methods:", failed_df['method'].tolist())
    
    # 1. Save detailed results
    df.to_csv(output_dir / "detailed_results.csv", index=False)
    
    # 2. Create summary table
    if len(successful_df) > 0:
        summary_columns = ['method', 'test_mse', 'test_r2', 'optimization_time', 'iterations', 'converged']
        summary_df = successful_df[summary_columns].copy()
        summary_df = summary_df.sort_values('test_mse')
        summary_df.to_csv(output_dir / "summary_results.csv", index=False)
        
        # 3. Create performance rankings
        rankings = {
            'Best MSE': summary_df.nsmallest(3, 'test_mse')[['method', 'test_mse']],
            'Best R²': summary_df.nlargest(3, 'test_r2')[['method', 'test_r2']],
            'Fastest': summary_df.nsmallest(3, 'optimization_time')[['method', 'optimization_time']],
            'Fewest Iterations': summary_df.nsmallest(3, 'iterations')[['method', 'iterations']]
        }
        
        print("\n🏆 Performance Rankings:")
        for category, ranking in rankings.items():
            print(f"\n{category}:")
            for idx, (_, row) in enumerate(ranking.iterrows(), 1):
                method = row['method']
                value = row[ranking.columns[1]]
                print(f"  {idx}. {method}: {value:.6f}")
    
    # 4. Create visualizations
    create_comparison_plots(successful_df, output_dir)
    
    # 5. Save summary statistics
    if len(successful_df) > 0:
        summary_stats = {
            'total_methods_tested': len(df),
            'successful_methods': len(successful_df),
            'failed_methods': len(failed_df),
            'best_test_mse': successful_df['test_mse'].min(),
            'best_test_r2': successful_df['test_r2'].max(),
            'fastest_time': successful_df['optimization_time'].min(),
            'average_iterations': successful_df['iterations'].mean(),
            'convergence_rate': successful_df['converged'].mean(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_dir / "summary_stats.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)


def create_comparison_plots(successful_df, output_dir):
    """
    Tạo comprehensive comparison plots
    """
    if len(successful_df) == 0:
        print("No successful results to plot")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Test MSE comparison
    plt.subplot(2, 3, 1)
    methods = successful_df['method']
    test_mses = successful_df['test_mse']
    
    bars = plt.bar(range(len(methods)), test_mses)
    plt.title('Test MSE Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Test MSE')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, mse) in enumerate(zip(bars, test_mses)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mse:.6f}', ha='center', va='bottom', fontsize=8)
    
    # 2. R² Score comparison
    plt.subplot(2, 3, 2)
    r2_scores = successful_df['test_r2']
    
    bars = plt.bar(range(len(methods)), r2_scores, color='green', alpha=0.7)
    plt.title('Test R² Score Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Test R² Score')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Runtime comparison
    plt.subplot(2, 3, 3)
    runtimes = successful_df['optimization_time']
    
    bars = plt.bar(range(len(methods)), runtimes, color='orange', alpha=0.7)
    plt.title('Runtime Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, time_val) in enumerate(zip(bars, runtimes)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # 4. Iterations comparison
    plt.subplot(2, 3, 4)
    iterations = successful_df['iterations']
    
    bars = plt.bar(range(len(methods)), iterations, color='red', alpha=0.7)
    plt.title('Iterations to Convergence', fontsize=14, fontweight='bold')
    plt.ylabel('Iterations')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, iter_val) in enumerate(zip(bars, iterations)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(iter_val)}', ha='center', va='bottom', fontsize=8)
    
    # 5. MSE vs Runtime scatter
    plt.subplot(2, 3, 5)
    plt.scatter(successful_df['optimization_time'], successful_df['test_mse'], 
               s=100, alpha=0.7, c=successful_df['iterations'], cmap='viridis')
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Test MSE')
    plt.title('MSE vs Runtime (color = iterations)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar()
    cbar.set_label('Iterations')
    
    # Add method labels
    for i, row in successful_df.iterrows():
        plt.annotate(row['method'], (row['optimization_time'], row['test_mse']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    # 6. Method family comparison (boxplot)
    plt.subplot(2, 3, 6)
    
    # Group methods by family
    method_families = {}
    for _, row in successful_df.iterrows():
        method = row['method']
        if 'Newton' in method and 'Damped' not in method:
            family = 'Newton'
        elif 'Damped Newton' in method:
            family = 'Damped Newton'
        elif 'BFGS' in method and 'L-BFGS' not in method:
            family = 'BFGS'
        elif 'L-BFGS' in method:
            family = 'L-BFGS'
        elif 'SR1' in method:
            family = 'SR1'
        else:
            family = 'Other'
        
        if family not in method_families:
            method_families[family] = []
        method_families[family].append(row['test_mse'])
    
    # Create boxplot
    families = list(method_families.keys())
    mse_data = [method_families[family] for family in families]
    
    bp = plt.boxplot(mse_data, labels=families)
    plt.title('MSE Distribution by Method Family', fontsize=14, fontweight='bold')
    plt.ylabel('Test MSE')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "comprehensive_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Comprehensive comparison plot saved to {plot_path}")


def main():
    """Main function để chạy full comparison"""
    print("=" * 80)
    print("COMPREHENSIVE OPTIMIZATION METHODS COMPARISON")
    print("=" * 80)
    
    try:
        # Setup
        output_dir = setup_output_dir()
        
        # Load data
        X_train, X_test, y_train, y_test = load_sampled_data()
        
        # Run comparison
        results = run_comprehensive_comparison(X_train, y_train, X_test, y_test)
        
        # Create comprehensive report
        create_comprehensive_report(results, output_dir)
        
        print(f"\n✅ COMPREHENSIVE COMPARISON COMPLETED!")
        print(f"📁 All results saved to {output_dir}")
        print("\nFiles created:")
        print("  - detailed_results.csv (all method results)")
        print("  - summary_results.csv (successful methods summary)")
        print("  - summary_stats.json (overall statistics)")
        print("  - comprehensive_comparison.png (visualization)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in comprehensive comparison: {e}")
        raise


if __name__ == "__main__":
    results = main()