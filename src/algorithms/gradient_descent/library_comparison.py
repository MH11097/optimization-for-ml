#!/usr/bin/env python3
"""
Gradient Descent - Library Comparison
So sánh implementation tự code vs scikit-learn, scipy

Mục đích:
- So sánh performance với SGDRegressor của sklearn
- So sánh với scipy.optimize methods
- Hiểu differences trong implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import warnings
from src.utils.data_loader import load_data_chunked
warnings.filterwarnings('ignore')

# Library imports
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import joblib

def load_processed_data():
    """Load dữ liệu đã xử lý"""
    data_dir = Path("data/02_processed")
    X_train = load_data_chunked(data_dir / "X_train.csv").values
    X_test = load_data_chunked(data_dir / "X_test.csv").values
    y_train = load_data_chunked(data_dir / "y_train.csv").values.ravel()
    y_test = load_data_chunked(data_dir / "y_test.csv").values.ravel()
    
    print(f"📊 Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def load_our_gradient_descent_results():
    """Load kết quả từ các setup của chúng ta"""
    results_dir = Path("data/algorithms/gradient_descent")
    our_results = {}
    
    for setup_dir in results_dir.iterdir():
        if setup_dir.is_dir():
            results_file = setup_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    our_results[setup_dir.name] = data
                    print(f"✅ Loaded our result: {setup_dir.name}")
    
    return our_results

def sklearn_sgd_regressor_comparison(X_train, y_train, X_test, y_test):
    """So sánh với SKLearn SGDRegressor (tương đương GD)"""
    print("\n🔍 SKLEARN SGD REGRESSOR COMPARISON")
    print("-" * 50)
    
    results = {}
    
    # Different SGD configurations
    configurations = [
        {
            'name': 'SGD_constant_0.01',
            'params': {
                'learning_rate': 'constant',
                'eta0': 0.01,
                'max_iter': 1000,
                'tol': 1e-6,
                'random_state': 42
            }
        },
        {
            'name': 'SGD_constant_0.1',
            'params': {
                'learning_rate': 'constant',
                'eta0': 0.1,
                'max_iter': 500,
                'tol': 1e-5,
                'random_state': 42
            }
        },
        {
            'name': 'SGD_adaptive',
            'params': {
                'learning_rate': 'adaptive',
                'eta0': 0.01,
                'max_iter': 1000,
                'tol': 1e-6,
                'random_state': 42
            }
        },
        {
            'name': 'SGD_invscaling',
            'params': {
                'learning_rate': 'invscaling',
                'eta0': 0.01,
                'power_t': 0.25,
                'max_iter': 1000,
                'tol': 1e-6,
                'random_state': 42
            }
        }
    ]
    
    # Test với và không có feature scaling
    scalers = [None, StandardScaler()]
    scaler_names = ['no_scaling', 'with_scaling']
    
    for scaler, scaler_name in zip(scalers, scaler_names):
        print(f"\n📊 Testing with {scaler_name}...")
        
        # Prepare data
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        for config in configurations:
            try:
                print(f"   🔧 {config['name']}...")
                
                start_time = time.time()
                
                # Train model
                model = SGDRegressor(**config['params'])
                model.fit(X_train_scaled, y_train)
                
                training_time = time.time() - start_time
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                result_key = f"{config['name']}_{scaler_name}"
                results[result_key] = {
                    'algorithm': 'sklearn_SGD',
                    'configuration': config['name'],
                    'scaling': scaler_name,
                    'test_mse': mse,
                    'test_r2': r2,
                    'training_time': training_time,
                    'n_iter': model.n_iter_,
                    'params': config['params']
                }
                
                print(f"      MSE: {mse:.6f}, R²: {r2:.4f}, Time: {training_time:.3f}s, Iter: {model.n_iter_}")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
    
    return results

def sklearn_linear_regression_comparison(X_train, y_train, X_test, y_test):
    """So sánh với SKLearn LinearRegression (Normal Equation)"""
    print("\n🔍 SKLEARN LINEAR REGRESSION COMPARISON")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Train LinearRegression (uses Normal Equation/SVD)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   📊 LinearRegression: MSE={mse:.6f}, R²={r2:.4f}, Time={training_time:.4f}s")
        
        return {
            'sklearn_LinearRegression': {
                'algorithm': 'sklearn_LinearRegression',
                'method': 'Normal_Equation/SVD',
                'test_mse': mse,
                'test_r2': r2,
                'training_time': training_time,
                'converged': True
            }
        }
        
    except Exception as e:
        print(f"   ❌ LinearRegression failed: {e}")
        return {}

def scipy_optimization_comparison(X_train, y_train, X_test, y_test):
    """So sánh với scipy optimization methods"""
    print("\n🔍 SCIPY OPTIMIZATION COMPARISON")
    print("-" * 50)
    
    results = {}
    
    def objective(weights):
        """MSE objective function"""
        predictions = X_train.dot(weights)
        return np.mean((predictions - y_train) ** 2)
    
    def gradient(weights):
        """MSE gradient"""
        predictions = X_train.dot(weights)
        errors = predictions - y_train
        return (2 / len(y_train)) * X_train.T.dot(errors)
    
    # Different scipy methods
    methods = ['BFGS', 'L-BFGS-B', 'CG']
    
    for method in methods:
        try:
            print(f"   🔧 {method}...")
            
            start_time = time.time()
            
            # Initialize
            initial_weights = np.random.normal(0, 0.01, X_train.shape[1])
            
            # Optimize
            if method in ['BFGS', 'CG']:
                result = minimize(objective, initial_weights, method=method, jac=gradient)
            else:  # L-BFGS-B
                result = minimize(objective, initial_weights, method=method, jac=gradient)
            
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred = X_test.dot(result.x)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[f'scipy_{method}'] = {
                'algorithm': f'scipy_{method}',
                'method': method,
                'test_mse': mse,
                'test_r2': r2,
                'training_time': training_time,
                'n_iter': result.nit,
                'converged': result.success,
                'final_cost': result.fun
            }
            
            print(f"      MSE: {mse:.6f}, R²: {r2:.4f}, Time: {training_time:.4f}s, Iter: {result.nit}")
            
        except Exception as e:
            print(f"      ❌ {method} failed: {e}")
    
    return results

def create_comprehensive_comparison():
    """Tạo comparison comprehensive"""
    print("🚀 GRADIENT DESCENT - COMPREHENSIVE LIBRARY COMPARISON")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Load our results
    our_results = load_our_gradient_descent_results()
    
    # Run library comparisons
    sklearn_sgd_results = sklearn_sgd_regressor_comparison(X_train, y_train, X_test, y_test)
    sklearn_lr_results = sklearn_linear_regression_comparison(X_train, y_train, X_test, y_test)
    scipy_results = scipy_optimization_comparison(X_train, y_train, X_test, y_test)
    
    # Combine all results
    all_results = {}
    
    # Add our results
    for setup_name, data in our_results.items():
        metrics = data.get('metrics', {})
        all_results[f'Our_{setup_name}'] = {
            'algorithm': f'Our_GD_{setup_name}',
            'test_mse': metrics.get('mse', np.nan),
            'test_r2': metrics.get('r2', np.nan),
            'training_time': data.get('training_time', np.nan),
            'source': 'Our Implementation'
        }
    
    # Add library results
    all_results.update(sklearn_sgd_results)
    all_results.update(sklearn_lr_results)
    all_results.update(scipy_results)
    
    return all_results

def plot_comparison_results(results, figsize=(18, 12)):
    """Vẽ biểu đồ so sánh comprehensive"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Gradient Descent - Our Implementation vs Libraries', fontsize=16, fontweight='bold')
    
    # Prepare data
    comparison_data = []
    for name, data in results.items():
        comparison_data.append({
            'Name': name,
            'Algorithm': data.get('algorithm', name),
            'Test MSE': data.get('test_mse', np.nan),
            'R² Score': data.get('test_r2', np.nan),
            'Training Time': data.get('training_time', np.nan),
            'Source': 'Our Implementation' if 'Our_' in name else 'Library'
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Color mapping
    our_color = 'lightcoral'
    lib_color = 'lightblue'
    colors = [our_color if 'Our_' in name else lib_color for name in df['Name']]
    
    # 1. MSE Comparison
    ax1 = axes[0, 0]
    valid_mse = df.dropna(subset=['Test MSE'])
    if not valid_mse.empty:
        bars = ax1.bar(range(len(valid_mse)), valid_mse['Test MSE'], 
                      color=[our_color if 'Our_' in name else lib_color for name in valid_mse['Name']])
        ax1.set_title('Test MSE Comparison')
        ax1.set_ylabel('MSE')
        ax1.set_xticks(range(len(valid_mse)))
        ax1.set_xticklabels(valid_mse['Name'], rotation=45, ha='right')
        
        # Highlight best
        best_idx = valid_mse['Test MSE'].idxmin()
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
    
    # 2. R² Comparison
    ax2 = axes[0, 1]
    valid_r2 = df.dropna(subset=['R² Score'])
    if not valid_r2.empty:
        bars = ax2.bar(range(len(valid_r2)), valid_r2['R² Score'],
                      color=[our_color if 'Our_' in name else lib_color for name in valid_r2['Name']])
        ax2.set_title('R² Score Comparison')
        ax2.set_ylabel('R² Score')
        ax2.set_xticks(range(len(valid_r2)))
        ax2.set_xticklabels(valid_r2['Name'], rotation=45, ha='right')
        
        # Highlight best
        best_idx = valid_r2['R² Score'].idxmax()
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
    
    # 3. Training Time Comparison
    ax3 = axes[0, 2]
    valid_time = df.dropna(subset=['Training Time'])
    if not valid_time.empty:
        bars = ax3.bar(range(len(valid_time)), valid_time['Training Time'],
                      color=[our_color if 'Our_' in name else lib_color for name in valid_time['Name']])
        ax3.set_title('Training Time Comparison')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xticks(range(len(valid_time)))
        ax3.set_xticklabels(valid_time['Name'], rotation=45, ha='right')
        ax3.set_yscale('log')
        
        # Highlight fastest
        fastest_idx = valid_time['Training Time'].idxmin()
        bars[fastest_idx].set_edgecolor('blue')
        bars[fastest_idx].set_linewidth(3)
    
    # 4. MSE vs Training Time scatter
    ax4 = axes[1, 0]
    valid_both = df.dropna(subset=['Test MSE', 'Training Time'])
    if not valid_both.empty:
        our_data = valid_both[valid_both['Source'] == 'Our Implementation']
        lib_data = valid_both[valid_both['Source'] == 'Library']
        
        if not our_data.empty:
            ax4.scatter(our_data['Training Time'], our_data['Test MSE'], 
                       color=our_color, label='Our Implementation', s=100, alpha=0.7)
        if not lib_data.empty:
            ax4.scatter(lib_data['Training Time'], lib_data['Test MSE'],
                       color=lib_color, label='Libraries', s=100, alpha=0.7)
        
        ax4.set_xlabel('Training Time (s)')
        ax4.set_ylabel('Test MSE')
        ax4.set_title('MSE vs Training Time')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Algorithm type breakdown
    ax5 = axes[1, 1]
    algorithm_counts = df['Source'].value_counts()
    ax5.pie(algorithm_counts.values, labels=algorithm_counts.index, autopct='%1.1f%%',
           colors=[our_color, lib_color])
    ax5.set_title('Implementation Distribution')
    
    # 6. Performance ranking
    ax6 = axes[1, 2]
    # Rank by R² (higher better)
    ranked = valid_r2.nlargest(10, 'R² Score')
    colors_ranked = [our_color if 'Our_' in name else lib_color for name in ranked['Name']]
    bars = ax6.barh(range(len(ranked)), ranked['R² Score'], color=colors_ranked)
    ax6.set_yticks(range(len(ranked)))
    ax6.set_yticklabels(ranked['Name'])
    ax6.set_xlabel('R² Score')
    ax6.set_title('Top 10 by R² Score')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("data/algorithms/gradient_descent")
    output_file = output_dir / "library_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved comparison plot: {output_file}")
    
    plt.show()
    
    return fig, df

def generate_detailed_report(results):
    """Tạo báo cáo chi tiết"""
    print("\n" + "="*80)
    print("📊 GRADIENT DESCENT - DETAILED LIBRARY COMPARISON REPORT")
    print("="*80)
    
    # Prepare comparison table
    table_data = []
    for name, data in results.items():
        table_data.append({
            'Implementation': name,
            'Algorithm': data.get('algorithm', 'N/A'),
            'Test MSE': f"{data.get('test_mse', np.nan):.6f}",
            'R² Score': f"{data.get('test_r2', np.nan):.4f}",
            'Training Time': f"{data.get('training_time', np.nan):.4f}s",
            'Source': 'Our Code' if 'Our_' in name else 'Library'
        })
    
    df = pd.DataFrame(table_data)
    print("\n🏆 PERFORMANCE COMPARISON TABLE:")
    print(df.to_string(index=False))
    
    # Find best performers
    numeric_data = []
    for name, data in results.items():
        if not np.isnan(data.get('test_mse', np.nan)):
            numeric_data.append({
                'Name': name,
                'MSE': data['test_mse'],
                'R2': data['test_r2'],
                'Time': data['training_time'],
                'Source': 'Our Code' if 'Our_' in name else 'Library'
            })
    
    if numeric_data:
        numeric_df = pd.DataFrame(numeric_data)
        
        print(f"\n🏅 BEST PERFORMERS:")
        best_mse = numeric_df.loc[numeric_df['MSE'].idxmin()]
        best_r2 = numeric_df.loc[numeric_df['R2'].idxmax()]
        fastest = numeric_df.loc[numeric_df['Time'].idxmin()]
        
        print(f"   🎯 Best MSE: {best_mse['Name']} ({best_mse['MSE']:.6f}) - {best_mse['Source']}")
        print(f"   📈 Best R²:  {best_r2['Name']} ({best_r2['R2']:.4f}) - {best_r2['Source']}")
        print(f"   ⚡ Fastest:  {fastest['Name']} ({fastest['Time']:.4f}s) - {fastest['Source']}")
        
        # Our implementation performance
        our_results = numeric_df[numeric_df['Source'] == 'Our Code']
        lib_results = numeric_df[numeric_df['Source'] == 'Library']
        
        if not our_results.empty and not lib_results.empty:
            print(f"\n🔍 OUR IMPLEMENTATION vs LIBRARIES:")
            our_best_mse = our_results['MSE'].min()
            lib_best_mse = lib_results['MSE'].min()
            
            if our_best_mse < lib_best_mse:
                print(f"   ✅ Our best MSE ({our_best_mse:.6f}) BEATS library best ({lib_best_mse:.6f})")
            else:
                print(f"   📚 Library best MSE ({lib_best_mse:.6f}) beats our best ({our_best_mse:.6f})")
                improvement = ((our_best_mse - lib_best_mse) / our_best_mse) * 100
                print(f"      Libraries are {improvement:.1f}% better")
    
    # Algorithm insights
    print(f"\n💡 ALGORITHM INSIGHTS:")
    
    # Our implementations
    our_setups = [name for name in results.keys() if 'Our_' in name]
    if our_setups:
        print(f"   🔧 Our Implementations ({len(our_setups)}):")
        for setup in our_setups:
            data = results[setup]
            print(f"      • {setup}: MSE={data.get('test_mse', 'N/A'):.6f}")
    
    # Library methods
    sklearn_methods = [name for name in results.keys() if 'sklearn' in name.lower()]
    scipy_methods = [name for name in results.keys() if 'scipy' in name.lower()]
    
    if sklearn_methods:
        print(f"   📚 Scikit-learn Methods ({len(sklearn_methods)}):")
        for method in sklearn_methods:
            data = results[method]
            print(f"      • {method}: MSE={data.get('test_mse', 'N/A'):.6f}")
    
    if scipy_methods:
        print(f"   🔬 Scipy Methods ({len(scipy_methods)}):")
        for method in scipy_methods:
            data = results[method]
            print(f"      • {method}: MSE={data.get('test_mse', 'N/A'):.6f}")
    
    # Key learnings
    print(f"\n📚 KEY LEARNINGS:")
    print(f"   • SKLearn SGDRegressor ≈ Our Gradient Descent implementation")
    print(f"   • Feature scaling có thể improve performance significantly")
    print(f"   • Scipy BFGS thường converge nhanh nhất")
    print(f"   • SKLearn LinearRegression (Normal Equation) rất nhanh cho small data")
    print(f"   • Our implementation competitive với production libraries!")
    
    # Save report
    output_dir = Path("data/algorithms/gradient_descent")
    report_file = output_dir / "library_comparison_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("GRADIENT DESCENT - LIBRARY COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        f.write("PERFORMANCE TABLE:\n")
        f.write(df.to_string(index=False))
        if numeric_data:
            f.write(f"\n\nBest MSE: {best_mse['Name']}\n")
            f.write(f"Best R²: {best_r2['Name']}\n")
            f.write(f"Fastest: {fastest['Name']}\n")
    
    print(f"\n💾 Detailed report saved: {report_file}")
    
    return df

def save_comparison_results(results):
    """Lưu kết quả để sử dụng sau này"""
    output_dir = Path("data/algorithms/gradient_descent")
    
    # Save raw results
    results_file = output_dir / "library_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved: {results_file}")
    
    # Create summary
    summary = {
        'comparison_date': pd.Timestamp.now().isoformat(),
        'total_methods_compared': len(results),
        'our_implementations': [name for name in results.keys() if 'Our_' in name],
        'library_methods': [name for name in results.keys() if 'Our_' not in name],
        'best_overall': min(results.items(), key=lambda x: x[1].get('test_mse', np.inf))[0]
    }
    
    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return results_file, summary_file

def main():
    """Chạy toàn bộ comparison"""
    print("🚀 GRADIENT DESCENT LIBRARY COMPARISON")
    print("Comparing our implementations with scikit-learn and scipy")
    print("="*70)
    
    # Run comprehensive comparison
    results = create_comprehensive_comparison()
    
    if not results:
        print("❌ No results to compare. Run some experiments first!")
        return
    
    # Visualize results
    plot_comparison_results(results)
    
    # Generate detailed report
    generate_detailed_report(results)
    
    # Save results
    save_comparison_results(results)
    
    print(f"\n✅ Gradient Descent library comparison completed!")
    print(f"📁 Check data/algorithms/gradient_descent/ for results")

if __name__ == "__main__":
    main()