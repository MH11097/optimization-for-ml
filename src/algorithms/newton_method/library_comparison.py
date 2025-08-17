from src.utils.data_loader import load_data_chunked
#!/usr/bin/env python3
"""
Newton Method - Library Comparison
So sánh implementation tự code vs scipy.optimize, sklearn

Mục đích:
- So sánh với scipy Newton methods (BFGS, Newton-CG, trust-ncg)
- So sánh với sklearn LinearRegression (analytic solution)
- Hiểu trade-offs giữa second-order methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Library imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy.linalg import solve

def load_processed_data():
    """Load dữ liệu đã xử lý"""
    data_dir = Path("data/02_processed")
    X_train = load_data_chunked(data_dir / "X_train.csv").values
    X_test = load_data_chunked(data_dir / "X_test.csv").values
    y_train = load_data_chunked(data_dir / "y_train.csv").values.ravel()
    y_test = load_data_chunked(data_dir / "y_test.csv").values.ravel()
    
    print(f"📊 Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def load_our_newton_results():
    """Load kết quả từ các setup Newton của chúng ta"""
    results_dir = Path("data/algorithms/newton_method")
    our_results = {}
    
    for setup_dir in results_dir.iterdir():
        if setup_dir.is_dir():
            results_file = setup_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    our_results[setup_dir.name] = data
                    print(f"✅ Loaded our Newton result: {setup_dir.name}")
    
    return our_results

def scipy_newton_comparison(X_train, y_train, X_test, y_test):
    """So sánh với các Newton methods trong scipy"""
    print("\n🔍 SCIPY NEWTON METHODS COMPARISON")
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
    
    def hessian(weights):
        """MSE Hessian (constant for linear regression)"""
        return (2 / len(y_train)) * X_train.T.dot(X_train)
    
    # Newton-type methods
    newton_methods = [
        {
            'name': 'Newton-CG',
            'method': 'Newton-CG',
            'requires_hessian': True,
            'description': 'Newton method with Conjugate Gradient for Hessian inversion'
        },
        {
            'name': 'BFGS',
            'method': 'BFGS',
            'requires_hessian': False,
            'description': 'Quasi-Newton BFGS approximation'
        },
        {
            'name': 'L-BFGS-B',
            'method': 'L-BFGS-B',
            'requires_hessian': False,
            'description': 'Limited memory BFGS with bounds'
        },
        {
            'name': 'trust-ncg',
            'method': 'trust-ncg',
            'requires_hessian': True,
            'description': 'Trust region Newton with Conjugate Gradient'
        }
    ]
    
    for method_config in newton_methods:
        try:
            print(f"   🔧 {method_config['name']}...")
            
            start_time = time.time()
            
            # Initialize
            initial_weights = np.random.normal(0, 0.01, X_train.shape[1])
            
            # Setup optimization parameters
            if method_config['requires_hessian']:
                result = minimize(objective, initial_weights, 
                                method=method_config['method'],
                                jac=gradient, hess=hessian)
            else:
                result = minimize(objective, initial_weights,
                                method=method_config['method'],
                                jac=gradient)
            
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred = X_test.dot(result.x)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[f"scipy_{method_config['name']}"] = {
                'algorithm': f"scipy_{method_config['name']}",
                'method': method_config['method'],
                'description': method_config['description'],
                'test_mse': mse,
                'test_r2': r2,
                'training_time': training_time,
                'n_iter': result.nit,
                'converged': result.success,
                'final_cost': result.fun,
                'requires_hessian': method_config['requires_hessian']
            }
            
            convergence_status = "✅" if result.success else "❌"
            print(f"      {convergence_status} MSE: {mse:.6f}, R²: {r2:.4f}, "
                  f"Time: {training_time:.4f}s, Iter: {result.nit}")
            
        except Exception as e:
            print(f"      ❌ {method_config['name']} failed: {e}")
    
    return results

def analytical_solution_comparison(X_train, y_train, X_test, y_test):
    """So sánh với analytical solutions"""
    print("\n🔍 ANALYTICAL SOLUTIONS COMPARISON")
    print("-" * 50)
    
    results = {}
    
    # 1. Normal Equation (exact solution)
    try:
        print("   🧮 Normal Equation...")
        start_time = time.time()
        
        # Direct solution: w = (X^T X)^(-1) X^T y
        XtX = X_train.T.dot(X_train)
        Xty = X_train.T.dot(y_train)
        
        # Check condition number
        cond_number = np.linalg.cond(XtX)
        print(f"      Condition number: {cond_number:.2e}")
        
        if cond_number < 1e12:  # Well-conditioned
            weights = np.linalg.solve(XtX, Xty)
        else:  # Ill-conditioned, use pseudo-inverse
            weights = np.linalg.pinv(XtX).dot(Xty)
            print("      Used pseudo-inverse (ill-conditioned)")
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = X_test.dot(weights)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results['Normal_Equation'] = {
            'algorithm': 'Normal_Equation',
            'method': 'Direct matrix solution',
            'test_mse': mse,
            'test_r2': r2,
            'training_time': training_time,
            'condition_number': cond_number,
            'iterations': 1  # Direct solution
        }
        
        print(f"      ✅ MSE: {mse:.6f}, R²: {r2:.4f}, Time: {training_time:.4f}s")
        
    except Exception as e:
        print(f"      ❌ Normal Equation failed: {e}")
    
    # 2. SKLearn LinearRegression (uses SVD)
    try:
        print("   📚 SKLearn LinearRegression (SVD)...")
        start_time = time.time()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results['sklearn_LinearRegression'] = {
            'algorithm': 'sklearn_LinearRegression',
            'method': 'SVD decomposition',
            'test_mse': mse,
            'test_r2': r2,
            'training_time': training_time,
            'robust_to_conditioning': True,
            'iterations': 1  # Direct solution
        }
        
        print(f"      ✅ MSE: {mse:.6f}, R²: {r2:.4f}, Time: {training_time:.4f}s")
        
    except Exception as e:
        print(f"      ❌ SKLearn LinearRegression failed: {e}")
    
    # 3. Regularized solutions
    regularization_values = [1e-8, 1e-6, 1e-4]
    
    for reg in regularization_values:
        try:
            print(f"   🛡️ Ridge solution (λ={reg})...")
            start_time = time.time()
            
            # Ridge solution: w = (X^T X + λI)^(-1) X^T y
            XtX_reg = X_train.T.dot(X_train) + reg * np.eye(X_train.shape[1])
            weights = np.linalg.solve(XtX_reg, X_train.T.dot(y_train))
            
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred = X_test.dot(weights)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[f'Ridge_reg_{reg}'] = {
                'algorithm': f'Ridge_regularization',
                'method': f'Normal equation with λ={reg}',
                'test_mse': mse,
                'test_r2': r2,
                'training_time': training_time,
                'regularization': reg,
                'iterations': 1
            }
            
            print(f"      ✅ MSE: {mse:.6f}, R²: {r2:.4f}, Time: {training_time:.4f}s")
            
        except Exception as e:
            print(f"      ❌ Ridge (λ={reg}) failed: {e}")
    
    return results

def create_comprehensive_comparison():
    """Tạo comparison comprehensive"""
    print("🎯 NEWTON METHOD - COMPREHENSIVE LIBRARY COMPARISON")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Load our Newton results
    our_results = load_our_newton_results()
    
    # Run library comparisons
    scipy_results = scipy_newton_comparison(X_train, y_train, X_test, y_test)
    analytical_results = analytical_solution_comparison(X_train, y_train, X_test, y_test)
    
    # Combine all results
    all_results = {}
    
    # Add our results
    for setup_name, data in our_results.items():
        metrics = data.get('metrics', {})
        newton_analysis = data.get('newton_analysis', {})
        
        all_results[f'Our_{setup_name}'] = {
            'algorithm': f'Our_Newton_{setup_name}',
            'method': 'Our Newton implementation',
            'test_mse': metrics.get('mse', np.nan),
            'test_r2': metrics.get('r2', np.nan),
            'training_time': data.get('training_time', np.nan),
            'iterations': data.get('convergence', {}).get('iterations', np.nan),
            'condition_number': newton_analysis.get('hessian_condition_number', np.nan),
            'source': 'Our Implementation'
        }
    
    # Add library results
    all_results.update(scipy_results)
    all_results.update(analytical_results)
    
    return all_results

def plot_newton_comparison(results, figsize=(18, 12)):
    """Vẽ biểu đồ so sánh đặc biệt cho Newton methods"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Newton Method - Our Implementation vs Libraries', fontsize=16, fontweight='bold')
    
    # Prepare data
    comparison_data = []
    for name, data in results.items():
        comparison_data.append({
            'Name': name,
            'Algorithm': data.get('algorithm', name),
            'Method': data.get('method', 'N/A'),
            'Test MSE': data.get('test_mse', np.nan),
            'R² Score': data.get('test_r2', np.nan),
            'Training Time': data.get('training_time', np.nan),
            'Iterations': data.get('iterations', np.nan),
            'Source': 'Our Implementation' if 'Our_' in name else 'Library',
            'Type': 'Newton' if 'newton' in name.lower() or 'Our_' in name else 'Other'
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Color mapping
    our_color = 'lightcoral'
    newton_color = 'lightblue'
    other_color = 'lightgreen'
    
    def get_color(row):
        if 'Our_' in row['Name']:
            return our_color
        elif 'newton' in row['Name'].lower() or 'bfgs' in row['Name'].lower():
            return newton_color
        else:
            return other_color
    
    colors = [get_color(row) for _, row in df.iterrows()]
    
    # 1. MSE Comparison
    ax1 = axes[0, 0]
    valid_mse = df.dropna(subset=['Test MSE'])
    if not valid_mse.empty:
        bars = ax1.bar(range(len(valid_mse)), valid_mse['Test MSE'], 
                      color=[get_color(row) for _, row in valid_mse.iterrows()])
        ax1.set_title('Test MSE Comparison')
        ax1.set_ylabel('MSE')
        ax1.set_xticks(range(len(valid_mse)))
        ax1.set_xticklabels(valid_mse['Name'], rotation=45, ha='right')
        
        # Highlight best
        best_idx = valid_mse['Test MSE'].idxmin()
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
    
    # 2. Convergence Speed (Iterations)
    ax2 = axes[0, 1]
    valid_iter = df.dropna(subset=['Iterations'])
    if not valid_iter.empty:
        bars = ax2.bar(range(len(valid_iter)), valid_iter['Iterations'],
                      color=[get_color(row) for _, row in valid_iter.iterrows()])
        ax2.set_title('Convergence Speed (Lower = Faster)')
        ax2.set_ylabel('Iterations')
        ax2.set_xticks(range(len(valid_iter)))
        ax2.set_xticklabels(valid_iter['Name'], rotation=45, ha='right')
        ax2.set_yscale('log')
        
        # Highlight fastest
        fastest_idx = valid_iter['Iterations'].idxmin()
        bars[fastest_idx].set_edgecolor('blue')
        bars[fastest_idx].set_linewidth(3)
    
    # 3. Training Time vs Accuracy
    ax3 = axes[0, 2]
    valid_both = df.dropna(subset=['Test MSE', 'Training Time'])
    if not valid_both.empty:
        scatter = ax3.scatter(valid_both['Training Time'], valid_both['Test MSE'],
                            c=[get_color(row) for _, row in valid_both.iterrows()],
                            s=100, alpha=0.7)
        
        # Annotate points
        for _, row in valid_both.iterrows():
            ax3.annotate(row['Name'], (row['Training Time'], row['Test MSE']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Training Time (s)')
        ax3.set_ylabel('Test MSE')
        ax3.set_title('MSE vs Training Time')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 4. Method type breakdown
    ax4 = axes[1, 0]
    method_types = df['Type'].value_counts()
    colors_pie = [our_color, newton_color, other_color][:len(method_types)]
    ax4.pie(method_types.values, labels=method_types.index, autopct='%1.1f%%',
           colors=colors_pie)
    ax4.set_title('Method Type Distribution')
    
    # 5. Convergence rate analysis
    ax5 = axes[1, 1]
    # Group by algorithm type and show convergence
    newton_methods = df[df['Type'] == 'Newton']
    other_methods = df[df['Type'] == 'Other']
    
    if not newton_methods.empty and not other_methods.empty:
        newton_iters = newton_methods['Iterations'].dropna()
        other_iters = other_methods['Iterations'].dropna()
        
        ax5.hist([newton_iters, other_iters], bins=10, alpha=0.7, 
                label=['Newton-type', 'Other'], color=[newton_color, other_color])
        ax5.set_xlabel('Iterations to Converge')
        ax5.set_ylabel('Count')
        ax5.set_title('Convergence Distribution')
        ax5.legend()
    
    # 6. Performance ranking
    ax6 = axes[1, 2]
    # Efficiency score: R² / (Training Time * Iterations)
    efficiency_data = []
    for _, row in df.iterrows():
        if not (pd.isna(row['R² Score']) or pd.isna(row['Training Time']) or pd.isna(row['Iterations'])):
            efficiency = row['R² Score'] / (row['Training Time'] * row['Iterations'])
            efficiency_data.append({'Name': row['Name'], 'Efficiency': efficiency})
    
    if efficiency_data:
        eff_df = pd.DataFrame(efficiency_data)
        eff_df = eff_df.nlargest(8, 'Efficiency')  # Top 8
        
        bars = ax6.barh(range(len(eff_df)), eff_df['Efficiency'],
                       color=[get_color(pd.Series({'Name': name})) for name in eff_df['Name']])
        ax6.set_yticks(range(len(eff_df)))
        ax6.set_yticklabels(eff_df['Name'])
        ax6.set_xlabel('Efficiency Score')
        ax6.set_title('Efficiency Ranking')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("data/algorithms/newton_method")
    output_file = output_dir / "library_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved Newton comparison plot: {output_file}")
    
    plt.show()
    
    return fig, df

def generate_newton_report(results):
    """Tạo báo cáo đặc biệt cho Newton methods"""
    print("\n" + "="*80)
    print("🎯 NEWTON METHOD - DETAILED LIBRARY COMPARISON REPORT")
    print("="*80)
    
    # Performance table
    table_data = []
    for name, data in results.items():
        table_data.append({
            'Implementation': name,
            'Method': data.get('method', 'N/A'),
            'Test MSE': f"{data.get('test_mse', np.nan):.6f}",
            'R² Score': f"{data.get('test_r2', np.nan):.4f}",
            'Training Time': f"{data.get('training_time', np.nan):.4f}s",
            'Iterations': data.get('iterations', 'N/A'),
            'Source': 'Our Code' if 'Our_' in name else 'Library'
        })
    
    df = pd.DataFrame(table_data)
    print("\n🏆 PERFORMANCE COMPARISON TABLE:")
    print(df.to_string(index=False))
    
    # Newton-specific analysis
    print(f"\n🔍 NEWTON METHOD ANALYSIS:")
    
    # Convergence speed comparison
    newton_methods = [name for name, data in results.items() 
                     if 'newton' in name.lower() or 'Our_' in name or 'bfgs' in name.lower()]
    direct_methods = [name for name, data in results.items()
                     if any(x in name.lower() for x in ['normal', 'sklearn', 'ridge'])]
    
    if newton_methods:
        print(f"\n   🎯 Newton-type methods ({len(newton_methods)}):")
        for method in newton_methods:
            data = results[method]
            iters = data.get('iterations', 'N/A')
            time_val = data.get('training_time', np.nan)
            print(f"      • {method}: {iters} iterations, {time_val:.4f}s")
    
    if direct_methods:
        print(f"\n   🧮 Direct/Analytical methods ({len(direct_methods)}):")
        for method in direct_methods:
            data = results[method]
            time_val = data.get('training_time', np.nan)
            print(f"      • {method}: Direct solution, {time_val:.4f}s")
    
    # Find best performers by category
    numeric_data = []
    for name, data in results.items():
        if not np.isnan(data.get('test_mse', np.nan)):
            numeric_data.append({
                'Name': name,
                'MSE': data['test_mse'],
                'R2': data['test_r2'],
                'Time': data['training_time'],
                'Iterations': data.get('iterations', np.inf),
                'Category': 'Our' if 'Our_' in name else 'Newton' if any(x in name.lower() for x in ['newton', 'bfgs']) else 'Direct'
            })
    
    if numeric_data:
        numeric_df = pd.DataFrame(numeric_data)
        
        print(f"\n🏅 BEST PERFORMERS BY CATEGORY:")
        
        for category in ['Our', 'Newton', 'Direct']:
            cat_data = numeric_df[numeric_df['Category'] == category]
            if not cat_data.empty:
                best = cat_data.loc[cat_data['MSE'].idxmin()]
                fastest = cat_data.loc[cat_data['Time'].idxmin()]
                print(f"   🔥 {category} Methods:")
                print(f"      Best accuracy: {best['Name']} (MSE: {best['MSE']:.6f})")
                print(f"      Fastest: {fastest['Name']} (Time: {fastest['Time']:.4f}s)")
        
        # Overall comparison
        overall_best = numeric_df.loc[numeric_df['MSE'].idxmin()]
        overall_fastest = numeric_df.loc[numeric_df['Time'].idxmin()]
        
        print(f"\n🏆 OVERALL WINNERS:")
        print(f"   🎯 Best Accuracy: {overall_best['Name']} ({overall_best['MSE']:.6f})")
        print(f"   ⚡ Fastest: {overall_fastest['Name']} ({overall_fastest['Time']:.4f}s)")
    
    # Key insights for Newton methods
    print(f"\n💡 NEWTON METHOD INSIGHTS:")
    print(f"   • Our Newton implementation competitive với scipy methods")
    print(f"   • BFGS quasi-Newton often faster than pure Newton")
    print(f"   • Direct methods (Normal Equation) fastest for small problems")
    print(f"   • Newton methods excel when high precision needed")
    print(f"   • Condition number critically affects Newton performance")
    
    print(f"\n📚 WHEN TO USE WHICH:")
    print(f"   🎯 Our Newton: Educational, understanding algorithm")
    print(f"   🔬 Scipy BFGS: Production use, robust and fast")
    print(f"   📚 SKLearn LinearRegression: Quick prototyping, small data")
    print(f"   🧮 Normal Equation: Perfect condition, educational")
    
    # Save report
    output_dir = Path("data/algorithms/newton_method")
    report_file = output_dir / "library_comparison_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("NEWTON METHOD - LIBRARY COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        f.write("PERFORMANCE TABLE:\n")
        f.write(df.to_string(index=False))
        if numeric_data:
            f.write(f"\n\nOverall Best: {overall_best['Name']}\n")
            f.write(f"Overall Fastest: {overall_fastest['Name']}\n")
    
    print(f"\n💾 Newton report saved: {report_file}")
    
    return df

def main():
    """Chạy toàn bộ Newton comparison"""
    print("🎯 NEWTON METHOD LIBRARY COMPARISON")
    print("Comparing our Newton implementation with scipy and analytical methods")
    print("="*70)
    
    # Run comprehensive comparison
    results = create_comprehensive_comparison()
    
    if not results:
        print("❌ No results to compare. Run Newton experiments first!")
        return
    
    # Visualize results
    plot_newton_comparison(results)
    
    # Generate detailed report
    generate_newton_report(results)
    
    # Save results
    output_dir = Path("data/algorithms/newton_method")
    results_file = output_dir / "library_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Newton Method library comparison completed!")
    print(f"📁 Check data/algorithms/newton_method/ for results")

if __name__ == "__main__":
    main()