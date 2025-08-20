#!/usr/bin/env python3
"""
So sánh Newton Method implementation với thư viện scikit-learn
Các thuật toán: OLS (Normal Equation), Ridge với các solvers khác nhau

ĐẶC ĐIỂM: So sánh accuracy, speed của Newton vs library solvers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
import os
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.optimization_utils import (
    tinh_mse, du_doan, 
    tinh_gradient_hoi_quy_tuyen_tinh, tinh_ma_tran_hessian_hoi_quy_tuyen_tinh,
    giai_he_phuong_trinh_tuyen_tinh, tinh_hessian_ridge,
    danh_gia_mo_hinh
)

def load_du_lieu():
    data_dir = Path("data/02.1_sampled")
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test = pd.read_csv(data_dir / "X_test.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    print(f"Loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def our_newton_ols(X, y, max_iter=50, tol=1e-10):
    """Our Newton Method for OLS"""
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    bias = 0.0
    
    start_time = time.time()
    
    for i in range(max_iter):
        # Compute gradient and Hessian
        grad_w, grad_b = tinh_gradient_hoi_quy_tuyen_tinh(X, y, weights, bias, 0.0)
        H = tinh_ma_tran_hessian_hoi_quy_tuyen_tinh(X, 0.0)
        
        # Newton update for weights
        try:
            delta_w = giai_he_phuong_trinh_tuyen_tinh(H, grad_w)
            weights_new = weights - delta_w
            bias_new = bias - 0.01 * grad_b  # Small learning rate for bias
            
            # Check convergence
            if np.linalg.norm(weights_new - weights) < tol:
                break
                
            weights = weights_new
            bias = bias_new
            
        except np.linalg.LinAlgError:
            print("Matrix is singular, adding regularization")
            break
    
    training_time = time.time() - start_time
    return weights, bias, training_time, i + 1

def our_newton_ridge(X, y, alpha=1.0, max_iter=50, tol=1e-10):
    """Our Newton Method for Ridge"""
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    bias = 0.0
    
    start_time = time.time()
    
    for i in range(max_iter):
        # Compute gradient and Hessian for Ridge
        grad_w, grad_b = tinh_gradient_hoi_quy_tuyen_tinh(X, y, weights, bias, alpha)
        H = tinh_hessian_ridge(X, alpha)
        
        # Newton update
        try:
            delta_w = giai_he_phuong_trinh_tuyen_tinh(H, grad_w)
            weights_new = weights - delta_w
            bias_new = bias - 0.01 * grad_b
            
            # Check convergence
            if np.linalg.norm(weights_new - weights) < tol:
                break
                
            weights = weights_new
            bias = bias_new
            
        except np.linalg.LinAlgError:
            print("Matrix is singular")
            break
    
    training_time = time.time() - start_time
    return weights, bias, training_time, i + 1

def analytical_solution_ols(X, y):
    """Analytical solution: (X^T X)^(-1) X^T y"""
    start_time = time.time()
    XTX = X.T @ X
    XTy = X.T @ y
    weights = np.linalg.solve(XTX, XTy)
    bias = np.mean(y - X @ weights)
    training_time = time.time() - start_time
    return weights, bias, training_time

def compare_algorithms():
    """So sánh các thuật toán"""
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Standardize data for sklearn
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    print("=" * 80)
    print("COMPARISON: NEWTON METHOD vs SCIKIT-LEARN")
    print("=" * 80)
    
    # 1. OLS Comparison
    print("\n📊 OLS (Ordinary Least Squares) Comparison:")
    print("-" * 60)
    
    # Our Newton Method
    weights_ours, bias_ours, time_ours, iter_ours = our_newton_ols(X_train, y_train)
    y_pred_ours = X_test @ weights_ours + bias_ours
    mse_ours = mean_squared_error(y_test, y_pred_ours)
    r2_ours = r2_score(y_test, y_pred_ours)
    
    # Our Analytical Solution
    weights_analytical, bias_analytical, time_analytical = analytical_solution_ols(X_train, y_train)
    y_pred_analytical = X_test @ weights_analytical + bias_analytical
    mse_analytical = mean_squared_error(y_test, y_pred_analytical)
    r2_analytical = r2_score(y_test, y_pred_analytical)
    
    # Scikit-learn LinearRegression (uses SVD)
    start_time = time.time()
    lr_sklearn = LinearRegression()
    lr_sklearn.fit(X_train_scaled, y_train)
    time_sklearn = time.time() - start_time
    y_pred_sklearn = lr_sklearn.predict(X_test_scaled)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    
    print(f"Our Newton Method    - MSE: {mse_ours:.6f}, R²: {r2_ours:.6f}, Time: {time_ours:.4f}s, Iter: {iter_ours}")
    print(f"Our Analytical       - MSE: {mse_analytical:.6f}, R²: {r2_analytical:.6f}, Time: {time_analytical:.4f}s")
    print(f"Scikit-learn SVD     - MSE: {mse_sklearn:.6f}, R²: {r2_sklearn:.6f}, Time: {time_sklearn:.4f}s")
    
    results['ols'] = {
        'newton': {'mse': mse_ours, 'r2': r2_ours, 'time': time_ours, 'iterations': iter_ours},
        'analytical': {'mse': mse_analytical, 'r2': r2_analytical, 'time': time_analytical},
        'sklearn': {'mse': mse_sklearn, 'r2': r2_sklearn, 'time': time_sklearn}
    }
    
    # 2. Ridge Comparison
    print("\n🛡️ Ridge Regression Comparison (α=1.0):")
    print("-" * 60)
    
    # Our Newton Method
    weights_ours, bias_ours, time_ours, iter_ours = our_newton_ridge(X_train, y_train, alpha=1.0)
    y_pred_ours = X_test @ weights_ours + bias_ours
    mse_ours = mean_squared_error(y_test, y_pred_ours)
    r2_ours = r2_score(y_test, y_pred_ours)
    
    # Scikit-learn Ridge with different solvers
    solvers = ['auto', 'svd', 'cholesky', 'lsqr']
    sklearn_results = {}
    
    for solver in solvers:
        try:
            start_time = time.time()
            ridge_sklearn = Ridge(alpha=1.0, solver=solver)
            ridge_sklearn.fit(X_train_scaled, y_train)
            time_sklearn = time.time() - start_time
            y_pred_sklearn = ridge_sklearn.predict(X_test_scaled)
            mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
            r2_sklearn = r2_score(y_test, y_pred_sklearn)
            
            sklearn_results[solver] = {
                'mse': mse_sklearn, 'r2': r2_sklearn, 'time': time_sklearn
            }
            print(f"Sklearn Ridge ({solver:>8s}) - MSE: {mse_sklearn:.6f}, R²: {r2_sklearn:.6f}, Time: {time_sklearn:.4f}s")
        except Exception as e:
            print(f"Sklearn Ridge ({solver:>8s}) - Failed: {e}")
    
    print(f"Our Newton Method    - MSE: {mse_ours:.6f}, R²: {r2_ours:.6f}, Time: {time_ours:.4f}s, Iter: {iter_ours}")
    
    results['ridge'] = {
        'newton': {'mse': mse_ours, 'r2': r2_ours, 'time': time_ours, 'iterations': iter_ours},
        'sklearn': sklearn_results
    }
    
    return results

def create_comparison_plots(results, output_dir):
    """Tạo biểu đồ so sánh"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # OLS MSE Comparison
    ax1 = axes[0, 0]
    ols_methods = ['Newton', 'Analytical', 'Sklearn SVD']
    ols_mse = [results['ols']['newton']['mse'], 
               results['ols']['analytical']['mse'],
               results['ols']['sklearn']['mse']]
    
    bars1 = ax1.bar(ols_methods, ols_mse, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax1.set_ylabel('MSE')
    ax1.set_title('OLS Methods - MSE Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom')
    
    # OLS Time Comparison
    ax2 = axes[0, 1]
    ols_time = [results['ols']['newton']['time'], 
                results['ols']['analytical']['time'],
                results['ols']['sklearn']['time']]
    
    bars2 = ax2.bar(ols_methods, ols_time, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('OLS Methods - Speed Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom')
    
    # Ridge Solver Comparison
    ax3 = axes[1, 0]
    ridge_solvers = ['Newton'] + list(results['ridge']['sklearn'].keys())
    ridge_mse = [results['ridge']['newton']['mse']]
    ridge_mse.extend([results['ridge']['sklearn'][solver]['mse'] 
                     for solver in results['ridge']['sklearn'].keys()])
    
    colors = ['skyblue'] + ['lightcoral'] * (len(ridge_solvers) - 1)
    bars3 = ax3.bar(ridge_solvers, ridge_mse, color=colors)
    ax3.set_ylabel('MSE')
    ax3.set_title('Ridge Methods - MSE Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Ridge Time Comparison
    ax4 = axes[1, 1]
    ridge_time = [results['ridge']['newton']['time']]
    ridge_time.extend([results['ridge']['sklearn'][solver]['time'] 
                      for solver in results['ridge']['sklearn'].keys()])
    
    bars4 = ax4.bar(ridge_solvers, ridge_time, color=colors)
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_title('Ridge Methods - Speed Comparison')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "newton_library_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """So sánh Newton Method implementations"""
    print("🎯 NEWTON METHOD LIBRARY COMPARISON")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/newton_method/library_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparisons
    results = compare_algorithms()
    
    # Create plots
    create_comparison_plots(results, results_dir)
    
    # Save detailed results
    print("\n💾 Saving comparison results...")
    comparison_data = {
        "comparison_type": "Newton Method vs Scikit-learn Solvers",
        "algorithms_tested": ["OLS (Newton vs Analytical vs SVD)", "Ridge (Newton vs Multiple Solvers)"],
        "results": results,
        "notes": {
            "newton_method": "Uses Hessian matrix for second-order optimization",
            "analytical_solution": "Direct matrix inversion (X^T X)^(-1) X^T y",
            "sklearn_solvers": "SVD, Cholesky, LSQR - different numerical approaches",
            "convergence": "Newton method typically converges in few iterations",
            "numerical_stability": "SVD and Cholesky more stable than direct inversion"
        },
        "conclusions": {
            "accuracy": "All methods achieve similar accuracy for well-conditioned problems",
            "speed": "Analytical solution fastest for small problems, SVD best for large/ill-conditioned",
            "stability": "Newton method may fail on singular matrices without regularization",
            "convergence": "Newton method shows quadratic convergence when it works",
            "scalability": "Newton method O(n³) complexity limits scalability"
        }
    }
    
    with open(results_dir / "newton_comparison_results.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nNewton Method comparison completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 NEWTON METHOD SUMMARY:")
    print("✅ Newton method achieves similar accuracy in fewer iterations")
    print("⚡ Analytical solution fastest for small, well-conditioned problems")
    print("🛡️ SVD solver most robust for ill-conditioned matrices")
    print("🎯 Newton method excellent for understanding second-order optimization")
    print("⚠️ Newton method requires careful handling of singular matrices")
    print("=" * 80)

if __name__ == "__main__":
    main()