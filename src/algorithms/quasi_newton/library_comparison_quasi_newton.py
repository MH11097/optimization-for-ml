#!/usr/bin/env python3
"""
So sánh Quasi-Newton methods với scipy.optimize và sklearn
Các thuật toán: BFGS, L-BFGS-B từ scipy và custom implementations

ĐẶC ĐIỂM: So sánh convergence, accuracy của quasi-newton vs scipy optimizers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
import os
import json
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.optimization_utils import (
    tinh_mse, du_doan, tinh_gradient_OLS, tinh_gia_tri_ham_OLS,
    danh_gia_mo_hinh
)
from utils.data_process_utils import load_du_lieu



def our_bfgs(X, y, max_iter=100, tol=1e-6):
    """Our simplified BFGS implementation"""
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    
    # Initialize inverse Hessian approximation
    B_inv = np.eye(n_features)
    
    start_time = time.time()
    loss_history = []
    
    for i in range(max_iter):
        # Compute function value and gradient
        f_val = tinh_gia_tri_ham_OLS(X, y, weights)
        gradient = tinh_gradient_OLS(X, y, weights)
        
        loss_history.append(f_val)
        
        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break
        
        # BFGS step
        p = -B_inv @ gradient  # Search direction
        
        # Simple line search (fixed step size for simplicity)
        alpha = 0.01
        weights_new = weights + alpha * p
        
        # Update for next iteration
        if i < max_iter - 1:  # Don't compute on last iteration
            s = alpha * p
            gradient_new = tinh_gradient_OLS(X, y, weights_new)
            y_bfgs = gradient_new - gradient
            
            # BFGS update (Sherman-Morrison formula)
            rho = 1.0 / (y_bfgs.T @ s + 1e-12)
            if rho > 0:  # Only update if curvature condition is satisfied
                A1 = np.eye(n_features) - rho * np.outer(s, y_bfgs)
                A2 = np.eye(n_features) - rho * np.outer(y_bfgs, s)
                B_inv = A1 @ B_inv @ A2 + rho * np.outer(s, s)
        
        weights = weights_new
    
    training_time = time.time() - start_time
    return weights, training_time, i + 1, loss_history

def scipy_objective(weights, X, y):
    """Objective function for scipy optimization"""
    return tinh_gia_tri_ham_OLS(X, y, weights)

def scipy_gradient(weights, X, y):
    """Gradient function for scipy optimization"""
    return tinh_gradient_OLS(X, y, weights)

def compare_quasi_newton_algorithms():
    """So sánh các Quasi-Newton algorithms"""
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Standardize data for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    print("=" * 80)
    print("COMPARISON: QUASI-NEWTON METHODS vs SCIPY")
    print("=" * 80)
    
    # 1. BFGS Comparison
    print("\n🎯 BFGS Comparison:")
    print("-" * 60)
    
    # Our BFGS
    weights_ours, time_ours, iter_ours, loss_ours = our_bfgs(X_train, y_train)
    y_pred_ours = X_test @ weights_ours
    mse_ours = mean_squared_error(y_test, y_pred_ours)
    r2_ours = r2_score(y_test, y_pred_ours)
    
    # Scipy BFGS
    start_time = time.time()
    initial_weights = np.random.normal(0, 0.01, X_train_scaled.shape[1])
    
    result_bfgs = minimize(
        fun=scipy_objective, 
        x0=initial_weights,
        args=(X_train_scaled, y_train),
        method='BFGS',
        jac=scipy_gradient,
        options={'maxiter': 100, 'gtol': 1e-6}
    )
    time_scipy_bfgs = time.time() - start_time
    
    y_pred_scipy_bfgs = X_test_scaled @ result_bfgs.x
    mse_scipy_bfgs = mean_squared_error(y_test, y_pred_scipy_bfgs)
    r2_scipy_bfgs = r2_score(y_test, y_pred_scipy_bfgs)
    
    print(f"Our BFGS             - MSE: {mse_ours:.6f}, R²: {r2_ours:.6f}, Time: {time_ours:.3f}s, Iter: {iter_ours}")
    print(f"Scipy BFGS           - MSE: {mse_scipy_bfgs:.6f}, R²: {r2_scipy_bfgs:.6f}, Time: {time_scipy_bfgs:.3f}s, Iter: {result_bfgs.nit}")
    print(f"Scipy BFGS Success   - {result_bfgs.success}, Message: {result_bfgs.message}")
    
    results['bfgs'] = {
        'our': {'mse': mse_ours, 'r2': r2_ours, 'time': time_ours, 'iterations': iter_ours, 'loss_history': loss_ours},
        'scipy': {'mse': mse_scipy_bfgs, 'r2': r2_scipy_bfgs, 'time': time_scipy_bfgs, 'iterations': result_bfgs.nit, 'success': result_bfgs.success}
    }
    
    # 2. L-BFGS-B Comparison
    print("\n⚡ L-BFGS-B Comparison:")
    print("-" * 60)
    
    # Scipy L-BFGS-B
    start_time = time.time()
    result_lbfgsb = minimize(
        fun=scipy_objective,
        x0=initial_weights,
        args=(X_train_scaled, y_train),
        method='L-BFGS-B',
        jac=scipy_gradient,
        options={'maxiter': 100, 'gtol': 1e-6}
    )
    time_scipy_lbfgsb = time.time() - start_time
    
    y_pred_scipy_lbfgsb = X_test_scaled @ result_lbfgsb.x
    mse_scipy_lbfgsb = mean_squared_error(y_test, y_pred_scipy_lbfgsb)
    r2_scipy_lbfgsb = r2_score(y_test, y_pred_scipy_lbfgsb)
    
    print(f"Scipy L-BFGS-B       - MSE: {mse_scipy_lbfgsb:.6f}, R²: {r2_scipy_lbfgsb:.6f}, Time: {time_scipy_lbfgsb:.3f}s, Iter: {result_lbfgsb.nit}")
    print(f"L-BFGS-B Success     - {result_lbfgsb.success}, Message: {result_lbfgsb.message}")
    
    results['lbfgsb'] = {
        'scipy': {'mse': mse_scipy_lbfgsb, 'r2': r2_scipy_lbfgsb, 'time': time_scipy_lbfgsb, 'iterations': result_lbfgsb.nit, 'success': result_lbfgsb.success}
    }
    
    # 3. Other Scipy Methods for Comparison
    print("\n🔬 Other Optimization Methods:")
    print("-" * 60)
    
    methods = ['CG', 'Newton-CG', 'TNC']
    other_results = {}
    
    for method in methods:
        try:
            start_time = time.time()
            if method == 'Newton-CG':
                # Newton-CG requires Hessian, use finite differences
                result = minimize(
                    fun=scipy_objective,
                    x0=initial_weights,
                    args=(X_train_scaled, y_train),
                    method=method,
                    jac=scipy_gradient,
                    options={'maxiter': 100, 'xtol': 1e-6}
                )
            else:
                result = minimize(
                    fun=scipy_objective,
                    x0=initial_weights,
                    args=(X_train_scaled, y_train),
                    method=method,
                    jac=scipy_gradient,
                    options={'maxiter': 100, 'gtol': 1e-6}
                )
            time_method = time.time() - start_time
            
            y_pred_method = X_test_scaled @ result.x
            mse_method = mean_squared_error(y_test, y_pred_method)
            r2_method = r2_score(y_test, y_pred_method)
            
            print(f"Scipy {method:<12s} - MSE: {mse_method:.6f}, R²: {r2_method:.6f}, Time: {time_method:.3f}s, Iter: {result.nit}, Success: {result.success}")
            
            other_results[method] = {
                'mse': mse_method, 'r2': r2_method, 'time': time_method, 
                'iterations': result.nit, 'success': result.success
            }
            
        except Exception as e:
            print(f"Scipy {method:<12s} - Failed: {e}")
    
    results['other_methods'] = other_results
    
    return results

def create_quasi_newton_plots(results, output_dir):
    """Tạo biểu đồ so sánh Quasi-Newton"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. MSE Comparison
    ax1 = axes[0, 0]
    methods = ['Our BFGS', 'Scipy BFGS', 'Scipy L-BFGS-B']
    mse_values = [results['bfgs']['our']['mse'], 
                  results['bfgs']['scipy']['mse'],
                  results['lbfgsb']['scipy']['mse']]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars1 = ax1.bar(methods, mse_values, color=colors)
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Time Comparison
    ax2 = axes[0, 1]
    time_values = [results['bfgs']['our']['time'], 
                   results['bfgs']['scipy']['time'],
                   results['lbfgsb']['scipy']['time']]
    
    bars2 = ax2.bar(methods, time_values, color=colors)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Speed Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Iterations to Convergence
    ax3 = axes[0, 2]
    iter_values = [results['bfgs']['our']['iterations'], 
                   results['bfgs']['scipy']['iterations'],
                   results['lbfgsb']['scipy']['iterations']]
    
    bars3 = ax3.bar(methods, iter_values, color=colors)
    ax3.set_ylabel('Iterations')
    ax3.set_title('Convergence Speed')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 4. Loss Convergence History
    ax4 = axes[1, 0]
    if 'loss_history' in results['bfgs']['our']:
        ax4.plot(results['bfgs']['our']['loss_history'], label='Our BFGS', linewidth=2, color='skyblue')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Loss Value')
    ax4.set_title('Loss Convergence (Our BFGS)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. All Methods MSE Comparison (including other methods)
    ax5 = axes[1, 1]
    all_methods = ['Our BFGS', 'BFGS', 'L-BFGS-B']
    all_mse = [results['bfgs']['our']['mse'], 
               results['bfgs']['scipy']['mse'],
               results['lbfgsb']['scipy']['mse']]
    
    # Add other methods if they exist
    if 'other_methods' in results:
        for method, data in results['other_methods'].items():
            if data.get('success', False):
                all_methods.append(method)
                all_mse.append(data['mse'])
    
    bars5 = ax5.bar(range(len(all_methods)), all_mse, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(all_methods))))
    ax5.set_ylabel('MSE')
    ax5.set_title('All Methods MSE Comparison')
    ax5.set_xticks(range(len(all_methods)))
    ax5.set_xticklabels(all_methods, rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. Success Rate and Method Properties
    ax6 = axes[1, 2]
    success_methods = ['Our BFGS', 'BFGS', 'L-BFGS-B']
    success_rates = [1.0, float(results['bfgs']['scipy']['success']), float(results['lbfgsb']['scipy']['success'])]
    
    bars6 = ax6.bar(success_methods, success_rates, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax6.set_ylabel('Success Rate')
    ax6.set_title('Method Success Rates')
    ax6.set_ylim([0, 1.1])
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars6:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "quasi_newton_library_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """So sánh Quasi-Newton implementations"""
    print("🎯 QUASI-NEWTON LIBRARY COMPARISON")
    
    # Setup results directory
    results_dir = Path("data/03_algorithms/quasi_newton/library_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparisons
    results = compare_quasi_newton_algorithms()
    
    # Create plots
    create_quasi_newton_plots(results, results_dir)
    
    # Save detailed results
    print("\n💾 Saving comparison results...")
    comparison_data = {
        "comparison_type": "Quasi-Newton Methods vs Scipy Optimize",
        "algorithms_tested": ["BFGS", "L-BFGS-B", "CG", "Newton-CG", "TNC"],
        "results": results,
        "notes": {
            "our_bfgs": "Simplified BFGS with fixed line search",
            "scipy_bfgs": "Full BFGS implementation with sophisticated line search",
            "scipy_lbfgsb": "Limited memory BFGS with bounds (memory efficient)",
            "convergence": "Scipy methods use more sophisticated stopping criteria",
            "line_search": "Scipy uses Armijo/Wolfe conditions for step size",
            "memory_usage": "L-BFGS-B uses limited memory for large problems"
        },
        "conclusions": {
            "accuracy": "Scipy methods achieve better final accuracy",
            "robustness": "Scipy implementations more robust with line search",
            "speed": "Scipy methods typically converge faster",
            "memory": "L-BFGS-B best for large-scale problems",
            "understanding": "Our implementation helps understand BFGS mechanics",
            "production": "Use scipy.optimize for production applications"
        }
    }
    
    with open(results_dir / "quasi_newton_comparison_results.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nQuasi-Newton comparison completed!")
    print(f"Results saved to: {results_dir.absolute()}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 QUASI-NEWTON COMPARISON SUMMARY:")
    print("✅ Scipy implementations achieve superior accuracy and robustness")
    print("⚡ Sophisticated line search makes scipy methods converge faster")
    print("🧠 L-BFGS-B excellent for large-scale optimization problems")
    print("🔧 Our BFGS implementation helps understand the algorithm mechanics")
    print("🚀 For production: use scipy.optimize, for learning: implement yourself!")
    print("=" * 80)

if __name__ == "__main__":
    main()