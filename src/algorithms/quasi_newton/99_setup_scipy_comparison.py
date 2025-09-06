#!/usr/bin/env python3
"""
Comprehensive comparison of our Quasi-Newton implementations vs SciPy
- BFGS vs scipy.optimize.minimize(method='BFGS')
- L-BFGS vs scipy.optimize.minimize(method='L-BFGS-B')
- SR1 vs scipy.optimize.minimize(method='trust-ncg') [closest equivalent]
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.quasi_newton.quasi_newton_model import QuasiNewtonModel
from utils.data_process_utils import load_du_lieu
from utils.optimization_utils import (
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, add_bias_column
)

# SciPy imports
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ Warning: SciPy not available. Skipping SciPy comparison.")


def scipy_objective_and_grad(weights, X, y, loss_type='ols', regularization=0.01):
    """SciPy-compatible objective function and gradient"""
    loss = tinh_gia_tri_ham_loss(loss_type, X, y, weights, None, regularization)
    grad, _ = tinh_gradient_ham_loss(loss_type, X, y, weights, None, regularization)
    return loss, grad


def run_scipy_optimization(X, y, method='BFGS', loss_type='ols', regularization=0.01, max_iter=1000):
    """Run SciPy optimization"""
    if not SCIPY_AVAILABLE:
        return None
    
    print(f"🔬 Running SciPy {method} for {loss_type.upper()}")
    
    # Add bias column
    X_with_bias = add_bias_column(X)
    n_features = X_with_bias.shape[1]
    
    # Initial weights
    x0 = np.random.normal(0, 0.01, n_features)
    
    start_time = time.time()
    
    # Setup method-specific options
    options = {'maxiter': max_iter, 'disp': True}
    if method == 'L-BFGS-B':
        options['maxcor'] = 10  # Memory size similar to our L-BFGS
    
    try:
        result = minimize(
            fun=lambda w: scipy_objective_and_grad(w, X_with_bias, y, loss_type, regularization),
            x0=x0,
            method=method,
            jac=True,  # We provide both function and gradient
            options=options
        )
        
        training_time = time.time() - start_time
        
        return {
            'success': result.success,
            'weights': result.x,
            'final_loss': result.fun,
            'final_gradient_norm': np.linalg.norm(result.jac),
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'training_time': training_time,
            'message': result.message,
            'method': method,
            'scipy_result': result
        }
        
    except Exception as e:
        print(f"❌ SciPy {method} failed: {e}")
        return None


def run_our_implementation(X, y, method='bfgs', loss_type='ols', regularization=0.01, max_iter=1000, memory_size=10):
    """Run our implementation"""
    print(f"🚀 Running Our {method.upper()} for {loss_type.upper()}")
    
    model = QuasiNewtonModel(
        ham_loss=loss_type,
        method=method,
        regularization=regularization,
        so_lan_thu=max_iter,
        diem_dung=1e-6,
        memory_size=memory_size,  # For L-BFGS
        sr1_skip_threshold=1e-8,  # For SR1
        convergence_check_freq=1  # Check every iteration for fair comparison
    )
    
    start_time = time.time()
    results = model.fit(X, y)
    training_time = time.time() - start_time
    
    return {
        'success': results['converged'],
        'weights': results['weights'],
        'final_loss': results['final_loss'],
        'final_gradient_norm': results['final_gradient_norm'],
        'iterations': results['final_iteration'],
        'training_time': training_time,
        'method': method,
        'our_results': results
    }


def compare_implementations():
    """Compare all implementations"""
    print("=" * 80)
    print("QUASI-NEWTON METHODS COMPARISON: Our Implementation vs SciPy")
    print("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Use smaller dataset for fair comparison
    if X_train.shape[0] > 5000:
        print(f"📊 Using first 3000 samples for comparison (original: {X_train.shape[0]})")
        X_train = X_train[:3000]
        y_train = y_train[:3000]
        X_test = X_test[:1000]
        y_test = y_test[:1000]
    
    print(f"📊 Dataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    print(f"📊 Features: {X_train.shape[1]}")
    
    results = {}
    max_iter = 100  # Reasonable limit for comparison
    
    # Test configurations
    test_configs = [
        ('ols', 0.0),
        ('ridge', 0.01),
    ]
    
    for loss_type, reg in test_configs:
        print(f"\n" + "="*60)
        print(f"TESTING {loss_type.upper()} (reg={reg})")
        print("="*60)
        
        results[f"{loss_type}_reg_{reg}"] = {}
        
        # 1. BFGS Comparison
        print(f"\n🔍 BFGS Comparison")
        print("-" * 40)
        
        # Our BFGS
        our_bfgs = run_our_implementation(X_train, y_train, 'bfgs', loss_type, reg, max_iter)
        results[f"{loss_type}_reg_{reg}"]["our_bfgs"] = our_bfgs
        
        # SciPy BFGS
        if SCIPY_AVAILABLE:
            scipy_bfgs = run_scipy_optimization(X_train, y_train, 'BFGS', loss_type, reg, max_iter)
            results[f"{loss_type}_reg_{reg}"]["scipy_bfgs"] = scipy_bfgs
        
        # 2. L-BFGS Comparison  
        print(f"\n🔍 L-BFGS Comparison")
        print("-" * 40)
        
        # Our L-BFGS
        our_lbfgs = run_our_implementation(X_train, y_train, 'lbfgs', loss_type, reg, max_iter, memory_size=10)
        results[f"{loss_type}_reg_{reg}"]["our_lbfgs"] = our_lbfgs
        
        # SciPy L-BFGS-B
        if SCIPY_AVAILABLE:
            scipy_lbfgs = run_scipy_optimization(X_train, y_train, 'L-BFGS-B', loss_type, reg, max_iter)
            results[f"{loss_type}_reg_{reg}"]["scipy_lbfgs"] = scipy_lbfgs
        
        # 3. SR1 (Our implementation only - no direct SciPy equivalent)
        print(f"\n🔍 SR1 (Our Implementation)")
        print("-" * 40)
        
        our_sr1 = run_our_implementation(X_train, y_train, 'sr1', loss_type, reg, max_iter)
        results[f"{loss_type}_reg_{reg}"]["our_sr1"] = our_sr1
    
    # Generate comparison report
    print(f"\n" + "="*80)
    print("COMPARISON SUMMARY")  
    print("="*80)
    
    comparison_data = []
    
    for config, methods in results.items():
        print(f"\n📊 Configuration: {config}")
        print("-" * 50)
        
        for method_name, result in methods.items():
            if result is not None:
                row = {
                    'Configuration': config,
                    'Method': method_name,
                    'Success': result['success'],
                    'Final Loss': f"{result['final_loss']:.8f}",
                    'Gradient Norm': f"{result['final_gradient_norm']:.2e}",
                    'Iterations': result['iterations'],
                    'Time (s)': f"{result['training_time']:.4f}"
                }
                
                if 'function_evaluations' in result:
                    row['Func Evals'] = result['function_evaluations']
                
                comparison_data.append(row)
                
                print(f"  {method_name:15}: Loss={result['final_loss']:.8f}, "
                      f"Grad={result['final_gradient_norm']:.2e}, "
                      f"Iter={result['iterations']:3d}, "
                      f"Time={result['training_time']:.4f}s, "
                      f"Success={result['success']}")
    
    # Save detailed results
    results_dir = Path("data/03_algorithms/quasi_newton/scipy_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    df = pd.DataFrame(comparison_data)
    df.to_csv(results_dir / "comparison_table.csv", index=False)
    print(f"\n💾 Comparison table saved to: {results_dir / 'comparison_table.csv'}")
    
    # Save detailed results
    with open(results_dir / "detailed_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for config, methods in results.items():
            json_results[config] = {}
            for method, result in methods.items():
                if result is not None:
                    json_result = result.copy()
                    # Remove non-serializable objects
                    json_result.pop('scipy_result', None)
                    json_result.pop('our_results', None)
                    # Convert numpy arrays
                    if isinstance(json_result.get('weights'), np.ndarray):
                        json_result['weights'] = json_result['weights'].tolist()
                    json_results[config][method] = json_result
        
        json.dump(json_results, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_dir / 'detailed_results.json'}")
    print(f"\n✅ Comparison completed! Check {results_dir.absolute()}")
    
    return results, df


def main():
    """Main comparison function"""
    if not SCIPY_AVAILABLE:
        print("❌ SciPy is required for comparison. Install with: pip install scipy")
        return
    
    try:
        results, df = compare_implementations()
        
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 50)
        print("✅ All quasi-Newton implementations completed")
        print("📊 Performance comparison available in results")
        print("🔍 Check the generated CSV and JSON files for detailed analysis")
        
        return results, df
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()