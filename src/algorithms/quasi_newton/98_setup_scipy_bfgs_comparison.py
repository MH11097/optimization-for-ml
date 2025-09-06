#!/usr/bin/env python3
"""
SciPy BFGS Comparison - So sánh kết quả với scipy.optimize.minimize(method='BFGS')
BFGS là thuật toán Quasi-Newton chuẩn với Wolfe line search
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import json
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_process_utils import load_du_lieu
from utils.optimization_utils import (
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, add_bias_column
)


def create_scipy_bfgs_functions(X, y, loss_type='ols', regularization=0.01):
    """
    Tạo objective và gradient functions cho SciPy BFGS
    
    Tham số:
        X: ma trận đặc trưng với bias đã thêm (n_samples, n_features + 1)
        y: vector target (n_samples,)
        loss_type: loại loss function ('ols', 'ridge', 'lasso')
        regularization: hệ số regularization
    
    Trả về:
        objective_func: hàm tính loss
        gradient_func: hàm tính gradient
    """
    def objective_func(w):
        """Tính giá trị loss function"""
        return tinh_gia_tri_ham_loss(loss_type, X, y, w, bias=None, regularization=regularization)
    
    def gradient_func(w):
        """Tính gradient"""
        grad_w, grad_b = tinh_gradient_ham_loss(loss_type, X, y, w, bias=None, regularization=regularization)
        return grad_w  # grad_b sẽ là 0 cho format mới với bias trong X
    
    return objective_func, gradient_func


def run_scipy_bfgs_optimization(X_train, y_train, loss_type='ols', regularization=0.01, max_iter=200):
    """
    Chạy tối ưu hóa bằng SciPy BFGS method
    
    Tham số:
        X_train: ma trận đặc trưng train (n_samples, n_features)
        y_train: vector target train (n_samples,)
        loss_type: loại loss function
        regularization: hệ số regularization
        max_iter: số iteration tối đa
    
    Trả về:
        dict: kết quả tối ưu hóa
    """
    print(f"\n🔬 Running SciPy BFGS optimization for {loss_type.upper()}...")
    
    # Thêm bias column
    X_with_bias = add_bias_column(X_train)
    n_features_with_bias = X_with_bias.shape[1]
    
    # Khởi tạo weights
    np.random.seed(42)
    initial_weights = np.random.randn(n_features_with_bias) * 0.01
    
    print(f"   Features (with bias): {n_features_with_bias}")
    print(f"   Initial weights range: [{initial_weights.min():.6f}, {initial_weights.max():.6f}]")
    print(f"   Loss type: {loss_type}")
    print(f"   Regularization: {regularization}")
    
    # Tạo objective và gradient functions
    objective_func, gradient_func = create_scipy_bfgs_functions(
        X_with_bias, y_train, loss_type, regularization
    )
    
    # Tính initial loss
    initial_loss = objective_func(initial_weights)
    initial_gradient_norm = np.linalg.norm(gradient_func(initial_weights))
    print(f"   Initial loss: {initial_loss:.8f}")
    print(f"   Initial gradient norm: {initial_gradient_norm:.2e}")
    
    # Chạy optimization với SciPy BFGS
    print(f"   Starting SciPy BFGS optimization...")
    start_time = time.time()
    
    try:
        result = minimize(
            fun=objective_func,
            x0=initial_weights,
            method='BFGS',  # BFGS với Wolfe line search (thuật toán gốc)
            jac=gradient_func,
            options={
                'maxiter': max_iter,
                'gtol': 1e-5,      # Gradient tolerance
                'norm': np.inf,    # Norm để tính gradient convergence
                'disp': False
            }
        )
        
        training_time = time.time() - start_time
        
        # Extract results
        final_weights = result.x
        final_loss = result.fun
        converged = result.success
        iterations = result.nit if hasattr(result, 'nit') else result.nfev
        final_gradient_norm = np.linalg.norm(gradient_func(final_weights))
        
        print(f"   ✅ Optimization completed!")
        print(f"   Training time: {training_time:.4f} seconds")
        print(f"   Converged: {converged}")
        print(f"   Iterations: {iterations}")
        print(f"   Function evaluations: {result.nfev}")
        print(f"   Final loss: {final_loss:.8f}")
        print(f"   Final gradient norm: {final_gradient_norm:.2e}")
        print(f"   Final weights range: [{final_weights.min():.6f}, {final_weights.max():.6f}]")
        print(f"   Bias (last weight): {final_weights[-1]:.6f}")
        
        # BFGS specific info
        if hasattr(result, 'hess_inv'):
            try:
                hess_inv_cond = np.linalg.cond(result.hess_inv)
                print(f"   Final inverse Hessian condition: {hess_inv_cond:.2e}")
            except:
                print(f"   Final inverse Hessian: Available but condition not computed")
        
        return {
            'weights': final_weights.tolist(),
            'final_loss': float(final_loss),
            'training_time': float(training_time),
            'converged': bool(converged),
            'iterations': int(iterations),
            'function_evaluations': int(result.nfev),
            'final_gradient_norm': float(final_gradient_norm),
            'algorithm': f'SciPy_BFGS_{loss_type.upper()}',
            'loss_type': loss_type,
            'regularization': float(regularization),
            'message': str(result.message),
            'initial_loss': float(initial_loss),
            'initial_gradient_norm': float(initial_gradient_norm)
        }
    
    except Exception as e:
        training_time = time.time() - start_time
        print(f"   ❌ Optimization failed: {e}")
        print(f"   Training time before failure: {training_time:.4f} seconds")
        
        # Return fallback result
        return {
            'weights': initial_weights.tolist(),
            'final_loss': float(initial_loss),
            'training_time': float(training_time),
            'converged': False,
            'iterations': 0,
            'function_evaluations': 0,
            'final_gradient_norm': float(initial_gradient_norm),
            'algorithm': f'SciPy_BFGS_{loss_type.upper()}',
            'loss_type': loss_type,
            'regularization': float(regularization),
            'message': f"Optimization failed: {e}",
            'initial_loss': float(initial_loss),
            'initial_gradient_norm': float(initial_gradient_norm),
            'error': str(e)
        }


def evaluate_scipy_results(result, X_test, y_test):
    """
    Đánh giá kết quả SciPy trên test set
    
    Tham số:
        result: kết quả từ run_scipy_bfgs_optimization
        X_test: ma trận đặc trưng test
        y_test: vector target test
    
    Trả về:
        dict: metrics đánh giá
    """
    # Thêm bias column cho X_test
    X_test_with_bias = add_bias_column(X_test)
    
    # Dự đoán
    weights = result['weights']
    predictions_log = X_test_with_bias @ weights
    
    # Evaluate on log scale
    mse_log = mean_squared_error(y_test, predictions_log)
    r2_log = r2_score(y_test, predictions_log)
    mae_log = mean_absolute_error(y_test, predictions_log)
    
    # Convert to original scale
    predictions_original = np.expm1(predictions_log)
    y_test_original = np.expm1(y_test)
    
    # Evaluate on original scale
    mse_original = mean_squared_error(y_test_original, predictions_original)
    r2_original = r2_score(y_test_original, predictions_original)
    mae_original = mean_absolute_error(y_test_original, predictions_original)
    rmse_original = np.sqrt(mse_original)
    
    # MAPE calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs((y_test_original - predictions_original) / y_test_original)
        valid_errors = percentage_errors[np.isfinite(percentage_errors)]
        mape = np.mean(valid_errors) * 100 if len(valid_errors) > 0 else float('inf')
    
    return {
        'predictions_log_min': float(predictions_log.min()),
        'predictions_log_max': float(predictions_log.max()),
        'predictions_original_min': float(predictions_original.min()),
        'predictions_original_max': float(predictions_original.max()),
        'metrics_log_scale': {
            'mse': float(mse_log),
            'r2': float(r2_log),
            'mae': float(mae_log)
        },
        'metrics_original_scale': {
            'mse': float(mse_original),
            'rmse': float(rmse_original),
            'mae': float(mae_original),
            'r2': float(r2_original),
            'mape': float(mape)
        }
    }


def run_comparison_experiments():
    """
    Chạy các thí nghiệm so sánh cho OLS, Ridge, Lasso với BFGS
    """
    print("="*80)
    print("🔬 SCIPY BFGS COMPARISON - QUASI-NEWTON BFGS ALGORITHM")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    print(f"   Data shapes: Train {X_train.shape}, Test {X_test.shape}")
    print(f"   Target range (log scale): y_train [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Configurations to test
    configs = [
        {'loss_type': 'ols', 'regularization': 0.0},
        {'loss_type': 'ridge', 'regularization': 0.01},
        {'loss_type': 'lasso', 'regularization': 0.01}
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n" + "="*50)
        print(f"🧪 EXPERIMENT: {config['loss_type'].upper()} BFGS")
        if config['regularization'] > 0:
            print(f"   Regularization: {config['regularization']}")
        print("="*50)
        
        # Run optimization
        result = run_scipy_bfgs_optimization(
            X_train, y_train, 
            loss_type=config['loss_type'],
            regularization=config['regularization'],
            max_iter=200  # BFGS thường cần nhiều iterations hơn Newton
        )
        
        # Evaluate
        print(f"\n📊 Evaluating on test set...")
        evaluation = evaluate_scipy_results(result, X_test, y_test)
        
        # Combine results
        combined_result = {
            **result,
            **evaluation,
            'parameters': config
        }
        
        # Print evaluation summary
        print(f"   📈 LOG SCALE METRICS:")
        print(f"      MSE: {evaluation['metrics_log_scale']['mse']:.8f}")
        print(f"      R²:  {evaluation['metrics_log_scale']['r2']:.6f}")
        print(f"      MAE: {evaluation['metrics_log_scale']['mae']:.6f}")
        
        print(f"   🎯 ORIGINAL SCALE METRICS:")
        print(f"      MSE:  {evaluation['metrics_original_scale']['mse']:,.2f}")
        print(f"      RMSE: {evaluation['metrics_original_scale']['rmse']:,.2f}")
        print(f"      MAE:  {evaluation['metrics_original_scale']['mae']:,.2f}")
        print(f"      R²:   {evaluation['metrics_original_scale']['r2']:.6f}")
        if evaluation['metrics_original_scale']['mape'] != float('inf'):
            print(f"      MAPE: {evaluation['metrics_original_scale']['mape']:.2f}%")
        else:
            print(f"      MAPE: N/A")
        
        all_results.append(combined_result)
        
        # Save individual result
        output_dir = Path(f"data/03_algorithms/quasi_newton/scipy_bfgs_{config['loss_type']}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(combined_result, f, indent=2)
        
        print(f"   💾 Results saved to: {output_dir / 'results.json'}")
    
    # Summary comparison
    print(f"\n" + "="*80)
    print("📋 SUMMARY COMPARISON - BFGS")
    print("="*80)
    
    print(f"\n{'Algorithm':<25} {'Loss Type':<10} {'Converged':<10} {'Iters':<8} {'FuncEval':<10} {'R² (log)':<10} {'Time (s)':<10}")
    print("-"*88)
    
    for result in all_results:
        algorithm = result['algorithm'][:24]  # Truncate for display
        loss_type = result['loss_type']
        converged = "✅" if result['converged'] else "❌"
        iters = result['iterations']
        func_eval = result['function_evaluations']
        r2_log = result['metrics_log_scale']['r2']
        time_s = result['training_time']
        
        print(f"{algorithm:<25} {loss_type:<10} {converged:<10} {iters:<8} {func_eval:<10} {r2_log:<10.6f} {time_s:<10.4f}")
    
    print(f"\n✅ All SciPy BFGS comparisons completed!")
    print(f"📁 Results saved in data/03_algorithms/quasi_newton/scipy_bfgs_*/")
    
    # BFGS specific insights
    print(f"\n🔍 BFGS ALGORITHM INSIGHTS:")
    print(f"   • BFGS sử dụng Wolfe line search để đảm bảo convergence")
    print(f"   • Xây dựng approximation của inverse Hessian qua iterations")
    print(f"   • Thường cần nhiều iterations hơn Newton nhưng ít hơn Gradient Descent")
    print(f"   • Rất hiệu quả cho các bài toán smooth optimization")
    print(f"   • Không cần tính toán Hessian matrix trực tiếp")
    
    return all_results


def main():
    """Main function"""
    results = run_comparison_experiments()
    return results


if __name__ == "__main__":
    main()