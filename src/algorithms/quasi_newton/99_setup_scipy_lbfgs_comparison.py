#!/usr/bin/env python3
"""
SciPy L-BFGS-B Comparison - So sánh kết quả với scipy.optimize.minimize(method='L-BFGS-B')
L-BFGS-B với bounds=None trở thành pure L-BFGS (Limited-memory BFGS)
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


def create_scipy_lbfgs_functions(X, y, loss_type='ols', regularization=0.01):
    """
    Tạo objective và gradient functions cho SciPy L-BFGS-B
    
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


def run_scipy_lbfgs_optimization(X_train, y_train, loss_type='ols', regularization=0.01, 
                                 max_iter=200, memory_size=10):
    """
    Chạy tối ưu hóa bằng SciPy L-BFGS-B method (pure L-BFGS)
    
    Tham số:
        X_train: ma trận đặc trưng train (n_samples, n_features)
        y_train: vector target train (n_samples,)
        loss_type: loại loss function
        regularization: hệ số regularization
        max_iter: số iteration tối đa
        memory_size: memory size cho L-BFGS (tham số m)
    
    Trả về:
        dict: kết quả tối ưu hóa
    """
    print(f"\n🔬 Running SciPy L-BFGS optimization for {loss_type.upper()}...")
    
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
    print(f"   Memory size (m): {memory_size}")
    
    # Tạo objective và gradient functions
    objective_func, gradient_func = create_scipy_lbfgs_functions(
        X_with_bias, y_train, loss_type, regularization
    )
    
    # Tính initial loss
    initial_loss = objective_func(initial_weights)
    initial_gradient_norm = np.linalg.norm(gradient_func(initial_weights))
    print(f"   Initial loss: {initial_loss:.8f}")
    print(f"   Initial gradient norm: {initial_gradient_norm:.2e}")
    
    # Chạy optimization với SciPy L-BFGS-B
    print(f"   Starting SciPy L-BFGS-B optimization (pure L-BFGS)...")
    start_time = time.time()
    
    try:
        result = minimize(
            fun=objective_func,
            x0=initial_weights,
            method='L-BFGS-B',  # L-BFGS-B với bounds=None = pure L-BFGS
            jac=gradient_func,
            bounds=None,        # Không có bounds => pure L-BFGS
            options={
                'maxiter': max_iter,
                'maxcor': memory_size,    # Số vector lưu trong memory (tham số m)
                'maxls': 20,              # Max line search steps
                'gtol': 1e-5,             # Gradient tolerance
                'ftol': 2.220446049250313e-09,  # Function tolerance
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
        
        return {
            'weights': final_weights,
            'final_loss': final_loss,
            'training_time': training_time,
            'converged': converged,
            'iterations': iterations,
            'function_evaluations': result.nfev,
            'final_gradient_norm': final_gradient_norm,
            'algorithm': f'SciPy_L_BFGS_{loss_type.upper()}',
            'loss_type': loss_type,
            'regularization': regularization,
            'memory_size': memory_size,
            'message': result.message,
            'initial_loss': initial_loss,
            'initial_gradient_norm': initial_gradient_norm
        }
    
    except Exception as e:
        training_time = time.time() - start_time
        print(f"   ❌ Optimization failed: {e}")
        print(f"   Training time before failure: {training_time:.4f} seconds")
        
        # Return fallback result
        return {
            'weights': initial_weights,
            'final_loss': initial_loss,
            'training_time': training_time,
            'converged': False,
            'iterations': 0,
            'function_evaluations': 0,
            'final_gradient_norm': initial_gradient_norm,
            'algorithm': f'SciPy_L_BFGS_{loss_type.upper()}',
            'loss_type': loss_type,
            'regularization': regularization,
            'memory_size': memory_size,
            'message': f"Optimization failed: {e}",
            'initial_loss': initial_loss,
            'initial_gradient_norm': initial_gradient_norm,
            'error': str(e)
        }


def evaluate_scipy_results(result, X_test, y_test):
    """
    Đánh giá kết quả SciPy trên test set
    
    Tham số:
        result: kết quả từ run_scipy_lbfgs_optimization
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
    Chạy các thí nghiệm so sánh cho OLS, Ridge, Lasso với L-BFGS
    """
    print("="*80)
    print("🔬 SCIPY L-BFGS COMPARISON - LIMITED-MEMORY BFGS ALGORITHM")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    print(f"   Data shapes: Train {X_train.shape}, Test {X_test.shape}")
    print(f"   Target range (log scale): y_train [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Configurations to test - bao gồm các memory sizes khác nhau
    configs = [
        {'loss_type': 'ols', 'regularization': 0.0, 'memory_size': 10},
        {'loss_type': 'ridge', 'regularization': 0.01, 'memory_size': 10},
        {'loss_type': 'lasso', 'regularization': 0.01, 'memory_size': 10},
        # Test memory size khác
        {'loss_type': 'ridge', 'regularization': 0.01, 'memory_size': 20},  # Larger memory
        {'loss_type': 'ridge', 'regularization': 0.01, 'memory_size': 5},   # Smaller memory
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n" + "="*50)
        print(f"🧪 EXPERIMENT: {config['loss_type'].upper()} L-BFGS")
        print(f"   Regularization: {config['regularization']}")
        print(f"   Memory size: {config['memory_size']}")
        print("="*50)
        
        # Run optimization
        result = run_scipy_lbfgs_optimization(
            X_train, y_train, 
            loss_type=config['loss_type'],
            regularization=config['regularization'],
            max_iter=200,
            memory_size=config['memory_size']
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
        memory_suffix = f"_mem{config['memory_size']}" if config['memory_size'] != 10 else ""
        output_dir = Path(f"data/03_algorithms/quasi_newton/scipy_lbfgs_{config['loss_type']}{memory_suffix}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(combined_result, f, indent=2)
        
        print(f"   💾 Results saved to: {output_dir / 'results.json'}")
    
    # Summary comparison
    print(f"\n" + "="*90)
    print("📋 SUMMARY COMPARISON - L-BFGS")
    print("="*90)
    
    print(f"\n{'Algorithm':<20} {'Loss':<7} {'Mem':<4} {'Conv':<5} {'Iters':<6} {'FEval':<6} {'R²(log)':<9} {'Time(s)':<8}")
    print("-"*90)
    
    for result in all_results:
        algorithm = result['algorithm'][:19]  # Truncate for display
        loss_type = result['loss_type'][:6]
        memory_size = result['memory_size']
        converged = "✅" if result['converged'] else "❌"
        iters = result['iterations']
        func_eval = result['function_evaluations']
        r2_log = result['metrics_log_scale']['r2']
        time_s = result['training_time']
        
        print(f"{algorithm:<20} {loss_type:<7} {memory_size:<4} {converged:<5} {iters:<6} {func_eval:<6} {r2_log:<9.6f} {time_s:<8.4f}")
    
    print(f"\n✅ All SciPy L-BFGS comparisons completed!")
    print(f"📁 Results saved in data/03_algorithms/quasi_newton/scipy_lbfgs_*/")
    
    # L-BFGS specific insights
    print(f"\n🔍 L-BFGS ALGORITHM INSIGHTS:")
    print(f"   • L-BFGS sử dụng limited memory để lưu trữ Hessian approximation")
    print(f"   • Memory size (m) thường từ 3-20, default = 10")
    print(f"   • Rất hiệu quả cho large-scale optimization problems")
    print(f"   • Sử dụng ít memory hơn BFGS nhưng vẫn hội tụ nhanh")
    print(f"   • Line search strategy quan trọng cho convergence")
    
    # Memory size analysis
    ridge_results = [r for r in all_results if r['loss_type'] == 'ridge']
    if len(ridge_results) >= 3:
        print(f"\n📊 MEMORY SIZE IMPACT (Ridge regression):")
        for result in ridge_results:
            mem_size = result['memory_size']
            iters = result['iterations']
            time_s = result['training_time']
            r2 = result['metrics_log_scale']['r2']
            print(f"   Memory {mem_size:2d}: {iters:3d} iters, {time_s:6.3f}s, R²={r2:.6f}")
    
    return all_results


def main():
    """Main function"""
    results = run_comparison_experiments()
    return results


if __name__ == "__main__":
    main()