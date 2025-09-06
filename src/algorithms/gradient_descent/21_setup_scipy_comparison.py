#!/usr/bin/env python3
"""
SciPy CG Comparison - So sánh kết quả với scipy.optimize.minimize(method='CG')
CG với restart=0 trở thành Steepest Descent (gần với pure Gradient Descent)
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
from utils.optimization_utils import tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, add_bias_column


def create_scipy_objective_function(X, y, loss_type='ols', regularization=0.01):
    """
    Tạo objective function và gradient function cho SciPy
    
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


def run_scipy_cg_optimization(X_train, y_train, loss_type='ols', regularization=0.01, max_iter=1000):
    """
    Chạy tối ưu hóa bằng SciPy CG method
    
    Tham số:
        X_train: ma trận đặc trưng train (n_samples, n_features)
        y_train: vector target train (n_samples,)
        loss_type: loại loss function
        regularization: hệ số regularization
        max_iter: số iteration tối đa
    
    Trả về:
        dict: kết quả tối ưu hóa
    """
    print(f"\n🔬 Running SciPy CG optimization for {loss_type.upper()}...")
    
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
    objective_func, gradient_func = create_scipy_objective_function(
        X_with_bias, y_train, loss_type, regularization
    )
    
    # Tính initial loss
    initial_loss = objective_func(initial_weights)
    print(f"   Initial loss: {initial_loss:.8f}")
    
    # Chạy optimization với SciPy CG
    print(f"   Starting SciPy CG optimization...")
    start_time = time.time()
    
    result = minimize(
        fun=objective_func,
        x0=initial_weights,
        method='CG',  # Conjugate Gradient - gần với Gradient Descent cho non-quadratic functions
        jac=gradient_func,
        options={
            'maxiter': max_iter,
            'gtol': 1e-6,  # Gradient tolerance
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
        'final_gradient_norm': final_gradient_norm,
        'algorithm': f'SciPy_CG_{loss_type.upper()}',
        'loss_type': loss_type,
        'regularization': regularization,
        'message': result.message
    }


def evaluate_scipy_results(result, X_test, y_test):
    """
    Đánh giá kết quả SciPy trên test set
    
    Tham số:
        result: kết quả từ run_scipy_cg_optimization
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
    Chạy các thí nghiệm so sánh cho OLS, Ridge, Lasso
    """
    print("="*80)
    print("🔬 SCIPY CG COMPARISON - GRADIENT DESCENT ALGORITHMS")
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
        print(f"🧪 EXPERIMENT: {config['loss_type'].upper()}")
        if config['regularization'] > 0:
            print(f"   Regularization: {config['regularization']}")
        print("="*50)
        
        # Run optimization
        result = run_scipy_cg_optimization(
            X_train, y_train, 
            loss_type=config['loss_type'],
            regularization=config['regularization'],
            max_iter=1000
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
        print(f"      MAPE: {evaluation['metrics_original_scale']['mape']:.2f}%")
        
        all_results.append(combined_result)
        
        # Save individual result
        output_dir = Path(f"data/03_algorithms/gradient_descent/scipy_cg_{config['loss_type']}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(combined_result, f, indent=2)
        
        print(f"   💾 Results saved to: {output_dir / 'results.json'}")
    
    # Summary comparison
    print(f"\n" + "="*80)
    print("📋 SUMMARY COMPARISON")
    print("="*80)
    
    print(f"\n{'Algorithm':<20} {'Loss Type':<10} {'R² (log)':<10} {'R² (orig)':<10} {'MAPE':<10} {'Time (s)':<10}")
    print("-"*80)
    
    for result in all_results:
        algorithm = result['algorithm']
        loss_type = result['loss_type']
        r2_log = result['metrics_log_scale']['r2']
        r2_orig = result['metrics_original_scale']['r2']
        mape = result['metrics_original_scale']['mape']
        time_s = result['training_time']
        
        print(f"{algorithm:<20} {loss_type:<10} {r2_log:<10.6f} {r2_orig:<10.6f} {mape:<10.2f} {time_s:<10.4f}")
    
    print(f"\n✅ All SciPy CG comparisons completed!")
    print(f"📁 Results saved in data/03_algorithms/gradient_descent/scipy_cg_*/")
    
    return all_results


def main():
    """Main function"""
    results = run_comparison_experiments()
    return results


if __name__ == "__main__":
    main()