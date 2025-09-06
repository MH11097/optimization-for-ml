#!/usr/bin/env python3
"""
SciPy Newton-CG Comparison - So sánh kết quả với scipy.optimize.minimize(method='Newton-CG')
Newton-CG là phiên bản truncated Newton method, gần với pure Newton's method
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
    tinh_gia_tri_ham_loss, tinh_gradient_ham_loss, tinh_hessian_ham_loss, 
    add_bias_column
)


def create_scipy_newton_functions(X, y, loss_type='ols', regularization=0.01):
    """
    Tạo objective, gradient và hessian functions cho SciPy Newton-CG
    
    Tham số:
        X: ma trận đặc trưng với bias đã thêm (n_samples, n_features + 1)
        y: vector target (n_samples,)
        loss_type: loại loss function ('ols', 'ridge', 'lasso')
        regularization: hệ số regularization
    
    Trả về:
        objective_func: hàm tính loss
        gradient_func: hàm tính gradient
        hessian_func: hàm tính hessian
    """
    def objective_func(w):
        """Tính giá trị loss function"""
        return tinh_gia_tri_ham_loss(loss_type, X, y, w, bias=None, regularization=regularization)
    
    def gradient_func(w):
        """Tính gradient"""
        grad_w, grad_b = tinh_gradient_ham_loss(loss_type, X, y, w, bias=None, regularization=regularization)
        return grad_w  # grad_b sẽ là 0 cho format mới với bias trong X
    
    def hessian_func(w):
        """Tính hessian matrix"""
        if loss_type == 'lasso':
            # Lasso cần weights để tính Hessian
            return tinh_hessian_ham_loss(loss_type, X, w, regularization=regularization)
        else:
            # OLS và Ridge không cần weights
            return tinh_hessian_ham_loss(loss_type, X, regularization=regularization)
    
    return objective_func, gradient_func, hessian_func


def run_scipy_newton_optimization(X_train, y_train, loss_type='ols', regularization=0.01, max_iter=100):
    """
    Chạy tối ưu hóa bằng SciPy Newton-CG method
    
    Tham số:
        X_train: ma trận đặc trưng train (n_samples, n_features)
        y_train: vector target train (n_samples,)
        loss_type: loại loss function
        regularization: hệ số regularization
        max_iter: số iteration tối đa
    
    Trả về:
        dict: kết quả tối ưu hóa
    """
    print(f"\n🔬 Running SciPy Newton-CG optimization for {loss_type.upper()}...")
    
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
    
    # Tạo objective, gradient và hessian functions
    objective_func, gradient_func, hessian_func = create_scipy_newton_functions(
        X_with_bias, y_train, loss_type, regularization
    )
    
    # Tính initial loss
    initial_loss = objective_func(initial_weights)
    initial_gradient_norm = np.linalg.norm(gradient_func(initial_weights))
    print(f"   Initial loss: {initial_loss:.8f}")
    print(f"   Initial gradient norm: {initial_gradient_norm:.2e}")
    
    # Kiểm tra Hessian matrix condition
    try:
        initial_hessian = hessian_func(initial_weights)
        hessian_cond = np.linalg.cond(initial_hessian)
        hessian_det = np.linalg.det(initial_hessian)
        print(f"   Initial Hessian condition number: {hessian_cond:.2e}")
        print(f"   Initial Hessian determinant: {hessian_det:.2e}")
    except Exception as e:
        print(f"   Warning: Could not compute initial Hessian info: {e}")
    
    # Chạy optimization với SciPy Newton-CG
    print(f"   Starting SciPy Newton-CG optimization...")
    start_time = time.time()
    
    try:
        result = minimize(
            fun=objective_func,
            x0=initial_weights,
            method='Newton-CG',  # Newton's method với Conjugate Gradient để giải hệ tuyến tính
            jac=gradient_func,
            hess=hessian_func,
            options={
                'maxiter': max_iter,
                'xtol': 1e-8,    # Tolerance for convergence in parameters
                'gtol': 1e-6,    # Gradient tolerance
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
            'weights': final_weights.tolist(),
            'final_loss': float(final_loss),
            'training_time': float(training_time),
            'converged': bool(converged),
            'iterations': int(iterations),
            'final_gradient_norm': float(final_gradient_norm),
            'algorithm': f'SciPy_Newton_CG_{loss_type.upper()}',
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
            'final_gradient_norm': float(initial_gradient_norm),
            'algorithm': f'SciPy_Newton_CG_{loss_type.upper()}',
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
        result: kết quả từ run_scipy_newton_optimization
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
    Chạy các thí nghiệm so sánh cho OLS, Ridge, Lasso với Newton's method
    """
    print("="*80)
    print("🔬 SCIPY NEWTON-CG COMPARISON - NEWTON'S METHOD ALGORITHMS")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    print(f"   Data shapes: Train {X_train.shape}, Test {X_test.shape}")
    print(f"   Target range (log scale): y_train [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Configurations to test
    # Note: Newton's method work best with regularization để tránh singular Hessian
    configs = [
        {'loss_type': 'ols', 'regularization': 1e-6},     # Tiny regularization for numerical stability
        {'loss_type': 'ridge', 'regularization': 0.01},
        {'loss_type': 'lasso', 'regularization': 0.01}    # Lasso có thể khó hội tụ với Newton
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n" + "="*50)
        print(f"🧪 EXPERIMENT: {config['loss_type'].upper()} Newton's Method")
        print(f"   Regularization: {config['regularization']}")
        print("="*50)
        
        # Run optimization
        result = run_scipy_newton_optimization(
            X_train, y_train, 
            loss_type=config['loss_type'],
            regularization=config['regularization'],
            max_iter=100  # Newton thường hội tụ nhanh hơn
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
        output_dir = Path(f"data/03_algorithms/newton_method/scipy_newton_{config['loss_type']}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(combined_result, f, indent=2)
        
        print(f"   💾 Results saved to: {output_dir / 'results.json'}")
    
    # Summary comparison
    print(f"\n" + "="*80)
    print("📋 SUMMARY COMPARISON - NEWTON'S METHOD")
    print("="*80)
    
    print(f"\n{'Algorithm':<25} {'Loss Type':<10} {'Converged':<10} {'R² (log)':<10} {'R² (orig)':<10} {'Time (s)':<10}")
    print("-"*85)
    
    for result in all_results:
        algorithm = result['algorithm'][:24]  # Truncate for display
        loss_type = result['loss_type']
        converged = "✅" if result['converged'] else "❌"
        r2_log = result['metrics_log_scale']['r2']
        r2_orig = result['metrics_original_scale']['r2']
        time_s = result['training_time']
        
        print(f"{algorithm:<25} {loss_type:<10} {converged:<10} {r2_log:<10.6f} {r2_orig:<10.6f} {time_s:<10.4f}")
    
    print(f"\n✅ All SciPy Newton-CG comparisons completed!")
    print(f"📁 Results saved in data/03_algorithms/newton_method/scipy_newton_*/")
    
    # Newton's method specific insights
    print(f"\n🔍 NEWTON'S METHOD INSIGHTS:")
    print(f"   • Newton's method thường hội tụ rất nhanh (vài iterations)")
    print(f"   • Yêu cầu Hessian matrix phải positive definite")
    print(f"   • SciPy Newton-CG sử dụng Conjugate Gradient để giải hệ tuyến tính")
    print(f"   • Thích hợp cho các bài toán smooth và well-conditioned")
    
    return all_results


def main():
    """Main function"""
    results = run_comparison_experiments()
    return results


if __name__ == "__main__":
    main()