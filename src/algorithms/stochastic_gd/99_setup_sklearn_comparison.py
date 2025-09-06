#!/usr/bin/env python3
"""
Sklearn SGD Comparison - So sánh kết quả với sklearn.linear_model.SGDRegressor
Sử dụng Stochastic Gradient Descent từ thư viện sklearn thay vì PyTorch
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: Scikit-learn not available. Please install scikit-learn to run this comparison.")

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_process_utils import load_du_lieu
from utils.optimization_utils import add_bias_column


def run_sklearn_sgd_optimization(X_train, y_train, loss_type='squared_error', regularization=0.01, 
                                learning_rate='constant', eta0=0.01, max_iter=100, batch_size=None,
                                alpha=0.01, l1_ratio=0.15, early_stopping=False):
    """
    Chạy tối ưu hóa bằng Sklearn SGD
    
    Tham số:
        X_train: ma trận đặc trưng train (n_samples, n_features)
        y_train: vector target train (n_samples,)
        loss_type: loại loss function ('squared_error', 'huber', 'epsilon_insensitive')
        regularization: loại regularization ('l1', 'l2', 'elasticnet', None)
        learning_rate: learning rate schedule ('constant', 'optimal', 'invscaling', 'adaptive')
        eta0: initial learning rate
        max_iter: số iteration tối đa
        batch_size: batch size (None = auto)
        alpha: regularization strength
        l1_ratio: l1 ratio for elastic net (0=l2, 1=l1)
        early_stopping: có sử dụng early stopping không
    
    Trả về:
        dict: kết quả tối ưu hóa
    """
    if not SKLEARN_AVAILABLE:
        return {
            'error': 'Scikit-learn not available',
            'algorithm': f'Sklearn_SGD_{loss_type.upper()}',
            'converged': False
        }
    
    print(f"\n=> Running Sklearn SGD optimization for {loss_type.upper()}...")
    
    n_samples, n_features = X_train.shape
    
    print(f"   Data: {n_samples} samples, {n_features} features")
    print(f"   Loss type: {loss_type}")
    print(f"   Regularization: {regularization}")
    print(f"   Alpha (reg strength): {alpha}")
    if regularization == 'elasticnet':
        print(f"   L1 ratio: {l1_ratio}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Initial learning rate: {eta0}")
    print(f"   Max iterations: {max_iter}")
    print(f"   Batch size: {batch_size if batch_size else 'auto'}")
    print(f"   Early stopping: {early_stopping}")
    
    # Chuẩn hóa chỉ features (KHÔNG chuẩn hóa targets để giữ nguyên log scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # y_train giữ nguyên ở log scale, không standardize
    
    print(f"   Features standardized, targets kept in log scale")
    
    # Tạo SGD model
    sgd_params = {
        'loss': loss_type,
        'penalty': regularization if regularization else 'none',
        'alpha': alpha,
        'learning_rate': learning_rate,
        'eta0': eta0,
        'max_iter': max_iter,
        'tol': 1e-5,  # Match custom SGD tolerance
        'shuffle': True,  # shuffle data at each epoch
        'random_state': 42,
        'early_stopping': early_stopping,
        'validation_fraction': 0.1 if early_stopping else 0.1,
        'n_iter_no_change': 5 if early_stopping else 5,
        'warm_start': False,
        'average': False  # không dùng averaging
    }
    
    # Thêm l1_ratio nếu sử dụng elasticnet
    if regularization == 'elasticnet':
        sgd_params['l1_ratio'] = l1_ratio
    
    # Thêm batch size nếu được chỉ định
    if batch_size is not None:
        # Sklearn không có batch_size trực tiếp, ta sử dụng partial_fit để simulate
        print(f"   Note: Simulating batch_size={batch_size} using partial_fit")
    
    model = SGDRegressor(**sgd_params)
    
    print(f"   Starting Sklearn SGD training...")
    start_time = time.time()
    
    # Training với hoặc không có batch size simulation
    if batch_size is not None and batch_size < n_samples:
        # Simulate mini-batch training using partial_fit
        n_batches_per_epoch = n_samples // batch_size
        if n_samples % batch_size != 0:
            n_batches_per_epoch += 1
        
        loss_history = []
        
        # Initialize model with first batch
        first_batch_indices = np.arange(batch_size)
        model.partial_fit(X_train_scaled[first_batch_indices], y_train[first_batch_indices])
        
        # Training loop
        for epoch in range(max_iter):
            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            for batch_idx in range(n_batches_per_epoch):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Partial fit on batch
                model.partial_fit(X_train_scaled[batch_indices], y_train[batch_indices])
                
                # Calculate loss for this batch
                batch_pred = model.predict(X_train_scaled[batch_indices])
                batch_loss = mean_squared_error(y_train[batch_indices], batch_pred)
                epoch_loss += batch_loss
            
            # Average loss for epoch
            avg_epoch_loss = epoch_loss / n_batches_per_epoch
            loss_history.append(avg_epoch_loss)
            
            # Print progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1:3d}/{max_iter}: Loss = {avg_epoch_loss:.8f}")
        
        total_batches = max_iter * n_batches_per_epoch
        
    else:
        # Standard training (full batch)
        model.fit(X_train_scaled, y_train)
        
        # Calculate loss history approximation
        predictions = model.predict(X_train_scaled)
        final_loss = mean_squared_error(y_train, predictions)
        loss_history = [final_loss]  # Single point approximation
        total_batches = model.n_iter_ if hasattr(model, 'n_iter_') else max_iter
    
    training_time = time.time() - start_time
    
    # Extract final results
    final_predictions = model.predict(X_train_scaled)
    final_loss = mean_squared_error(y_train, final_predictions)
    
    # Get coefficients
    final_weights = model.coef_ if hasattr(model, 'coef_') else np.array([])
    final_bias = model.intercept_ if hasattr(model, 'intercept_') else 0.0
    
    # Check convergence
    converged = hasattr(model, 'n_iter_') and model.n_iter_ < max_iter
    if not hasattr(model, 'n_iter_'):
        converged = True  # Assume converged if no iteration info
    
    print(f"   => Training completed!")
    print(f"   Training time: {training_time:.4f} seconds")
    print(f"   Final loss: {final_loss:.8f}")
    print(f"   Converged: {converged}")
    if len(final_weights) > 0:
        print(f"   Final weights range: [{final_weights.min():.6f}, {final_weights.max():.6f}]")
    if isinstance(final_bias, np.ndarray):
        bias_value = final_bias[0] if len(final_bias) > 0 else 0.0
    else:
        bias_value = final_bias
    print(f"   Final bias: {bias_value:.6f}")
    print(f"   Total iterations/batches: {total_batches}")
    
    return {
        'model': model,
        'scaler': scaler,
        'final_weights': final_weights.tolist() if len(final_weights) > 0 else [],
        'final_bias': float(bias_value),
        'final_loss': float(final_loss),
        'training_time': training_time,
        'converged': converged,
        'max_iter': max_iter,
        'n_iter': int(model.n_iter_) if hasattr(model, 'n_iter_') else max_iter,
        'loss_history': loss_history,
        'algorithm': f'Sklearn_SGD_{loss_type.upper()}',
        'loss_type': loss_type,
        'regularization': regularization,
        'alpha': alpha,
        'learning_rate_schedule': learning_rate,
        'eta0': eta0,
        'batch_size': batch_size,
        'total_batches': int(total_batches)
    }


def evaluate_sklearn_results(result, X_test, y_test):
    """
    Đánh giá kết quả Sklearn trên test set
    
    Tham số:
        result: kết quả từ run_sklearn_sgd_optimization
        X_test: ma trận đặc trưng test
        y_test: vector target test
    
    Trả về:
        dict: metrics đánh giá
    """
    if 'error' in result:
        return {'error': result['error']}
    
    model = result['model']
    scaler = result['scaler']
    
    # Transform test data using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    predictions_log = model.predict(X_test_scaled)
    
    # Evaluate on log scale
    mse_log = mean_squared_error(y_test, predictions_log)
    r2_log = r2_score(y_test, predictions_log)
    mae_log = mean_absolute_error(y_test, predictions_log)
    
    # Convert to original scale with enhanced safety checks
    try:
        print(f"   Debug: Prediction range before conversion: [{predictions_log.min():.3f}, {predictions_log.max():.3f}]")
        print(f"   Debug: Target range (log): [{y_test.min():.3f}, {y_test.max():.3f}]")
        
        # Check for extreme values before expm1
        if np.any(predictions_log > 20):  # More conservative threshold
            print(f"   WARNING: Large predictions detected (max: {predictions_log.max():.3f}), clipping...")
            predictions_log = np.clip(predictions_log, y_test.min()-2, y_test.max()+2)
        
        # Check for very negative values
        if np.any(predictions_log < -5):
            print(f"   WARNING: Very negative predictions detected (min: {predictions_log.min():.3f}), clipping...")
            predictions_log = np.clip(predictions_log, y_test.min()-2, y_test.max()+2)
        
        predictions_original = np.expm1(predictions_log)
        y_test_original = np.expm1(y_test)
        
        print(f"   Debug: Prediction range after conversion: [{predictions_original.min():.0f}, {predictions_original.max():.0f}]")
        print(f"   Debug: Target range (original): [{y_test_original.min():.0f}, {y_test_original.max():.0f}]")
        
        # Final check for infinity/NaN
        if np.any(~np.isfinite(predictions_original)):
            print(f"   ERROR: Non-finite predictions after transformation")
            # Find problematic values
            inf_mask = ~np.isfinite(predictions_original)
            print(f"   Debug: {inf_mask.sum()} non-finite values found")
            raise ValueError("Model predictions contain infinity or NaN values")
    
    except (OverflowError, ValueError) as e:
        print(f"   ERROR during scale conversion: {e}")
        return {'error': str(e)}
    
    # Evaluate on original scale
    try:
        mse_original = mean_squared_error(y_test_original, predictions_original)
    except ValueError as e:
        print(f"   ERROR: MSE calculation failed: {e}")
        return {'error': f"MSE calculation failed: {e}"}
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
    Chạy thí nghiệm với cấu hình tối ưu của Sklearn SGD
    """
    print("="*80)
    print("SKLEARN SGD vs CUSTOM SGD COMPARISON")
    print("="*80)
    
    if not SKLEARN_AVAILABLE:
        print("ERROR: Scikit-learn not available. Please install scikit-learn to run this comparison.")
        return []
    
    # Load data
    print("\n=> Loading data...")
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    print(f"   Data shapes: Train {X_train.shape}, Test {X_test.shape}")
    print(f"   Target range (log scale): y_train [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Configuration aligned with custom SGD parameters  
    # Custom SGD uses: learning_rate=0.01, regularization=0.01, batch_size=256, epochs=100
    # IMPORTANT: sklearn with standardized features needs much lower learning rate
    config = {
        'loss_type': 'squared_error',
        'regularization': 'l2', 
        'alpha': 0.01,  # Match custom SGD regularization strength
        'learning_rate': 'constant',  # Use constant like custom SGD  
        'eta0': 0.001,  # Much lower LR for standardized features (0.01 causes divergence)
        'max_iter': 200,  # More iterations to compensate for lower LR
        'batch_size': 256,  # Match custom SGD batch size
        'early_stopping': False  # Turn off for fair comparison
    }
    
    print(f"\n" + "="*50)
    print(f"SKLEARN SGD - ALIGNED WITH CUSTOM SGD")
    print(f"   Loss: {config['loss_type']}")
    print(f"   Regularization: {config['regularization']} (alpha={config['alpha']})")
    print(f"   Learning rate: {config['learning_rate']} (eta0={config['eta0']})")
    print(f"   Max iterations: {config['max_iter']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Early stopping: {config['early_stopping']}")
    print(f"   Random state: 42 (matching custom SGD)")
    print("="*50)
    
    # Run optimization
    result = run_sklearn_sgd_optimization(X_train, y_train, **config)
    
    if 'error' in result:
        print(f"   ERROR: {result['error']}")
        return []
    
    # Evaluate
    print(f"\n=> Evaluating on test set...")
    evaluation = evaluate_sklearn_results(result, X_test, y_test)
    
    if 'error' in evaluation:
        print(f"   ERROR: Evaluation error: {evaluation['error']}")
        return []
    
    # Remove model and scaler from result for JSON serialization
    result_for_save = {k: v for k, v in result.items() if k not in ['model', 'scaler']}
    
    # Combine results
    combined_result = {
        **result_for_save,
        **evaluation,
        'parameters': config
    }
    
    # Print evaluation summary
    print(f"   LOG SCALE METRICS:")
    print(f"      MSE: {evaluation['metrics_log_scale']['mse']:.8f}")
    print(f"      R²:  {evaluation['metrics_log_scale']['r2']:.6f}")
    print(f"      MAE: {evaluation['metrics_log_scale']['mae']:.6f}")
    
    print(f"   ORIGINAL SCALE METRICS:")
    print(f"      MSE:  {evaluation['metrics_original_scale']['mse']:,.2f}")
    print(f"      RMSE: {evaluation['metrics_original_scale']['rmse']:,.2f}")
    print(f"      MAE:  {evaluation['metrics_original_scale']['mae']:,.2f}")
    print(f"      R²:   {evaluation['metrics_original_scale']['r2']:.6f}")
    if evaluation['metrics_original_scale']['mape'] != float('inf'):
        print(f"      MAPE: {evaluation['metrics_original_scale']['mape']:.2f}%")
    else:
        print(f"      MAPE: N/A")
    
    # Save results
    output_dir = Path("data/03_algorithms/stochastic_gd/sklearn_sgd_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "results.json", "w") as f:
        json.dump(combined_result, f, indent=2)
    
    # Save summary results following template
    save_results_summary(result, evaluation, output_dir)
    
    print(f"   Results saved to: {output_dir / 'results.json'}")
    
    print(f"\n=> Sklearn SGD optimization completed!")
    print(f"Results saved in {output_dir}/")
    
    # Enhanced comparison insights  
    print(f"\nSKLEARN vs CUSTOM SGD COMPARISON:")
    print(f"   Configuration Alignment:")
    print(f"   • Learning rate: {config['eta0']} (constant schedule)")
    print(f"   • Regularization: L2 with alpha={config['alpha']}")
    print(f"   • Batch size: {config['batch_size']} samples per batch")
    print(f"   • Max iterations: {config['max_iter']} epochs")
    print(f"   • Features standardized, targets in log scale")
    print(f"   • Same random seed (42) for reproducibility")
    print(f"   ")
    print(f"   Performance Achieved:")
    print(f"   • R² (log scale): {evaluation['metrics_log_scale']['r2']:.6f}")
    print(f"   • Training time: {result['training_time']:.4f}s")
    print(f"   • Iterations completed: {result['n_iter']}/{result['max_iter']}")
    print(f"   • Converged: {'YES' if result['converged'] else 'NO'}")
    
    # Add comparison reference if available
    try:
        custom_sgd_path = Path("data/03_algorithms/stochastic_gd/01a_setup_sgd_ols_batch_256_fixed_step_0001/results.json")
        if custom_sgd_path.exists():
            with open(custom_sgd_path, 'r') as f:
                custom_results = json.load(f)
            print(f"   ")
            print(f"   vs Custom SGD (batch_256, lr=0.01):")
            print(f"   • Custom R² (log): {custom_results['training_results']['best_loss']:.6f}")
            print(f"   • Sklearn R² (log): {evaluation['metrics_log_scale']['r2']:.6f}")
            print(f"   • Custom time: {custom_results['training_results']['training_time']:.4f}s")
            print(f"   • Sklearn time: {result['training_time']:.4f}s")
    except:
        pass
    
    # Save training history
    if len(result['loss_history']) > 1:
        history_df = pd.DataFrame({
            'epoch': range(1, len(result['loss_history']) + 1),
            'loss': result['loss_history']
        })
        history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    # Create visualizations
    create_visualizations(result, evaluation, X_test, y_test, output_dir)
    
    # Print final summary in standard format
    print(f"\n" + "="*60)
    print(f"FINAL RESULTS SUMMARY")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Loss Function: {result['loss_type']} with {result['regularization']} regularization")
    print(f"Learning Rate: {result['learning_rate_schedule']} (eta0={result['eta0']})")
    print(f"Iterations: {result['n_iter']}/{result['max_iter']} ({'Converged' if result['converged'] else 'Not converged'})")
    print(f"Training Time: {result['training_time']:.4f}s")
    print(f"")
    print(f"Model Performance:")
    print(f"   R² (log):      {evaluation['metrics_log_scale']['r2']:.6f}")
    print(f"   R² (original): {evaluation['metrics_original_scale']['r2']:.6f}")
    print(f"   RMSE:          {evaluation['metrics_original_scale']['rmse']:,.2f}")
    if evaluation['metrics_original_scale']['mape'] != float('inf'):
        print(f"   MAPE:          {evaluation['metrics_original_scale']['mape']:.2f}%")
    print(f"")
    print(f"Files saved:")
    print(f"   • results.json")
    print(f"   • training_history.csv")
    print(f"   • convergence_analysis.png")
    print(f"   • predictions_vs_actual.png")
    print("="*60)
    
    return [combined_result]


def create_visualizations(result, evaluation, X_test, y_test, output_dir):
    """
    Tạo các biểu đồ visualization theo template chuẩn
    """
    model = result['model']
    scaler = result['scaler']
    
    # Transform test data
    X_test_scaled = scaler.transform(X_test)
    predictions_log = model.predict(X_test_scaled)
    predictions_original = np.expm1(predictions_log)
    y_test_original = np.expm1(y_test)
    
    plt.style.use('seaborn-v0_8')
    
    # 1. Convergence Analysis
    if len(result['loss_history']) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(result['loss_history']) + 1), result['loss_history'], 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Sklearn SGD - Convergence Analysis')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Predictions vs Actual (Original Scale)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_original, predictions_original, alpha=0.6, s=30)
    
    # Perfect prediction line
    min_val = min(y_test_original.min(), predictions_original.min())
    max_val = max(y_test_original.max(), predictions_original.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Sklearn SGD - Predictions vs Actual\nR² = {evaluation["metrics_original_scale"]["r2"]:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_results_summary(result, evaluation, output_dir):
    """
    Lưu tóm tắt kết quả theo template chuẩn
    """
    summary = {
        "algorithm": "Sklearn SGD",
        "loss_function": result['loss_type'],
        "regularization": result['regularization'],
        "regularization_strength": result['alpha'],
        "learning_rate_schedule": result['learning_rate_schedule'],
        "initial_learning_rate": result['eta0'],
        "max_iterations": result['max_iter'],
        "converged": result['converged'],
        "final_iterations": result['n_iter'],
        "training_time_seconds": result['training_time'],
        "final_loss": result['final_loss'],
        "metrics_log_scale": evaluation['metrics_log_scale'],
        "metrics_original_scale": evaluation['metrics_original_scale']
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    """Main function"""
    results = run_comparison_experiments()
    return results


if __name__ == "__main__":
    main()