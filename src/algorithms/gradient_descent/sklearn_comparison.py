#!/usr/bin/env python3
"""
Sklearn Comparison Test - So sánh kết quả với sklearn LinearRegression
Để kiểm tra tính đúng đắn của implementation
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_process_utils import load_du_lieu


def main():
    """So sánh sklearn LinearRegression với implementation hiện tại"""
    print("="*80)
    print("🔬 SKLEARN COMPARISON TEST")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    print(f"   Data shapes: Train {X_train.shape}, Test {X_test.shape}")
    print(f"   Target range (log scale): y_train [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"   Target range (log scale): y_test [{y_test.min():.3f}, {y_test.max():.3f}]")
    
    # Train sklearn LinearRegression
    print("\n🤖 Training sklearn LinearRegression...")
    start_time = time.time()
    
    sklearn_model = LinearRegression(fit_intercept=True)
    sklearn_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Learned bias (intercept): {sklearn_model.intercept_:.6f}")
    print(f"   Number of features: {len(sklearn_model.coef_)}")
    print(f"   Weight range: [{sklearn_model.coef_.min():.6f}, {sklearn_model.coef_.max():.6f}]")
    
    # Make predictions
    print("\n🎯 Making predictions...")
    sklearn_predictions = sklearn_model.predict(X_test)
    
    print(f"   Prediction range (log scale): [{sklearn_predictions.min():.3f}, {sklearn_predictions.max():.3f}]")
    
    # Evaluate on log scale
    print("\n📊 EVALUATION ON LOG SCALE:")
    mse_log = mean_squared_error(y_test, sklearn_predictions)
    r2_log = r2_score(y_test, sklearn_predictions)
    mae_log = mean_absolute_error(y_test, sklearn_predictions)
    
    print(f"   MSE (log scale): {mse_log:.6f}")
    print(f"   R² (log scale):  {r2_log:.6f}")
    print(f"   MAE (log scale): {mae_log:.6f}")
    
    # Convert to original scale for evaluation
    print("\n🔄 Converting to original scale for evaluation...")
    predictions_original = np.expm1(sklearn_predictions)  # inverse of log1p
    y_test_original = np.expm1(y_test)                    # inverse of log1p
    
    print(f"   Original predictions range: [{predictions_original.min():.0f}, {predictions_original.max():.0f}]")
    print(f"   Original targets range: [{y_test_original.min():.0f}, {y_test_original.max():.0f}]")
    
    # Evaluate on original scale
    print("\n📊 EVALUATION ON ORIGINAL SCALE:")
    mse_original = mean_squared_error(y_test_original, predictions_original)
    r2_original = r2_score(y_test_original, predictions_original)
    mae_original = mean_absolute_error(y_test_original, predictions_original)
    rmse_original = np.sqrt(mse_original)
    
    print(f"   MSE:  {mse_original:,.2f}")
    print(f"   RMSE: {rmse_original:,.2f}")
    print(f"   MAE:  {mae_original:,.2f}")
    print(f"   R²:   {r2_original:.6f}")
    
    # MAPE calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs((y_test_original - predictions_original) / y_test_original)
        valid_errors = percentage_errors[np.isfinite(percentage_errors)]
        mape = np.mean(valid_errors) * 100 if len(valid_errors) > 0 else float('inf')
    
    print(f"   MAPE: {mape:.2f}%")
    
    # Model assessment
    print("\n🏆 MODEL QUALITY ASSESSMENT:")
    if r2_original >= 0.8:
        r2_assessment = "EXCELLENT (R² ≥ 0.8)"
    elif r2_original >= 0.6:
        r2_assessment = "GOOD (R² ≥ 0.6)"
    elif r2_original >= 0.5:
        r2_assessment = "ACCEPTABLE (R² ≥ 0.5)"
    else:
        r2_assessment = "POOR (R² < 0.5)"
    
    if mape <= 10:
        mape_assessment = "EXCELLENT (MAPE ≤ 10%)"
    elif mape <= 20:
        mape_assessment = "GOOD (MAPE ≤ 20%)"
    else:
        mape_assessment = "NEEDS IMPROVEMENT (MAPE > 20%)"
    
    print(f"   📈 R² Assessment: {r2_assessment}")
    print(f"   📊 MAPE Assessment: {mape_assessment}")
    
    # Save results for comparison
    results = {
        'algorithm': 'sklearn_LinearRegression',
        'training_time': training_time,
        'bias': float(sklearn_model.intercept_),
        'n_features': len(sklearn_model.coef_),
        'predictions_log_min': float(sklearn_predictions.min()),
        'predictions_log_max': float(sklearn_predictions.max()),
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
        },
        'assessment': {
            'r2': r2_assessment,
            'mape': mape_assessment
        }
    }
    
    # Save to file for comparison
    output_dir = Path("data/03_algorithms/gradient_descent/sklearn_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "sklearn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir / 'sklearn_results.json'}")
    
    print("\n" + "="*80)
    print("✅ SKLEARN COMPARISON COMPLETED!")
    print("="*80)
    print("\n🔍 KEY INSIGHTS:")
    print(f"   • Sklearn bias: {sklearn_model.intercept_:.6f}")
    print(f"   • Log scale R²: {r2_log:.6f}")
    print(f"   • Original scale R²: {r2_original:.6f}")
    print(f"   • Training time: {training_time:.2f}s")
    print("\n📋 Use these results as benchmark for gradient descent implementation!")


if __name__ == "__main__":
    main()