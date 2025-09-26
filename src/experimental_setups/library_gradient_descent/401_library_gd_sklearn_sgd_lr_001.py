#!/usr/bin/env python3
"""
401 - Library Gradient Descent: sklearn SGD with LR 0.001
Sử dụng sklearn SGDRegressor để so sánh với custom gradient descent implementation.

Expected naming pattern: 4XX_library_gd_{library}_{variant}_{params}.py
"""

import sys
import os
from pathlib import Path
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import configuration and utilities
from config import load_config
from utils.data_utils import tach_du_lieu_train_test
from utils.optimization_utils import luu_ket_qua_experiment
from src.optimization.external.library_gradient_descent import (
    create_sklearn_sgd_optimizer, 
    SKLEARN_AVAILABLE
)

def main():

    if not SKLEARN_AVAILABLE:
        print("ERROR: sklearn is not available. Please install: pip install scikit-learn")
        return
    
    print("=" * 80)
    print("401 - LIBRARY GRADIENT DESCENT: SKLEARN SGD LR 0.001")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Thiết lập experiment parameters
    experiment_params = {
        'experiment_id': '401_library_gd_sklearn_sgd_lr_001',
        'library_name': 'sklearn',
        'algorithm_name': 'SGD',
        'learning_rate': 0.001,
        'loss_type': 'ols',
        'regularization': 0.0,  # OLS không có regularization
        'max_iterations': 10000,
        'convergence_tolerance': 1e-6,
        'random_state': 42,
        
        # sklearn SGD specific parameters
        'momentum': 0.0,
        'learning_rate_schedule': 'constant',
        'early_stopping': False,
        'feature_scaling': True,  # sklearn SGD works better with scaling
        
        # Experiment metadata
        'experiment_type': 'library_comparison',
        'baseline_comparison': 'custom_gradient_descent'
    }
    
    print("\\nExperiment Parameters:")
    for key, value in experiment_params.items():
        print(f"  {key}: {value}")
    
    # Tách dữ liệu
    print("\\nLoading and splitting data...")
    X_train, X_test, y_train, y_test = tach_du_lieu_train_test(
        config.processed_data_path,
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Tạo optimizer
    print("\\nCreating sklearn SGD optimizer...")
    optimizer = create_sklearn_sgd_optimizer(
        learning_rate=experiment_params['learning_rate'],
        loss_type=experiment_params['loss_type'],
        regularization=experiment_params['regularization'],
        max_iterations=experiment_params['max_iterations'],
        convergence_tolerance=experiment_params['convergence_tolerance'],
        random_state=experiment_params['random_state'],
        momentum=experiment_params['momentum'],
        learning_rate_schedule=experiment_params['learning_rate_schedule'],
        early_stopping=experiment_params['early_stopping'],
        feature_scaling=experiment_params['feature_scaling']
    )
    
    # Training
    print("\\nStarting training...")
    try:
        training_results = optimizer.fit(X_train, y_train)
        print("Training completed successfully!")
        
        # Evaluation
        print("\\nEvaluating model...")
        ml_metrics = optimizer.evaluate(X_test, y_test)
        
        # Prepare results for saving
        results = {
            'experiment_id': experiment_params['experiment_id'],
            'algorithm': f"{experiment_params['library_name']} {experiment_params['algorithm_name']}",
            'loss_function': experiment_params['loss_type'],
            'parameters': experiment_params,
            'training_results': {
                'converged': training_results['converged'],
                'final_iteration': training_results['final_iteration'],
                'training_time': training_results['training_time'],
                'final_loss': training_results['final_loss'],
                'final_gradient_norm': training_results['final_gradient_norm'],
                'best_loss': training_results.get('best_loss', training_results['final_loss']),
                'best_iteration': training_results.get('best_iteration', training_results['final_iteration']),
                'best_gradient_norm': training_results.get('best_gradient_norm', training_results['final_gradient_norm'])
            },
            'ml_metrics': ml_metrics,
            'algorithm_specific': training_results.get('algorithm_specific', {}),
            'computational_complexity': training_results.get('computational_complexity', {})
        }
        
        # Create output directory
        output_dir = project_root / "data" / "03_algorithms" / "library_gradient_descent" / experiment_params['experiment_id']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        print(f"\\nSaving results to {output_dir}")
        luu_ket_qua_experiment(
            results, 
            training_results['loss_history'], 
            training_results['gradient_norms'],
            str(output_dir),
            create_visualizations=True
        )
        
        print("\\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("Results summary:")
        print(f"  - Converged: {'Yes' if results['training_results']['converged'] else 'No'}")
        print(f"  - Final loss: {results['training_results']['final_loss']:.6f}")
        print(f"  - Training time: {results['training_results']['training_time']:.4f}s")
        print(f"  - Test RMSE: {ml_metrics['rmse']:.6f}")
        print(f"  - Test R²: {ml_metrics['r2']:.6f}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\\nERROR during training: {str(e)}")
        print("Experiment failed!")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()