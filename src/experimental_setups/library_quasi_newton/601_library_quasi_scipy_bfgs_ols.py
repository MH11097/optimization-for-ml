#!/usr/bin/env python3
"""
601 - Library Quasi-Newton: SciPy BFGS with OLS
Sử dụng SciPy BFGS để so sánh với custom quasi-Newton implementation.

Expected naming pattern: 6XX_library_quasi_{library}_{variant}_{params}.py
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
from src.optimization.external.library_quasi_newton import (
    create_scipy_bfgs_optimizer, 
    SCIPY_AVAILABLE
)

def main():
    if not SCIPY_AVAILABLE:
        print("ERROR: SciPy is not available. Please install: pip install scipy")
        return
    
    print("=" * 80)
    print("601 - LIBRARY QUASI-NEWTON: SCIPY BFGS OLS")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Thiết lập experiment parameters
    experiment_params = {
        'experiment_id': '601_library_quasi_scipy_bfgs_ols',
        'library_name': 'scipy',
        'algorithm_name': 'BFGS',
        'loss_type': 'ols',
        'regularization': 0.0,  # OLS không có regularization
        'max_iterations': 10000,
        'convergence_tolerance': 1e-6,
        'random_state': 42,
        
        # SciPy BFGS specific parameters
        'method': 'BFGS',
        'gtol': 1e-6,  # Gradient tolerance
        'norm': float('inf'),  # Infinity norm
        'eps': 1.4901161193847656e-08,  # Step size for finite differences
        
        # Experiment metadata
        'experiment_type': 'library_comparison',
        'baseline_comparison': 'custom_quasi_newton'
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
    print("\\nCreating SciPy BFGS optimizer...")
    optimizer = create_scipy_bfgs_optimizer(
        method=experiment_params['method'],
        loss_type=experiment_params['loss_type'],
        regularization=experiment_params['regularization'],
        max_iterations=experiment_params['max_iterations'],
        convergence_tolerance=experiment_params['convergence_tolerance'],
        random_state=experiment_params['random_state'],
        gtol=experiment_params['gtol'],
        norm=experiment_params['norm'],
        eps=experiment_params['eps']
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
        output_dir = project_root / "data" / "03_algorithms" / "library_quasi_newton" / experiment_params['experiment_id']
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
        print(f"  - Iterations: {results['training_results']['final_iteration']}")
        print(f"  - Test RMSE: {ml_metrics['rmse']:.6f}")
        print(f"  - Test R²: {ml_metrics['r2']:.6f}")
        
        # Show algorithm-specific info
        if 'scipy_result' in results['algorithm_specific']:
            scipy_info = results['algorithm_specific']['scipy_result']
            print(f"  - SciPy success: {scipy_info.get('success', 'Unknown')}")
            print(f"  - Function evaluations: {scipy_info.get('nfev', 'Unknown')}")
            print(f"  - Gradient evaluations: {scipy_info.get('njev', 'Unknown')}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\\nERROR during training: {str(e)}")
        print("Experiment failed!")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()