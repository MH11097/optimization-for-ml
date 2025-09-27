#!/usr/bin/env python3
"""
Library-based experimental setup - Scipy L-BFGS-B
Algorithm: L-BFGS-B method using scipy.optimize.minimize
Configuration: L-BFGS-B for linear regression optimization
Generated for comparison with custom BFGS implementation
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

# Add the src directory to path
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.data_process_utils import load_du_lieu

def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def save_results(results, experiment_name):
    """Lưu kết quả experiment"""
    results_dir = Path(f"data/03_algorithms/library/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results.json
    with open(results_dir / "results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save training history if available
    if 'training_history' in results and results['training_history']:
        history_df = []
        iteration_times = results.get('iteration_times', [])
        for i, loss in enumerate(results['training_history']):
            # Handle case where iteration_times might be shorter than training_history
            time_value = iteration_times[i] if i < len(iteration_times) else 0.0
            history_df.append({
                'iteration': i + 1,
                'loss': loss,
                'time': time_value
            })
        
        import pandas as pd
        pd.DataFrame(history_df).to_csv(results_dir / "training_history.csv", index=False)
    
    return results_dir

def create_plots(X_test, y_test, y_pred, results, experiment_name):
    """Tạo các plots cho kết quả"""
    results_dir = Path(f"data/03_algorithms/library/{experiment_name}")
    
    # Plot 1: Predictions vs Actual
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values - Scipy L-BFGS-B')
    plt.savefig(results_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training Loss (if available)
    if 'training_history' in results and results['training_history']:
        plt.figure(figsize=(10, 6))
        plt.plot(results['training_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss - Scipy L-BFGS-B')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

class LBFGSRegressor:
    """L-BFGS-B wrapper for linear regression using scipy.optimize.minimize"""
    
    def __init__(self, tol=1e-6, maxiter=1000):
        self.tol = tol
        self.maxiter = maxiter
        self.coef_ = None
        self.intercept_ = None
        self.training_history_ = []
        self.iteration_times_ = []
        self.n_iter_ = 0
        
    def _objective_function(self, params, X, y):
        """Objective function: Mean Squared Error"""
        predictions = X @ params
        mse = np.mean((y - predictions) ** 2)
        self.training_history_.append(mse)
        return mse
    
    def _gradient(self, params, X, y):
        """Gradient of the objective function"""
        predictions = X @ params
        gradient = -2 * X.T @ (y - predictions) / len(y)
        return gradient
    
    def fit(self, X, y):
        """Fit the model using L-BFGS-B method"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize parameters
        initial_params = np.zeros(X_with_intercept.shape[1])
        
        # Track iteration times
        self.training_history_ = []
        self.iteration_times_ = []
        
        def callback(params):
            """Callback function to track iterations"""
            iter_start = time.time()
            # Function evaluation is already done in objective function
            iter_time = time.time() - iter_start
            self.iteration_times_.append(iter_time)
        
        # Optimize using L-BFGS-B
        start_time = time.time()
        
        result = minimize(
            fun=self._objective_function,
            x0=initial_params,
            args=(X_with_intercept, y),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.maxiter, 'ftol': self.tol, 'gtol': self.tol},
            callback=callback
        )
        
        # Extract parameters
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]
        self.n_iter_ = result.nit
        self.success_ = result.success
        self.final_loss_ = result.fun
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return X @ self.coef_ + self.intercept_

def main():
    print(f"\n============================================================")
    print(f"LIBRARY-BASED EXPERIMENTAL SETUP")
    print(f"Algorithm: L-BFGS-B method using scipy.optimize.minimize")
    print(f"Configuration: L-BFGS-B for linear regression")
    print(f"============================================================")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Standardize features (recommended for BFGS methods)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create L-BFGS-B model
    model = LBFGSRegressor(tol=1e-6, maxiter=1000)
    
    # Training
    print("Starting training...")
    start_time = time.time()
    
    model.fit(X_train_scaled, y_train)
    
    total_time = time.time() - start_time
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    final_loss = model.final_loss_
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Prepare results
    results = {
        'algorithm': 'Scipy L-BFGS-B',
        'library': 'scipy',
        'configuration': {
            'method': 'L-BFGS-B',
            'tolerance': 1e-6,
            'max_iterations': 1000,
            'standardized': True
        },
        'metrics': {
            'final_loss': final_loss,
            'test_mse': mse,
            'test_r2': r2,
            'training_time_seconds': total_time,
            'total_iterations': model.n_iter_,
            'success': model.success_
        },
        'training_history': model.training_history_,
        'iteration_times': model.iteration_times_
    }
    
    # Save results
    experiment_name = get_experiment_name()
    results_dir = save_results(results, experiment_name)
    create_plots(X_test, y_test, y_pred, results, experiment_name)
    
    print(f"\nScipy L-BFGS-B completed successfully!")
    print(f"Configuration: L-BFGS-B with standardized features")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R²: {r2:.6f}")
    print(f"Training Time: {total_time:.2f} seconds")
    print(f"Total Iterations: {model.n_iter_}")
    print(f"Convergence: {'Success' if model.success_ else 'Failed'}")
    print(f"Results saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    main()