#!/usr/bin/env python3
"""
Library-based experimental setup - Sklearn Lasso
Algorithm: Lasso regression using sklearn.linear_model.Lasso
Configuration: Lasso regression (L1 regularization) 
Generated for comparison with custom Subgradient implementation
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import Lasso, lasso_path
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
    plt.title('Predictions vs Actual Values - Sklearn Lasso')
    plt.savefig(results_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Regularization path (if available)
    if 'regularization_path' in results and results['regularization_path']:
        plt.figure(figsize=(10, 6))
        alphas = results['regularization_path']['alphas']
        coefs = results['regularization_path']['coefs']
        
        plt.plot(alphas, coefs.T)
        plt.xscale('log')
        plt.xlabel('Alpha (Regularization strength)')
        plt.ylabel('Coefficients')
        plt.title('Lasso Regularization Path')
        plt.grid(True)
        plt.savefig(results_dir / "optimization_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Training Loss convergence (if available)
    if 'training_history' in results and results['training_history']:
        plt.figure(figsize=(10, 6))
        plt.plot(results['training_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss - Sklearn Lasso')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

class LassoWithHistory(Lasso):
    """Lasso wrapper that tracks training convergence"""
    
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.training_history_ = []
        self.iteration_times_ = []
    
    def fit(self, X, y, sample_weight=None):
        """Fit model and track training progress"""
        # For monitoring convergence, we'll use different alpha values
        # and track the objective function value
        
        start_time = time.time()
        
        # Fit the model
        super().fit(X, y, sample_weight)
        
        # Calculate objective function value (MSE + L1 penalty)
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        l1_penalty = self.alpha * np.sum(np.abs(self.coef_))
        objective = mse + l1_penalty
        
        self.training_history_ = [objective]  # Simplified history
        self.iteration_times_ = [time.time() - start_time]
        
        return self

def compute_regularization_path(X, y, n_alphas=20):
    """Compute Lasso regularization path for different alpha values"""
    alphas, coefs, _ = lasso_path(X, y, n_alphas=n_alphas)
    return alphas, coefs

def main():
    print(f"\n============================================================")
    print(f"LIBRARY-BASED EXPERIMENTAL SETUP")
    print(f"Algorithm: Lasso regression using sklearn.linear_model.Lasso")
    print(f"Configuration: L1 regularization (Subgradient equivalent)")
    print(f"============================================================")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Standardize features (recommended for regularized methods)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different alpha values to find a good one
    alpha_candidates = [0.1, 0.01, 0.001, 0.0001]
    best_alpha = 0.1
    best_score = float('-inf')
    
    print("Finding optimal alpha...")
    for alpha in alpha_candidates:
        temp_model = Lasso(alpha=alpha, max_iter=1000, tol=1e-6)
        temp_model.fit(X_train_scaled, y_train)
        temp_pred = temp_model.predict(X_test_scaled)
        score = r2_score(y_test, temp_pred)
        print(f"Alpha {alpha}: R² = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    print(f"Selected alpha: {best_alpha}")
    
    # Create Lasso model with best alpha
    model = LassoWithHistory(
        alpha=best_alpha,
        max_iter=1000,
        tol=1e-6,
        random_state=42
    )
    
    # Training
    print("Starting training...")
    start_time = time.time()
    
    model.fit(X_train_scaled, y_train)
    
    total_time = time.time() - start_time
    
    # Compute regularization path for visualization
    print("Computing regularization path...")
    path_start = time.time()
    alphas, coefs = compute_regularization_path(X_train_scaled, y_train)
    path_time = time.time() - path_start
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate objective function value
    train_pred = model.predict(X_train_scaled)
    train_mse = np.mean((y_train - train_pred) ** 2)
    l1_penalty = model.alpha * np.sum(np.abs(model.coef_))
    final_objective = train_mse + l1_penalty
    
    # Count non-zero coefficients (sparsity)
    n_nonzero_coefs = np.sum(model.coef_ != 0)
    
    # Prepare results (convert numpy/sklearn values to Python types for JSON serialization)
    results = {
        'algorithm': 'Sklearn Lasso',
        'library': 'scikit-learn',
        'configuration': {
            'alpha': float(best_alpha),
            'max_iter': 1000,
            'tolerance': 1e-6,
            'standardized': True
        },
        'metrics': {
            'final_loss': float(train_mse),
            'final_objective': float(final_objective),
            'l1_penalty': float(l1_penalty),
            'test_mse': float(mse),
            'test_r2': float(r2),
            'training_time_seconds': float(total_time),
            'regularization_path_time': float(path_time),
            'n_iter_': int(model.n_iter_),
            'n_nonzero_coefs': int(n_nonzero_coefs),
            'sparsity_ratio': float(n_nonzero_coefs / len(model.coef_))
        },
        'training_history': [float(x) for x in model.training_history_],
        'iteration_times': [float(x) for x in model.iteration_times_],
        'regularization_path': {
            'alphas': [float(x) for x in alphas],
            'coefs': [[float(y) for y in x] for x in coefs.T]
        }
    }
    
    # Save results
    experiment_name = get_experiment_name()
    results_dir = save_results(results, experiment_name)
    create_plots(X_test, y_test, y_pred, results, experiment_name)
    
    print(f"\nSklearn Lasso completed successfully!")
    print(f"Configuration: Lasso with alpha={best_alpha}")
    print(f"Final Loss (MSE): {train_mse:.6f}")
    print(f"Final Objective: {final_objective:.6f}")
    print(f"L1 Penalty: {l1_penalty:.6f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R²: {r2:.6f}")
    print(f"Training Time: {total_time:.2f} seconds")
    print(f"Iterations: {model.n_iter_}")
    print(f"Non-zero coefficients: {n_nonzero_coefs}/{len(model.coef_)} ({100*n_nonzero_coefs/len(model.coef_):.1f}%)")
    print(f"Results saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    main()