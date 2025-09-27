#!/usr/bin/env python3
"""
Library-based experimental setup - Sklearn SGD
Algorithm: Stochastic Gradient Descent using sklearn.linear_model.SGDRegressor
Configuration: SGD with default sklearn parameters
Generated for comparison with custom SGD implementation
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import SGDRegressor
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
    plt.title('Predictions vs Actual Values - Sklearn SGD')
    plt.savefig(results_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training Loss (if available)
    if 'training_history' in results and results['training_history']:
        plt.figure(figsize=(10, 6))
        plt.plot(results['training_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss - Sklearn SGD')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

class SGDWithHistory(SGDRegressor):
    """SGDRegressor wrapper that tracks training history"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history_ = []
        self.iteration_times_ = []
    
    def fit(self, X, y, sample_weight=None):
        """Fit model and track training progress"""
        start_time = time.time()
        
        # For tracking loss, we need to use partial_fit in a loop
        self.loss_history_ = []
        self.iteration_times_ = []
        
        # Initialize the model with first sample
        iter_start = time.time()
        super().partial_fit(X[:1], y[:1])
        iter_time = time.time() - iter_start
        self.iteration_times_.append(iter_time)
        
        # Calculate initial loss
        y_pred = self.predict(X)
        initial_loss = mean_squared_error(y, y_pred)
        self.loss_history_.append(initial_loss)
        
        # Continue training with all data in mini-batches
        n_samples = len(X)
        batch_size = min(100, n_samples // 10)  # Adaptive batch size
        
        for epoch in range(self.max_iter):
            epoch_start = time.time()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                self.partial_fit(X_batch, y_batch)
            
            # Calculate loss after each epoch
            y_pred = self.predict(X)
            epoch_loss = mean_squared_error(y, y_pred)
            self.loss_history_.append(epoch_loss)
            
            epoch_time = time.time() - epoch_start
            self.iteration_times_.append(epoch_time)
            
            # Early stopping check
            if len(self.loss_history_) > 10:
                recent_losses = self.loss_history_[-5:]
                if max(recent_losses) - min(recent_losses) < 1e-6:
                    print(f"Converged at epoch {epoch}")
                    break
        
        return self

def main():
    print(f"\n============================================================")
    print(f"LIBRARY-BASED EXPERIMENTAL SETUP")
    print(f"Algorithm: Stochastic Gradient Descent using sklearn")
    print(f"Configuration: SGDRegressor with default parameters")
    print(f"============================================================")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Standardize features (recommended for SGD)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create SGD model
    model = SGDWithHistory(
        loss='squared_error',
        learning_rate='constant',
        eta0=0.01,
        max_iter=1000,
        tol=1e-6,
        random_state=42
    )
    
    # Training
    print("Starting training...")
    start_time = time.time()
    
    model.fit(X_train_scaled, y_train)
    
    total_time = time.time() - start_time
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    final_loss = model.loss_history_[-1] if model.loss_history_ else None
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Prepare results
    results = {
        'algorithm': 'Sklearn SGDRegressor',
        'library': 'scikit-learn',
        'configuration': {
            'loss': 'squared_error',
            'learning_rate': 'constant',
            'eta0': 0.01,
            'max_iter': 1000,
            'tol': 1e-6,
            'standardized': True
        },
        'metrics': {
            'final_loss': final_loss,
            'test_mse': mse,
            'test_r2': r2,
            'training_time_seconds': total_time,
            'total_iterations': len(model.loss_history_),
            'n_iter_': getattr(model, 'n_iter_', None)
        },
        'training_history': model.loss_history_,
        'iteration_times': model.iteration_times_
    }
    
    # Save results
    experiment_name = get_experiment_name()
    results_dir = save_results(results, experiment_name)
    create_plots(X_test, y_test, y_pred, results, experiment_name)
    
    print(f"\nSklearn SGD completed successfully!")
    print(f"Configuration: SGDRegressor with standardized features")
    print(f"Final Loss: {final_loss:.6f}" if final_loss else "Final Loss: N/A")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R²: {r2:.6f}")
    print(f"Training Time: {total_time:.2f} seconds")
    print(f"Total Iterations: {len(model.loss_history_)}")
    print(f"Results saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    main()