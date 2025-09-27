#!/usr/bin/env python3
"""
Library-based experimental setup - PyTorch GD
Algorithm: Gradient Descent using PyTorch SGD
Configuration: PyTorch SGD with full batch (simulating GD)
Generated for comparison with custom GD implementation
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
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

class LinearModel(nn.Module):
    """Simple linear regression model for PyTorch"""
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def save_results(results, experiment_name):
    """Lưu kết quả experiment"""
    results_dir = Path(f"data/03_algorithms/library/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results.json
    with open(results_dir / "results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save training history
    history_df = []
    for i, loss in enumerate(results.get('training_history', [])):
        history_df.append({
            'iteration': i + 1,
            'loss': loss,
            'time': results.get('iteration_times', [0] * len(results.get('training_history', [])))[i]
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
    plt.title('Predictions vs Actual Values - PyTorch GD')
    plt.savefig(results_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training Loss
    if 'training_history' in results and results['training_history']:
        plt.figure(figsize=(10, 6))
        plt.plot(results['training_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss - PyTorch GD')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(results_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print(f"\n============================================================")
    print(f"LIBRARY-BASED EXPERIMENTAL SETUP")
    print(f"Algorithm: Gradient Descent using PyTorch SGD")
    print(f"Configuration: Full batch SGD (simulating GD)")
    print(f"============================================================")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
    # Create model
    input_dim = X_train.shape[1]
    model = LinearModel(input_dim)
    
    # Use SGD with full batch size (simulating GD)
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    # Training
    print("Starting training...")
    start_time = time.time()
    
    training_history = []
    iteration_times = []
    max_iterations = 1000
    
    model.train()
    for epoch in range(max_iterations):
        iter_start = time.time()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        iter_time = time.time() - iter_start
        training_history.append(loss.item())
        iteration_times.append(iter_time)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
        # Early stopping
        if epoch > 10 and abs(training_history[-1] - training_history[-2]) < 1e-8:
            print(f"Converged at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy().flatten()
    
    # Calculate metrics
    final_loss = training_history[-1]
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Prepare results (convert numpy/torch values to Python types for JSON serialization)
    results = {
        'algorithm': 'PyTorch SGD (simulating GD)',
        'library': 'PyTorch',
        'configuration': {
            'learning_rate': 0.01,
            'batch_size': len(X_train),
            'optimizer': 'SGD',
            'max_iterations': max_iterations
        },
        'metrics': {
            'final_loss': float(final_loss),
            'test_mse': float(mse),
            'test_r2': float(r2),
            'training_time_seconds': float(total_time),
            'total_iterations': len(training_history)
        },
        'training_history': [float(x) for x in training_history],
        'iteration_times': [float(x) for x in iteration_times]
    }
    
    # Save results
    experiment_name = get_experiment_name()
    results_dir = save_results(results, experiment_name)
    create_plots(X_test, y_test, y_pred, results, experiment_name)
    
    print(f"\nPyTorch GD completed successfully!")
    print(f"Configuration: PyTorch SGD with full batch")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R²: {r2:.6f}")
    print(f"Training Time: {total_time:.2f} seconds")
    print(f"Total Iterations: {len(training_history)}")
    print(f"Results saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    main()