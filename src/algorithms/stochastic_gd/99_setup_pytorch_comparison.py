#!/usr/bin/env python3
"""
PyTorch SGD Comparison - So sánh kết quả với torch.optim.SGD(momentum=0)
Pure SGD không momentum là thuật toán Stochastic Gradient Descent gốc
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Please install torch to run this comparison.")

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_process_utils import load_du_lieu
from utils.optimization_utils import add_bias_column


class LinearRegressionModel(nn.Module):
    """
    Simple Linear Regression model for PyTorch
    """
    def __init__(self, n_features):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze()


def create_loss_function(loss_type='ols', regularization=0.01):
    """
    Tạo loss function cho PyTorch
    
    Tham số:
        loss_type: loại loss function ('ols', 'ridge', 'lasso')
        regularization: hệ số regularization
    
    Trả về:
        loss_func: function tính loss
    """
    base_criterion = nn.MSELoss()
    
    def loss_func(model, predictions, targets):
        # Base MSE loss
        mse_loss = base_criterion(predictions, targets)
        
        if loss_type == 'ols':
            return mse_loss
        elif loss_type == 'ridge':
            # L2 regularization (Ridge)
            l2_penalty = 0
            for param in model.parameters():
                l2_penalty += torch.sum(param ** 2)
            return mse_loss + regularization * l2_penalty
        elif loss_type == 'lasso':
            # L1 regularization (Lasso) 
            l1_penalty = 0
            for param in model.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            return mse_loss + regularization * l1_penalty
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss_func


def run_pytorch_sgd_optimization(X_train, y_train, loss_type='ols', regularization=0.01, 
                                batch_size=32, learning_rate=0.01, epochs=100):
    """
    Chạy tối ưu hóa bằng PyTorch SGD
    
    Tham số:
        X_train: ma trận đặc trưng train (n_samples, n_features)
        y_train: vector target train (n_samples,)
        loss_type: loại loss function
        regularization: hệ số regularization
        batch_size: batch size cho SGD
        learning_rate: learning rate
        epochs: số epochs
    
    Trả về:
        dict: kết quả tối ưu hóa
    """
    if not PYTORCH_AVAILABLE:
        return {
            'error': 'PyTorch not available',
            'algorithm': f'PyTorch_SGD_{loss_type.upper()}',
            'converged': False
        }
    
    print(f"\n🔬 Running PyTorch SGD optimization for {loss_type.upper()}...")
    
    # Convert to torch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    
    n_samples, n_features = X_train.shape
    
    print(f"   Data: {n_samples} samples, {n_features} features")
    print(f"   Loss type: {loss_type}")
    print(f"   Regularization: {regularization}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    
    # Create model
    model = LinearRegressionModel(n_features)
    
    # Initialize weights
    torch.manual_seed(42)
    with torch.no_grad():
        model.linear.weight.normal_(0, 0.01)
        model.linear.bias.normal_(0, 0.01)
    
    # Get initial weights for comparison
    initial_weights = model.linear.weight.clone().detach().numpy().flatten()
    initial_bias = model.linear.bias.clone().detach().numpy().item()
    
    print(f"   Initial weights range: [{initial_weights.min():.6f}, {initial_weights.max():.6f}]")
    print(f"   Initial bias: {initial_bias:.6f}")
    
    # Create data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create optimizer (Pure SGD without momentum)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=0,        # Pure SGD gốc
        weight_decay=0     # Không dùng weight decay built-in, dùng manual regularization
    )
    
    # Create loss function
    loss_func = create_loss_function(loss_type, regularization)
    
    # Training
    print(f"   Starting PyTorch SGD training...")
    start_time = time.time()
    
    model.train()
    loss_history = []
    
    # Tính initial loss
    with torch.no_grad():
        initial_predictions = model(X_tensor)
        initial_loss = loss_func(model, initial_predictions, y_tensor).item()
        print(f"   Initial loss: {initial_loss:.8f}")
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X)
            
            # Compute loss
            loss = loss_func(model, predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for epoch
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:3d}/{epochs}: Loss = {avg_epoch_loss:.8f}")
    
    training_time = time.time() - start_time
    
    # Extract final results
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_tensor)
        final_loss = loss_func(model, final_predictions, y_tensor).item()
    
    final_weights = model.linear.weight.clone().detach().numpy().flatten()
    final_bias = model.linear.bias.clone().detach().numpy().item()
    
    # Simple convergence check (last 10 epochs)
    if len(loss_history) >= 10:
        recent_losses = loss_history[-10:]
        loss_std = np.std(recent_losses)
        converged = loss_std < 1e-6
    else:
        converged = False
    
    print(f"   ✅ Training completed!")
    print(f"   Training time: {training_time:.4f} seconds")
    print(f"   Final loss: {final_loss:.8f}")
    print(f"   Converged: {converged}")
    print(f"   Final weights range: [{final_weights.min():.6f}, {final_weights.max():.6f}]")
    print(f"   Final bias: {final_bias:.6f}")
    print(f"   Total batches processed: {epochs * len(dataloader)}")
    
    return {
        'model': model,
        'final_weights': final_weights,
        'final_bias': final_bias,
        'final_loss': final_loss,
        'training_time': training_time,
        'converged': converged,
        'epochs': epochs,
        'loss_history': loss_history,
        'algorithm': f'PyTorch_SGD_{loss_type.upper()}',
        'loss_type': loss_type,
        'regularization': regularization,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'initial_loss': initial_loss,
        'total_batches': epochs * len(dataloader)
    }


def evaluate_pytorch_results(result, X_test, y_test):
    """
    Đánh giá kết quả PyTorch trên test set
    
    Tham số:
        result: kết quả từ run_pytorch_sgd_optimization
        X_test: ma trận đặc trưng test
        y_test: vector target test
    
    Trả về:
        dict: metrics đánh giá
    """
    if 'error' in result:
        return {'error': result['error']}
    
    model = result['model']
    
    # Convert test data to tensor
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions_log = model(X_test_tensor).numpy()
    
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
    Chạy các thí nghiệm so sánh cho OLS, Ridge, Lasso với PyTorch SGD
    """
    print("="*80)
    print("🔬 PYTORCH SGD COMPARISON - STOCHASTIC GRADIENT DESCENT")
    print("="*80)
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. Please install torch to run this comparison.")
        return []
    
    # Load data
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    print(f"   Data shapes: Train {X_train.shape}, Test {X_test.shape}")
    print(f"   Target range (log scale): y_train [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Configurations to test
    configs = [
        {'loss_type': 'ols', 'regularization': 0.0, 'batch_size': 32, 'learning_rate': 0.01, 'epochs': 100},
        {'loss_type': 'ridge', 'regularization': 0.01, 'batch_size': 32, 'learning_rate': 0.01, 'epochs': 100},
        {'loss_type': 'lasso', 'regularization': 0.01, 'batch_size': 32, 'learning_rate': 0.01, 'epochs': 100},
        # Test different batch sizes
        {'loss_type': 'ridge', 'regularization': 0.01, 'batch_size': 64, 'learning_rate': 0.01, 'epochs': 100},
        {'loss_type': 'ridge', 'regularization': 0.01, 'batch_size': 128, 'learning_rate': 0.01, 'epochs': 100},
        # Test different learning rates
        {'loss_type': 'ridge', 'regularization': 0.01, 'batch_size': 32, 'learning_rate': 0.001, 'epochs': 150},
        {'loss_type': 'ridge', 'regularization': 0.01, 'batch_size': 32, 'learning_rate': 0.1, 'epochs': 50},
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n" + "="*50)
        print(f"🧪 EXPERIMENT: {config['loss_type'].upper()} PyTorch SGD")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print(f"   Epochs: {config['epochs']}")
        if config['regularization'] > 0:
            print(f"   Regularization: {config['regularization']}")
        print("="*50)
        
        # Run optimization
        result = run_pytorch_sgd_optimization(
            X_train, y_train, 
            loss_type=config['loss_type'],
            regularization=config['regularization'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs']
        )
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            continue
        
        # Evaluate
        print(f"\n📊 Evaluating on test set...")
        evaluation = evaluate_pytorch_results(result, X_test, y_test)
        
        if 'error' in evaluation:
            print(f"   ❌ Evaluation error: {evaluation['error']}")
            continue
        
        # Remove model from result for JSON serialization
        result_for_save = {k: v for k, v in result.items() if k != 'model'}
        
        # Combine results
        combined_result = {
            **result_for_save,
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
        config_suffix = f"_bs{config['batch_size']}_lr{config['learning_rate']:g}_ep{config['epochs']}"
        if config['regularization'] != 0.01 and config['regularization'] != 0.0:
            config_suffix += f"_reg{config['regularization']:g}"
        
        output_dir = Path(f"data/03_algorithms/stochastic_gd/pytorch_sgd_{config['loss_type']}{config_suffix}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(combined_result, f, indent=2)
        
        print(f"   💾 Results saved to: {output_dir / 'results.json'}")
    
    # Summary comparison
    print(f"\n" + "="*100)
    print("📋 SUMMARY COMPARISON - PYTORCH SGD")
    print("="*100)
    
    print(f"\n{'Algorithm':<18} {'Loss':<6} {'BS':<4} {'LR':<6} {'Ep':<4} {'Conv':<5} {'Batches':<8} {'R²(log)':<9} {'Time(s)':<8}")
    print("-"*100)
    
    for result in all_results:
        algorithm = result['algorithm'][:17]  # Truncate for display
        loss_type = result['loss_type'][:5]
        batch_size = result['batch_size']
        learning_rate = result['learning_rate']
        epochs = result['epochs']
        converged = "✅" if result['converged'] else "❌"
        total_batches = result['total_batches']
        r2_log = result['metrics_log_scale']['r2']
        time_s = result['training_time']
        
        print(f"{algorithm:<18} {loss_type:<6} {batch_size:<4} {learning_rate:<6.3f} {epochs:<4} {converged:<5} {total_batches:<8} {r2_log:<9.6f} {time_s:<8.4f}")
    
    print(f"\n✅ All PyTorch SGD comparisons completed!")
    print(f"📁 Results saved in data/03_algorithms/stochastic_gd/pytorch_sgd_*/")
    
    # SGD specific insights
    print(f"\n🔍 STOCHASTIC GRADIENT DESCENT INSIGHTS:")
    print(f"   • SGD với momentum=0 là thuật toán SGD gốc")
    print(f"   • Batch size ảnh hưởng đến noise và convergence")
    print(f"   • Learning rate cần điều chỉnh cẩn thận để tránh divergence")
    print(f"   • Số epochs cần đủ để model có thể hội tụ")
    print(f"   • SGD thích hợp cho large datasets với mini-batches")
    
    return all_results


def main():
    """Main function"""
    results = run_comparison_experiments()
    return results


if __name__ == "__main__":
    main()