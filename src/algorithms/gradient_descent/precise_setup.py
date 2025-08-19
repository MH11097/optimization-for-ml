#!/usr/bin/env python3
"""
Gradient Descent - Precise Setup

=== ỨNG DỤNG THỰC TẾ: GRADIENT DESCENT CHÍNH XÁC ===

THAM SỐ:
- Learning Rate: 0.001 (thấp)
- Max Iterations: 2000
- Tolerance: 1e-8

ĐẶC ĐIỂM:
- Hội tụ chậm nhưng rất chính xác
- Learning rate thấp, rất ổn định
- Tìm được minimum tốt nhất
- Phù hợp cho production, nghiên cứu
- Sử dụng dữ liệu từ 02.1_sampled
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys
import os

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.optimization_utils import tinh_mse, compute_r2_score, predict
from utils.visualization_utils import ve_duong_hoi_tu, ve_so_sanh_thuc_te_du_doan

def setup_output_dir():
    """Tạo thư mục output"""
    output_dir = Path("data/03_algorithms/gradient_descent/precise_setup")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_sampled_data():
    """Load dữ liệu từ 02.1_sampled (consistent với workflow hiện tại)"""
    data_dir = Path("data/02.1_sampled")
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Sampled data not found: {data_dir / file}")
    
    print("📂 Loading sampled data...")
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test = pd.read_csv(data_dir / "X_test.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    print(f"✅ Loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def tinh_chi_phi(X, y, weights, bias=0.0):
    """Tính Mean Squared Error cost function"""
    predictions = X.dot(weights) + bias
    errors = predictions - y
    cost = np.mean(errors ** 2)
    return cost

def tinh_gradient(X, y, weights, bias=0.0):
    """Tính gradient của MSE cost function"""
    n_samples = X.shape[0]
    predictions = X.dot(weights) + bias
    errors = predictions - y
    gradient_w = (2 / n_samples) * X.T.dot(errors)
    gradient_b = (2 / n_samples) * np.sum(errors)
    return gradient_w, gradient_b

def gradient_descent_chinh_xac(X, y, learning_rate=0.001, max_iterations=2000, tolerance=1e-8):
    """
    Gradient Descent Chính xác - Tối ưu hóa cao nhất
    
    Tham số:
    - learning_rate: 0.001 (thấp, cho chính xác)
    - max_iterations: 2000 (nhiều, tìm kiếm kỹ)
    - tolerance: 1e-8 (rất nghiêm ngặt)
    """
    print("🎯 Training Precise Gradient Descent...")
    print(f"   Learning rate: {learning_rate} (THẤP - cho chính xác)")
    print(f"   Max iterations: {max_iterations} (nhiều để tìm kiếm kỹ)")
    print(f"   Tolerance: {tolerance} (rất nghiêm ngặt)")
    
    # Khởi tạo parameters
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    bias = 0.0
    
    cost_history = []
    gradient_norms = []
    train_mse_history = []
    cost_improvements = []  # Track improvement per iteration
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Tính cost và gradient
        cost = tinh_chi_phi(X, y, weights, bias)
        gradient_w, gradient_b = tinh_gradient(X, y, weights, bias)
        
        # Cập nhật parameters
        weights = weights - learning_rate * gradient_w
        bias = bias - learning_rate * gradient_b
        
        # Lưu lịch sử
        cost_history.append(cost)
        gradient_norm = np.sqrt(np.sum(gradient_w**2) + gradient_b**2)
        gradient_norms.append(gradient_norm)
        train_mse_history.append(cost)
        
        # Track improvement
        if iteration > 0:
            improvement = cost_history[-2] - cost_history[-1]
            cost_improvements.append(improvement)
        
        # Check convergence (very strict)
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"✅ Hội tụ chính xác cao sau {iteration + 1} iterations")
            break
        
        # Progress update (more frequent for long training)
        if (iteration + 1) % 400 == 0:
            recent_improvement = np.mean(cost_improvements[-100:]) if len(cost_improvements) >= 100 else 0
            print(f"   Iteration {iteration + 1}: Cost = {cost:.8f}, Avg improvement = {recent_improvement:.2e}")
    
    training_time = time.time() - start_time
    
    if iteration == max_iterations - 1:
        print(f"⚠️ Đạt maximum iterations ({max_iterations})")
        print("   Cân nhận tăng max_iterations để có chính xác cao hơn")
    
    print(f"⏳ Training time: {training_time:.2f} seconds (kỹ lưỡng)")
    print(f"📉 Final cost: {cost_history[-1]:.8f} (chính xác cao)")
    print(f"📏 Final gradient norm: {gradient_norms[-1]:.2e}")
    
    # Precision analysis
    if len(cost_improvements) > 100:
        recent_improvements = cost_improvements[-100:]
        avg_recent_improvement = np.mean(recent_improvements)
        print(f"🔍 Recent improvement rate: {avg_recent_improvement:.2e} per iteration")
        
        stability = np.std(recent_improvements) / np.mean(recent_improvements) if np.mean(recent_improvements) > 0 else 0
        print(f"📊 Stability score: {1/(1+stability):.4f} (gần 1 = ổn định hơn)")
    
    return weights, bias, cost_history, gradient_norms, cost_improvements, training_time

def du_doan(X, weights, bias):
    """Dự đoán kết quả với trọng số và bias"""
    return X.dot(weights) + bias

def save_results(results, output_dir):
    """Lưu kết quả vào files với precision analysis"""
    
    # Advanced precision analysis
    precision_analysis = {}
    if len(results.get('cost_improvements', [])) > 100:
        recent_improvements = results['cost_improvements'][-100:]
        precision_analysis = {
            'final_convergence_rate': np.mean(recent_improvements),
            'convergence_stability': np.std(recent_improvements),
            'total_cost_reduction': results['cost_history'][0] - results['cost_history'][-1],
            'precision_score': len(results['cost_history']) / 2000,  # How much of max iterations used
            'gradient_reduction_ratio': results['gradient_norms'][0] / results['gradient_norms'][-1]
        }
    
    # 1. Save training history with improvements
    history_data = {
        'iteration': range(len(results['cost_history'])),
        'cost': results['cost_history'],
        'gradient_norm': results['gradient_norms'],
        'train_mse': results['train_mse_history']
    }
    
    if results.get('cost_improvements'):
        improvements_padded = [0] + results['cost_improvements']  # Add 0 for first iteration
        history_data['cost_improvement'] = improvements_padded
    
    history_df = pd.DataFrame(history_data)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    # 2. Save comprehensive results summary
    summary = {
        'method': 'Gradient Descent',
        'setup': 'precise_setup',
        'setup_description': 'Low learning rate for maximum precision',
        'parameters': {
            'learning_rate': 0.001,
            'max_iterations': 2000,
            'tolerance': 1e-8
        },
        'final_train_mse': results['final_train_mse'],
        'final_test_mse': results['final_test_mse'],
        'final_train_r2': results['final_train_r2'],
        'final_test_r2': results['final_test_r2'],
        'optimization_time': results['optimization_time'],
        'convergence_iterations': results['convergence_iterations'],
        'final_gradient_norm': results['final_gradient_norm'],
        'n_weights': len(results['weights']),
        'bias': results['bias'],
        'precision_analysis': precision_analysis,
        'notes': {
            'pros': ['Highest precision', 'Most stable', 'Best final result', 'Research quality'],
            'cons': ['Slow training', 'More computation', 'May be overkill'],
            'recommendations': 'Use when: precision is critical, have time, production systems'
        }
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # 3. Save weights
    weights_df = pd.DataFrame({
        'feature_index': range(len(results['weights'])),
        'weight_value': results['weights']
    })
    weights_df.to_csv(output_dir / "learned_weights.csv", index=False)
    
    print(f"💾 Results saved to {output_dir}")

def main():
    """Main function để chạy Gradient Descent precise setup"""
    print("🎯 GRADIENT DESCENT - PRECISE SETUP")
    print("Low learning rate configuration for maximum precision")
    
    try:
        # Setup
        output_dir = setup_output_dir()
        
        # Load data
        X_train, X_test, y_train, y_test = load_sampled_data()
        
        # Run optimization
        print("🚀 Starting Gradient Descent precise optimization...")
        weights, bias, cost_history, gradient_norms, cost_improvements, training_time = gradient_descent_chinh_xac(
            X_train, y_train,
            learning_rate=0.001,    # Precise setup
            max_iterations=2000,
            tolerance=1e-8
        )
        
        # Final evaluation
        train_predictions = du_doan(X_train, weights, bias)
        test_predictions = du_doan(X_test, weights, bias)
        
        final_train_mse = tinh_mse(y_train, train_predictions)
        final_test_mse = tinh_mse(y_test, test_predictions)
        
        final_train_r2 = compute_r2_score(y_train, train_predictions)
        final_test_r2 = compute_r2_score(y_test, test_predictions)
        
        results = {
            'weights': weights,
            'bias': bias,
            'cost_history': cost_history,
            'gradient_norms': gradient_norms,
            'cost_improvements': cost_improvements,
            'train_mse_history': cost_history,  # Same as cost for MSE
            'final_train_mse': final_train_mse,
            'final_test_mse': final_test_mse,
            'final_train_r2': final_train_r2,
            'final_test_r2': final_test_r2,
            'optimization_time': training_time,
            'convergence_iterations': len(cost_history),
            'final_gradient_norm': gradient_norms[-1] if gradient_norms else 0.0
        }
        
        # Save results with precision analysis
        save_results(results, output_dir)
        
        # Plot results - use Vietnamese visualization functions
        print("📈 Creating visualization...")
        ve_duong_hoi_tu(cost_history, gradient_norms, "Gradient Descent - Precise Setup")
        ve_so_sanh_thuc_te_du_doan(y_test, test_predictions, "Precise Setup Test Predictions")
        
        print("\n" + "="*50)
        print("🎯 PRECISE SETUP - EVALUATION RESULTS")
        print("="*50)
        print(f"Test MSE:      {final_test_mse:.8f} (chính xác cao)")
        print(f"Test R²:       {final_test_r2:.6f}")
        
        # Additional precision metrics
        residuals = y_test - test_predictions
        max_error = np.max(np.abs(residuals))
        error_95th = np.percentile(np.abs(residuals), 95)
        prediction_std = np.std(residuals)
        
        print(f"\n🔍 PRECISION METRICS:")
        print(f"   📏 Max Error:     {max_error:.6f}")
        print(f"   📊 95th %ile Err: {error_95th:.6f}")
        print(f"   📐 Pred Std:      {prediction_std:.6f}")
        
        print(f"\n🎯 PRECISION CHARACTERISTICS:")
        print(f"   🐌 Learning Rate: 0.001 (THẤP cho chính xác)")
        print(f"   ⏱️ Training Time: {training_time:.2f}s (kỹ lưỡng)")
        print(f"   🔄 Iterations Used: {len(cost_history)}/2000")
        print(f"   ✅ Tolerance: 1e-8 (rất nghiêm ngặt)")
        
        # Precision score
        precision_score = min(final_test_r2, 1.0)
        if precision_score > 0.95:
            precision_rating = "XUẤT SẮC"
            color = "🟢"
        elif precision_score > 0.90:
            precision_rating = "RẤT TỐT"
            color = "🟡"
        elif precision_score > 0.85:
            precision_rating = "TỐT"
            color = "🟠"
        else:
            precision_rating = "CẦN CẢI THIỆN"
            color = "🔴"
        
        print(f"   {color} Precision Rating: {precision_rating}")
        
        print(f"\n🎯 KHI NÀO SỬ DỤNG PRECISE SETUP:")
        print(f"   • Hệ thống production đòi hỏi độ chính xác cao")
        print(f"   • Nghiên cứu khi chính xác quan trọng")
        print(f"   • Training model cuối cùng")
        print(f"   • Khi thời gian tính toán không phải vấn đề")
        
        print(f"\n💡 OPTIMIZATION INSIGHTS:")
        print(f"   • Hội tụ rất ổn định")
        print(f"   • Rủi ro tối thiểu overshoot")
        print(f"   • Tốt nhất để tìm global minimum")
        print(f"   • Phù hợp ứng dụng nhạy cảm")
        
        print(f"\n✅ Results saved to: {output_dir}")
        print("🎉 Gradient Descent precise optimization completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in Gradient Descent precise optimization: {e}")
        raise

if __name__ == "__main__":
    results = main()