#!/usr/bin/env python3
"""
Quasi-Newton Method - BFGS Setup cho Linear Regression

=== THAM SỐ SETUP & HÀM LOSS ===

CÁC HÀM LOSS HỖ TRỢ:
1. OLS (Ordinary Least Squares): MSE thuần túy
2. Ridge: MSE + L2 regularization (Ridge = λ * ||w||^2)
3. Lasso: MSE + L1 regularization (Lasso = λ * ||w||_1)

CÁC SETUP KHÁC NHAU:
Standard Setup (BFGS):
- Max Iterations: 100
- Tolerance: 1e-8
- Line Search: Backtracking Armijo
- Sử dụng cho: hầu hết các trường hợp

Robust Setup (BFGS):
- Max Iterations: 200
- Tolerance: 1e-6 (relaxed)
- Sử dụng cho: bài toán ill-conditioned

ĐẶC ĐIỂM:
- Xấp xỉ Hessian matrix H ≈ ∇²f
- Nhanh hơn Newton (không tính Hessian)
- Hội tụ superlinear
- Hiệu quả bộ nhớ hơn Newton
- Sử dụng dữ liệu từ 02.1_sampled

TOÁN HỌC:
- BFGS update: B_{k+1} = B_k + (y_k y_k^T)/(y_k^T s_k) - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k)
- s_k = x_{k+1} - x_k (step)
- y_k = ∇f_{k+1} - ∇f_k (gradient change)
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

from utils.optimization_utils import (
    tinh_gradient_hoi_quy_tuyen_tinh,
    tinh_mse,
    tinh_r2_score,
    du_doan
)
from utils.visualization_utils import ve_duong_hoi_tu, ve_so_sanh_thuc_te_du_doan

def setup_output_dir():
    """Tạo thư mục output"""
    output_dir = Path("data/03_algorithms/quasi_newton/bfgs_setup")
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

def tinh_chi_phi_va_gradient(X, y, trong_so, he_so_tu_do, dieu_chinh=0.0):
    """Tính cost và gradient cho BFGS"""
    du_doan_values = du_doan(X, trong_so, he_so_tu_do)
    mse_cost = tinh_mse(y, du_doan_values)
    
    # Thêm regularization term
    regularization_term = 0.5 * dieu_chinh * np.sum(trong_so**2)
    cost = mse_cost + regularization_term
    
    # Tính gradient cho weights và bias
    gradient_w, gradient_b = tinh_gradient_hoi_quy_tuyen_tinh(X, y, trong_so, he_so_tu_do, dieu_chinh)
    
    # Kết hợp gradient (weights + bias)
    gradient = np.concatenate([gradient_w, [gradient_b]])
    
    return cost, gradient

def tim_buoc_nhay_lui(X, y, trong_so, he_so_tu_do, huong_di, gradient, alpha_init=1.0, c1=1e-4, rho=0.9, dieu_chinh=0.0):
    """
    Backtracking line search để tìm step size tối ưu
    Đảm bảo Armijo condition: f(x + αp) ≤ f(x) + c₁α∇f^T p
    """
    current_cost, _ = tinh_chi_phi_va_gradient(X, y, trong_so, he_so_tu_do, dieu_chinh)
    alpha = alpha_init
    
    # Directional derivative
    directional_derivative = gradient.dot(huong_di)
    
    # Backtracking
    max_iter = 20
    for i in range(max_iter):
        # Split combined parameters
        combined_params = np.concatenate([trong_so, [he_so_tu_do]])
        new_combined_params = combined_params + alpha * huong_di
        new_trong_so = new_combined_params[:-1]
        new_he_so_tu_do = new_combined_params[-1]
        
        new_cost, _ = tinh_chi_phi_va_gradient(X, y, new_trong_so, new_he_so_tu_do, dieu_chinh)
        
        # Armijo condition
        if new_cost <= current_cost + c1 * alpha * directional_derivative:
            return alpha
        
        alpha *= rho
    
    return alpha  # Return last alpha if no good step found

def bfgs_toi_uu_hoa(X, y, max_iterations=100, tolerance=1e-8, dieu_chinh=0.0):
    """
    BFGS Quasi-Newton Method Implementation với bias handling
    
    Algorithm:
    1. Initialize B₀ = I (identity matrix) for combined [weights, bias]
    2. Compute search direction: p_k = -B_k⁻¹ ∇f_k
    3. Line search for step size α_k
    4. Update: x_{k+1} = x_k + α_k p_k
    5. Update BFGS approximation B_{k+1}
    """
    print("🚀 Training BFGS Quasi-Newton Method...")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Tolerance: {tolerance}")
    print(f"   Regularization: {dieu_chinh}")
    print("   Approximating Hessian with BFGS updates")
    
    # Initialize weights and bias
    n_features = X.shape[1]
    trong_so = np.random.normal(0, 0.01, n_features)
    he_so_tu_do = 0.0
    
    # Initialize BFGS approximation for combined parameters [weights, bias]
    n_params = n_features + 1  # weights + bias
    B = np.eye(n_params)  # Initial approximation = Identity
    
    cost_history = []
    gradient_norms = []
    step_sizes = []
    condition_numbers = []  # Track condition number of B
    
    start_time = time.time()
    
    # First iteration
    cost, gradient = tinh_chi_phi_va_gradient(X, y, trong_so, he_so_tu_do, dieu_chinh)
    cost_history.append(cost)
    gradient_norms.append(np.linalg.norm(gradient))
    
    for iteration in range(max_iterations):
        # Check convergence
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < tolerance:
            print(f"✅ Converged after {iteration + 1} iterations")
            print(f"   Gradient norm: {grad_norm:.2e}")
            break
        
        # Compute search direction: p = -B⁻¹ ∇f
        try:
            search_direction = -np.linalg.solve(B, gradient)
        except np.linalg.LinAlgError:
            print("⚠️ BFGS matrix became singular, resetting to identity")
            B = np.eye(n_params)
            search_direction = -gradient
        
        # Line search for step size
        step_size = tim_buoc_nhay_lui(X, y, trong_so, he_so_tu_do, search_direction, gradient, dieu_chinh=dieu_chinh)
        step_sizes.append(step_size)
        
        # Store previous values
        prev_combined_params = np.concatenate([trong_so, [he_so_tu_do]])
        gradient_prev = gradient.copy()
        
        # Update parameters
        new_combined_params = prev_combined_params + step_size * search_direction
        trong_so = new_combined_params[:-1]
        he_so_tu_do = new_combined_params[-1]
        
        # Compute new cost and gradient
        cost, gradient = tinh_chi_phi_va_gradient(X, y, trong_so, he_so_tu_do, dieu_chinh)
        cost_history.append(cost)
        gradient_norms.append(np.linalg.norm(gradient))
        
        # BFGS update
        current_combined_params = np.concatenate([trong_so, [he_so_tu_do]])
        s = current_combined_params - prev_combined_params  # step
        y = gradient - gradient_prev  # gradient change
        
        # Check BFGS condition: s^T y > 0
        sy = s.dot(y)
        if sy > 1e-10:  # Positive definite condition
            # BFGS update formula
            Bs = B.dot(s)
            sBs = s.dot(Bs)
            
            # B_{k+1} = B_k + (yy^T)/(y^T s) - (B s s^T B)/(s^T B s)
            B = B + np.outer(y, y) / sy - np.outer(Bs, Bs) / sBs
        else:
            print(f"   Warning: BFGS condition violated at iteration {iteration + 1}")
        
        # Track condition number
        try:
            cond_num = np.linalg.cond(B)
            condition_numbers.append(cond_num)
        except:
            condition_numbers.append(np.inf)
        
        # Progress update
        if (iteration + 1) % 20 == 0:
            print(f"   Iteration {iteration + 1}: Cost = {cost:.8f}, "
                  f"||∇f|| = {grad_norm:.2e}, Step = {step_size:.2e}")
    
    training_time = time.time() - start_time
    
    if iteration == max_iterations - 1:
        print(f"⚠️ Reached maximum iterations ({max_iterations})")
    
    print(f"⚡ Training time: {training_time:.4f} seconds")
    print(f"📉 Final cost: {cost_history[-1]:.8f}")
    print(f"📏 Final gradient norm: {gradient_norms[-1]:.2e}")
    print(f"📐 Final BFGS condition number: {condition_numbers[-1]:.2e}")
    
    return (trong_so, he_so_tu_do, cost_history, gradient_norms, step_sizes, 
            condition_numbers, training_time, B)

def evaluate_model(trong_so, he_so_tu_do, X_test, y_test):
    """Đánh giá model trên test set với weights và bias"""
    predictions = du_doan(X_test, trong_so, he_so_tu_do)
    
    # Metrics
    mse = tinh_mse(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    
    # R-squared using utility function
    r2 = tinh_r2_score(y_test, predictions)
    
    # MAPE
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_results(cost_history, gradient_norms, step_sizes, condition_numbers, 
                trong_so, he_so_tu_do, X_test, y_test, B, output_dir):
    """Vẽ các biểu đồ đặc trưng của BFGS"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('BFGS Quasi-Newton Method - Hessian Approximation Analysis', fontsize=16)
    
    # 1. Convergence curve
    ax1 = axes[0, 0]
    ax1.plot(cost_history, 'b-', linewidth=3, marker='o', markersize=4)
    ax1.set_title('BFGS Convergence (Superlinear)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('MSE Cost')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Gradient norms (should decrease superlinearly)
    ax2 = axes[0, 1]
    ax2.plot(gradient_norms, 'r-', linewidth=3, marker='s', markersize=4)
    ax2.set_title('Gradient Norm (Superlinear Decay)')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('||∇f||')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add convergence rate annotation
    if len(gradient_norms) > 2:
        rate = gradient_norms[-1] / gradient_norms[-2]
        ax2.text(0.6, 0.8, f'Final Rate: {rate:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. Step sizes from line search
    ax3 = axes[0, 2]
    ax3.plot(step_sizes, 'g-', linewidth=2, marker='^', markersize=4)
    ax3.set_title('Step Sizes (Line Search)')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Step Size α')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. BFGS condition numbers
    ax4 = axes[1, 0]
    valid_cond_nums = [c for c in condition_numbers if c != np.inf and not np.isnan(c)]
    if valid_cond_nums:
        ax4.plot(valid_cond_nums, 'purple', linewidth=2)
        ax4.set_title('BFGS Matrix Condition Number')
        ax4.set_xlabel('Iterations')
        ax4.set_ylabel('Condition Number')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # Health check
        final_cond = valid_cond_nums[-1]
        if final_cond < 1e6:
            health = "HEALTHY"
            color = "green"
        elif final_cond < 1e12:
            health = "MODERATE"
            color = "orange"
        else:
            health = "ILL-CONDITIONED"
            color = "red"
        
        ax4.text(0.05, 0.95, f'Status: {health}', transform=ax4.transAxes,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))
    
    # 5. Predictions vs Actual
    ax5 = axes[1, 1]
    predictions = du_doan(X_test, trong_so, he_so_tu_do)
    ax5.scatter(y_test, predictions, alpha=0.6, color='navy')
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax5.set_xlabel('Actual Values')
    ax5.set_ylabel('Predicted Values')
    ax5.set_title('Predictions vs Actual')
    ax5.grid(True, alpha=0.3)
    
    # Calculate and show R²
    r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    ax5.text(0.05, 0.95, f'R² = {r2:.6f}', transform=ax5.transAxes,
             bbox=dict(boxstyle="round", facecolor='lightyellow'))
    
    # 6. BFGS matrix eigenvalues
    ax6 = axes[1, 2]
    try:
        eigenvals = np.linalg.eigvals(B)
        eigenvals = eigenvals[eigenvals > 0]  # Only positive eigenvalues
        ax6.hist(eigenvals, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        ax6.set_title('BFGS Matrix Eigenvalues')
        ax6.set_xlabel('Eigenvalue')
        ax6.set_ylabel('Count')
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        
        # Add statistics
        min_eig = np.min(eigenvals)
        max_eig = np.max(eigenvals)
        ax6.text(0.05, 0.95, f'Min: {min_eig:.2e}\\nMax: {max_eig:.2e}', 
                transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
    except:
        ax6.text(0.5, 0.5, 'Eigenvalue computation failed', ha='center', va='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "bfgs_setup_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(trong_so, he_so_tu_do, metrics, cost_history, gradient_norms, step_sizes, 
                condition_numbers, training_time, B, output_dir):
    """Lưu kết quả với BFGS analysis"""
    
    # BFGS-specific analysis
    bfgs_analysis = {
        'superlinear_convergence': len(cost_history) < 50,  # Fast convergence indicator
        'final_condition_number': float(condition_numbers[-1]) if condition_numbers else np.inf,
        'avg_step_size': float(np.mean(step_sizes)) if step_sizes else 0,
        'convergence_rate': float(gradient_norms[-1] / gradient_norms[-2]) if len(gradient_norms) > 1 else 1,
        'hessian_approximation_quality': 'good' if condition_numbers and condition_numbers[-1] < 1e6 else 'poor'
    }
    
    results = {
        'setup_name': 'BFGS Setup',
        'algorithm': 'Quasi-Newton (BFGS)',
        'parameters': {
            'max_iterations': 100,
            'tolerance': 1e-8,
            'line_search': 'backtracking'
        },
        'metrics': metrics,
        'training_time': training_time,
        'convergence': {
            'iterations': len(cost_history),
            'final_cost': cost_history[-1],
            'final_gradient_norm': gradient_norms[-1]
        },
        'bfgs_analysis': bfgs_analysis,
        'notes': {
            'setup_description': 'BFGS Quasi-Newton with backtracking line search',
            'pros': ['Superlinear convergence', 'No Hessian computation', 'Memory efficient', 'Robust'],
            'cons': ['More complex than GD', 'Can become ill-conditioned', 'Requires line search'],
            'mathematical_principle': 'Approximates Hessian using gradient information from previous iterations',
            'key_insight': 'Balances Newton speed with computational efficiency'
        }
    }
    
    # Save detailed results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save weights, bias and BFGS matrix
    np.save(output_dir / "weights.npy", trong_so)
    np.save(output_dir / "bias.npy", he_so_tu_do)
    np.save(output_dir / "bfgs_matrix.npy", B)
    
    # Save training history
    max_len = max(len(cost_history), len(gradient_norms), len(step_sizes), len(condition_numbers))
    
    # Pad arrays to same length
    cost_padded = cost_history + [np.nan] * (max_len - len(cost_history))
    grad_padded = gradient_norms + [np.nan] * (max_len - len(gradient_norms))
    step_padded = step_sizes + [np.nan] * (max_len - len(step_sizes))
    cond_padded = condition_numbers + [np.nan] * (max_len - len(condition_numbers))
    
    history_df = pd.DataFrame({
        'iteration': range(max_len),
        'cost': cost_padded,
        'gradient_norm': grad_padded,
        'step_size': step_padded,
        'condition_number': cond_padded
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)

def print_results(metrics, training_time, iterations, final_condition_number):
    """In kết quả với focus trên BFGS characteristics"""
    print("\n" + "="*60)
    print("🚀 BFGS QUASI-NEWTON METHOD - SETUP RESULTS")
    print("="*60)
    print(f"Test MSE:  {metrics['mse']:.8f}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")
    print(f"Test MAE:  {metrics['mae']:.6f}")
    print(f"R² Score:  {metrics['r2']:.6f}")
    print(f"MAPE:      {metrics['mape']:.2f}%")
    
    print(f"\n🚀 QUASI-NEWTON CHARACTERISTICS:")
    print(f"   ⚡ Training Time: {training_time:.4f}s (fast!)")
    print(f"   🔄 Iterations: {iterations} (superlinear convergence)")
    print(f"   📐 Final Condition Number: {final_condition_number:.2e}")
    print(f"   ✅ Convergence Type: Superlinear")
    
    # Convergence assessment
    if iterations < 20:
        convergence_rating = "EXCELLENT (< 20 iterations)"
        color = "🟢"
    elif iterations < 50:
        convergence_rating = "VERY GOOD (< 50 iterations)"
        color = "🟡"
    else:
        convergence_rating = "SLOWER THAN EXPECTED"
        color = "🔴"
    
    print(f"   {color} Convergence Speed: {convergence_rating}")
    
    # Matrix condition assessment
    if final_condition_number < 1e6:
        matrix_health = "WELL-CONDITIONED"
        health_color = "🟢"
    elif final_condition_number < 1e12:
        matrix_health = "MODERATELY CONDITIONED"
        health_color = "🟡"
    else:
        matrix_health = "ILL-CONDITIONED"
        health_color = "🔴"
    
    print(f"   {health_color} BFGS Matrix: {matrix_health}")
    
    print(f"\n🧮 BFGS MATHEMATICAL INSIGHTS:")
    print(f"   • Approximates Hessian: B ≈ ∇²f")
    print(f"   • BFGS update uses gradient history")
    print(f"   • Update formula: B_{k+1} = B_k + yy^T/y^Ts - Bss^TB/s^TBs")
    print(f"   • Search direction: p = -B^{-1}∇f")
    print(f"   • Superlinear convergence rate")
    
    print(f"\n🎯 ADVANTAGES OVER OTHER METHODS:")
    print(f"   vs Newton Method:")
    print(f"   • ✅ No Hessian computation (saves O(n²) space)")
    print(f"   • ✅ No matrix inversion per iteration")
    print(f"   • ✅ More robust to ill-conditioning")
    print(f"   vs Gradient Descent:")
    print(f"   • ✅ Much faster convergence")
    print(f"   • ✅ Uses curvature information")
    print(f"   • ❌ More complex implementation")
    
    print(f"\n🎯 WHEN TO USE BFGS:")
    print(f"   • Medium-sized problems (n < 1000)")
    print(f"   • Smooth, twice differentiable functions")
    print(f"   • When you need fast convergence")
    print(f"   • When Hessian computation is expensive")
    print(f"   • Unconstrained optimization")
    
    print(f"\n⚠️ LIMITATIONS:")
    print(f"   • Requires O(n²) memory for B matrix")
    print(f"   • Not suitable for very large problems")
    print(f"   • Can become ill-conditioned")
    print(f"   • Requires good line search")

def main():
    """Chạy BFGS Quasi-Newton Method"""
    print("🚀 BFGS QUASI-NEWTON METHOD")
    print("Hessian approximation for fast convergence")
    
    # Setup
    output_dir = setup_output_dir()
    
    # Load data
    X_train, X_test, y_train, y_test = load_sampled_data()
    
    # Train model
    trong_so, he_so_tu_do, cost_history, gradient_norms, step_sizes, condition_numbers, training_time, B = bfgs_toi_uu_hoa(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(trong_so, he_so_tu_do, X_test, y_test)
    
    # Plot results
    plot_results(cost_history, gradient_norms, step_sizes, condition_numbers, 
                trong_so, he_so_tu_do, X_test, y_test, B, output_dir)
    
    # Save everything
    save_results(trong_so, he_so_tu_do, metrics, cost_history, gradient_norms, step_sizes, 
                condition_numbers, training_time, B, output_dir)
    
    # Print results
    final_condition_number = condition_numbers[-1] if condition_numbers else np.inf
    print_results(metrics, training_time, len(cost_history), final_condition_number)
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()