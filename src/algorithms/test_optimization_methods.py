"""Comprehensive test suite cho tất cả optimization methods"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all optimization methods
from newton_method.pure_newton import newton_standard_setup as newton_pure
from newton_method.damped_newton import damped_newton_standard_setup as newton_damped
from quasi_newton.bfgs import bfgs_standard_setup
from quasi_newton.lbfgs import lbfgs_standard_setup
from quasi_newton.sr1 import sr1_standard_setup

from utils.optimization_utils import (
    tinh_gradient_hoi_quy_tuyen_tinh,
    tinh_ma_tran_hessian_hoi_quy_tuyen_tinh,
    xac_minh_gradient,
    xac_minh_hessian,
    tinh_mse,
    compute_r2_score,
    predict
)


class OptimizationTester:
    """
    Comprehensive tester cho các optimization methods
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = {}
        
    def generate_test_data(self, n_samples: int = 100, n_features: int = 5, 
                          noise_level: float = 0.1, seed: int = 42) -> dict:
        """Generate synthetic test data"""
        np.random.seed(seed)
        
        # Generate feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Generate true weights and bias
        true_weights = np.random.randn(n_features)
        true_bias = np.random.randn()
        
        # Generate target with noise
        noise = noise_level * np.random.randn(n_samples)
        y = X @ true_weights + true_bias + noise
        
        return {
            'X': X,
            'y': y,
            'true_weights': true_weights,
            'true_bias': true_bias,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'n_features': n_features
        }
    
    def test_gradient_hessian_correctness(self, test_data: dict) -> dict:
        """Test correctness của gradient và Hessian calculations"""
        if self.verbose:
            print("=== Testing Gradient and Hessian Correctness ===")
        
        X, y = test_data['X'], test_data['y']
        n_features = test_data['n_features']
        
        # Test point
        weights = np.random.randn(n_features)
        bias = np.random.randn()
        
        # Define objective function
        def objective(params):
            w = params[:-1]
            b = params[-1]
            predictions = X @ w + b
            return 0.5 * np.mean((predictions - y)**2)
        
        # Define gradient function
        def gradient_func(params):
            w = params[:-1]
            b = params[-1]
            grad_w, grad_b = tinh_gradient_hoi_quy_tuyen_tinh(X, y, w, b, 0.0)
            return np.concatenate([grad_w, [grad_b]])
        
        # Test gradient
        test_params = np.concatenate([weights, [bias]])
        gradient_correct = xac_minh_gradient(objective, gradient_func, test_params)
        
        # Test Hessian (for weights only, since it's constant)
        def hessian_func(w):
            return tinh_ma_tran_hessian_hoi_quy_tuyen_tinh(X, 0.0)
        
        def objective_weights_only(w):
            predictions = X @ w + bias
            return 0.5 * np.mean((predictions - y)**2)
        
        hessian_correct = xac_minh_hessian(objective_weights_only, hessian_func, weights)
        
        results = {
            'gradient_correct': gradient_correct,
            'hessian_correct': hessian_correct
        }
        
        if self.verbose:
            print(f"Gradient correctness: {'✅ PASS' if gradient_correct else '❌ FAIL'}")
            print(f"Hessian correctness: {'✅ PASS' if hessian_correct else '❌ FAIL'}")
            print()
        
        return results
    
    def run_single_method_test(self, method_name: str, method_func, test_data: dict) -> dict:
        """Run test for a single optimization method"""
        if self.verbose:
            print(f"Testing {method_name}...")
        
        X, y = test_data['X'], test_data['y']
        true_weights, true_bias = test_data['true_weights'], test_data['true_bias']
        
        # Run optimization
        start_time = time.time()
        try:
            result = method_func(X, y, verbose=False)
            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
        
        end_time = time.time()
        
        if not success:
            return {
                'method': method_name,
                'success': False,
                'error': error_message,
                'runtime': end_time - start_time
            }
        
        # Extract results
        learned_weights = result['weights']
        learned_bias = result['bias']
        final_mse = result.get('final_mse', tinh_mse(y, result['predictions']))
        
        # Compute metrics
        weights_error = np.linalg.norm(learned_weights - true_weights)
        bias_error = abs(learned_bias - true_bias)
        
        # Compute R² score
        predictions = predict(X, learned_weights, learned_bias)
        r2_score = compute_r2_score(y, predictions)
        
        # Convergence info
        convergence_info = result.get('convergence_info', {})
        iterations = convergence_info.get('iterations', len(result.get('cost_history', [])))
        converged = convergence_info.get('converged', True)
        
        test_result = {
            'method': method_name,
            'success': success,
            'runtime': end_time - start_time,
            'final_mse': final_mse,
            'r2_score': r2_score,
            'weights_error': weights_error,
            'bias_error': bias_error,
            'iterations': iterations,
            'converged': converged,
            'convergence_info': convergence_info
        }
        
        if self.verbose:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} - MSE: {final_mse:.6f}, R²: {r2_score:.4f}, "
                  f"Iterations: {iterations}, Time: {test_result['runtime']:.3f}s")
        
        return test_result
    
    def run_comprehensive_test(self, test_data: dict) -> dict:
        """Run comprehensive test trên tất cả methods"""
        if self.verbose:
            print("=== Comprehensive Optimization Methods Test ===")
            print(f"Problem: {test_data['n_samples']} samples, {test_data['n_features']} features")
            print()
        
        # Test gradient and Hessian first
        grad_hess_results = self.test_gradient_hessian_correctness(test_data)
        
        # Define methods to test
        methods = {
            'Pure Newton': newton_pure,
            'Damped Newton': newton_damped,
            'BFGS': bfgs_standard_setup,
            'L-BFGS': lbfgs_standard_setup,
            'SR1': sr1_standard_setup
        }
        
        # Run tests for all methods
        method_results = {}
        for method_name, method_func in methods.items():
            method_results[method_name] = self.run_single_method_test(
                method_name, method_func, test_data
            )
        
        # Compile comprehensive results
        results = {
            'test_data_info': {
                'n_samples': test_data['n_samples'],
                'n_features': test_data['n_features'],
                'noise_level': test_data['noise_level']
            },
            'gradient_hessian_tests': grad_hess_results,
            'method_results': method_results,
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return results
    
    def compare_methods(self, results: dict) -> pd.DataFrame:
        """Tạo comparison table của các methods"""
        method_results = results['method_results']
        
        comparison_data = []
        for method_name, result in method_results.items():
            if result['success']:
                comparison_data.append({
                    'Method': method_name,
                    'Success': '✅',
                    'MSE': result['final_mse'],
                    'R²': result['r2_score'],
                    'Iterations': result['iterations'],
                    'Runtime (s)': result['runtime'],
                    'Weights Error': result['weights_error'],
                    'Bias Error': result['bias_error'],
                    'Converged': '✅' if result['converged'] else '❌'
                })
            else:
                comparison_data.append({
                    'Method': method_name,
                    'Success': '❌',
                    'MSE': np.nan,
                    'R²': np.nan,
                    'Iterations': np.nan,
                    'Runtime (s)': result['runtime'],
                    'Weights Error': np.nan,
                    'Bias Error': np.nan,
                    'Converged': '❌',
                    'Error': result.get('error', 'Unknown')
                })
        
        return pd.DataFrame(comparison_data)
    
    def plot_convergence_comparison(self, methods_to_plot: dict, test_data: dict, save_path: str = None):
        """Plot convergence comparison của các methods"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        X, y = test_data['X'], test_data['y']
        
        # Collect results
        method_results = {}
        for method_name, method_func in methods_to_plot.items():
            try:
                result = method_func(X, y, verbose=False)
                method_results[method_name] = result
            except Exception as e:
                if self.verbose:
                    print(f"Error in {method_name}: {e}")
                continue
        
        # Plot cost history
        ax = axes[0, 0]
        for method_name, result in method_results.items():
            cost_history = result.get('cost_history', [])
            if cost_history:
                ax.plot(cost_history, label=method_name, linewidth=2)
        ax.set_title('Cost Function Convergence')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot gradient norms
        ax = axes[0, 1]
        for method_name, result in method_results.items():
            gradient_norms = result.get('gradient_norms', [])
            if gradient_norms:
                ax.plot(gradient_norms, label=method_name, linewidth=2)
        ax.set_title('Gradient Norm Convergence')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||Gradient||')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot runtime comparison
        ax = axes[1, 0]
        methods = list(method_results.keys())
        runtimes = [result['optimization_time'] for result in method_results.values()]
        bars = ax.bar(methods, runtimes)
        ax.set_title('Runtime Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, runtime in zip(bars, runtimes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{runtime:.3f}s', ha='center', va='bottom')
        
        # Plot final MSE comparison
        ax = axes[1, 1]
        final_mses = [result.get('final_mse', compute_mse(y, result['predictions'])) 
                     for result in method_results.values()]
        bars = ax.bar(methods, final_mses)
        ax.set_title('Final MSE Comparison')
        ax.set_ylabel('MSE')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mse in zip(bars, final_mses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{mse:.6f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Convergence comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: dict, output_dir: str):
        """Save test results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(output_path / "test_results.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            json_results = convert_numpy_types(results)
            json.dump(json_results, f, indent=2)
        
        # Save comparison table
        comparison_df = self.compare_methods(results)
        comparison_df.to_csv(output_path / "methods_comparison.csv", index=False)
        
        if self.verbose:
            print(f"\n📁 Test results saved to {output_path}")
            print("Files created:")
            print("  - test_results.json (detailed results)")
            print("  - methods_comparison.csv (summary table)")


def run_basic_tests():
    """Run basic tests với default settings"""
    tester = OptimizationTester(verbose=True)
    
    print("🧪 Running Basic Optimization Methods Tests\n")
    
    # Generate test data
    test_data = tester.generate_test_data(n_samples=100, n_features=5, noise_level=0.1)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(test_data)
    
    # Display comparison table
    comparison_df = tester.compare_methods(results)
    print("\n📊 Methods Comparison:")
    print(comparison_df.to_string(index=False, float_format='%.6f'))
    
    # Plot convergence comparison
    methods_to_plot = {
        'Pure Newton': newton_pure,
        'Damped Newton': newton_damped,
        'BFGS': bfgs_standard_setup,
        'L-BFGS': lbfgs_standard_setup,
        'SR1': sr1_standard_setup
    }
    
    tester.plot_convergence_comparison(
        methods_to_plot, test_data, 
        save_path="data/03_algorithms/convergence_comparison.png"
    )
    
    # Save results
    tester.save_results(results, "data/03_algorithms/test_results")
    
    return results


def run_stress_tests():
    """Run stress tests với different problem sizes và conditions"""
    tester = OptimizationTester(verbose=True)
    
    print("🔥 Running Stress Tests\n")
    
    test_configs = [
        {'n_samples': 50, 'n_features': 3, 'noise_level': 0.01, 'name': 'Small Clean'},
        {'n_samples': 200, 'n_features': 10, 'noise_level': 0.1, 'name': 'Medium Noisy'},
        {'n_samples': 500, 'n_features': 20, 'noise_level': 0.2, 'name': 'Large Very Noisy'},
        {'n_samples': 100, 'n_features': 50, 'noise_level': 0.1, 'name': 'High Dimensional'}
    ]
    
    all_stress_results = {}
    
    for config in test_configs:
        print(f"\n=== {config['name']} Problem ===")
        test_data = tester.generate_test_data(
            n_samples=config['n_samples'],
            n_features=config['n_features'],
            noise_level=config['noise_level']
        )
        
        results = tester.run_comprehensive_test(test_data)
        all_stress_results[config['name']] = results
        
        # Display results for this configuration
        comparison_df = tester.compare_methods(results)
        print(f"\n{config['name']} Results:")
        print(comparison_df[['Method', 'Success', 'MSE', 'Iterations', 'Runtime (s)']].to_string(
            index=False, float_format='%.4f'))
    
    return all_stress_results


if __name__ == "__main__":
    # Run basic tests
    print("=" * 80)
    print("OPTIMIZATION METHODS TEST SUITE")
    print("=" * 80)
    
    basic_results = run_basic_tests()
    
    print("\n" + "=" * 80)
    print("STRESS TESTS")
    print("=" * 80)
    
    stress_results = run_stress_tests()
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)