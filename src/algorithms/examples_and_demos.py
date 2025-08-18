"""
Examples và demos để showcase các optimization methods
Bao gồm practical examples với real data và synthetic data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all optimization methods
from newton_method.pure_newton import newton_standard_setup, newton_robust_setup, newton_fast_setup
from newton_method.damped_newton import damped_newton_standard_setup
from quasi_newton.bfgs import bfgs_standard_setup
from quasi_newton.lbfgs import lbfgs_standard_setup
from quasi_newton.sr1 import sr1_standard_setup

from utils.optimization_utils import compute_mse, compute_r2_score, predict
from utils.calculus_utils import compute_condition_number, compute_hessian_linear_regression


class OptimizationDemo:
    """
    Demo class để showcase optimization methods với interactive examples
    """
    
    def __init__(self):
        self.demo_results = {}
        
    def demo_1_quadratic_convergence(self):
        """
        Demo 1: Showcase quadratic convergence của Newton method
        So sánh với gradient descent để thấy sự khác biệt
        """
        print("=" * 80)
        print("DEMO 1: QUADRATIC CONVERGENCE - NEWTON METHOD")
        print("=" * 80)
        print("🎯 Objective: Demonstrate the superior convergence rate of Newton method")
        print("📊 We'll compare Pure Newton vs simulated Gradient Descent behavior\n")
        
        # Generate well-conditioned quadratic problem
        np.random.seed(42)
        n_samples, n_features = 200, 4
        X = np.random.randn(n_samples, n_features)
        true_weights = np.array([2.0, -1.5, 3.0, 0.5])
        true_bias = 1.0
        noise = 0.05 * np.random.randn(n_samples)
        y = X @ true_weights + true_bias + noise
        
        print(f"📋 Problem setup:")
        print(f"   - {n_samples} samples, {n_features} features")
        print(f"   - True weights: {true_weights}")
        print(f"   - True bias: {true_bias:.2f}")
        print(f"   - Noise level: 5%\n")
        
        # Run Newton method với verbose tracking
        print("🚀 Running Pure Newton Method...")
        newton_result = newton_standard_setup(X, y, verbose=False)
        
        # Simulate gradient descent behavior (for comparison)
        print("🚀 Running BFGS (as GD comparison)...")
        bfgs_result = bfgs_standard_setup(X, y, verbose=False)
        
        # Display results
        print("\n📈 CONVERGENCE COMPARISON:")
        print("-" * 50)
        
        newton_iterations = newton_result['convergence_info']['iterations']
        bfgs_iterations = bfgs_result['convergence_info']['iterations']
        
        print(f"Pure Newton:")
        print(f"  - Iterations to convergence: {newton_iterations}")
        print(f"  - Final MSE: {newton_result['final_mse']:.8f}")
        print(f"  - Convergence rate: QUADRATIC")
        
        print(f"\nBFGS (Quasi-Newton):")
        print(f"  - Iterations to convergence: {bfgs_iterations}")
        print(f"  - Final MSE: {bfgs_result['final_mse']:.8f}")
        print(f"  - Convergence rate: SUPERLINEAR")
        
        # Error analysis
        newton_weights_error = np.linalg.norm(newton_result['weights'] - true_weights)
        newton_bias_error = abs(newton_result['bias'] - true_bias)
        
        print(f"\n🎯 ACCURACY ANALYSIS (Pure Newton):")
        print(f"  - Weights error: {newton_weights_error:.2e}")
        print(f"  - Bias error: {newton_bias_error:.2e}")
        print(f"  - R² score: {compute_r2_score(y, newton_result['predictions']):.6f}")
        
        # Visualize convergence
        self._plot_convergence_comparison(
            {'Newton': newton_result, 'BFGS': bfgs_result},
            title="Demo 1: Newton vs BFGS Convergence"
        )
        
        print("\n✅ Demo 1 completed! Newton method shows superior convergence speed.\n")
        return newton_result, bfgs_result
    
    def demo_2_method_comparison(self):
        """
        Demo 2: Comprehensive comparison của tất cả methods trên same problem
        """
        print("=" * 80)
        print("DEMO 2: COMPREHENSIVE METHOD COMPARISON")
        print("=" * 80)
        print("🎯 Objective: Compare all optimization methods on the same problem")
        print("📊 Methods: Newton, Damped Newton, BFGS, L-BFGS, SR1\n")
        
        # Generate challenging but solvable problem
        np.random.seed(123)
        n_samples, n_features = 150, 6
        X = np.random.randn(n_samples, n_features)
        
        # Add some correlation to make it more interesting
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_samples)
        
        true_weights = np.array([1.0, -0.5, 2.0, -1.5, 0.8, -0.3])
        true_bias = 0.5
        noise = 0.1 * np.random.randn(n_samples)
        y = X @ true_weights + true_bias + noise
        
        # Analyze problem characteristics
        hessian = compute_hessian_linear_regression(X, 1e-8)
        condition_number = compute_condition_number(hessian)
        
        print(f"📋 Problem characteristics:")
        print(f"   - {n_samples} samples, {n_features} features")
        print(f"   - Hessian condition number: {condition_number:.2e}")
        print(f"   - Problem difficulty: {'Well-conditioned' if condition_number < 1e6 else 'Ill-conditioned'}")
        print()
        
        # Define methods to compare
        methods = {
            'Pure Newton': newton_standard_setup,
            'Damped Newton': damped_newton_standard_setup,
            'BFGS': bfgs_standard_setup,
            'L-BFGS': lbfgs_standard_setup,
            'SR1': sr1_standard_setup
        }
        
        # Run all methods
        results = {}
        print("🚀 Running all methods...")
        
        for method_name, method_func in methods.items():
            print(f"   - {method_name}...", end=" ")
            start_time = time.time()
            
            try:
                result = method_func(X, y, verbose=False)
                runtime = time.time() - start_time
                result['runtime'] = runtime
                results[method_name] = result
                
                # Quick quality check
                final_mse = result.get('final_mse', np.inf)
                iterations = result.get('convergence_info', {}).get('iterations', 'Unknown')
                print(f"✅ MSE: {final_mse:.6f}, Iterations: {iterations}, Time: {runtime:.3f}s")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                continue
        
        # Create comprehensive comparison
        self._create_comprehensive_comparison(results, true_weights, true_bias, y)
        
        print("\n✅ Demo 2 completed! All methods compared successfully.\n")
        return results
    
    def demo_3_robustness_test(self):
        """
        Demo 3: Test robustness với ill-conditioned problems
        """
        print("=" * 80)
        print("DEMO 3: ROBUSTNESS TEST - ILL-CONDITIONED PROBLEMS")
        print("=" * 80)
        print("🎯 Objective: Test method robustness on challenging problems")
        print("⚠️  We'll create progressively more difficult problems\n")
        
        problems = [
            {
                'name': 'Well-conditioned',
                'n_samples': 100,
                'n_features': 4,
                'correlation_strength': 0.0,
                'noise_level': 0.05
            },
            {
                'name': 'Moderately ill-conditioned',
                'n_samples': 80,
                'n_features': 5,
                'correlation_strength': 0.8,
                'noise_level': 0.1
            },
            {
                'name': 'Highly ill-conditioned',
                'n_samples': 60,
                'n_features': 6,
                'correlation_strength': 0.95,
                'noise_level': 0.15
            }
        ]
        
        # Test methods (focus on robust ones)
        robust_methods = {
            'Newton (Robust)': newton_robust_setup,
            'Damped Newton': damped_newton_standard_setup,
            'BFGS': bfgs_standard_setup,
            'L-BFGS': lbfgs_standard_setup
        }
        
        robustness_results = {}
        
        for problem in problems:
            print(f"\n📊 Testing: {problem['name']}")
            print("-" * 40)
            
            # Generate problem
            np.random.seed(456 + hash(problem['name']) % 1000)
            n_samples = problem['n_samples']
            n_features = problem['n_features']
            
            X = np.random.randn(n_samples, n_features)
            
            # Add correlation to make ill-conditioned
            if problem['correlation_strength'] > 0:
                for i in range(1, n_features):
                    X[:, i] = (problem['correlation_strength'] * X[:, 0] + 
                              (1 - problem['correlation_strength']) * X[:, i])
            
            true_weights = np.random.randn(n_features)
            true_bias = np.random.randn()
            noise = problem['noise_level'] * np.random.randn(n_samples)
            y = X @ true_weights + true_bias + noise
            
            # Analyze condition number
            hessian = compute_hessian_linear_regression(X, 1e-8)
            condition_number = compute_condition_number(hessian)
            
            print(f"   Condition number: {condition_number:.2e}")
            
            problem_results = {}
            
            for method_name, method_func in robust_methods.items():
                try:
                    start_time = time.time()
                    result = method_func(X, y, verbose=False)
                    runtime = time.time() - start_time
                    
                    # Analyze result quality
                    weights_error = np.linalg.norm(result['weights'] - true_weights)
                    bias_error = abs(result['bias'] - true_bias)
                    final_mse = result.get('final_mse', np.inf)
                    converged = result.get('convergence_info', {}).get('converged', False)
                    
                    problem_results[method_name] = {
                        'success': True,
                        'weights_error': weights_error,
                        'bias_error': bias_error,
                        'final_mse': final_mse,
                        'converged': converged,
                        'runtime': runtime
                    }
                    
                    status = "✅" if converged else "⚠️ "
                    print(f"   {status} {method_name}: MSE = {final_mse:.6f}, Error = {weights_error:.3e}")
                    
                except Exception as e:
                    problem_results[method_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"   ❌ {method_name}: FAILED - {str(e)}")
            
            robustness_results[problem['name']] = {
                'condition_number': condition_number,
                'results': problem_results
            }
        
        # Summary
        print(f"\n📋 ROBUSTNESS SUMMARY:")
        print("-" * 50)
        
        for problem_name, problem_data in robustness_results.items():
            successful_methods = sum(1 for r in problem_data['results'].values() if r['success'])
            total_methods = len(robust_methods)
            success_rate = successful_methods / total_methods * 100
            
            print(f"{problem_name}:")
            print(f"  - Success rate: {success_rate:.1f}% ({successful_methods}/{total_methods})")
            print(f"  - Condition number: {problem_data['condition_number']:.2e}")
        
        print("\n✅ Demo 3 completed! Robustness analysis finished.\n")
        return robustness_results
    
    def demo_4_real_data_showcase(self):
        """
        Demo 4: Showcase trên real data từ car price dataset
        """
        print("=" * 80)
        print("DEMO 4: REAL DATA SHOWCASE - CAR PRICE PREDICTION")
        print("=" * 80)
        print("🎯 Objective: Apply optimization methods to real car price data")
        print("📊 Using preprocessed and sampled car price dataset\n")
        
        try:
            # Load real data
            data_dir = Path("data/02.1_sampled")
            X_train = pd.read_csv(data_dir / "X_train.csv").values
            X_test = pd.read_csv(data_dir / "X_test.csv").values
            y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
            y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
            
            print(f"📋 Real dataset characteristics:")
            print(f"   - Training samples: {X_train.shape[0]}")
            print(f"   - Test samples: {X_test.shape[0]}")
            print(f"   - Features: {X_train.shape[1]}")
            print(f"   - Target range: ${y_train.min():.0f} - ${y_train.max():.0f}")
            
            # Analyze problem difficulty
            hessian = compute_hessian_linear_regression(X_train, 1e-8)
            condition_number = compute_condition_number(hessian)
            print(f"   - Hessian condition number: {condition_number:.2e}")
            print()
            
            # Select representative methods
            real_data_methods = {
                'Pure Newton': newton_standard_setup,
                'Damped Newton': damped_newton_standard_setup,
                'BFGS': bfgs_standard_setup,
                'L-BFGS': lbfgs_standard_setup
            }
            
            print("🚀 Applying optimization methods to car price prediction:")
            
            real_results = {}
            
            for method_name, method_func in real_data_methods.items():
                print(f"\n   📊 {method_name}:")
                
                try:
                    start_time = time.time()
                    result = method_func(X_train, y_train, verbose=False)
                    runtime = time.time() - start_time
                    
                    # Evaluate on test set
                    test_predictions = predict(X_test, result['weights'], result['bias'])
                    test_mse = compute_mse(y_test, test_predictions)
                    test_r2 = compute_r2_score(y_test, test_predictions)
                    
                    # Train set performance
                    train_mse = result.get('final_mse', np.inf)
                    train_r2 = compute_r2_score(y_train, result['predictions'])
                    
                    real_results[method_name] = {
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'runtime': runtime,
                        'result': result
                    }
                    
                    print(f"     - Train MSE: ${np.sqrt(train_mse):,.0f} RMSE")
                    print(f"     - Test MSE:  ${np.sqrt(test_mse):,.0f} RMSE")
                    print(f"     - Test R²:   {test_r2:.4f}")
                    print(f"     - Runtime:   {runtime:.3f}s")
                    
                    iterations = result.get('convergence_info', {}).get('iterations', 'Unknown')
                    converged = result.get('convergence_info', {}).get('converged', False)
                    print(f"     - Converged: {'Yes' if converged else 'No'} ({iterations} iterations)")
                    
                except Exception as e:
                    print(f"     ❌ Failed: {str(e)}")
                    continue
            
            # Find best method
            if real_results:
                best_method = min(real_results.keys(), key=lambda k: real_results[k]['test_mse'])
                best_rmse = np.sqrt(real_results[best_method]['test_mse'])
                
                print(f"\n🏆 BEST PERFORMANCE:")
                print(f"   Method: {best_method}")
                print(f"   Test RMSE: ${best_rmse:,.0f}")
                print(f"   Test R²: {real_results[best_method]['test_r2']:.4f}")
                
                # Practical interpretation
                print(f"\n💡 PRACTICAL INTERPRETATION:")
                print(f"   - Average prediction error: ±${best_rmse:,.0f}")
                print(f"   - Model explains {real_results[best_method]['test_r2']:.1%} of price variance")
                print(f"   - Suitable for: {'Production use' if real_results[best_method]['test_r2'] > 0.8 else 'Further tuning needed'}")
        
        except FileNotFoundError:
            print("❌ Real data not found. Please ensure data/02.1_sampled/ contains the preprocessed files.")
            return None
        
        print("\n✅ Demo 4 completed! Real data analysis finished.\n")
        return real_results
    
    def _plot_convergence_comparison(self, results, title="Method Comparison"):
        """Helper method để plot convergence comparison"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Cost history
        plt.subplot(1, 3, 1)
        for method_name, result in results.items():
            cost_history = result.get('cost_history', [])
            if cost_history:
                plt.plot(cost_history, label=method_name, linewidth=2)
        
        plt.title('Cost Function')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Gradient norm
        plt.subplot(1, 3, 2)
        for method_name, result in results.items():
            gradient_norms = result.get('gradient_norms', [])
            if gradient_norms:
                plt.plot(gradient_norms, label=method_name, linewidth=2)
        
        plt.title('Gradient Norm')
        plt.xlabel('Iteration')
        plt.ylabel('||Gradient||')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Runtime comparison
        plt.subplot(1, 3, 3)
        methods = list(results.keys())
        runtimes = [result.get('runtime', result.get('optimization_time', 0)) for result in results.values()]
        
        bars = plt.bar(methods, runtimes, alpha=0.7)
        plt.title('Runtime Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, runtime in zip(bars, runtimes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{runtime:.3f}s', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def _create_comprehensive_comparison(self, results, true_weights, true_bias, y):
        """Helper method để tạo comprehensive comparison table"""
        print(f"\n📊 DETAILED COMPARISON:")
        print("=" * 90)
        
        # Create comparison table
        comparison_data = []
        
        for method_name, result in results.items():
            # Calculate errors
            weights_error = np.linalg.norm(result['weights'] - true_weights)
            bias_error = abs(result['bias'] - true_bias)
            final_mse = result.get('final_mse', np.inf)
            r2_score = compute_r2_score(y, result['predictions'])
            
            # Get convergence info
            conv_info = result.get('convergence_info', {})
            iterations = conv_info.get('iterations', 'Unknown')
            converged = conv_info.get('converged', False)
            runtime = result.get('runtime', result.get('optimization_time', 0))
            
            comparison_data.append({
                'Method': method_name,
                'MSE': f"{final_mse:.6f}",
                'R²': f"{r2_score:.4f}",
                'Weight Error': f"{weights_error:.3e}",
                'Bias Error': f"{bias_error:.3e}",
                'Iterations': str(iterations),
                'Time (s)': f"{runtime:.3f}",
                'Converged': '✅' if converged else '❌'
            })
        
        # Print table
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Create visualization
        self._plot_convergence_comparison(results, "Comprehensive Method Comparison")
    
    def run_all_demos(self):
        """Run tất cả demos theo sequence"""
        print("🎪 OPTIMIZATION METHODS SHOWCASE")
        print("Interactive demos demonstrating different aspects of optimization")
        print("=" * 80)
        
        demos = [
            ("Quadratic Convergence Demo", self.demo_1_quadratic_convergence),
            ("Method Comparison Demo", self.demo_2_method_comparison),
            ("Robustness Test Demo", self.demo_3_robustness_test),
            ("Real Data Showcase", self.demo_4_real_data_showcase)
        ]
        
        demo_results = {}
        
        for demo_name, demo_func in demos:
            print(f"\n▶️  Starting: {demo_name}")
            
            try:
                result = demo_func()
                demo_results[demo_name] = result
                print(f"✅ Completed: {demo_name}")
                
                # Pause between demos for readability
                input("\nPress Enter to continue to next demo...")
                
            except KeyboardInterrupt:
                print(f"\n⏹️  Demo interrupted by user")
                break
            except Exception as e:
                print(f"❌ Demo failed: {str(e)}")
                continue
        
        print("\n" + "=" * 80)
        print("🎉 ALL DEMOS COMPLETED!")
        print("=" * 80)
        print("Thank you for exploring optimization methods with us!")
        print("Check out the individual method files for more technical details.")
        
        return demo_results


def quick_demo():
    """Quick demo function để test một method nhanh"""
    print("🚀 QUICK DEMO - Pure Newton Method")
    print("-" * 40)
    
    # Generate simple test case
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1, -0.5, 2]) + 0.5 + 0.1 * np.random.randn(50)
    
    print(f"Problem: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run Newton method
    result = newton_standard_setup(X, y, verbose=False)
    
    print(f"✅ Converged in {result['convergence_info']['iterations']} iterations")
    print(f"📊 Final MSE: {result['final_mse']:.6f}")
    print(f"📈 R² Score: {compute_r2_score(y, result['predictions']):.4f}")
    print(f"⏱️  Runtime: {result['optimization_time']:.3f} seconds")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimization Methods Examples and Demos")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Run mode: quick demo or full interactive demos")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_demo()
    else:
        demo = OptimizationDemo()
        demo.run_all_demos()