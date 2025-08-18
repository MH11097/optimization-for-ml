#!/usr/bin/env python3
"""
Master script để chạy complete optimization suite
Bao gồm validation, testing, comparison, và demos
"""

import sys
import os
import time
from pathlib import Path
import argparse

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    parser = argparse.ArgumentParser(
        description="Complete Optimization Methods Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_optimization_suite.py --validate --test --compare
  python run_full_optimization_suite.py --demo --quick
  python run_full_optimization_suite.py --all
        """
    )
    
    parser.add_argument("--validate", action="store_true",
                       help="Run mathematical validation tests")
    parser.add_argument("--test", action="store_true",
                       help="Run comprehensive test suite")
    parser.add_argument("--compare", action="store_true",
                       help="Run method comparison on real data")
    parser.add_argument("--demo", action="store_true",
                       help="Run interactive demonstrations")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick version of selected operations")
    parser.add_argument("--all", action="store_true",
                       help="Run all operations (validation + testing + comparison)")
    
    args = parser.parse_args()
    
    # If no specific args, show help
    if not any([args.validate, args.test, args.compare, args.demo, args.all]):
        parser.print_help()
        return
    
    print("🚀 OPTIMIZATION METHODS SUITE")
    print("=" * 60)
    print("Comprehensive implementation of Newton and Quasi-Newton methods")
    print("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # Set operations based on args
    if args.all:
        operations = ['validate', 'test', 'compare']
    else:
        operations = []
        if args.validate: operations.append('validate')
        if args.test: operations.append('test')
        if args.compare: operations.append('compare')
        if args.demo: operations.append('demo')
    
    # Run operations
    for operation in operations:
        print(f"\n▶️  Starting: {operation.upper()}")
        print("-" * 40)
        
        try:
            if operation == 'validate':
                results['validation'] = run_validation(args.quick)
            elif operation == 'test':
                results['testing'] = run_testing(args.quick)
            elif operation == 'compare':
                results['comparison'] = run_comparison(args.quick)
            elif operation == 'demo':
                results['demo'] = run_demo(args.quick)
                
            print(f"✅ Completed: {operation.upper()}")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Operation '{operation}' interrupted by user")
            break
        except Exception as e:
            print(f"❌ Operation '{operation}' failed: {str(e)}")
            results[operation] = {'error': str(e)}
            continue
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 SUITE EXECUTION SUMMARY")
    print("=" * 60)
    
    for operation, result in results.items():
        if isinstance(result, dict) and 'error' in result:
            print(f"❌ {operation.upper()}: FAILED - {result['error']}")
        else:
            print(f"✅ {operation.upper()}: SUCCESS")
            
            # Show specific metrics if available
            if operation == 'validation' and isinstance(result, dict):
                success_rate = result.get('success_rate', 0)
                print(f"   - Tests passed: {success_rate:.1f}%")
                
            elif operation == 'comparison' and isinstance(result, dict):
                successful_methods = sum(1 for r in result if r.get('success', False))
                total_methods = len(result)
                print(f"   - Methods succeeded: {successful_methods}/{total_methods}")
    
    print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
    print("\n🎉 Suite execution completed!")
    
    return results


def run_validation(quick=False):
    """Run validation tests"""
    print("🧪 Running mathematical validation...")
    
    from validate_implementations import ValidationSuite
    
    validator = ValidationSuite(verbose=not quick)
    results = validator.run_all_tests()
    
    if not quick:
        success_rate = results['success_rate']
        if success_rate == 100:
            print("🎉 All validations passed!")
        elif success_rate >= 90:
            print("✅ Most validations passed.")
        else:
            print("⚠️  Some validations failed.")
    
    return results


def run_testing(quick=False):
    """Run comprehensive testing"""
    print("🔬 Running comprehensive tests...")
    
    if quick:
        # Quick synthetic tests only
        from test_optimization_methods import OptimizationTester
        
        tester = OptimizationTester(verbose=False)
        test_data = tester.generate_test_data(n_samples=50, n_features=3)
        results = tester.run_comprehensive_test(test_data)
        
        print("Quick test completed ✅")
        
    else:
        # Full test suite
        from test_optimization_methods import run_basic_tests, run_stress_tests
        
        print("Running basic tests...")
        basic_results = run_basic_tests()
        
        print("\nRunning stress tests...")
        stress_results = run_stress_tests()
        
        results = {
            'basic': basic_results,
            'stress': stress_results
        }
        
        print("Full test suite completed ✅")
    
    return results


def run_comparison(quick=False):
    """Run method comparison on real data"""
    print("📊 Running method comparison...")
    
    try:
        from run_all_methods_comparison import main as run_comparison_main
        
        if quick:
            print("Note: Quick mode still runs full comparison (data loading required)")
        
        results = run_comparison_main()
        print("Method comparison completed ✅")
        
        return results
        
    except FileNotFoundError as e:
        print(f"❌ Data not found: {e}")
        print("Please ensure data/02.1_sampled/ contains the required CSV files")
        raise
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        raise


def run_demo(quick=False):
    """Run demonstrations"""
    print("🎪 Running demonstrations...")
    
    from examples_and_demos import OptimizationDemo, quick_demo
    
    if quick:
        # Quick demo only
        result = quick_demo()
        print("Quick demo completed ✅")
        return result
    else:
        # Full interactive demos
        demo = OptimizationDemo()
        results = demo.run_all_demos()
        print("Interactive demos completed ✅")
        return results


def setup_environment():
    """Setup environment and check dependencies"""
    required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install with: pip install " + ' '.join(missing_packages))
        return False
    
    return True


if __name__ == "__main__":
    # Check environment
    if not setup_environment():
        sys.exit(1)
    
    # Run main
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Suite interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Suite failed with error: {e}")
        sys.exit(1)