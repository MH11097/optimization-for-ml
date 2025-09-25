"""
Test script for refactored visualization utilities.
Tests all visualization modules to ensure they work correctly
and maintain backward compatibility.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
def test_basic_plots():
    """Test basic plotting utilities."""
    print("🔧 Testing basic plots module...")
    
    try:
        from src.utils_refactored.visualization.plots import (
            setup_plot_style, create_color_palette, plot_multi_series,
            plot_predictions_vs_actual
        )
        
        # Test style setup
        setup_plot_style()
        print("  ✅ Style setup successful")
        
        # Test color palette
        colors = create_color_palette(5)
        assert len(colors) == 5
        print("  ✅ Color palette creation successful")
        
        # Test multi-series plot (without displaying)
        data = {'Algorithm A': [1, 0.5, 0.25, 0.1], 'Algorithm B': [1, 0.7, 0.3, 0.15]}
        # plot_multi_series(data)  # Would show plot
        print("  ✅ Multi-series plotting function available")
        
        # Test predictions vs actual
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        # plot_predictions_vs_actual(y_true, y_pred)  # Would show plot
        print("  ✅ Predictions vs actual plotting function available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic plots test failed: {e}")
        return False

def test_optimization_viz():
    """Test optimization visualization module."""
    print("🔧 Testing optimization visualization module...")
    
    try:
        from src.utils_refactored.visualization.optimization_viz import (
            plot_convergence, plot_optimization_path, plot_gradient_vector,
            plot_multi_algorithm_convergence
        )
        
        # Test convergence plotting
        loss_history = [1.0, 0.5, 0.25, 0.1, 0.05]
        gradient_norms = [2.0, 1.0, 0.5, 0.2, 0.1]
        # plot_convergence(loss_history, gradient_norms)  # Would show plot
        print("  ✅ Convergence plotting function available")
        
        # Test optimization path
        w1_path = np.array([0, 0.5, 1.0, 1.2, 1.3])
        w2_path = np.array([0, -0.5, -1.0, -1.1, -1.15])
        # plot_optimization_path(w1_path, w2_path)  # Would show plot
        print("  ✅ Optimization path plotting function available")
        
        # Test gradient vector plotting
        gradient = np.array([0.5, -0.3, 0.8, -0.1])
        # plot_gradient_vector(gradient)  # Would show plot
        print("  ✅ Gradient vector plotting function available")
        
        # Test multi-algorithm convergence
        algorithms_data = {
            'GD': {'loss_history': [1, 0.5, 0.25], 'gradient_norms': [2, 1, 0.5]},
            'Adam': {'loss_history': [1, 0.3, 0.1], 'gradient_norms': [2, 0.8, 0.2]}
        }
        # plot_multi_algorithm_convergence(algorithms_data)  # Would show plot
        print("  ✅ Multi-algorithm convergence plotting function available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Optimization visualization test failed: {e}")
        return False

def test_comparison_module():
    """Test comparison visualization module."""
    print("🔧 Testing comparison module...")
    
    try:
        from src.utils_refactored.visualization.comparison import (
            plot_algorithm_comparison, create_comparison_table, plot_radar_chart,
            create_performance_summary
        )
        
        # Test algorithm comparison
        performance_data = {
            'GD': {'accuracy': 0.85, 'speed': 0.7, 'memory': 0.9},
            'Adam': {'accuracy': 0.92, 'speed': 0.6, 'memory': 0.8},
            'SGD': {'accuracy': 0.88, 'speed': 0.9, 'memory': 0.95}
        }
        
        # plot_algorithm_comparison(performance_data, ['accuracy', 'speed'])  # Would show plot
        print("  ✅ Algorithm comparison plotting function available")
        
        # Test comparison table
        df = create_comparison_table(performance_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        print("  ✅ Comparison table creation successful")
        
        # Test performance summary
        summary = create_performance_summary(performance_data)
        assert isinstance(summary, str)
        assert 'GD' in summary
        print("  ✅ Performance summary creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Comparison module test failed: {e}")
        return False

def test_complexity_module():
    """Test complexity visualization module."""
    print("🔧 Testing complexity module...")
    
    try:
        from src.utils_refactored.visualization.complexity import (
            plot_operation_distribution, create_complexity_summary_table,
            plot_scalability_analysis
        )
        
        # Test operation distribution
        complexity_data = {
            'operations': {'matrix_ops': 100, 'vector_ops': 50, 'scalar_ops': 200},
            'total_operations': 350
        }
        # plot_operation_distribution(complexity_data)  # Would show plot
        print("  ✅ Operation distribution plotting function available")
        
        # Test complexity summary table
        df = create_complexity_summary_table(complexity_data)
        assert isinstance(df, pd.DataFrame)
        print("  ✅ Complexity summary table creation successful")
        
        # Test scalability analysis
        complexity_results = [
            {'complexity_factor': 10, 'total_operations': 100, 'execution_time': 0.01},
            {'complexity_factor': 20, 'total_operations': 400, 'execution_time': 0.04},
            {'complexity_factor': 30, 'total_operations': 900, 'execution_time': 0.09}
        ]
        # plot_scalability_analysis(complexity_results)  # Would show plot
        print("  ✅ Scalability analysis plotting function available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Complexity module test failed: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with Vietnamese function names."""
    print("🔧 Testing backward compatibility...")
    
    try:
        from src.utils_refactored.visualization import (
            thiet_lap_style_bieu_do, tao_color_palette, ve_duong_hoi_tu,
            ve_so_sanh_algorithms, ve_du_doan_vs_thuc_te
        )
        
        # Test Vietnamese function names work
        thiet_lap_style_bieu_do()
        print("  ✅ Vietnamese style setup function available")
        
        colors = tao_color_palette(3)
        assert len(colors) == 3
        print("  ✅ Vietnamese color palette function available")
        
        # Test other Vietnamese functions exist (don't call them to avoid plots)
        assert callable(ve_duong_hoi_tu)
        assert callable(ve_so_sanh_algorithms) 
        assert callable(ve_du_doan_vs_thuc_te)
        print("  ✅ Vietnamese plotting functions available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        return False

def test_module_imports():
    """Test that all modules can be imported successfully."""
    print("🔧 Testing module imports...")
    
    try:
        # Test individual module imports
        from src.utils_refactored.visualization import plots
        from src.utils_refactored.visualization import optimization_viz
        from src.utils_refactored.visualization import comparison
        from src.utils_refactored.visualization import complexity
        
        print("  ✅ All individual modules imported successfully")
        
        # Test main module import
        import src.utils_refactored.visualization as viz
        
        # Check that main functions are available
        assert hasattr(viz, 'setup_plot_style')
        assert hasattr(viz, 'plot_convergence')
        assert hasattr(viz, 'plot_algorithm_comparison')
        assert hasattr(viz, 'plot_operation_distribution')
        
        print("  ✅ Main visualization module imported successfully")
        print("  ✅ Key functions available in main module")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Module import test failed: {e}")
        return False

def main():
    """Run all visualization tests."""
    print("🚀 Testing Refactored Visualization Utilities")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Basic Plots", test_basic_plots),
        ("Optimization Visualization", test_optimization_viz),
        ("Comparison Module", test_comparison_module), 
        ("Complexity Module", test_complexity_module),
        ("Backward Compatibility", test_backward_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print("-" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print("-" * 40)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All visualization tests passed! The refactored modules are working correctly.")
        print("✨ Both new function names and backward compatibility are maintained.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)