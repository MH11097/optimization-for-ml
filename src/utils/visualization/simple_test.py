"""
Simple test for visualization modules only.
Tests visualization functions directly without core dependencies.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_plots_module():
    """Test plots module directly."""
    print("🔧 Testing plots module...")
    
    try:
        from plots import setup_plot_style, create_color_palette
        
        # Test style setup
        setup_plot_style()
        print("  ✅ Style setup successful")
        
        # Test color palette
        colors = create_color_palette(5)
        assert len(colors) == 5
        print("  ✅ Color palette creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Plots module test failed: {e}")
        return False

def test_optimization_viz_module():
    """Test optimization_viz module directly."""
    print("🔧 Testing optimization_viz module...")
    
    try:
        from optimization_viz import plot_convergence, plot_gradient_vector
        
        # Test that functions are callable
        assert callable(plot_convergence)
        assert callable(plot_gradient_vector)
        print("  ✅ Optimization visualization functions available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Optimization visualization test failed: {e}")
        return False

def test_comparison_module():
    """Test comparison module directly."""
    print("🔧 Testing comparison module...")
    
    try:
        from comparison import create_comparison_table, create_performance_summary
        
        # Test comparison table
        performance_data = {
            'GD': {'accuracy': 0.85, 'speed': 0.7},
            'Adam': {'accuracy': 0.92, 'speed': 0.6}
        }
        
        df = create_comparison_table(performance_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
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
    """Test complexity module directly."""
    print("🔧 Testing complexity module...")
    
    try:
        from complexity import create_complexity_summary_table
        
        # Test complexity summary table
        complexity_data = {
            'operations': {'matrix_ops': 100, 'vector_ops': 50},
            'total_operations': 150
        }
        
        df = create_complexity_summary_table(complexity_data)
        assert isinstance(df, pd.DataFrame)
        print("  ✅ Complexity summary table creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Complexity module test failed: {e}")
        return False

def main():
    """Run all visualization tests."""
    print("🚀 Testing Visualization Modules Directly")
    print("=" * 50)
    
    tests = [
        ("Plots Module", test_plots_module),
        ("Optimization Viz", test_optimization_viz_module), 
        ("Comparison Module", test_comparison_module),
        ("Complexity Module", test_complexity_module)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Print results summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    print("-" * 30)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print("-" * 30)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All visualization module tests passed!")
        print("✨ The refactored visualization modules are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()