#!/usr/bin/env python3
"""
Demo script để kiểm tra library comparison framework
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def demo_available_libraries():
    """Demo các thư viện có sẵn"""
    print("=" * 80)
    print("DEMO: AVAILABLE LIBRARY OPTIMIZERS")
    print("=" * 80)
    
    # Check library gradient descent
    try:
        from src.optimization.external.library_gradient_descent import get_available_optimizers
        gd_optimizers = get_available_optimizers()
        print("\n📚 Library Gradient Descent:")
        for name, available in gd_optimizers.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"  - {name}: {status}")
    except ImportError as e:
        print(f"\n❌ Library Gradient Descent module not available: {e}")
    
    # Check library newton
    try:
        from src.optimization.external.library_newton import get_available_optimizers
        newton_optimizers = get_available_optimizers()
        print("\n🔍 Library Newton Methods:")
        for name, available in newton_optimizers.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"  - {name}: {status}")
    except ImportError as e:
        print(f"\n❌ Library Newton module not available: {e}")
    
    # Check library quasi-newton
    try:
        from src.optimization.external.library_quasi_newton import get_available_optimizers
        quasi_optimizers = get_available_optimizers()
        print("\n⚡ Library Quasi-Newton Methods:")
        for name, available in quasi_optimizers.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"  - {name}: {status}")
    except ImportError as e:
        print(f"\n❌ Library Quasi-Newton module not available: {e}")

def demo_algorithm_comparator():
    """Demo AlgorithmComparator với library support"""
    print("\n" + "=" * 80)
    print("DEMO: ALGORITHM COMPARATOR WITH LIBRARY SUPPORT")
    print("=" * 80)
    
    try:
        from src.algorithm_comparator import AlgorithmComparator
        
        print("\n🔧 AlgorithmComparator Features:")
        print("  ✅ Automatic library algorithm detection")
        print("  ✅ Library vs Custom comparison reports")
        print("  ✅ Enhanced CSV export with library-specific columns")
        print("  ✅ Library-specific parameter extraction")
        print("  ✅ Multi-library performance analysis")
        
        print("\n📊 Usage Examples:")
        print("  # Compare custom gradient descent algorithms (111-140)")
        print("  comparator = AlgorithmComparator('gradient_descent', 111, 140)")
        print("  comparator.run_comparison()")
        print()
        print("  # Compare library gradient descent algorithms (401-450)")
        print("  comparator = AlgorithmComparator('library_gradient_descent', 401, 450)")
        print("  comparator.run_comparison()")
        print()
        print("  # Compare library quasi-newton algorithms (601-650)")
        print("  comparator = AlgorithmComparator('library_quasi_newton', 601, 650)")
        print("  comparator.run_comparison()")
        
    except ImportError as e:
        print(f"\n❌ AlgorithmComparator not available: {e}")

def demo_experiment_templates():
    """Demo experiment templates"""
    print("\n" + "=" * 80)
    print("DEMO: EXPERIMENT TEMPLATES")
    print("=" * 80)
    
    print("\n📝 Available Experiment Templates:")
    
    templates = [
        {
            'path': 'src/experimental_setups/library_gradient_descent/401_library_gd_sklearn_sgd_lr_001.py',
            'description': 'sklearn SGD với learning rate 0.001',
            'series': '4XX - Library Gradient Descent'
        },
        {
            'path': 'src/experimental_setups/library_quasi_newton/601_library_quasi_scipy_bfgs_ols.py', 
            'description': 'SciPy BFGS với OLS loss function',
            'series': '6XX - Library Quasi-Newton'
        }
    ]
    
    for template in templates:
        template_path = Path(template['path'])
        if template_path.exists():
            print(f"\n  ✅ {template['series']}")
            print(f"     File: {template['path']}")
            print(f"     Description: {template['description']}")
            print(f"     Status: Template ready")
        else:
            print(f"\n  ❌ {template['series']}")
            print(f"     File: {template['path']}")  
            print(f"     Status: Template not found")

def main():
    """Main demo function"""
    print("🚀 LIBRARY COMPARISON FRAMEWORK DEMO")
    print("=" * 80)
    print("Framework để so sánh custom algorithms với library implementations")
    print("Hỗ trợ sklearn, pytorch, tensorflow, scipy, jax và các thư viện khác")
    
    # Demo available libraries
    demo_available_libraries()
    
    # Demo algorithm comparator
    demo_algorithm_comparator()
    
    # Demo experiment templates
    demo_experiment_templates()
    
    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS:")
    print("=" * 80)
    print("1. Install required dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Run sample experiments:")
    print("   python src/experimental_setups/library_gradient_descent/401_library_gd_sklearn_sgd_lr_001.py")
    print()
    print("3. Compare results:")
    print("   python src/algorithm_comparator.py library_gradient_descent 401 410")
    print()
    print("4. View comprehensive comparison:")
    print("   # Sẽ tạo library_vs_custom_comparison.md nếu có cả library và custom algorithms")
    
    print("\n" + "=" * 80)
    print("✅ FRAMEWORK SETUP COMPLETE!")
    print("Ready để so sánh custom vs library optimization algorithms!")
    print("=" * 80)

if __name__ == "__main__":
    main()