# Optimization Algorithms Implementation

A comprehensive implementation of various optimization methods for linear regression, including Newton methods and Quasi-Newton methods.

## 🎯 Overview

This project implements and compares multiple optimization algorithms:

### Newton Methods
- **Pure Newton Method**: Classic Newton optimization using exact Hessian
- **Damped Newton Method**: Newton with line search for global convergence

### Quasi-Newton Methods
- **BFGS**: Broyden-Fletcher-Goldfarb-Shanno algorithm
- **L-BFGS**: Limited Memory BFGS for large-scale problems
- **SR1**: Symmetric Rank-1 update method

## 📁 Project Structure

```
src/algorithms/
├── newton_method/
│   ├── pure_newton.py                 # Pure Newton implementation
│   ├── damped_newton.py              # Damped Newton with line search
│   ├── standard_setup_fixed.py      # Standard setup for real data
│   └── README.md                     # Newton methods documentation
├── quasi_newton/
│   ├── bfgs.py                       # BFGS implementation
│   ├── lbfgs.py                      # L-BFGS implementation
│   ├── sr1.py                        # SR1 implementation
│   └── README.md                     # Quasi-Newton documentation
├── test_optimization_methods.py      # Comprehensive test suite
├── run_all_methods_comparison.py     # Real data comparison
├── validate_implementations.py       # Mathematical validation
├── examples_and_demos.py            # Interactive demos
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Basic Usage

```python
import numpy as np
from newton_method.pure_newton import newton_standard_setup
from quasi_newton.bfgs import bfgs_standard_setup

# Generate sample data
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Run Pure Newton
newton_result = newton_standard_setup(X, y, verbose=True)
print(f"Newton MSE: {newton_result['final_mse']:.6f}")

# Run BFGS
bfgs_result = bfgs_standard_setup(X, y, verbose=True)
print(f"BFGS MSE: {bfgs_result['final_mse']:.6f}")
```

### 2. Compare All Methods

```python
from run_all_methods_comparison import main

# Run comprehensive comparison on real data
results = main()
```

### 3. Interactive Demos

```python
from examples_and_demos import OptimizationDemo

demo = OptimizationDemo()
demo.run_all_demos()  # Interactive demonstrations
```

### 4. Validation and Testing

```python
from validate_implementations import main as validate
from test_optimization_methods import run_basic_tests

# Validate mathematical correctness
validate()

# Run comprehensive tests
run_basic_tests()
```

## 🎯 Method Comparison

| Method | Convergence Rate | Memory | Best For |
|--------|------------------|--------|----------|
| **Pure Newton** | Quadratic | O(n²) | Small problems, high precision |
| **Damped Newton** | Quadratic | O(n²) | Robust Newton with global convergence |
| **BFGS** | Superlinear | O(n²) | Medium problems, good approximation |
| **L-BFGS** | Superlinear | O(mn) | Large problems, memory efficient |
| **SR1** | Superlinear | O(n²) | Indefinite Hessians, trust region |

### Performance Characteristics

#### Newton Methods
```
✅ Advantages:
  - Extremely fast convergence (quadratic)
  - Optimal for quadratic functions
  - High final precision
  
❌ Disadvantages:
  - Expensive Hessian computation O(n³)
  - Large memory requirements O(n²)
  - Sensitive to conditioning
```

#### Quasi-Newton Methods
```
✅ Advantages:
  - Good convergence (superlinear)
  - No Hessian computation needed
  - More robust than Newton
  
❌ Disadvantages:
  - Slower than pure Newton
  - Approximation errors accumulate
  - May need periodic restarts
```

## 🛠️ Implementation Features

### Core Features
- **Mathematical Correctness**: All implementations verified against analytical solutions
- **Robust Error Handling**: Graceful handling of singular matrices and numerical issues
- **Comprehensive Logging**: Detailed convergence tracking and diagnostics
- **Flexible Configurations**: Multiple setups (standard, robust, fast) for each method
- **Real Data Integration**: Works with preprocessed car price dataset

### Advanced Features
- **Line Search**: Backtracking line search with Armijo conditions
- **Regularization**: Automatic regularization for ill-conditioned problems
- **Memory Management**: Efficient memory usage with configurable limits
- **Convergence Analysis**: Multiple stopping criteria and convergence diagnostics
- **Performance Monitoring**: Detailed timing and iteration tracking

## 📊 Usage Examples

### Example 1: Method Selection Guide

```python
# Small, well-conditioned problem → Pure Newton
if n_features < 100 and condition_number < 1e6:
    result = newton_standard_setup(X, y)

# Medium problem → BFGS
elif n_features < 1000:
    result = bfgs_standard_setup(X, y)

# Large problem → L-BFGS
else:
    result = lbfgs_standard_setup(X, y)
```

### Example 2: Custom Configuration

```python
from newton_method.pure_newton import PureNewtonOptimizer

# Custom Newton optimizer
optimizer = PureNewtonOptimizer(
    regularization=1e-6,    # Higher regularization
    max_iterations=100,     # More iterations
    tolerance=1e-8,         # Higher precision
    verbose=True
)

result = optimizer.optimize(X, y)
```

### Example 3: Robustness Testing

```python
from validate_implementations import ValidationSuite

validator = ValidationSuite(verbose=True)
validation_results = validator.run_all_tests()

print(f"Success rate: {validation_results['success_rate']:.1f}%")
```

## 📈 Real Data Application

The implementation is tested on a real car price prediction dataset:

```bash
# Run on car price data
python run_all_methods_comparison.py
```

Expected performance on car price dataset:
- **Test RMSE**: ~$2,000-3,000
- **R² Score**: 0.85-0.90
- **Convergence**: <50 iterations for most methods

## 🧪 Testing and Validation

### Comprehensive Test Suite
```bash
# Run all tests
python test_optimization_methods.py

# Validate implementations
python validate_implementations.py

# Interactive demos
python examples_and_demos.py --mode full
```

### Test Coverage
- ✅ Mathematical correctness verification
- ✅ Gradient and Hessian accuracy
- ✅ Convergence behavior testing
- ✅ Edge case handling
- ✅ Robustness on ill-conditioned problems
- ✅ Performance benchmarking
- ✅ Real data validation

## 🎓 Educational Resources

### Theory and Intuition
Each method includes comprehensive documentation:
- **Mathematical foundations**
- **Algorithm derivations**
- **Convergence proofs**
- **Implementation details**
- **Practical considerations**

### Interactive Learning
```python
from examples_and_demos import OptimizationDemo

demo = OptimizationDemo()

# Individual demos
demo.demo_1_quadratic_convergence()  # Newton convergence showcase
demo.demo_2_method_comparison()      # Side-by-side comparison
demo.demo_3_robustness_test()        # Ill-conditioning tests
demo.demo_4_real_data_showcase()     # Real world application
```

## 🔧 Configuration Options

### Newton Methods
```python
# Standard: balanced performance
newton_standard_setup(X, y)

# Robust: better conditioning
newton_robust_setup(X, y)

# Fast: quick convergence
newton_fast_setup(X, y)
```

### Quasi-Newton Methods
```python
# BFGS configurations
bfgs_standard_setup(X, y)    # Standard BFGS
bfgs_robust_setup(X, y)      # With restarts
bfgs_fast_setup(X, y)        # Fewer iterations

# L-BFGS configurations
lbfgs_standard_setup(X, y)           # memory=10
lbfgs_memory_efficient_setup(X, y)   # memory=5
lbfgs_high_memory_setup(X, y)        # memory=20

# SR1 configurations
sr1_standard_setup(X, y)     # Balanced
sr1_robust_setup(X, y)       # Conservative
sr1_aggressive_setup(X, y)   # Permissive
```

## 🐛 Troubleshooting

### Common Issues

1. **Singular Hessian**
```python
# Solution: Increase regularization
optimizer = PureNewtonOptimizer(regularization=1e-4)
```

2. **Slow Convergence**
```python
# Solution: Check conditioning or try different method
condition_number = compute_condition_number(hessian)
if condition_number > 1e12:
    use_lbfgs_instead()
```

3. **Memory Issues**
```python
# Solution: Use L-BFGS for large problems
result = lbfgs_memory_efficient_setup(X, y)
```

### Debug Mode
```python
# Enable detailed logging
result = newton_standard_setup(X, y, verbose=True)

# Check convergence details
print(result['convergence_info'])
print(f"Condition number: {result['hessian_condition_number']}")
```

## 📄 License and Citation

This implementation is for educational and research purposes. If you use this code in research, please cite:

```bibtex
@misc{optimization_methods_2024,
  title={Comprehensive Implementation of Newton and Quasi-Newton Methods},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional optimization methods (Conjugate Gradient, Adam, etc.)
- GPU acceleration
- Sparse matrix support
- More comprehensive benchmarks
- Extended real-world applications

## 📚 References

1. Nocedal, J., & Wright, S. (2006). *Numerical Optimization*. Springer.
2. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
3. Bertsekas, D. P. (2016). *Nonlinear Programming*. Athena Scientific.

---

*Built with ❤️ for optimization enthusiasts and machine learning practitioners*