# Quasi-Newton Methods Implementation

This directory contains comprehensive implementations of three major Quasi-Newton optimization algorithms: **BFGS**, **L-BFGS**, and **SR1** for machine learning applications.

## 🎯 Overview

Quasi-Newton methods are second-order optimization algorithms that approximate the Hessian matrix to achieve superlinear convergence while avoiding the computational cost of computing the actual Hessian. They are particularly effective for unconstrained optimization problems in machine learning.

## 🧮 Algorithms Implemented

### 1. BFGS (Broyden-Fletcher-Goldfarb-Shanno)
- **Method**: `method='bfgs'`
- **Memory**: O(n²) storage
- **Convergence**: Superlinear
- **Best for**: Small-medium problems with fast convergence needs
- **Features**: Guaranteed positive definite updates, full Hessian approximation

### 2. L-BFGS (Limited-memory BFGS) 
- **Method**: `method='lbfgs'`
- **Memory**: O(mn) storage where m is memory size
- **Convergence**: Near-superlinear
- **Best for**: Large-scale problems, memory-constrained environments
- **Features**: Two-loop recursion, configurable memory size

### 3. SR1 (Symmetric Rank-1)
- **Method**: `method='sr1'` 
- **Memory**: O(n²) storage
- **Convergence**: Linear to superlinear
- **Best for**: Problems with indefinite Hessians, experimental use
- **Features**: Can handle non-convex problems, skip condition for stability

## 🚀 Quick Start

```python
from algorithms.quasi_newton.quasi_newton_model import QuasiNewtonModel
from utils.data_process_utils import load_du_lieu

# Load data
X_train, X_test, y_train, y_test = load_du_lieu()

# Initialize model (choose your method)
model = QuasiNewtonModel(
    ham_loss='ols',           # 'ols', 'ridge', 'lasso'
    method='bfgs',            # 'bfgs', 'lbfgs', 'sr1' 
    diem_dung=1e-5,          # Convergence tolerance
    memory_size=10,          # L-BFGS memory (ignored for other methods)
    sr1_skip_threshold=1e-8  # SR1 skip condition (ignored for other methods)
)

# Train and evaluate
results = model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)

# Save comprehensive results
model.save_results("experiment_name")
model.plot_results(X_test, y_test, "experiment_name")
```

## 📁 Setup Scripts

Pre-configured experiments for different algorithm combinations:

### BFGS Experiments
- `01a_setup_bfgs_ols.py` - BFGS with Ordinary Least Squares
- `01b_setup_bfgs_ridge.py` - BFGS with Ridge regularization

### L-BFGS Experiments  
- `02a_setup_lbfgs_ols_m5.py` - L-BFGS (memory=5) with OLS
- `02b_setup_lbfgs_ols_m10.py` - L-BFGS (memory=10) with OLS
- `02c_setup_lbfgs_ridge_m5.py` - L-BFGS (memory=5) with Ridge

### SR1 Experiments
- `03a_setup_sr1_ols.py` - SR1 with OLS
- `03b_setup_sr1_ridge.py` - SR1 with Ridge regularization

### Benchmarking
- `99_setup_scipy_comparison.py` - Comprehensive comparison with SciPy implementations

## 🔬 Algorithm Mathematics

### BFGS Update Formula
BFGS maintains inverse Hessian approximation $B_k^{-1}$:

$$B_{k+1}^{-1} = (I - \\rho_k s_k y_k^T) B_k^{-1} (I - \\rho_k y_k s_k^T) + \\rho_k s_k s_k^T$$

Where:
- $s_k = x_{k+1} - x_k$ (parameter step)
- $y_k = \\nabla f_{k+1} - \\nabla f_k$ (gradient change)
- $\\rho_k = 1/(y_k^T s_k)$ (curvature measure)

### L-BFGS Two-Loop Recursion
Computes search direction without forming full inverse Hessian:

**Backward Loop:**
```
for i = m-1 down to 0:
    α_i = ρ_i · (s_i^T · q)
    q = q - α_i · y_i
end
```

**Forward Loop:**
```
for i = 0 to m-1:
    β_i = ρ_i · (y_i^T · r)
    r = r + (α_i - β_i) · s_i
end
```

### SR1 Update Formula
Symmetric rank-1 updates:

$$B_{k+1}^{-1} = B_k^{-1} + \\frac{(s_k - B_k^{-1} y_k)(s_k - B_k^{-1} y_k)^T}{(s_k - B_k^{-1} y_k)^T y_k}$$

**Skip Condition** (for numerical stability):
$$|(s_k - B_k^{-1} y_k)^T y_k| < \\epsilon \\|s_k - B_k^{-1} y_k\\| \\|y_k\\|$$

## ⚙️ Parameters Guide

### Core Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `ham_loss` | Loss function | 'ols' | 'ols', 'ridge', 'lasso' |
| `method` | Algorithm | 'bfgs' | 'bfgs', 'lbfgs', 'sr1' |
| `so_lan_thu` | Max iterations | 10000 | 100-50000 |
| `diem_dung` | Convergence tolerance | 1e-5 | 1e-8 to 1e-3 |
| `regularization` | L1/L2 penalty | 0.01 | 0.001-0.1 |

### Line Search Parameters
| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `armijo_c1` | Armijo constant | 1e-4 | 1e-6 to 1e-2 |
| `wolfe_c2` | Wolfe curvature | 0.9 | 0.1 to 0.99 |
| `backtrack_rho` | Step reduction | 0.8 | 0.1 to 0.9 |
| `max_line_search_iter` | Max line search | 50 | 10-100 |

### Method-Specific Parameters
| Method | Parameter | Description | Default | Recommended |
|--------|-----------|-------------|---------|-------------|
| L-BFGS | `memory_size` | History vectors | 10 | 5-20 |
| SR1 | `sr1_skip_threshold` | Skip condition | 1e-8 | 1e-10 to 1e-6 |
| All | `damping` | Numerical stability | 1e-8 | 1e-12 to 1e-6 |

## 📊 Results Analysis

Each experiment generates comprehensive analysis:

### Output Files
- **`results.json`**: Complete algorithm metrics and parameters
- **`training_history.csv`**: Iteration-by-iteration progress data
- **`convergence_analysis.png`**: Loss/gradient convergence visualization
- **`predictions_vs_actual.png`**: Model prediction quality assessment
- **`optimization_trajectory.png`**: Parameter space optimization path

### Key Performance Metrics

#### Convergence Metrics
- **Final gradient norm**: Measure of optimization convergence
- **Loss reduction ratio**: Initial loss / final loss
- **Iterations to convergence**: Efficiency measure
- **Convergence rate**: Linear vs superlinear classification

#### Computational Metrics
- **Training time**: Wall-clock optimization time
- **Function evaluations**: Total objective function calls
- **Line search efficiency**: Average line search iterations
- **Memory usage**: For L-BFGS memory utilization

#### Algorithm-Specific Metrics
- **BFGS**: Condition number evolution
- **L-BFGS**: Memory usage vs allocated memory
- **SR1**: Skip rate and numerical stability

## 🎯 Algorithm Selection Guide

### Problem Size Recommendations

| Problem Size | Features (n) | Recommended Method | Memory Size (L-BFGS) |
|--------------|--------------|--------------------|-----------------------|
| Small | < 100 | BFGS | - |
| Medium | 100-1000 | BFGS or L-BFGS | 10-20 |
| Large | 1000-10000 | L-BFGS | 5-10 |
| Very Large | > 10000 | L-BFGS | 3-7 |

### Use Case Guidelines

**Choose BFGS when:**
- ✅ Fast convergence is critical
- ✅ Problem size is manageable (n < 1000)
- ✅ Memory is not a constraint
- ✅ High accuracy is required

**Choose L-BFGS when:**
- ✅ Problem is large-scale (n > 1000)
- ✅ Memory is limited
- ✅ Good balance of speed/memory needed
- ✅ Most general-purpose applications

**Choose SR1 when:**
- ✅ Hessian may be indefinite
- ✅ Non-convex optimization
- ✅ Experimental/research applications
- ✅ Diagonal dominance in problem structure

## 🔄 Parameter Tuning Guidelines

### L-BFGS Memory Size
- **m=3-5**: Very memory constrained, may need more iterations
- **m=10**: Balanced choice (recommended default)
- **m=15-20**: Better Hessian approximation, more memory
- **m>20**: Diminishing returns, approaching full BFGS

### SR1 Skip Threshold
- **1e-10**: Very conservative, fewer risky updates
- **1e-8**: Standard choice (recommended)
- **1e-6**: More aggressive, higher update rate
- **1e-4**: Risk numerical instability

### Convergence Tolerance
- **1e-3**: Fast termination, lower accuracy
- **1e-5**: Balanced accuracy (recommended)
- **1e-7**: High accuracy applications
- **1e-9**: Research-grade precision

## 🏆 Performance Benchmarking

### Running Comparisons
```bash
# Compare all methods against SciPy
python 99_setup_scipy_comparison.py

# Run individual experiments
python 01a_setup_bfgs_ols.py
python 02b_setup_lbfgs_ols_m10.py  
python 03a_setup_sr1_ols.py
```

### Typical Performance Characteristics

| Algorithm | Iterations | Convergence Rate | Memory | Function Evals |
|-----------|------------|------------------|--------|----------------|
| BFGS | 5-25 | Superlinear | O(n²) | 10-50 |
| L-BFGS | 10-100 | Near-superlinear | O(mn) | 20-200 |
| SR1 | 10-200 | Linear-superlinear | O(n²) | 20-400 |

### Convergence Comparison
```
BFGS:    ████████████████████████████ Fastest
L-BFGS:  ██████████████████████ Fast  
SR1:     ████████████ Variable
```

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions

**Slow Convergence:**
- Check gradient computation accuracy
- Reduce convergence tolerance temporarily
- Try different line search parameters
- Verify problem scaling

**Memory Issues (BFGS):**
- Switch to L-BFGS with small memory size
- Reduce dataset size for testing
- Check for memory leaks in gradient computation

**Numerical Instability (SR1):**
- Increase skip threshold (`sr1_skip_threshold`)
- Check condition number evolution
- Verify problem conditioning

**Line Search Failures:**
- Reduce initial step size
- Adjust Wolfe parameters (`armijo_c1`, `wolfe_c2`)
- Increase `max_line_search_iter`

### Diagnostic Tools

Monitor these metrics during training:
- Gradient norm trajectory
- Condition number evolution (BFGS/SR1)
- Step size history
- Line search iterations per step
- Skip rate (SR1)

## 📚 References and Theory

### Essential Papers
1. **BFGS**: Broyden, C.G. (1970). *The convergence of a class of double-rank minimization algorithms*
2. **L-BFGS**: Liu, D.C. & Nocedal, J. (1989). *On the limited memory BFGS method for large scale optimization*
3. **SR1**: Conn, A.R., Gould, N.I. & Toint, P.L. (2000). *Trust Region Methods*

### Recommended Textbooks
- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.) - **The definitive reference**
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization* - **Theoretical foundations**
- Dennis, J.E. & Schnabel, R.B. (1996). *Numerical Methods for Unconstrained Optimization*

### Online Resources
- [Optimization Online](http://www.optimization-online.org/) - Latest research
- [NEOS Guide](https://neos-guide.org/optimization-tree) - Algorithm selection guide
- [SciPy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) - Reference implementation

## 🚦 Advanced Usage

### Custom Loss Functions
Extend the framework by adding custom loss functions to `optimization_utils.py`:

```python
# Add to tinh_gia_tri_ham_loss and tinh_gradient_ham_loss
if ham_loss == 'custom':
    # Implement your custom loss and gradient
    pass
```

### Monitoring Training
```python
model = QuasiNewtonModel(
    method='lbfgs',
    convergence_check_freq=1  # Check convergence every iteration
)

results = model.fit(X_train, y_train)

# Access detailed training history
print(f"Loss history: {results['loss_history']}")
print(f"Gradient norms: {results['gradient_norms']}")
print(f"Step sizes: {results['step_sizes']}")
```

### Batch Processing
```python
# Process multiple configurations
configs = [
    {'method': 'bfgs', 'ham_loss': 'ols'},
    {'method': 'lbfgs', 'memory_size': 5, 'ham_loss': 'ridge'},
    {'method': 'sr1', 'sr1_skip_threshold': 1e-6, 'ham_loss': 'ols'}
]

results = []
for config in configs:
    model = QuasiNewtonModel(**config)
    result = model.fit(X_train, y_train)
    results.append(result)
```

---

**🎉 Ready to optimize? Start with `01a_setup_bfgs_ols.py` for a quick introduction, or dive into `99_setup_scipy_comparison.py` for comprehensive benchmarking!**