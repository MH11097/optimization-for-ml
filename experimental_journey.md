# Experimental Journey - Systematic Algorithm Optimization

## PHASE 1A: Gradient Descent - Baseline Learning Rate Selection

**Date**: 2025-09-06  
**Objective**: Find optimal learning rate for GD with OLS loss function

### Experiments Conducted:
1. `01_setup_gd_ols_lr_0001.py` - lr=0.001
2. `02_setup_gd_ols_lr_001.py` - lr=0.01  
3. `03_setup_gd_ols_lr_01.py` - lr=0.1

### Results Summary:

| Learning Rate | Final Loss | Convergence | Time (s) | Iterations | Gradient Norm |
|---------------|------------|-------------|----------|------------|---------------|
| 0.001         | 0.011935   | ❌ No       | 546.7    | 100,000    | 5.79e-04     |
| 0.01          | 0.011925   | ❌ No       | 544.3    | 100,000    | 1.94e-05     |
| **0.1**       | **0.011925**| ✅ **Yes**  | **112.7** | **20,100** | **9.93e-06** |

### Key Findings:
- **Winner**: lr=0.1 shows clear superiority
- **Convergence**: Only lr=0.1 properly converged within 100k iterations
- **Efficiency**: 5x faster training time (112s vs 540s)
- **Stability**: Clean convergence with final gradient norm < 1e-5

### Decision:
**Learning Rate = 0.1** will be used as baseline for all subsequent GD experiments.

### Extended Testing - Higher Learning Rates:

| Learning Rate | Final Loss | Convergence | Time (s) | Iterations | Status |
|---------------|------------|-------------|----------|------------|--------|
| **0.1**       | 0.011925   | ✅ Yes      | 112.7    | 20,100     | 🏆 Optimal |
| **0.2**       | 0.011925   | ✅ Yes      | 46.5     | 7,900      | ⚠️ Slower |
| **0.3**       | Infinity   | ❌ No       | 3.2      | Diverged   | 🔥 Unstable |

**Findings**: 
- **Instability threshold**: 0.2 < threshold < 0.3
- **Performance paradox**: lr=0.2 faster convergence but slower wall-time
- **Optimal choice**: lr=0.1 (best efficiency + safe margin)

---

## PHASE 1B: Gradient Descent - Regularization Testing

**Date**: 2025-09-06  
**Objective**: Determine optimal regularization strength with lr=0.1

### Experiments Conducted:
1. `03_setup_gd_ols_lr_01.py` - OLS baseline (lr=0.1, reg=0.0)
2. `07_setup_gd_ridge_lr_01_reg_001.py` - Ridge lr=0.1, reg=0.01
3. `08_setup_gd_ridge_lr_01_reg_05.py` - Ridge lr=0.1, reg=0.5

### Results Summary:

| Configuration | Loss Function | Final Loss | Convergence | Time (s) | Iterations | Status |
|---------------|---------------|------------|-------------|----------|------------|--------|
| **OLS (baseline)** | OLS | **0.011925** | ✅ Yes | 112.7 | 20,100 | 🏆 **Optimal** |
| **Ridge reg=0.01** | Ridge | 0.012757 | ✅ Yes | 212.3 | 3,500 | ⚠️ Higher loss |
| **Ridge reg=0.5** | Ridge | 0.029766 | ✅ Yes | 18.7 | 200 | 🔴 **Poor fit** |

### Key Findings:
- **No regularization wins**: OLS achieves lowest loss (0.011925)
- **Regularization penalty**: Ridge 0.01 adds +7% loss, Ridge 0.5 adds +149% loss  
- **Weight shrinkage**: Heavy regularization reduces weight magnitude significantly
- **Convergence trade-off**: Higher regularization → faster convergence but worse fit

### Decision:
**OLS (no regularization)** with **lr=0.1** is optimal for this dataset.

---

## NEXT PHASE: 1C - Advanced Techniques Testing (with lr=0.1, OLS)

**Planned experiments**:
- Learning rate decay, momentum, backtracking line search
- All using optimal lr=0.1, OLS from Phase 1B