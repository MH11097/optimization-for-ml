# Proximal Gradient Descent

## Mathematical Foundation

### Definition

Proximal gradient descent is an optimization algorithm designed for composite objective functions of the form:

$$\min_{x} f(x) + g(x)$$

where:
- $f(x)$ is a smooth, convex function with Lipschitz continuous gradient
- $g(x)$ is a convex but potentially non-smooth function (typically a regularizer)

This decomposition naturally arises in regularized machine learning problems where $f$ represents the data fitting term and $g$ represents the regularization penalty.

### Algorithm Formulation

The proximal gradient algorithm alternates between two steps:

1. **Gradient Step**: $z_k = x_k - \alpha_k \nabla f(x_k)$
2. **Proximal Step**: $x_{k+1} = \text{prox}_{\alpha_k g}(z_k)$

where the proximal operator is defined as:

$$\text{prox}_{\lambda h}(v) = \underset{x}{\arg\min} \left\{ h(x) + \frac{1}{2\lambda} \|x - v\|^2 \right\}$$

### Proximal Operator Properties

The proximal operator has several important properties:

1. **Uniqueness**: For convex $h$, the proximal operator has a unique solution
2. **Non-expansive**: $\|\text{prox}_{\lambda h}(u) - \text{prox}_{\lambda h}(v)\| \leq \|u - v\|$
3. **Identity for smooth functions**: If $h$ is differentiable, $\text{prox}_{\lambda h}(v) = v - \lambda \nabla h(v)$

### L1 Regularization Case

For the important case of L1 regularization, consider:

$$\min_{w} \frac{1}{2n} \|Xw - y\|^2 + \lambda \|w\|_1$$

Here:
- $f(w) = \frac{1}{2n} \|Xw - y\|^2$ (smooth data fitting term)
- $g(w) = \lambda \|w\|_1$ (non-smooth L1 penalty)

The gradient of the smooth part is:

$$\nabla f(w) = \frac{1}{n} X^T (Xw - y)$$

The proximal operator for L1 regularization is the soft thresholding operator:

$$\text{prox}_{\lambda |\cdot|}(v) = \text{sign}(v) \odot \max(|v| - \lambda, 0)$$

where $\odot$ denotes element-wise multiplication.

### Soft Thresholding Function

The soft thresholding operator applies element-wise:

$$\text{soft}_\lambda(v_i) = \begin{cases}
v_i - \lambda & \text{if } v_i > \lambda \\
0 & \text{if } |v_i| \leq \lambda \\
v_i + \lambda & \text{if } v_i < -\lambda
\end{cases}$$

This operator induces sparsity by setting small coefficients to exactly zero.

## Convergence Analysis

### Theoretical Guarantees

For the composite optimization problem with $f$ having Lipschitz continuous gradient (constant $L$) and both $f$ and $g$ convex:

- **Convergence rate**: $O(1/k)$ where $k$ is the iteration count
- **Step size condition**: $\alpha \leq 1/L$ ensures convergence
- **Objective decrease**: The algorithm monotonically decreases the objective function

### Convergence Conditions

The algorithm converges under the following conditions:

1. $f$ is convex with $L$-Lipschitz continuous gradient
2. $g$ is convex (possibly non-smooth)
3. Step size satisfies $0 < \alpha \leq 1/L$
4. The optimal solution exists

## Algorithm Configurations

### Standard Configuration

This configuration provides a balance between sparsity and fitting quality.

**Parameters:**
- Learning rate: $\alpha = 0.01$
- L1 regularization: $\lambda = 0.01$
- Maximum iterations: 1000
- Convergence tolerance: $10^{-6}$

**Characteristics:**
- Moderate sparsity induction (20-40% zeros typically)
- Good balance between data fitting and regularization
- Stable convergence properties
- Suitable for most L1-regularized problems

### High Sparsity Configuration

This configuration emphasizes sparse solutions through stronger regularization.

**Parameters:**
- Learning rate: $\alpha = 0.01$
- L1 regularization: $\lambda = 0.1$
- Maximum iterations: 1500
- Convergence tolerance: $10^{-7}$

**Characteristics:**
- Strong sparsity induction (60-80% zeros)
- Automatic feature selection capability
- May sacrifice some fitting accuracy for interpretability
- Requires more iterations for convergence
- Ideal for high-dimensional feature selection problems

### Dense Configuration

This configuration applies minimal regularization for problems where sparsity is not desired.

**Parameters:**
- Learning rate: $\alpha = 0.01$
- L1 regularization: $\lambda = 0.001$
- Maximum iterations: 800
- Convergence tolerance: $10^{-5}$

**Characteristics:**
- Minimal sparsity induction (0-10% zeros)
- Preserves most features in the solution
- Faster convergence due to lighter regularization
- Approaches ridge regression behavior
- Suitable when all features are considered important

## Regularization Parameter Analysis

### L1 Regularization Effects

The regularization parameter $\lambda$ controls the sparsity-accuracy tradeoff:

| $\lambda$ Value | Sparsity Level | Accuracy | Use Case |
|-----------------|----------------|----------|----------|
| $0.001$ | Low (0-10%) | High | Dense feature retention |
| $0.01$ | Moderate (20-40%) | Good | Balanced selection |
| $0.1$ | High (60-80%) | Moderate | Aggressive selection |
| $1.0$ | Very high (90%+) | Lower | Extreme sparsity |

### Feature Selection Mechanism

L1 regularization induces sparsity through the geometry of the constraint set. The L1 ball has sharp corners at the coordinate axes, making it likely that the optimal solution lies on these axes (i.e., has zero components).

### Regularization Path

The solution path $w(\lambda)$ traces how the optimal weights change as a function of the regularization parameter. This path is piecewise linear, with breakpoints corresponding to features entering or leaving the active set.

## Implementation Guidelines

### Soft Thresholding Implementation

```python
def soft_threshold(x, threshold):
    """Apply soft thresholding operator element-wise."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def proximal_l1(x, alpha, lambda_l1):
    """Proximal operator for L1 regularization."""
    return soft_threshold(x, alpha * lambda_l1)
```

### Full Algorithm Implementation

```python
def proximal_gradient_descent(X, y, lambda_l1, alpha=0.01, max_iter=1000, tol=1e-6):
    """Proximal gradient descent for L1-regularized least squares."""
    n, p = X.shape
    w = np.zeros(p)
    
    # Precompute for efficiency
    XtX = X.T @ X
    Xty = X.T @ y
    
    costs = []
    
    for iteration in range(max_iter):
        # Gradient of smooth part
        gradient = (XtX @ w - Xty) / n
        
        # Gradient step
        z = w - alpha * gradient
        
        # Proximal step (soft thresholding)
        w_new = soft_threshold(z, alpha * lambda_l1)
        
        # Compute objective value
        residual = X @ w_new - y
        smooth_cost = 0.5 * np.mean(residual**2)
        regularization_cost = lambda_l1 * np.sum(np.abs(w_new))
        total_cost = smooth_cost + regularization_cost
        costs.append(total_cost)
        
        # Check convergence
        if np.linalg.norm(w_new - w) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
            
        w = w_new
    
    return w, costs
```

### Adaptive Step Size Selection

```python
def backtracking_line_search(X, y, w, gradient, lambda_l1, alpha_init=1.0, 
                             beta=0.5, c=1e-4):
    """Backtracking line search for proximal gradient descent."""
    alpha = alpha_init
    
    # Current objective value
    current_obj = compute_objective(X, y, w, lambda_l1)
    
    while True:
        # Proposed update
        z = w - alpha * gradient
        w_new = soft_threshold(z, alpha * lambda_l1)
        new_obj = compute_objective(X, y, w_new, lambda_l1)
        
        # Sufficient decrease condition (adapted for non-smooth case)
        if new_obj <= current_obj - c * alpha * np.dot(gradient, w - w_new):
            return alpha
        
        alpha *= beta
        
        if alpha < 1e-12:
            break
    
    return alpha
```

## Theoretical Insights

### Relationship to LASSO

Proximal gradient descent for L1-regularized least squares is equivalent to solving the LASSO problem:

$$\min_{w} \frac{1}{2n} \|Xw - y\|^2 + \lambda \|w\|_1$$

This connection provides theoretical guarantees and extensive literature on convergence properties.

### Sparsity Conditions

A feature $j$ will be set to zero if and only if:

$$|\nabla f(w^*)_j| \leq \lambda$$

where $w^*$ is the optimal solution and $\nabla f(w^*)_j$ is the $j$-th component of the gradient at the optimum.

### Active Set Dynamics

The algorithm can be viewed as performing coordinate-wise updates on the active set (non-zero components) while maintaining sparsity through the proximal operator.

## Troubleshooting Guide

### Common Issues and Solutions

| Problem | Symptoms | Recommended Solution |
|---------|----------|----------------------|
| Insufficient sparsity | Few zero coefficients | Increase $\lambda$ parameter |
| Over-sparsity | Too many zero coefficients | Decrease $\lambda$ parameter |
| Slow convergence | Many iterations required | Increase step size or use acceleration |
| Oscillatory behavior | Non-monotonic objective | Decrease step size |
| Poor solution quality | High objective value | Check data preprocessing and scaling |

### Diagnostic Procedures

```python
def diagnose_proximal_gd(X, y, w, lambda_l1, costs):
    """Comprehensive diagnostics for proximal gradient descent."""
    
    # Sparsity analysis
    sparsity_ratio = np.mean(np.abs(w) < 1e-8)
    active_features = np.sum(np.abs(w) >= 1e-8)
    
    print(f"Sparsity ratio: {sparsity_ratio:.1%}")
    print(f"Active features: {active_features}/{len(w)}")
    
    # Convergence analysis
    if len(costs) > 10:
        recent_improvement = (costs[-10] - costs[-1]) / costs[-10]
        print(f"Recent improvement (last 10 iter): {recent_improvement:.2%}")
    
    # Solution quality
    residual = X @ w - y
    mse = np.mean(residual**2)
    l1_penalty = lambda_l1 * np.sum(np.abs(w))
    
    print(f"Mean squared error: {mse:.6f}")
    print(f"L1 penalty term: {l1_penalty:.6f}")
    print(f"Total objective: {mse + l1_penalty:.6f}")
    
    # Feature importance ranking
    feature_importance = np.abs(w)
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    print("Top 5 most important features:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        if feature_importance[idx] > 1e-8:
            print(f"  Feature {idx}: {w[idx]:.4f}")
```

## Advanced Variants

### Accelerated Proximal Gradient (FISTA)

The Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) achieves $O(1/k^2)$ convergence rate:

```python
def fista(X, y, lambda_l1, alpha=0.01, max_iter=1000, tol=1e-6):
    """Fast Iterative Shrinkage-Thresholding Algorithm."""
    n, p = X.shape
    w = np.zeros(p)
    w_prev = w.copy()
    t = 1.0
    
    for iteration in range(max_iter):
        # Nesterov momentum extrapolation
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        beta = (t - 1) / t_new
        y_momentum = w + beta * (w - w_prev)
        
        # Proximal gradient step on extrapolated point
        gradient = X.T @ (X @ y_momentum - y) / n
        z = y_momentum - alpha * gradient
        w_new = soft_threshold(z, alpha * lambda_l1)
        
        # Check convergence
        if np.linalg.norm(w_new - w) < tol:
            break
            
        w_prev = w
        w = w_new
        t = t_new
    
    return w
```

### Adaptive Regularization

Dynamic adjustment of the regularization parameter based on convergence behavior:

```python
def adaptive_proximal_gd(X, y, lambda_init=0.01, adaptation_rate=0.95):
    """Proximal GD with adaptive regularization parameter."""
    lambda_current = lambda_init
    w_best = None
    best_cv_score = float('inf')
    
    for lambda_current in [lambda_init * (adaptation_rate ** i) for i in range(20)]:
        w, costs = proximal_gradient_descent(X, y, lambda_current)
        cv_score = cross_validate_score(X, y, w)
        
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            w_best = w
            best_lambda = lambda_current
    
    return w_best, best_lambda
```

## Applications and Use Cases

### Feature Selection

Proximal gradient descent with L1 regularization naturally performs feature selection by driving irrelevant coefficients to zero. This is particularly valuable in high-dimensional settings where interpretability is important.

### Sparse Signal Recovery

In signal processing applications, the algorithm can recover sparse signals from noisy measurements, leveraging the sparsity-inducing properties of the L1 penalty.

### Compressed Sensing

The method is fundamental to compressed sensing, where the goal is to recover sparse signals from fewer measurements than traditionally required.

## References and Further Reading

### Foundational Literature
1. Parikh, N., & Boyd, S. (2014). Proximal algorithms. *Foundations and Trends in Optimization*, 1(3), 127-239.
2. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
3. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society*, 58(1), 267-288.

### Advanced Topics
- Combettes, P. L., & Pesquet, J. C. (2011). Proximal splitting methods in signal processing. *Fixed-Point Algorithms for Inverse Problems in Science and Engineering*.
- Moreau, J. J. (1962). Fonctions convexes duales et points proximaux dans un espace hilbertien. *Comptes Rendus de l'Académie des Sciences*, 255, 2897-2899.

## Summary

Proximal gradient descent provides an elegant framework for optimizing composite objective functions that combine smooth data fitting terms with non-smooth regularization penalties. Through the proximal operator, the algorithm naturally handles non-smooth terms like L1 regularization while maintaining convergence guarantees. The method's ability to induce sparsity makes it particularly valuable for feature selection and interpretable machine learning, while its theoretical foundations ensure reliable performance across diverse applications.