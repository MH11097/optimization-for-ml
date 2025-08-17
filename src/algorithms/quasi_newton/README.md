# BFGS Quasi-Newton Method

## Mathematical Foundation

### Definition

The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is a quasi-Newton optimization method that approximates the Newton's method by iteratively building an approximation to the Hessian matrix using gradient information from previous iterations.

### Algorithm Formulation

The BFGS update rule follows the general quasi-Newton framework:

$$x_{k+1} = x_k - \alpha_k H_k \nabla f(x_k)$$

where:
- $x_k$ is the current iterate
- $\alpha_k$ is the step size (typically from line search)
- $H_k$ is the approximate inverse Hessian
- $\nabla f(x_k)$ is the gradient at $x_k$

### Secant Condition

The fundamental requirement for quasi-Newton methods is the secant condition:

$$B_{k+1} s_k = y_k$$

where:
- $B_{k+1}$ is the approximate Hessian at iteration $k+1$
- $s_k = x_{k+1} - x_k$ (step vector)
- $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ (gradient difference)

This condition ensures that the Hessian approximation is consistent with the observed gradient changes.

### BFGS Update Formula

The BFGS method updates the inverse Hessian approximation $H_k = B_k^{-1}$ using:

$$H_{k+1} = H_k + \frac{s_k s_k^T}{s_k^T y_k} - \frac{H_k y_k y_k^T H_k}{y_k^T H_k y_k} + \frac{u_k u_k^T}{u_k^T y_k}$$

where $u_k = \frac{s_k}{s_k^T y_k} - \frac{H_k y_k}{y_k^T H_k y_k}$.

The more commonly used Sherman-Morrison-Woodbury form is:

$$H_{k+1} = \left(I - \rho_k s_k y_k^T\right) H_k \left(I - \rho_k y_k s_k^T\right) + \rho_k s_k s_k^T$$

where $\rho_k = \frac{1}{y_k^T s_k}$.

### Convergence Properties

BFGS enjoys several favorable convergence properties:

1. **Superlinear convergence**: For strongly convex functions with Lipschitz continuous Hessian
2. **Global convergence**: When combined with appropriate line search conditions
3. **Finite termination**: On quadratic functions (like Newton's method)
4. **Positive definiteness**: $H_k$ remains positive definite under suitable conditions

## Algorithm Configurations

### Standard Configuration

This configuration provides robust performance across a wide range of optimization problems.

**Parameters:**
- Maximum iterations: 100
- Convergence tolerance: $10^{-6}$
- Line search: Backtracking with Armijo condition
- Initial Hessian: Identity matrix

**Characteristics:**
- Balances convergence speed with numerical stability
- Suitable for well-conditioned problems
- Requires moderate memory storage ($O(n^2)$)
- Robust to poor initial guesses

### Fast Configuration

This configuration prioritizes rapid convergence for well-behaved problems.

**Parameters:**
- Maximum iterations: 50
- Convergence tolerance: $10^{-5}$
- Line search: Backtracking
- Initial Hessian: Scaled identity based on first gradient

**Characteristics:**
- Emphasizes speed over robustness
- Better initial Hessian approximation through scaling
- May struggle with ill-conditioned problems
- Suitable for prototyping and well-understood problems

### Robust Configuration

This configuration emphasizes stability and reliability for challenging optimization landscapes.

**Parameters:**
- Maximum iterations: 200
- Convergence tolerance: $10^{-8}$
- Line search: Strong Wolfe conditions
- Initial Hessian: Diagonal scaling
- Restart frequency: Every 50 iterations

**Characteristics:**
- High precision convergence
- Robust line search conditions ensure global convergence
- Periodic restarts prevent accumulation of approximation errors
- Suitable for ill-conditioned or noisy problems

## Line Search Theory

### Importance of Line Search

Unlike Newton's method, BFGS typically requires careful step size selection because the search direction may not be optimal. Line search ensures global convergence and numerical stability.

### Armijo Condition

The Armijo condition ensures sufficient decrease in the objective function:

$$f(x_k + \alpha_k d_k) \leq f(x_k) + c_1 \alpha_k \nabla f(x_k)^T d_k$$

where:
- $d_k = -H_k \nabla f(x_k)$ is the search direction
- $c_1 \in (0, 1)$ is a small constant (typically $10^{-4}$)

### Wolfe Conditions

The strong Wolfe conditions combine sufficient decrease with curvature conditions:

1. **Armijo condition**: $f(x_k + \alpha_k d_k) \leq f(x_k) + c_1 \alpha_k \nabla f(x_k)^T d_k$
2. **Curvature condition**: $|\nabla f(x_k + \alpha_k d_k)^T d_k| \leq c_2 |\nabla f(x_k)^T d_k|$

where $0 < c_1 < c_2 < 1$ (typically $c_1 = 10^{-4}$, $c_2 = 0.9$).

## Implementation Guidelines

### Core BFGS Implementation

```python
def bfgs_optimization(objective, gradient, x0, max_iter=100, tol=1e-6):
    """BFGS optimization algorithm."""
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # Initial inverse Hessian approximation
    
    grad = gradient(x)
    costs = [objective(x)]
    
    for iteration in range(max_iter):
        # Check convergence
        if np.linalg.norm(grad) < tol:
            print(f"Converged after {iteration} iterations")
            break
        
        # Compute search direction
        direction = -H @ grad
        
        # Line search
        alpha = backtracking_line_search(objective, gradient, x, direction)
        
        # Update position
        x_new = x + alpha * direction
        grad_new = gradient(x_new)
        
        # BFGS update
        s = x_new - x
        y = grad_new - grad
        
        # Check curvature condition
        if y.T @ s > 1e-8:
            H = bfgs_update(H, s, y)
        
        # Prepare for next iteration
        x = x_new
        grad = grad_new
        costs.append(objective(x))
    
    return x, costs

def bfgs_update(H, s, y):
    """BFGS inverse Hessian update."""
    rho = 1.0 / (y.T @ s)
    I = np.eye(len(s))
    
    # Sherman-Morrison-Woodbury formula
    H_new = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
    
    return H_new
```

### Line Search Implementation

```python
def backtracking_line_search(objective, gradient, x, direction, 
                           alpha_init=1.0, c1=1e-4, rho=0.5, max_iter=50):
    """Backtracking line search with Armijo condition."""
    alpha = alpha_init
    fx = objective(x)
    grad_x = gradient(x)
    armijo_threshold = c1 * grad_x.T @ direction
    
    for _ in range(max_iter):
        x_new = x + alpha * direction
        fx_new = objective(x_new)
        
        # Check Armijo condition
        if fx_new <= fx + alpha * armijo_threshold:
            return alpha
        
        alpha *= rho
    
    # Return small step if line search fails
    return alpha
```

### Numerical Stability Enhancements

```python
def robust_bfgs_update(H, s, y, damping=True):
    """BFGS update with numerical stability enhancements."""
    
    # Check curvature condition
    sy = y.T @ s
    if sy <= 1e-8:
        if damping:
            # Damped BFGS update
            theta = 0.8
            y_damped = theta * y + (1 - theta) * H @ s
            sy = y_damped.T @ s
            y = y_damped
        else:
            # Skip update
            return H
    
    rho = 1.0 / sy
    I = np.eye(len(s))
    
    # Compute update
    A = I - rho * np.outer(s, y)
    H_new = A @ H @ A.T + rho * np.outer(s, s)
    
    # Ensure positive definiteness
    eigenvals = np.linalg.eigvals(H_new)
    if np.min(eigenvals) <= 0:
        print("Warning: Non-positive definite Hessian approximation")
        # Add regularization
        H_new += 1e-8 * np.eye(len(s))
    
    return H_new
```

## Theoretical Analysis

### Convergence Rate

For strongly convex functions with Lipschitz continuous Hessian, BFGS achieves superlinear convergence:

$$\lim_{k \to \infty} \frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|} = 0$$

This rate is faster than linear but slower than the quadratic rate of Newton's method.

### Memory Requirements

BFGS requires $O(n^2)$ memory to store the inverse Hessian approximation, which can be prohibitive for large-scale problems. This limitation motivates limited-memory variants like L-BFGS.

### Positive Definiteness

Under appropriate conditions (particularly the curvature condition $y_k^T s_k > 0$), the BFGS update preserves positive definiteness of the Hessian approximation, ensuring descent directions.

## Troubleshooting Guide

### Common Issues and Solutions

| Problem | Symptoms | Recommended Solution |
|---------|----------|----------------------|
| Slow initial convergence | High cost for first 10-20 iterations | Normal behavior; wait for Hessian buildup |
| Line search failures | Very small step sizes | Check gradient accuracy; increase damping |
| Memory limitations | Out of memory errors | Switch to L-BFGS for large problems |
| Poor final accuracy | Stagnation before tolerance | Check problem conditioning |
| Non-descent directions | Increasing objective values | Verify curvature condition; add regularization |

### Diagnostic Procedures

```python
def diagnose_bfgs(H, grad, iteration, costs):
    """Comprehensive diagnostics for BFGS optimization."""
    
    # Check Hessian approximation quality
    cond_H = np.linalg.cond(H)
    eigenvals = np.linalg.eigvals(H)
    min_eigval = np.min(eigenvals)
    
    print(f"Iteration {iteration}:")
    print(f"  Condition number of H: {cond_H:.2e}")
    print(f"  Minimum eigenvalue: {min_eigval:.2e}")
    print(f"  Gradient norm: {np.linalg.norm(grad):.6f}")
    
    # Convergence analysis
    if len(costs) > 1:
        improvement = (costs[-2] - costs[-1]) / abs(costs[-2])
        print(f"  Relative improvement: {improvement:.2e}")
    
    # Warning checks
    if cond_H > 1e12:
        print("  WARNING: Ill-conditioned Hessian approximation")
    if min_eigval <= 0:
        print("  WARNING: Non-positive definite Hessian")
    
    return {
        'condition_number': cond_H,
        'min_eigenvalue': min_eigval,
        'gradient_norm': np.linalg.norm(grad)
    }
```

### Performance Optimization

```python
def optimize_bfgs_parameters(problem_size, conditioning):
    """Suggest BFGS parameters based on problem characteristics."""
    
    if problem_size < 100:
        config = {
            'method': 'standard_bfgs',
            'line_search': 'strong_wolfe',
            'max_iter': 100
        }
    elif problem_size < 1000:
        config = {
            'method': 'bfgs_with_restart',
            'restart_frequency': 50,
            'line_search': 'backtracking',
            'max_iter': 200
        }
    else:
        config = {
            'method': 'l_bfgs',
            'memory_limit': 10,
            'line_search': 'backtracking',
            'max_iter': 500
        }
    
    # Adjust for conditioning
    if conditioning == 'poor':
        config['tolerance'] = 1e-4
        config['damping'] = True
    elif conditioning == 'good':
        config['tolerance'] = 1e-8
        config['damping'] = False
    
    return config
```

## Advanced Variants

### Limited Memory BFGS (L-BFGS)

For large-scale problems, L-BFGS maintains only a limited history of updates:

```python
class LBFGSOptimizer:
    def __init__(self, memory_size=10):
        self.m = memory_size
        self.s_history = []
        self.y_history = []
        self.rho_history = []
    
    def update_history(self, s, y):
        """Update limited memory with new (s, y) pair."""
        rho = 1.0 / (y.T @ s)
        
        if len(self.s_history) >= self.m:
            # Remove oldest entries
            self.s_history.pop(0)
            self.y_history.pop(0)
            self.rho_history.pop(0)
        
        # Add new entries
        self.s_history.append(s)
        self.y_history.append(y)
        self.rho_history.append(rho)
    
    def compute_direction(self, grad):
        """Compute search direction using two-loop recursion."""
        q = grad.copy()
        alpha = np.zeros(len(self.s_history))
        
        # First loop (backward)
        for i in reversed(range(len(self.s_history))):
            alpha[i] = self.rho_history[i] * self.s_history[i].T @ q
            q -= alpha[i] * self.y_history[i]
        
        # Apply initial Hessian approximation
        if len(self.s_history) > 0:
            gamma = (self.s_history[-1].T @ self.y_history[-1]) / (self.y_history[-1].T @ self.y_history[-1])
            z = gamma * q
        else:
            z = q
        
        # Second loop (forward)
        for i in range(len(self.s_history)):
            beta = self.rho_history[i] * self.y_history[i].T @ z
            z += (alpha[i] - beta) * self.s_history[i]
        
        return -z
```

### Trust Region BFGS

Combines BFGS with trust region methodology for enhanced robustness:

```python
def trust_region_bfgs(objective, gradient, hessian_approx, x, radius):
    """Trust region step with BFGS Hessian approximation."""
    grad = gradient(x)
    H = hessian_approx
    
    # Solve trust region subproblem: min_p 1/2 p^T H p + grad^T p, ||p|| <= radius
    try:
        # Try Cholesky decomposition
        L = np.linalg.cholesky(H)
        p_newton = -np.linalg.solve(H, grad)
        
        if np.linalg.norm(p_newton) <= radius:
            return p_newton
        else:
            # Solve constrained problem
            return solve_trust_region_constraint(H, grad, radius)
    except np.linalg.LinAlgError:
        # Fallback to steepest descent
        return -radius * grad / np.linalg.norm(grad)
```

## Applications and Use Cases

### Machine Learning

BFGS is particularly effective for:
- Logistic regression with moderate feature counts
- Neural network training (small to medium networks)
- Maximum likelihood estimation
- Support vector machine training

### Scientific Computing

Common applications include:
- Parameter estimation in physical models
- Inverse problems and data assimilation
- Engineering design optimization
- Statistical model fitting

### When to Use BFGS

Choose BFGS when:
- Problem dimension is moderate (< 10,000 variables)
- Gradients are available and accurate
- Function evaluations are expensive
- Superlinear convergence is desired
- Memory usage of $O(n^2)$ is acceptable

## References and Further Reading

### Foundational Literature
1. Broyden, C. G. (1970). The convergence of a class of double-rank minimization algorithms. *Journal of the Institute of Mathematics and its Applications*, 6(1), 76-90.
2. Fletcher, R. (1970). A new approach to variable metric algorithms. *The Computer Journal*, 13(3), 317-322.
3. Goldfarb, D. (1970). A family of variable-metric methods derived by variational means. *Mathematics of Computation*, 24(109), 23-26.
4. Shanno, D. F. (1970). Conditioning of quasi-Newton methods for function minimization. *Mathematics of Computation*, 24(111), 647-656.

### Advanced Topics
- Nocedal, J., & Wright, S. (2006). *Numerical Optimization*. Springer.
- Dennis Jr, J. E., & Moré, J. J. (1977). Quasi-Newton methods, motivation and theory. *SIAM Review*, 19(1), 46-89.
- Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1-3), 503-528.

## Summary

The BFGS quasi-Newton method provides an effective balance between the rapid convergence of Newton's method and the computational efficiency of gradient descent. By iteratively building an approximation to the inverse Hessian using only gradient information, BFGS achieves superlinear convergence while avoiding expensive Hessian computations. The method's robustness, combined with strong theoretical foundations and practical line search strategies, makes it a cornerstone algorithm for medium-scale optimization problems across diverse applications in machine learning, scientific computing, and engineering.