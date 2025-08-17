# Stochastic Gradient Descent

## Mathematical Foundation

### Definition

Stochastic Gradient Descent (SGD) is an iterative optimization algorithm that approximates the true gradient using a subset of training data at each iteration. Unlike batch gradient descent which computes the exact gradient using all training samples, SGD uses only a single sample or a small mini-batch.

### Algorithm Formulation

The general update rule for SGD is:

$w_{k+1} = w_k - \alpha_k \nabla f_i(w_k)$

where:
- $w_k$ represents the parameter vector at iteration $k$
- $\alpha_k$ is the learning rate (step size) at iteration $k$  
- $\nabla f_i(w_k)$ is the gradient of the loss function for sample $i$ (or mini-batch)
- $i$ is a randomly selected sample or mini-batch index

### Variant Comparison

The SGD family encompasses three main approaches:

| Method | Batch Size | Gradient Quality | Computational Cost | Memory Requirement |
|--------|------------|------------------|--------------------|-----------------|
| Batch GD | $n$ (full dataset) | Exact | $O(n)$ per iteration | $O(n)$ |
| Mini-batch SGD | $1 < b < n$ | Approximate | $O(b)$ per iteration | $O(b)$ |
| Stochastic SGD | $1$ (single sample) | Noisy | $O(1)$ per iteration | $O(1)$ |

### Gradient Computation

For the mean squared error loss function applied to a single sample:

$f_i(w) = \frac{1}{2}(x_i^T w - y_i)^2$

The corresponding gradient is:

$\nabla f_i(w) = x_i(x_i^T w - y_i)$

For a mini-batch $B$ containing $|B|$ samples:

$\nabla f_B(w) = \frac{1}{|B|} \sum_{i \in B} \nabla f_i(w)$

### Convergence Analysis

#### Theoretical Convergence Rate
For convex objective functions, SGD achieves a convergence rate of $O(1/\sqrt{k})$ where $k$ is the number of iterations. This is slower than the $O(1/k)$ rate of batch gradient descent.

#### Convergence Characteristics
- **Non-monotonic**: The objective function value does not decrease monotonically
- **Stochastic nature**: The algorithm oscillates around the optimum due to gradient noise
- **Asymptotic convergence**: $\mathbb{E}[w_k] \to w^*$ as $k \to \infty$ under appropriate conditions
- **Practical advantage**: Often achieves good solutions faster than batch methods in early iterations

## Algorithm Configurations

### Large Batch Configuration

This configuration uses relatively large mini-batches to balance computational efficiency with gradient quality.

**Parameters:**
- Batch size: 256 samples
- Learning rate: 0.01
- Epochs: 50
- Data shuffling: Enabled

**Characteristics:**
- Provides more stable gradient estimates due to larger sample size
- Enables efficient GPU utilization through vectorized operations
- Requires fewer epochs to converge due to higher quality gradients
- Demands higher memory allocation for batch processing
- Offers limited exploration of the parameter space due to reduced stochasticity

### Medium Batch Configuration

This configuration represents a balanced approach between gradient quality and computational requirements.

**Parameters:**
- Batch size: 32 samples
- Learning rate: 0.001 (with decay)
- Epochs: 100
- Learning rate decay: 0.95 per epoch

**Characteristics:**
- Achieves reasonable balance between gradient noise and computational efficiency
- Maintains moderate memory requirements suitable for most hardware configurations
- Demonstrates robust performance across diverse problem domains
- Requires careful learning rate scheduling for optimal convergence
- May exhibit slower convergence compared to larger batch configurations

### Small Batch Configuration (True SGD)

This configuration implements classical stochastic gradient descent using individual samples.

**Parameters:**
- Batch size: 1 sample
- Learning rate: 0.0001
- Epochs: 200
- Momentum: 0.9

**Characteristics:**
- Maximizes memory efficiency with minimal storage requirements
- Provides strong exploration capabilities, helping escape local minima
- Enables online learning scenarios where data arrives sequentially
- Exhibits highly noisy convergence patterns requiring careful monitoring
- Demands extensive epochs and momentum for practical convergence
- Necessitates precise hyperparameter tuning for stable performance

## Batch Size Analysis

### Impact of Batch Size Selection

| Batch Size | Gradient Quality | Memory Usage | Convergence Speed | Exploration Capability |
|------------|------------------|--------------|-------------------|----------------------|
| 1 | High noise | Minimal | Slow | Maximum |
| 32 | Moderate noise | Low | Balanced | Good |
| 256 | Low noise | Medium | Fast | Limited |
| Full dataset | Exact | High | Fastest per epoch | None |

### Batch Size Selection Strategy

The optimal batch size depends on dataset characteristics and computational constraints:

```python
def select_batch_size(dataset_size, memory_limit, feature_dimension):
    # Base selection on dataset size
    if dataset_size < 1000:
        batch_size = dataset_size  # Full batch for small datasets
    elif dataset_size < 10000:
        batch_size = 32           # Small batch for medium datasets
    else:
        batch_size = 256          # Large batch for large datasets
    
    # Apply memory constraints
    max_feasible_batch = memory_limit // (feature_dimension * 4)
    return min(batch_size, max_feasible_batch)
```

### Learning Rate Scaling

When adjusting batch sizes, the learning rate should be scaled accordingly to maintain convergence properties:

$\alpha_{\text{new}} = \alpha_{\text{old}} \times \frac{\text{batch}_{\text{new}}}{\text{batch}_{\text{old}}}$

This linear scaling rule helps preserve the effective step size in parameter space.

## Learning Rate Scheduling

### Common Learning Rate Schedules

#### Fixed Learning Rate
$\alpha_t = \alpha_0$

Maintains constant learning rate throughout training. Simple but may not achieve optimal convergence.

#### Step Decay
$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / s \rfloor}$

where $\gamma$ is the decay factor and $s$ is the step size.

#### Exponential Decay
$\alpha_t = \alpha_0 \cdot e^{-\lambda t}$

Provides smooth continuous decay with rate parameter $\lambda$.

#### Polynomial Decay
$\alpha_t = \alpha_0 \cdot (1 + \lambda t)^{-p}$

With $p = 0.5$, this achieves the theoretical $O(1/\sqrt{t})$ convergence rate.

#### Cosine Annealing
$\alpha_t = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \frac{1 + \cos(\pi t / T)}{2}$

Enables smooth decay with periodic "warm restarts" at period $T$.

## Variance Reduction Techniques

### Momentum Methods

#### Classical Momentum
$v_t = \beta v_{t-1} + \alpha \nabla f_i(w_t)$
$w_{t+1} = w_t - v_t$

where $\beta \in [0,1)$ controls the momentum term, typically set to 0.9.

#### Nesterov Accelerated Gradient
$v_t = \beta v_{t-1} + \alpha \nabla f_i(w_t + \beta v_{t-1})$
$w_{t+1} = w_t - v_t$

This "look-ahead" variant often provides superior convergence properties.

### Adaptive Learning Rate Methods

#### AdaGrad
$s_t = s_{t-1} + (\nabla f_i(w_t))^2$
$w_{t+1} = w_t - \frac{\alpha}{\sqrt{s_t + \epsilon}} \nabla f_i(w_t)$

#### Adam Optimizer
$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla f_i(w_t)$
$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla f_i(w_t))^2$
$w_{t+1} = w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected estimates.

## Theoretical Insights

### Comparison with Batch Methods

Stochastic gradient descent trades gradient accuracy for computational efficiency. While batch gradient descent computes the exact gradient direction, SGD uses noisy estimates that approximate the true gradient in expectation.

### Exploration vs Exploitation

The stochastic nature of SGD provides implicit regularization through gradient noise, often helping the algorithm escape local minima and find better solutions than deterministic methods.

### Learning Rate Dynamics

Learning rate scheduling addresses the exploration-exploitation tradeoff over time. Initially, larger learning rates enable rapid progress and exploration of the parameter space. As training progresses, smaller learning rates facilitate fine-tuning and convergence to local optima.

## Implementation Guidelines

### Data Shuffling

Proper data shuffling is crucial for SGD convergence. Without shuffling, the algorithm may exhibit systematic bias based on data ordering.

```python
def shuffle_data(X, y, random_state=None):
    """Shuffle training data for SGD."""
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

# Apply shuffling at each epoch
for epoch in range(num_epochs):
    X_shuffled, y_shuffled = shuffle_data(X, y)
    # Process mini-batches
```

### Mini-batch Construction

```python
def create_mini_batches(X, y, batch_size):
    """Create mini-batches from training data."""
    n_samples = len(X)
    batches = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X[i:end_idx]
        batch_y = y[i:end_idx]
        batches.append((batch_X, batch_y))
    
    return batches
```

### Convergence Monitoring

```python
def monitor_convergence(X_train, y_train, X_val, y_val, weights, 
                       train_losses, val_losses, patience=10):
    """Monitor training progress and implement early stopping."""
    
    # Compute current losses
    train_loss = compute_loss(X_train, y_train, weights)
    val_loss = compute_loss(X_val, y_val, weights)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Early stopping logic
    if len(val_losses) > patience:
        recent_improvement = min(val_losses[-patience:]) < min(val_losses[:-patience])
        if not recent_improvement:
            return True  # Stop training
    
    return False  # Continue training
```

## Troubleshooting Guide

### Common Issues and Solutions

| Problem | Symptoms | Recommended Solution |
|---------|----------|-----------------------|
| Loss oscillates excessively | Highly noisy loss curves | Reduce learning rate or increase batch size |
| Slow convergence | Minimal loss reduction per epoch | Increase learning rate or add momentum |
| Overfitting | Validation loss increases while training loss decreases | Apply regularization or early stopping |
| Loss divergence | Loss values become infinite or NaN | Significantly reduce learning rate |
| Training plateau | Loss stagnates for many epochs | Implement learning rate decay or restart |

### Diagnostic Procedures

```python
def diagnose_training(gradient, learning_rate, batch_size, n_samples, iteration):
    """Diagnostic checks for SGD training."""
    
    # Monitor gradient magnitudes
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > 10.0:  # Threshold may need adjustment
        print(f"Warning: Large gradient norm detected: {grad_norm:.6f}")
    
    # Track effective learning rate
    if iteration % 100 == 0:
        effective_lr = learning_rate * grad_norm
        print(f"Iteration {iteration}: Effective LR = {effective_lr:.6f}")
    
    # Batch size assessment
    batch_ratio = batch_size / n_samples
    if batch_size == 1:
        print("True SGD mode: Expect noisy but exploratory convergence")
    elif batch_ratio >= 0.1:
        print("Large batch detected: Consider full batch gradient descent")
```

### Hyperparameter Optimization Strategy

```python
def optimize_sgd_hyperparameters(X_train, y_train, X_val, y_val):
    """Systematic hyperparameter optimization for SGD."""
    
    # Phase 1: Learning rate search
    lr_candidates = [1e-1, 1e-2, 1e-3, 1e-4]
    stable_lr = None
    
    for lr in lr_candidates:
        if test_stability(X_train, y_train, learning_rate=lr):
            stable_lr = lr
            break
    
    # Phase 2: Batch size optimization
    batch_candidates = [1, 8, 32, 128, 256]
    best_config = grid_search_validation(
        X_train, y_train, X_val, y_val,
        learning_rates=[stable_lr],
        batch_sizes=batch_candidates
    )
    
    # Phase 3: Learning rate refinement
    refined_lr = fine_tune_learning_rate(
        X_train, y_train, X_val, y_val,
        base_lr=stable_lr,
        batch_size=best_config['batch_size']
    )
    
    return {
        'learning_rate': refined_lr,
        'batch_size': best_config['batch_size']
    }
```

## Advanced Variants

### Variance Reduction Methods

#### SVRG (Stochastic Variance Reduced Gradient)
Reduces gradient variance by incorporating periodic full gradient computations:

```python
def svrg_update(X, y, w, iteration, snapshot_frequency=100):
    """SVRG gradient update with variance reduction."""
    if iteration % snapshot_frequency == 0:
        full_gradient = compute_full_gradient(X, y, w)
        
    # Variance-reduced gradient estimate
    sample_idx = np.random.randint(len(X))
    current_sample_grad = compute_sample_gradient(X[sample_idx], y[sample_idx], w)
    old_sample_grad = compute_sample_gradient(X[sample_idx], y[sample_idx], w_snapshot)
    
    variance_reduced_grad = current_sample_grad - old_sample_grad + full_gradient
    return variance_reduced_grad
```

#### SAGA (Stochastic Average Gradient Accelerated)
Maintains a table of individual sample gradients for variance reduction:

```python
class SAGAOptimizer:
    def __init__(self, n_samples, n_features):
        self.gradient_table = np.zeros((n_samples, n_features))
        self.avg_gradient = np.zeros(n_features)
    
    def update(self, sample_idx, new_gradient):
        old_gradient = self.gradient_table[sample_idx]
        self.gradient_table[sample_idx] = new_gradient
        self.avg_gradient += (new_gradient - old_gradient) / len(self.gradient_table)
        return self.avg_gradient
```

## References and Further Reading

### Foundational Literature
1. Robbins, H., & Monro, S. (1951). A stochastic approximation method. *The Annals of Mathematical Statistics*, 22(3), 400-407.
2. Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. *Proceedings of COMPSTAT*.
3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

### Advanced Topics
- Johnson, R., & Zhang, T. (2013). Accelerating stochastic gradient descent using predictive variance reduction. *NIPS*.
- Defazio, A., Bach, F., & Lacoste-Julien, S. (2014). SAGA: A fast incremental gradient method. *NIPS*.
- Dean, J., et al. (2012). Large scale distributed deep networks. *NIPS*.

## Summary

Stochastic Gradient Descent represents a fundamental tradeoff between computational efficiency and gradient accuracy. By sacrificing perfect gradient information, SGD enables scalable optimization for large datasets while providing beneficial exploration properties through gradient noise. The algorithm's success depends critically on appropriate batch size selection, learning rate scheduling, and variance reduction techniques.