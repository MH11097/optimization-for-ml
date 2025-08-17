# Stochastic Gradient Descent - Kiến Thức Toán Học & Setups

## 📚 **Lý Thuyết Toán Học**

### **Định Nghĩa Cơ Bản**
Stochastic Gradient Descent (SGD) approximates gradient bằng cách sử dụng một subset (mini-batch) của data thay vì toàn bộ dataset.

**Công thức cập nhật:**
```
w_{k+1} = w_k - α_k ∇f_i(w_k)
```

Trong đó:
- `w_k`: weights hiện tại
- `α_k`: learning rate at step k
- `∇f_i(w_k)`: gradient of sample i (hoặc mini-batch)
- `i`: randomly selected sample/batch

### **Batch vs Mini-batch vs Stochastic**

| Method | Batch Size | Gradient | Noise | Memory |
|--------|------------|----------|-------|--------|
| **Full Batch GD** | n (all data) | Exact | None | O(n) |
| **Mini-batch SGD** | b (1 < b < n) | Approximate | Medium | O(b) |
| **Stochastic SGD** | 1 (single sample) | Very noisy | High | O(1) |

### **Stochastic Gradient Formula**
Cho MSE loss với single sample:
```
f_i(w) = (1/2)(x_i^T w - y_i)²
∇f_i(w) = x_i(x_i^T w - y_i)
```

Cho mini-batch B:
```
∇f_B(w) = (1/|B|) Σ_{i∈B} ∇f_i(w)
```

### **Convergence Properties**
- **Noisy convergence**: Oscillates around optimum
- **Rate**: O(1/√k) for convex functions
- **Asymptotic**: E[w_k] → w* as k → ∞
- **Practical**: Often faster initial progress than batch GD

## 🎯 **Các Setup và Ý Nghĩa**

### **1. Large Batch Setup (Mini-batch)**
```python
batch_size = 256          # Large mini-batches
learning_rate = 0.01      # Standard rate
epochs = 50               # Fewer epochs needed
shuffle = True            # Important for SGD
```

**🧠 Cách nhớ:**
- "256 = compromise giữa accuracy và efficiency"
- "0.01 vì gradient ít noisy hơn"
- "50 epochs vì mỗi epoch process nhiều data"

**⚖️ Trade-offs:**
- ✅ Stable gradient estimates
- ✅ Efficient GPU utilization
- ✅ Good convergence properties
- ❌ Higher memory usage
- ❌ Less exploration of parameter space

### **2. Medium Batch Setup**
```python
batch_size = 32           # Balanced choice
learning_rate = 0.001     # Lower due to noise
epochs = 100              # More epochs needed
learning_rate_decay = 0.95 # Gradual reduction
```

**🧠 Cách nhớ:**
- "32 = sweet spot cho most problems"
- "0.001 vì cần careful với noise"
- "Decay = từ từ giảm speed khi gần đích"

**⚖️ Trade-offs:**
- ✅ Good balance noise vs efficiency
- ✅ Reasonable memory requirements
- ✅ Works well in practice
- ❌ Needs learning rate tuning
- ❌ Convergence can be slow

### **3. Small Batch Setup (True SGD)**
```python
batch_size = 1            # Single sample
learning_rate = 0.0001    # Very small due to high noise
epochs = 200              # Many epochs needed
momentum = 0.9            # Essential for stability
```

**🧠 Cách nhớ:**
- "Batch size 1 = maximum noise"
- "0.0001 vì phải very careful"
- "200 epochs vì progress slow per epoch"
- "Momentum = smoothing the noisy path"

**⚖️ Trade-offs:**
- ✅ Maximum memory efficiency
- ✅ Good exploration (escape local minima)
- ✅ Online learning capability
- ❌ Very noisy convergence
- ❌ Slow overall convergence
- ❌ Requires careful tuning

## 📊 **Batch Size Deep Dive**

### **Batch Size Effects**

| Batch Size | Gradient Quality | Memory | Convergence | Exploration |
|------------|------------------|--------|-------------|-------------|
| **1** | Very noisy | Minimal | Slow but explores | Maximum |
| **32** | Moderate noise | Low | Good balance | Good |
| **256** | Low noise | Medium | Fast but smooth | Limited |
| **Full** | Exact | High | Fastest per epoch | None |

### **Optimal Batch Size Selection**
```python
# Rule of thumb
if dataset_size < 1000:
    batch_size = dataset_size  # Full batch
elif dataset_size < 10000:
    batch_size = 32           # Small batch
else:
    batch_size = 256          # Large batch

# Memory constraint
max_batch = memory_limit // (feature_size * 4)  # 4 bytes per float
batch_size = min(batch_size, max_batch)
```

### **Learning Rate Scaling**
```python
# Linear scaling rule
if new_batch_size > old_batch_size:
    new_lr = old_lr * (new_batch_size / old_batch_size)

# Examples:
# batch=1,   lr=0.0001
# batch=32,  lr=0.0032
# batch=256, lr=0.0256
```

## 🧮 **Learning Rate Schedules**

### **Fixed Learning Rate**
```python
α_t = α_0  # Constant throughout training
```
- **Pros**: Simple, no hyperparameters
- **Cons**: May not converge to optimum

### **Step Decay**
```python
α_t = α_0 * γ^(t // step_size)
```
- `γ = 0.1`: Decay factor
- `step_size = 50`: Decay every 50 epochs

### **Exponential Decay**
```python
α_t = α_0 * e^(-λt)
```
- Smooth continuous decay

### **Polynomial Decay**
```python
α_t = α_0 * (1 + λt)^(-p)
```
- `p = 0.5`: Gives O(1/√t) rate

### **Cosine Annealing**
```python
α_t = α_min + (α_max - α_min) * (1 + cos(πt/T)) / 2
```
- Smooth decay with "warm restarts"

## 🎯 **Variance Reduction Techniques**

### **Momentum**
```python
v_t = β * v_{t-1} + α * ∇f_i(w_t)
w_{t+1} = w_t - v_t
```
- `β = 0.9`: "Remember 90% of previous direction"
- Smooths out oscillations

### **Nesterov Momentum**
```python
v_t = β * v_{t-1} + α * ∇f_i(w_t + β * v_{t-1})
w_{t+1} = w_t - v_t
```
- "Look ahead" before computing gradient

### **Adaptive Methods**
```python
# AdaGrad
s_t = s_{t-1} + (∇f_i(w_t))²
w_{t+1} = w_t - α * ∇f_i(w_t) / (√s_t + ε)

# Adam (combines momentum + adaptive)
m_t = β₁ * m_{t-1} + (1-β₁) * ∇f_i(w_t)
v_t = β₂ * v_{t-1} + (1-β₂) * (∇f_i(w_t))²
w_{t+1} = w_t - α * m̂_t / (√v̂_t + ε)
```

## 🧠 **Memory Aids & Intuition**

### **SGD vs Hiking Analogy**
```
Full Batch GD = Helicopter view của entire mountain
- Perfect map nhưng expensive
- Optimal path but slow updates

Mini-batch SGD = Binoculars với partial view
- Good approximation, reasonable cost
- Balance between accuracy và speed

Stochastic SGD = Walking with flashlight
- Only see immediate area (single sample)
- Fast steps nhưng zigzag path
- Good exploration, might find hidden shortcuts
```

### **Batch Size như Group Travel**
```
Batch Size 1    = Solo traveler
                 - Flexible, exploratory
                 - Can change direction quickly
                 - But easily distracted

Batch Size 32   = Small group
                 - Good consensus on direction
                 - Still flexible enough

Batch Size 256  = Large tour group  
                 - Clear direction consensus
                 - Efficient movement
                 - Less flexible, might miss opportunities
```

### **Learning Rate Decay như Car Driving**
```
High LR (start) = Highway driving
                - Fast progress toward destination
                - Good for covering distance

Medium LR       = City driving  
                - Moderate speed, more careful
                - Navigate around obstacles

Low LR (end)    = Parking
                - Very careful, precise movements
                - Fine-tune to exact position
```

## 🔧 **Implementation Best Practices**

### **Data Shuffling**
```python
# Essential for SGD convergence
def shuffle_data(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

# Shuffle every epoch
for epoch in range(num_epochs):
    X_shuffled, y_shuffled = shuffle_data(X, y)
    # Process mini-batches...
```

### **Mini-batch Creation**
```python
def create_mini_batches(X, y, batch_size):
    n_samples = len(X)
    batches = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X[i:end_idx]
        batch_y = y[i:end_idx]
        batches.append((batch_X, batch_y))
    
    return batches
```

### **Convergence Monitoring**
```python
# Track both training and validation loss
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    epoch_train_loss = 0
    for batch_X, batch_y in mini_batches:
        # SGD update
        loss = compute_loss(batch_X, batch_y, weights)
        epoch_train_loss += loss
    
    # Validation (full batch)
    val_loss = compute_loss(X_val, y_val, weights)
    
    train_losses.append(epoch_train_loss)
    val_losses.append(val_loss)
    
    # Early stopping check
    if should_early_stop(val_losses):
        break
```

## 🔍 **Troubleshooting Guide**

### **Common Issues & Solutions**

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Loss oscillates wildly** | Very noisy loss curves | Decrease learning rate |
| **Slow convergence** | Loss decreases very slowly | Increase batch size or LR |
| **Overfitting** | Val loss increases while train decreases | Add regularization |
| **Divergence** | Loss explodes to infinity | Much smaller learning rate |
| **Plateau** | Loss stops improving | Learning rate decay |

### **Diagnostic Checks**
```python
# 1. Monitor gradient norms
grad_norm = np.linalg.norm(gradient)
if grad_norm > threshold:
    print("Warning: Large gradients detected")

# 2. Track learning rate effectiveness
if iteration % 100 == 0:
    effective_lr = learning_rate * grad_norm
    print(f"Effective LR: {effective_lr:.6f}")

# 3. Check batch size impact
if batch_size == 1:
    print("Using true SGD - expect noisy convergence")
elif batch_size >= 0.1 * n_samples:
    print("Large batch - consider full batch GD")
```

### **Hyperparameter Tuning Strategy**
```python
# Step 1: Find working learning rate
lr_candidates = [1e-1, 1e-2, 1e-3, 1e-4]
for lr in lr_candidates:
    if converges_without_explosion(lr):
        working_lr = lr
        break

# Step 2: Optimize batch size
batch_candidates = [1, 8, 32, 128, 256]
best_batch = grid_search(batch_candidates, working_lr)

# Step 3: Fine-tune learning rate
lr_fine = grid_search(working_lr * [0.3, 1.0, 3.0], best_batch)
```

## 📈 **Advanced Variants**

### **SVRG (Stochastic Variance Reduced Gradient)**
Reduces variance by using full gradient snapshots:
```python
# Every m iterations, compute full gradient
if iteration % m == 0:
    full_gradient = compute_full_gradient(X, y, w)

# Use variance-reduced gradient
gradient = sample_gradient - sample_gradient_old + full_gradient
```

### **SAGA (Stochastic Average Gradient)**
Maintains running average of gradients:
```python
# Update table of gradients
gradient_table[i] = compute_sample_gradient(x_i, y_i, w)

# Use averaged gradient
avg_gradient = np.mean(gradient_table)
w = w - lr * avg_gradient
```

### **Coordinate Descent SGD**
Update one coordinate at a time:
```python
for j in range(n_features):
    gradient_j = compute_coordinate_gradient(j)
    w[j] = w[j] - lr * gradient_j
```

## 📖 **Further Reading**

### **Key Papers**
1. **Robbins & Monro (1951)**: Original stochastic approximation
2. **Bottou (2010)**: Large-scale machine learning with SGD
3. **Kingma & Ba (2014)**: Adam optimizer

### **Modern Developments**
- **AdaGrad, RMSprop, Adam**: Adaptive learning rates
- **Variance reduction methods**: SVRG, SAGA, SAG
- **Distributed SGD**: Parallel and asynchronous variants

---

*"Stochastic Gradient Descent: Trade perfect information for computational efficiency and exploration ability"* 🎲⚡🎯