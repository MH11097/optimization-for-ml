# Proximal Gradient Descent - Kiến Thức Toán Học & Setups

## 📚 **Lý Thuyết Toán Học**

### **Định Nghĩa Cơ Bản**
Proximal Gradient Descent giải bài toán optimization có dạng:
```
minimize f(x) + g(x)
```
Trong đó:
- `f(x)`: smooth function (có gradient)
- `g(x)`: non-smooth function (thường là regularization)

### **Algorithm**
```
1. Forward step:  z_k = x_k - α ∇f(x_k)
2. Proximal step: x_{k+1} = prox_{αg}(z_k)
```

### **Proximal Operator**
```
prox_λh(v) = argmin_x { h(x) + (1/2λ)||x - v||² }
```

**Intuition:** Tìm điểm gần v nhưng minimize h(x)

### **Cho L1 Regularization**
Problem: `minimize (1/2)||Xw - y||² + λ||w||₁`

- `f(w) = (1/2)||Xw - y||²` (smooth)
- `g(w) = λ||w||₁` (non-smooth)

**Proximal operator của L1:**
```
prox_λ|·|(v) = sign(v) ⊙ max(|v| - λ, 0)
```

Đây chính là **soft thresholding function**!

### **Soft Thresholding**
```
soft_threshold(v, λ) = {
  v - λ,  if v > λ
  0,      if |v| ≤ λ  
  v + λ,  if v < -λ
}
```

**🧠 Intuition:** "Shrink về 0, set to 0 nếu quá nhỏ"

## 🎯 **Setup và Ý Nghĩa**

### **Standard Setup**
```python
learning_rate = 0.01     # Moderate
lambda_l1 = 0.01         # Light regularization
max_iterations = 1000    # Sufficient
tolerance = 1e-6         # Good precision
```

**🧠 Cách nhớ:**
- "0.01 learning rate = standard như GD"
- "0.01 lambda = 1% penalty trên weights"
- "Balance giữa fit và sparsity"

**⚖️ Trade-offs:**
- ✅ Good balance fit vs sparsity
- ✅ Moderate feature selection
- ✅ Stable convergence
- ❌ May not be sparse enough

**Expected sparsity:** ~20-40% weights = 0

### **Sparse Setup**
```python
learning_rate = 0.01     # Same
lambda_l1 = 0.1          # Higher regularization
max_iterations = 1500    # More iterations
tolerance = 1e-7         # Precise
```

**🧠 Cách nhớ:**
- "0.1 lambda = 10% penalty, khuyến khích sparsity"
- "1500 iterations vì cần thời gian để sparse"
- "Higher penalty → more zeros"

**⚖️ Trade-offs:**
- ✅ High sparsity (60-80% zeros)
- ✅ Automatic feature selection
- ✅ Interpretable models
- ❌ May sacrifice some accuracy
- ❌ Risk of underfitting

### **Dense Setup**
```python
learning_rate = 0.01     # Same
lambda_l1 = 0.001        # Light regularization
max_iterations = 800     # Fewer needed
tolerance = 1e-5         # Relaxed
```

**🧠 Cách nhớ:**
- "0.001 lambda = 0.1% penalty, gần như no regularization"
- "Gần như standard regression"
- "Ít sparsity, focus on accuracy"

**⚖️ Trade-offs:**
- ✅ High accuracy
- ✅ Keeps most features
- ✅ Fast convergence
- ❌ Little feature selection
- ❌ Less interpretable

## 🧮 **Mathematical Deep Dive**

### **Convergence Theory**
For `f` L-smooth và `g` convex:
```
f(x_k) + g(x_k) - [f(x*) + g(x*)] ≤ O(1/k)
```

**Key insight:** Same convergence rate as gradient descent!

### **Sparsity Analysis**
Soft thresholding tạo sparsity:
```
If |∇f(w)[i]| ≤ λ ⟹ w[i] = 0
```

**Intuition:** Feature i bị set về 0 nếu gradient nhỏ hơn penalty

### **Regularization Path**
Khi λ tăng:
```
λ = 0     → No sparsity (standard regression)
λ = small → Some sparsity
λ = large → High sparsity  
λ = ∞     → All weights = 0
```

## 📊 **Lambda Selection Guide**

### **Rule of Thumb**
```python
# Start with
λ_max = max(|X^T y|) / n  # Largest useful lambda
λ_start = 0.1 * λ_max     # Good starting point
```

### **Cross-Validation Strategy**
```python
lambdas = [0.001, 0.01, 0.1, 1.0]
for λ in lambdas:
    cv_score = cross_validate(λ)
    sparsity = count_zeros(λ)
    print(f"λ={λ}: CV={cv_score:.3f}, Sparsity={sparsity}%")
```

### **Lambda Values & Effects**

| λ | Sparsity | Accuracy | Use Case |
|---|----------|----------|----------|
| **0.001** | 0-10% | Highest | Dense features needed |
| **0.01** | 20-40% | High | Balanced |
| **0.1** | 60-80% | Medium | Feature selection |
| **1.0** | 90%+ | Lower | Extreme sparsity |

## 🎯 **Khi Nào Dùng Proximal GD**

### **Perfect Use Cases**
- ✅ **High-dimensional data** (p >> n)
- ✅ **Feature selection** cần thiết
- ✅ **Interpretable models** required
- ✅ **Sparse solutions** preferred
- ✅ **Overfitting** concerns

### **Avoid When**
- ❌ **All features important** 
- ❌ **Non-linear relationships** (use kernels)
- ❌ **Very small datasets** (may overregularize)
- ❌ **Need smooth solutions** (L2 better)

### **Comparison với Other Methods**

| Method | Regularization | Sparsity | Smoothness |
|--------|----------------|----------|------------|
| **Ridge** | L2: λ∑w²₍ᵢ₎ | No | Yes |
| **Lasso (Proximal GD)** | L1: λ∑\|wᵢ\| | Yes | No |
| **Elastic Net** | L1 + L2 | Moderate | Moderate |

## 🧠 **Memory Aids & Intuition**

### **Proximal GD như "Penalty Kicks"**
```
1. Forward step = Kick ball toward goal (gradient step)
2. Proximal step = Referee applies penalty rules (regularization)
   - Light penalty (small λ) = Ball continues mostly unchanged
   - Heavy penalty (large λ) = Ball gets stopped/redirected
```

### **Soft Thresholding Visualization**
```
Input:    -2  -1  -0.5  0  0.5  1  2
λ = 0.3:  -1.7 -0.7  0   0   0  0.7 1.7
λ = 1.0:  -1   0    0   0   0   0   1
```

### **Sparsity Intuition**
```
λ = 0.001 → "Gentle encouragement to be small"
λ = 0.01  → "Moderate pressure for sparsity"  
λ = 0.1   → "Strong push toward zeros"
λ = 1.0   → "Heavy penalty, force many zeros"
```

## 🔧 **Implementation Tips**

### **Efficient Soft Thresholding**
```python
def soft_threshold(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

# Vectorized for efficiency
def proximal_l1(x, alpha, lambda_l1):
    threshold = alpha * lambda_l1
    return soft_threshold(x, threshold)
```

### **Convergence Monitoring**
```python
# Track multiple metrics
metrics = {
    'total_cost': [],
    'smooth_cost': [],  
    'l1_penalty': [],
    'sparsity_count': [],
    'active_features': []
}
```

### **Warm Start Strategy**
```python
# Start with less regularization, gradually increase
lambdas = [0.001, 0.01, 0.1]
weights = initialize_random()

for λ in lambdas:
    weights = proximal_gd(X, y, λ, init_weights=weights)
```

## 🔍 **Troubleshooting Guide**

### **Common Issues**

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Not sparse enough** | Few zeros | Increase λ |
| **Too sparse** | Many important features = 0 | Decrease λ |
| **Poor accuracy** | High test error | Decrease λ or add L2 |
| **Slow convergence** | Many iterations | Check learning rate |
| **Oscillations** | Cost fluctuates | Decrease learning rate |

### **Debugging Sparsity**
```python
# Check sparsity evolution
sparsity_ratio = np.mean(np.abs(weights) < 1e-10)
print(f"Sparsity: {sparsity_ratio:.1%}")

# Check feature importance
important_features = np.where(np.abs(weights) > threshold)[0]
print(f"Active features: {len(important_features)}")
```

### **Feature Selection Analysis**
```python
# Sort features by importance
feature_importance = np.abs(weights)
sorted_indices = np.argsort(feature_importance)[::-1]

print("Top 10 most important features:")
for i in range(10):
    idx = sorted_indices[i]
    print(f"Feature {idx}: weight = {weights[idx]:.4f}")
```

## 📈 **Advanced Topics**

### **Accelerated Proximal GD (FISTA)**
```python
# Nesterov acceleration
t_prev = 1
w_prev = w
for iteration in range(max_iter):
    t = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
    y = w + ((t_prev - 1) / t) * (w - w_prev)
    
    # Standard proximal step on y
    z = y - alpha * gradient(y)
    w_new = soft_threshold(z, alpha * lambda_l1)
```

### **Adaptive Lambda Selection**
```python
# Decrease lambda during training
lambda_schedule = lambda_init * (decay_rate ** iteration)
```

---

*"Proximal GD: Combine smooth optimization with non-smooth penalties for automatic feature selection"* 🎯✂️