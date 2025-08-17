# Subgradient Methods - Kiến Thức Toán Học & Setups

## 📚 **Lý Thuyết Toán Học**

### **Định Nghĩa Cơ Bản**
Subgradient method để optimize **non-smooth functions** - functions không có gradient everywhere.

**Algorithm:**
```
x_{k+1} = x_k - α_k g_k
```
Trong đó `g_k ∈ ∂f(x_k)` là subgradient tại x_k

### **Subgradient Definition**
Vector g là subgradient của f tại x nếu:
```
f(y) ≥ f(x) + g^T(y - x)  ∀y
```

**Intuition:** g support hyperplane từ dưới lên function f

### **Subdifferential**
```
∂f(x) = {g : g is subgradient of f at x}
```

**Properties:**
- Nếu f differentiable tại x: `∂f(x) = {∇f(x)}`
- Nếu f non-smooth tại x: `∂f(x)` là set với nhiều elements

### **Subgradient của L1 Norm**
For `f(x) = |x|`:
```
∂|x| = {
  {1},     if x > 0
  {-1},    if x < 0  
  [-1,1],  if x = 0
}
```

For vector `f(w) = ||w||₁ = Σᵢ|wᵢ|`:
```
∂||w||₁ = (∂|w₁|, ∂|w₂|, ..., ∂|wₙ|)
```

### **Subgradient của Max Function**
For `f(x) = max{f₁(x), f₂(x)}`:
```
∂f(x) = conv{∇fᵢ(x) : i ∈ I(x)}
```
Trong đó `I(x) = {i : fᵢ(x) = f(x)}` (active set)

## 🎯 **Algorithm Properties**

### **Convergence Rate**
For convex f, subgradient method có:
```
f(x_k) - f* ≤ O(1/√k)
```

**So sánh:**
- Gradient descent: O(1/k) 
- Newton method: O(1/k²)
- **Subgradient: O(1/√k) - CHẬM HỚN!**

### **Non-Monotonic Convergence**
**Key difference:** Cost không giảm monotonically!
- Gradient descent: f(x_{k+1}) ≤ f(x_k)
- **Subgradient: f(x_{k+1}) có thể > f(x_k)**

**Solution:** Track best point found so far!

### **Step Size Rules**
1. **Constant:** αₖ = α (simple nhưng may not converge)
2. **Diminishing:** αₖ = α/√k (guarantees convergence)
3. **Square summable:** Σαₖ = ∞, Σαₖ² < ∞

## 🎯 **Setup và Ý Nghĩa**

### **Standard Setup**
```python
learning_rate = 0.01     # Constant step size
lambda_l1 = 0.01         # L1 regularization
max_iterations = 2000    # More iterations needed
tolerance = 1e-6         # For best point tracking
```

**🧠 Cách nhớ:**
- "0.01 như GD nhưng behavior khác"
- "2000 iterations vì convergence chậm"
- "Need patience với subgradient!"

**⚖️ Trade-offs:**
- ✅ Handles non-smooth functions
- ✅ Simple implementation
- ✅ General purpose
- ❌ Slow convergence O(1/√k)
- ❌ Non-monotonic cost

### **Diminishing Step Setup**
```python
learning_rate_init = 0.1    # Start larger
decay_rate = 0.99           # Gradual decay
max_iterations = 3000       # Even more iterations
```

**Formula:** `α_k = α_0 / √k` hoặc `α_k = α_0 * decay_rate^k`

**🧠 Cách nhớ:**
- "Start big, shrink over time"
- "Like cooling in simulated annealing"
- "3000 iterations vì cần thời gian"

**⚖️ Trade-offs:**
- ✅ Guaranteed convergence
- ✅ Better theoretical properties
- ❌ Very slow initially
- ❌ Complex tuning

### **Aggressive Setup**
```python
learning_rate = 0.05     # Higher step size
lambda_l1 = 0.05         # Higher regularization  
max_iterations = 1500    # Fewer iterations
early_stopping = True    # Stop if not improving
```

**🧠 Cách nhớ:**
- "Take bigger risks cho faster progress"
- "Stop early if không improve"
- "High risk, high reward"

**⚖️ Trade-offs:**
- ✅ Potentially faster
- ✅ More exploration
- ❌ Less stable
- ❌ May not converge

## 🧮 **Mathematical Examples**

### **L1 Regularized Regression**
Problem: `minimize (1/2)||Xw - y||² + λ||w||₁`

**Subgradient:**
```python
def compute_subgradient(X, y, w, lambda_l1):
    # Smooth part
    smooth_grad = X.T @ (X @ w - y)
    
    # L1 subgradient
    l1_subgrad = np.zeros_like(w)
    l1_subgrad[w > 0] = lambda_l1
    l1_subgrad[w < 0] = -lambda_l1
    # For w[i] = 0, choose 0 (could be any value in [-λ, λ])
    
    return smooth_grad + l1_subgrad
```

### **Hinge Loss (SVM)**
For `f(w) = max(0, 1 - y(w^T x))`:

**Subgradient:**
```python
def hinge_subgradient(X, y, w):
    margins = y * (X @ w)
    subgrad = np.zeros_like(w)
    
    # Active constraints (margin < 1)
    active = margins < 1
    subgrad = -np.sum(X[active] * y[active].reshape(-1, 1), axis=0)
    
    return subgrad
```

## 🎯 **Khi Nào Dùng Subgradient**

### **Perfect Use Cases**
- ✅ **Non-smooth objectives** (L1, L∞, max, min)
- ✅ **Constraint violations** (penalty methods)
- ✅ **Robust optimization** (minimax problems)
- ✅ **No smooth approximation** available
- ✅ **Simple implementation** needed

### **Avoid When**
- ❌ **Smooth alternatives** exist (use proximal GD)
- ❌ **Fast convergence** required
- ❌ **High precision** needed
- ❌ **Limited iterations** budget

### **Examples in Practice**

| Application | Non-smooth part | Why subgradient |
|-------------|-----------------|-----------------|
| **LASSO** | L1 penalty | Sparsity-inducing |
| **SVM** | Hinge loss | Margin violations |
| **Robust regression** | L1 loss | Outlier resistance |
| **TV denoising** | Total variation | Edge preservation |

## 🧠 **Memory Aids & Intuition**

### **Subgradient như "Blind Navigation"**
```
Gradient Descent = Đi với GPS chính xác
- Biết exact direction (gradient)
- Smooth path

Subgradient = Đi với compass gần đúng
- Direction roughly correct (subgradient)
- May zigzag, không smooth
- Eventually reach destination
```

### **Non-Monotonic Behavior**
```
Iteration 1: Cost = 100 ⬇️
Iteration 2: Cost = 80  ⬇️  
Iteration 3: Cost = 90  ⬆️ (worse!)
Iteration 4: Cost = 75  ⬇️ (better than best)
...

Keep tracking: best_cost = 75
```

### **Convergence Intuition**
```
O(1/√k) means:
k = 100  → Error ∝ 1/10 = 0.1
k = 10000 → Error ∝ 1/100 = 0.01

Need 100x more iterations for 10x better accuracy!
```

## 🔧 **Implementation Best Practices**

### **Best Point Tracking**
```python
best_weights = weights.copy()
best_cost = float('inf')

for iteration in range(max_iterations):
    # Subgradient step
    subgrad = compute_subgradient(X, y, weights, lambda_l1)
    weights -= learning_rate * subgrad
    
    # Evaluate and track best
    current_cost = compute_cost(X, y, weights, lambda_l1)
    if current_cost < best_cost:
        best_cost = current_cost
        best_weights = weights.copy()

# Use best_weights, not final weights!
return best_weights
```

### **Adaptive Step Size**
```python
def diminishing_step_size(iteration, alpha_0=0.1):
    return alpha_0 / np.sqrt(iteration + 1)

def scheduled_step_size(iteration, alpha_0=0.1, decay=0.99):
    return alpha_0 * (decay ** iteration)
```

### **Convergence Monitoring**
```python
# Don't use current cost (non-monotonic)
# Use best cost or moving average
window_size = 50
recent_costs = costs[-window_size:]
avg_recent_cost = np.mean(recent_costs)

if iteration > window_size:
    improvement = prev_avg - avg_recent_cost
    if improvement < tolerance:
        print("Converged based on moving average")
        break
```

## 🔍 **Troubleshooting Guide**

### **Common Issues**

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Oscillating costs** | Cost goes up/down | Normal! Track best point |
| **Very slow progress** | Little improvement | Increase learning rate |
| **Divergence** | Costs explode | Decrease learning rate |
| **Poor final result** | High error | Use diminishing step size |

### **Debugging Checklist**
```python
# 1. Check subgradient computation
subgrad = compute_subgradient(X, y, weights, lambda_l1)
print(f"Subgradient norm: {np.linalg.norm(subgrad):.6f}")

# 2. Monitor best vs current
print(f"Current cost: {current_cost:.6f}")
print(f"Best cost: {best_cost:.6f}")
print(f"Gap: {current_cost - best_cost:.6f}")

# 3. Check step size effect
costs_after_steps = []
for alpha in [0.001, 0.01, 0.1]:
    test_weights = weights - alpha * subgrad
    test_cost = compute_cost(X, y, test_weights, lambda_l1)
    costs_after_steps.append((alpha, test_cost))
```

### **Performance Tuning**
```python
# Experiment with step sizes
alphas = [0.001, 0.01, 0.1, 1.0]
best_alpha = None
best_final_cost = float('inf')

for alpha in alphas:
    final_weights, final_cost = subgradient_method(X, y, alpha)
    if final_cost < best_final_cost:
        best_final_cost = final_cost
        best_alpha = alpha

print(f"Best alpha: {best_alpha}")
```

## 📈 **Advanced Variations**

### **Projected Subgradient**
For constrained problems:
```python
# After subgradient step
weights = weights - alpha * subgrad
# Project onto constraint set
weights = project_onto_constraints(weights)
```

### **Averaging**
Polyak averaging for better convergence:
```python
# Keep running average of iterates
w_avg = (1/(k+1)) * ((k * w_avg) + w_k)
```

### **Bundle Methods**
Use multiple subgradients for better directions.

---

*"Subgradient Methods: Navigate non-smooth landscapes with approximate directions, patience required!"* 🗻🧭