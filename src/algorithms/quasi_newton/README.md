# BFGS (Quasi-Newton) Methods - Kiến Thức Toán Học & Setups

## 📚 **Lý Thuyết Toán Học**

### **Định Nghĩa Cơ Bản**
BFGS (Broyden-Fletcher-Goldfarb-Shanno) là quasi-Newton method approximates Hessian matrix thay vì tính trực tiếp.

**Công thức cập nhật:**
```
x_{k+1} = x_k - α_k B_k^{-1} ∇f(x_k)
```

Trong đó:
- `x_k`: điểm hiện tại
- `α_k`: step size (thường từ line search)
- `B_k`: approximate Hessian matrix
- `∇f(x_k)`: gradient tại x_k

### **BFGS Update Formula**
Thay vì tính Hessian trực tiếp, BFGS updates approximation:

**Secant condition:**
```
B_{k+1} s_k = y_k
```

Trong đó:
- `s_k = x_{k+1} - x_k` (step vector)
- `y_k = ∇f(x_{k+1}) - ∇f(x_k)` (gradient change)

**BFGS Update:**
```
B_{k+1} = B_k + (y_k y_k^T)/(y_k^T s_k) - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k)
```

### **Sherman-Morrison-Woodbury Formula**
Thay vì update B_k, ta update H_k = B_k^{-1}:
```
H_{k+1} = (I - ρ_k s_k y_k^T) H_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
```

Trong đó: `ρ_k = 1/(y_k^T s_k)`

### **Convergence Properties**
- **Superlinear convergence**: Faster than linear, slower than quadratic
- **Global convergence** với line search
- **Finite termination** cho quadratic functions (như Newton)
- **Memory efficient**: O(n²) thay vì O(n³) như Newton

## 🎯 **Các Setup và Ý Nghĩa**

### **1. Standard Setup**
```python
max_iterations = 100       # More than Newton
tolerance = 1e-6          # Good precision
line_search = True        # Essential for stability
initial_hessian = "identity"  # Simple start
```

**🧠 Cách nhớ:**
- "100 iterations vì cần build up Hessian approximation"
- "Line search như GPS để tìm optimal step"
- "Identity matrix = starting from scratch"

**⚖️ Trade-offs:**
- ✅ No Hessian computation needed
- ✅ Better than GD, cheaper than Newton
- ✅ Good for medium-sized problems
- ❌ Still needs O(n²) memory
- ❌ Slower initial convergence

### **2. Fast Setup**
```python
max_iterations = 50        # Fewer iterations
tolerance = 1e-5          # Slightly relaxed
line_search = "backtracking"  # Simple line search
initial_hessian = "scaled_identity"  # Better start
```

**🧠 Cách nhớ:**
- "50 iterations với better initialization"
- "Backtracking như trial-and-error for step size"
- "Scaled identity = educated guess"

**⚖️ Trade-offs:**
- ✅ Faster convergence initially
- ✅ Good for prototyping
- ❌ May not reach high precision
- ❌ Depends on good scaling

### **3. Robust Setup**
```python
max_iterations = 200       # Many iterations allowed
tolerance = 1e-8          # High precision
line_search = "strong_wolfe"  # Rigorous conditions
initial_hessian = "diagonal"  # Feature-based scaling
restart_frequency = 50     # Periodic restart
```

**🧠 Cách nhớ:**
- "200 iterations for worst-case scenarios"
- "Strong Wolfe = bulletproof line search"
- "Diagonal initialization considers feature scales"
- "Restart every 50 = fresh start prevention"

**⚖️ Trade-offs:**
- ✅ Very robust and reliable
- ✅ Handles ill-conditioned problems
- ✅ High final precision
- ❌ More computational overhead
- ❌ Complex parameter tuning

## 📊 **Line Search Deep Dive**

### **Why Line Search?**
BFGS direction có thể không optimal, cần find best step size:
```
α* = argmin_α f(x_k + α * d_k)
```

### **Line Search Types**

| Type | Conditions | Cost | Robustness |
|------|------------|------|------------|
| **Backtracking** | Armijo only | Low | Good |
| **Strong Wolfe** | Armijo + Curvature | Medium | Excellent |
| **Exact** | True minimum | High | Perfect |

### **Armijo Condition**
```
f(x_k + α d_k) ≤ f(x_k) + c₁ α ∇f(x_k)^T d_k
```
- `c₁ = 1e-4`: "Sufficient decrease parameter"
- Ensures progress in function value

### **Wolfe Curvature Condition**
```
∇f(x_k + α d_k)^T d_k ≥ c₂ ∇f(x_k)^T d_k
```
- `c₂ = 0.9`: "Curvature parameter"
- Ensures sufficient step size

## 🧮 **Mathematical Properties**

### **Convergence Rate**
**Superlinear convergence:**
```
lim_{k→∞} ||x_{k+1} - x*|| / ||x_k - x*|| = 0
```

**In practice:**
- Iterations 1-10: Linear-like (building Hessian)
- Iterations 10+: Near-quadratic (good approximation)

### **Memory Complexity**

| Operation | BFGS | Newton | GD |
|-----------|------|--------|----| 
| **Memory** | O(n²) | O(n²) | O(n) |
| **Per iteration** | O(n²) | O(n³) | O(n) |
| **Hessian computation** | No | Yes | No |

### **Comparison với Other Methods**

| Method | Convergence | Memory | Robustness | Use Case |
|--------|-------------|--------|------------|----------|
| **GD** | Linear | O(n) | High | Large problems |
| **Newton** | Quadratic | O(n²) | Medium | Small problems |
| **BFGS** | Superlinear | O(n²) | High | Medium problems |
| **L-BFGS** | Superlinear | O(m·n) | High | Large problems |

## 🎯 **Khi Nào Dùng BFGS**

### **Ideal Conditions**
- ✅ **Medium-sized problems** (100-10,000 variables)
- ✅ **Smooth functions** (twice differentiable)
- ✅ **Need faster than GD** but cheaper than Newton
- ✅ **Good gradient information** available

### **Avoid When**
- ❌ **Very large problems** (use L-BFGS instead)
- ❌ **Non-smooth functions** (use subgradient methods)
- ❌ **Memory constraints** (use GD)
- ❌ **Very noisy gradients** (use stochastic methods)

### **Perfect Use Cases**
1. **Logistic Regression** (smooth, medium-sized)
2. **Neural Network Training** (small networks)
3. **Parameter Estimation** (statistical models)
4. **Engineering Optimization** (design problems)

## 🔧 **Implementation Details**

### **Initial Hessian Strategies**

```python
# 1. Identity matrix (simple)
H_0 = np.eye(n)

# 2. Scaled identity (better)
γ = (y_0^T s_0) / (y_0^T y_0)
H_0 = γ * np.eye(n)

# 3. Diagonal scaling (best)
H_0 = np.diag(feature_scales)
```

### **Memory Management**
```python
# Store only necessary vectors
class BFGSState:
    def __init__(self, n):
        self.H = np.eye(n)  # Current inverse Hessian
        self.s_history = []  # Step vectors
        self.y_history = []  # Gradient changes
        
    def update(self, s_k, y_k):
        # BFGS update formula
        rho = 1.0 / (y_k.T @ s_k)
        # ... update H using Sherman-Morrison
```

### **Numerical Stability**
```python
# Check curvature condition
if y_k.T @ s_k > 1e-8:
    # Safe to update
    self.bfgs_update(s_k, y_k)
else:
    # Skip update or use damped BFGS
    self.skip_update()
```

## 🧠 **Memory Aids & Intuition**

### **BFGS vs Detective Analogy**
```
Gradient Descent = Rookie detective
- Chỉ có basic clues (gradient)
- Từ từ thu thập evidence

Newton Method = Experienced detective với lab
- Có full forensic analysis (Hessian)
- Expensive nhưng accurate

BFGS = Smart detective với experience
- Learn from previous cases (update approximation)
- Efficient với accumulated knowledge
```

### **Line Search như GPS Navigation**
```
No line search = Đi theo hướng mà không care distance
- Có thể overshoot hoặc understep

With line search = GPS tính optimal distance
- Take right direction with right step size
- Much more efficient journey
```

### **Convergence Pattern**
```
Phase 1 (0-10 iter):  🐢 Learning the landscape
Phase 2 (10-30 iter): 🐰 Good approximation built
Phase 3 (30+ iter):   🚀 Near-Newton performance
```

## 🔍 **Troubleshooting Guide**

### **Common Issues & Solutions**

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Slow initial convergence** | High cost first 10 iterations | Normal, wait for Hessian buildup |
| **Line search failures** | α becomes very small | Check gradient accuracy |
| **Memory issues** | Out of RAM | Switch to L-BFGS |
| **Poor final accuracy** | Stalls before tolerance | Check conditioning |

### **Diagnostic Checks**
```python
# 1. Check curvature condition
if y_k.T @ s_k <= 0:
    print("Warning: Negative curvature detected")
    
# 2. Monitor condition number
κ = np.linalg.cond(H_k)
if κ > 1e12:
    print("Warning: Ill-conditioned Hessian approximation")
    
# 3. Check step sizes
if α < 1e-8:
    print("Warning: Very small steps, possible convergence")
```

### **Performance Tuning**
```python
# 1. Adjust line search parameters
c1 = 1e-4  # Decrease for more aggressive steps
c2 = 0.9   # Increase for fewer line search iterations

# 2. Restart strategy
if iteration % restart_freq == 0:
    H = np.eye(n)  # Reset to identity
    
# 3. Scaling
if problem_has_different_scales:
    use_diagonal_initialization()
```

## 📈 **Advanced Variants**

### **L-BFGS (Limited Memory)**
For large problems, store only m recent (s,y) pairs:
```python
# Memory: O(m·n) instead of O(n²)
m = 10  # Typical value
history_size = min(m, iteration)
```

### **Damped BFGS**
For robustness, interpolate with previous Hessian:
```python
if y_k.T @ s_k < threshold:
    y_k_damped = θ * y_k + (1-θ) * B_k @ s_k
```

### **BFGS with Trust Region**
Combine with trust region for global convergence:
```python
step = solve_trust_region(B_k, gradient, radius)
```

## 📖 **Further Reading**

### **Key Papers**
1. **Broyden (1970)**: Original quasi-Newton idea
2. **Fletcher (1970)**: BFGS derivation
3. **Nocedal & Wright**: Numerical Optimization textbook

### **Modern Variations**
- **L-BFGS**: Limited memory version
- **BFGS-B**: Box-constrained version
- **Hessian-free methods**: For very large problems

---

*"BFGS: Learn the curvature as you go, combining the best of gradient descent and Newton methods"* 🧠⚡🎯