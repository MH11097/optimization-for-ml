# Newton Methods - Kiến Thức Toán Học & Setups

## 📚 **Lý Thuyết Toán Học**

### **Định Nghĩa Cơ Bản**

Newton Method sử dụng ma trận Hessian (xấp xỉ bậc 2 của hàm mục tiêu) để tìm cực tiểu của hàm f(x).
**Công thức cập nhật:**

```
x_{k+1} = x_k - H_k^{-1} ∇f(x_k)
```

Trong đó:

- `x_k`: điểm hiện tại
- `H_k = ∇²f(x_k)`: Hessian matrix tại x_k
- `∇f(x_k)`: gradient tại x_k

### **Intuition: Taylor Expansion**

Newton method approximates f(x) bằng quadratic Taylor expansion:

```
f(x) ≈ f(x_k) + ∇f(x_k)^T(x - x_k) + (1/2)(x - x_k)^T H_k (x - x_k)
```

Minimum của quadratic này là:

```
x* = x_k - H_k^{-1} ∇f(x_k)
```

### **Cho Linear Regression**

Function: `f(w) = (1/2n) ||Xw - y||²`
**Gradient:**

```
∇f(w) = (1/n) X^T (Xw - y)
```

**Hessian (constant!):**

```
H = ∇²f(w) = (1/n) X^T X
```

**Newton update:**

```
w_{k+1} = w_k - H^{-1} ∇f(w_k)
```

### **Convergence Properties**

- **Quadratic convergence**: `||x_{k+1} - x*|| ≤ C ||x_k - x*||²`
- **Local convergence**: Cần start gần optimal point
- **One-step convergence** cho quadratic functions!

## 🎯 **Setup và Ý Nghĩa**

### **Standard Setup**

```python
regularization = 1e-8    # Minimal
max_iterations = 50      # Usually enough
tolerance = 1e-10        # Very strict
```

**🧠 Cách nhớ:**

- "1e-8 như thêm 1 giọt nước vào biển"
- "50 bước vì Newton rất nhanh"
- "1e-10 vì có thể achieve high precision"
  **⚖️ Trade-offs:**
- ✅ Extremely fast convergence
- ✅ High precision
- ✅ Optimal for quadratic functions
- ❌ Expensive Hessian computation
- ❌ Memory intensive O(n²)

### **Robust Setup**

```python
regularization = 1e-6    # Higher for stability
max_iterations = 100     # More iterations allowed
tolerance = 1e-8         # Slightly relaxed
```

**🧠 Cách nhớ:**

- "1e-6 như insurance policy"
- "100 iterations cho worst case"
- "Still very precise"
  **⚖️ Trade-offs:**
- ✅ More robust to ill-conditioning
- ✅ Handles edge cases better
- ❌ Slightly less pure Newton behavior

## 📊 **Regularization Deep Dive**

### **Why Regularization?**

Hessian `H = X^T X` có thể singular hoặc ill-conditioned:
**Problems:**

1. **Singular H**: Det(H) = 0, không invert được
2. **Ill-conditioned**: Condition number κ(H) >> 1
   **Solution: Regularization**

```
H_reg = H + λI
```

### **Regularization Values**

| λ         | Khi nào dùng              | Effect                  |
| --------- | ------------------------- | ----------------------- |
| **1e-12** | Perfect conditioning      | Minimal impact          |
| **1e-8**  | Standard (good data)      | Barely noticeable       |
| **1e-6**  | Moderate ill-conditioning | Slight smoothing        |
| **1e-4**  | Poor conditioning         | Noticeable ridge effect |
| **1e-2**  | Very ill-conditioned      | Strong regularization   |

### **Condition Number Analysis**

```python
κ = λ_max / λ_min  # Condition number
if κ < 1e6:    # Well-conditioned
    λ = 1e-8
elif κ < 1e12: # Moderately conditioned
    λ = 1e-6
else:          # Ill-conditioned
    λ = 1e-4
```

## 🧮 **Mathematical Properties**

### **Convergence Rate**

**Quadratic convergence:**

```
||x_{k+1} - x*|| ≤ C ||x_k - x*||²
```

**In practice:**

- Iteration 1: Error = 0.1
- Iteration 2: Error = 0.01
- Iteration 3: Error = 0.0001
- Iteration 4: Error = 0.00000001

### **Linear Algebra Costs**

| Operation                 | Cost   | Memory |
| ------------------------- | ------ | ------ |
| **Hessian computation**   | O(n²m) | O(n²)  |
| **Matrix inversion**      | O(n³)  | O(n²)  |
| **Matrix-vector product** | O(n²)  | O(n)   |

**Total per iteration:** O(n²m + n³)

### **Comparison với Gradient Descent**

| Aspect                 | Newton    | Gradient Descent |
| ---------------------- | --------- | ---------------- |
| **Convergence rate**   | Quadratic | Linear           |
| **Iterations needed**  | ~5-20     | ~100-1000        |
| **Per-iteration cost** | O(n³)     | O(nm)            |
| **Memory**             | O(n²)     | O(n)             |
| **Robustness**         | Local     | Global           |

## 🎯 **Khi Nào Dùng Newton Method**

### **Ideal Conditions**

- ✅ **Small to medium n** (< 1000 features)
- ✅ **Quadratic/near-quadratic functions**
- ✅ **High precision required**
- ✅ **Good starting point**

### **Avoid When**

- ❌ **Large n** (> 10,000 features)
- ❌ **Ill-conditioned problems**
- ❌ **Non-convex functions**
- ❌ **Memory constraints**

### **Perfect Use Cases**

1. **Linear/Ridge Regression** (exactly quadratic)
2. **Logistic Regression** (well-conditioned)
3. **Small neural networks** (final layer tuning)
4. **Scientific computing** (high precision needed)

## 🔧 **Implementation Details**

### **Hessian Computation**

```python
# For f(w) = (1/2n) ||Xw - y||²
H = (1/n) * X.T @ X
# Add regularization
H_reg = H + λ * np.eye(n)
# Check condition number
κ = np.linalg.cond(H_reg)
```

### **Safe Matrix Inversion**

```python
try:
    H_inv = np.linalg.inv(H_reg)
except np.linalg.LinAlgError:
    # Increase regularization
    H_reg = H + (λ * 1000) * np.eye(n)
    H_inv = np.linalg.inv(H_reg)
```

### **Alternative: Solve Linear System**

Instead of computing H^{-1}, solve:

```python
# More numerically stable
newton_step = np.linalg.solve(H_reg, gradient)
weights_new = weights - newton_step
```

## 🧠 **Memory Aids & Intuition**

### **Newton vs Car Analogy**

```
Gradient Descent = Đi bộ với GPS
- Chỉ biết hướng (gradient)
- Từ từ, step by step
- An toàn nhưng chậm
Newton Method = Đi xe có bản đồ chi tiết
- Biết cả hướng và độ cong đường (Hessian)
- Có thể optimal route
- Nhanh nhưng cần sophisticated system
```

### **Regularization như Insurance**

```
λ = 1e-8   →  "Chỉ mua bảo hiểm minimum"
λ = 1e-6   →  "Bảo hiểm standard"
λ = 1e-4   →  "Bảo hiểm comprehensive"
```

### **Convergence Visualization**

```
Iteration 0: 🔴🎯  (far from target)
Iteration 1: 🟡🎯  (much closer)
Iteration 2: 🟢🎯  (very close)
Iteration 3: ✅🎯  (bullseye!)
```

## 🔍 **Troubleshooting Guide**

### **Common Issues & Solutions**

| Problem              | Symptoms        | Solution                 |
| -------------------- | --------------- | ------------------------ |
| **Singular Hessian** | LinAlgError     | Increase regularization  |
| **Slow convergence** | Many iterations | Check conditioning       |
| **Memory error**     | Out of RAM      | Use Quasi-Newton instead |
| **Poor accuracy**    | High final cost | Check data preprocessing |

### **Diagnostic Checks**

```python
# 1. Check condition number
κ = np.linalg.cond(H)
print(f"Condition number: {κ:.2e}")
# 2. Check eigenvalues
eigenvals = np.linalg.eigvals(H)
print(f"Min eigenvalue: {np.min(eigenvals):.2e}")
print(f"Max eigenvalue: {np.max(eigenvals):.2e}")
# 3. Check rank
rank = np.linalg.matrix_rank(H)
print(f"Rank: {rank}/{H.shape[0]}")
```

### **Performance Optimization**

```python
# 1. Use Cholesky for positive definite H
if is_positive_definite(H):
    L = np.linalg.cholesky(H)
    newton_step = solve_triangular(L, gradient)
# 2. Cache Hessian if constant
if problem_is_quadratic:
    H_inv = np.linalg.inv(H)  # Compute once
# 3. Use sparse matrices if applicable
if is_sparse(X):
    H = sparse_dot_product(X.T, X)
```

## 📈 **Advanced Variants**

### **Damped Newton Method**

```python
# Add line search
α = backtracking_line_search(...)
weights_new = weights - α * H_inv @ gradient
```

### **Trust Region Newton**

```python
# Constrain step size
newton_step = solve_trust_region(H, gradient, radius)
```

### **Limited Memory Newton**

## For large problems, approximate Hessian with limited memory.

_"Newton Method: Use the full curvature information to take optimal steps toward the minimum"_ 🎯⚡
