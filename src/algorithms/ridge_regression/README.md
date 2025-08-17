# Ridge Regression - Kiến Thức Toán Học & Setups

## 📚 **Lý Thuyết Toán Học**

### **Định Nghĩa Cơ Bản**
Ridge Regression thêm L2 regularization vào linear regression để prevent overfitting và handle multicollinearity.

**Objective function:**
```
f(w) = (1/2n) ||Xw - y||² + (λ/2) ||w||²
```

Trong đó:
- `(1/2n) ||Xw - y||²`: Original MSE loss
- `(λ/2) ||w||²`: L2 regularization penalty
- `λ ≥ 0`: Regularization parameter

### **Closed-Form Solution**
Ridge regression có analytical solution:
```
w* = (X^T X + λI)^{-1} X^T y
```

**So với OLS:** `w* = (X^T X)^{-1} X^T y`

### **Gradient và Hessian**
**Gradient:**
```
∇f(w) = (1/n) X^T (Xw - y) + λw
```

**Hessian:**
```
∇²f(w) = (1/n) X^T X + λI
```

**Key insight:** Hessian luôn positive definite khi λ > 0!

### **Regularization Effect**
L2 penalty "shrinks" weights toward zero:
- **Small λ**: Gần như OLS, risk overfitting
- **Large λ**: Weights gần 0, risk underfitting
- **Optimal λ**: Balance bias-variance tradeoff

## 🎯 **Các Setup và Ý Nghĩa**

### **1. Light Regularization Setup**
```python
lambda_reg = 0.01         # Small penalty
solver = "analytical"     # Closed-form solution
normalize_features = True # Important for Ridge
```

**🧠 Cách nhớ:**
- "0.01 = 1% penalty trên weights"
- "Analytical = tính exact solution"
- "Normalize vì Ridge sensitive to scale"

**⚖️ Trade-offs:**
- ✅ Fast convergence (closed-form)
- ✅ Minimal bias introduced
- ✅ Good for well-conditioned problems
- ❌ May not prevent overfitting enough
- ❌ Still sensitive to multicollinearity

### **2. Standard Regularization Setup**
```python
lambda_reg = 1.0          # Moderate penalty
solver = "gradient_descent" # Iterative method
max_iterations = 1000     # Sufficient for convergence
learning_rate = 0.01      # Standard GD rate
```

**🧠 Cách nhớ:**
- "λ = 1.0 = equal weight MSE vs regularization"
- "GD khi matrix too large for inversion"
- "1000 iterations for safety"

**⚖️ Trade-offs:**
- ✅ Good bias-variance balance
- ✅ Handles multicollinearity well
- ✅ Scalable to large datasets
- ❌ Requires hyperparameter tuning
- ❌ Slower than analytical

### **3. Strong Regularization Setup**
```python
lambda_reg = 100.0        # Heavy penalty
solver = "coordinate_descent" # Efficient for regularized
alpha_grid = [0.1, 1, 10, 100] # Cross-validation
cv_folds = 5              # K-fold validation
```

**🧠 Cách nhớ:**
- "λ = 100 = regularization dominates"
- "Coordinate descent efficient for Ridge"
- "Grid search để tìm optimal λ"

**⚖️ Trade-offs:**
- ✅ Strong overfitting prevention
- ✅ Very stable solutions
- ✅ Automatic λ selection
- ❌ High bias, may underfit
- ❌ Computational overhead from CV

## 📊 **Regularization Parameter Deep Dive**

### **Lambda Selection Strategies**

| Strategy | Method | Pros | Cons |
|----------|--------|------|------|
| **Fixed small** | λ = 0.01 | Fast, simple | May overfit |
| **Fixed moderate** | λ = 1.0 | Good default | Not optimal |
| **Cross-validation** | Grid search | Optimal for data | Expensive |
| **Analytical** | GCV/AIC/BIC | Fast approximation | Asymptotic |

### **Lambda Effects Visualization**
```
λ = 0:     ────────○ (OLS, possible overfitting)
λ = 0.01:  ──────○   (Light shrinkage)
λ = 1.0:   ────○     (Moderate shrinkage)  
λ = 100:   ○         (Heavy shrinkage)
λ = ∞:     •         (All weights → 0)
```

### **Bias-Variance Tradeoff**
```
Bias²    = ||E[ŵ] - w*||²     (Increases with λ)
Variance = E[||ŵ - E[ŵ]||²]   (Decreases with λ)
MSE      = Bias² + Variance + σ²
```

**Optimal λ minimizes total MSE**

## 🧮 **Mathematical Properties**

### **Condition Number Improvement**
Original: `κ(X^T X) = λ_max / λ_min`
Ridge: `κ(X^T X + λI) = (λ_max + λ) / (λ_min + λ)`

**Effect:** Ridge dramatically improves conditioning!

### **Effective Degrees of Freedom**
```
df(λ) = tr(X(X^T X + λI)^{-1} X^T) = Σ σᵢ²/(σᵢ² + λ)
```
- `σᵢ`: singular values of X
- `df(0) = p` (OLS)
- `df(∞) = 0` (null model)

### **Shrinkage Factors**
Each principal component shrunk by factor:
```
sᵢ = σᵢ²/(σᵢ² + λ)
```
- Large σᵢ (important directions): Less shrinkage
- Small σᵢ (noise directions): More shrinkage

## 🎯 **Solver Comparison**

### **Analytical vs Iterative**

| Aspect | Analytical | Gradient Descent | Coordinate Descent |
|--------|------------|------------------|--------------------|
| **Speed** | O(n³) one-shot | O(kn²) iterations | O(kn) iterations |
| **Memory** | O(n²) | O(n) | O(n) |
| **Accuracy** | Exact | Approximate | Approximate |
| **Scale limit** | n < 10K | Any n | Any n |

### **When to Use Each Solver**

```python
if n_features < 1000:
    solver = "analytical"     # Fast and exact
elif sparse_data:
    solver = "coordinate_descent"  # Efficient for sparse
else:
    solver = "gradient_descent"    # General purpose
```

## 🔧 **Implementation Details**

### **Feature Normalization**
```python
# Essential for Ridge!
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# Why? Ridge penalty ||w||² treats all features equally
# Without normalization: features with large scale get unfairly penalized
```

### **Analytical Solution Implementation**
```python
def ridge_analytical(X, y, lambda_reg):
    n, p = X.shape
    
    # Add regularization to normal equations
    A = X.T @ X + lambda_reg * np.eye(p)
    b = X.T @ y
    
    # Solve linear system (more stable than inversion)
    weights = np.linalg.solve(A, b)
    return weights
```

### **Cross-Validation for Lambda**
```python
def ridge_cv(X, y, lambda_grid, cv_folds=5):
    best_lambda = None
    best_score = float('inf')
    
    for lambda_reg in lambda_grid:
        cv_scores = []
        for train_idx, val_idx in kfold_split(X, cv_folds):
            # Train on fold
            w = ridge_fit(X[train_idx], y[train_idx], lambda_reg)
            # Validate
            val_error = mse(X[val_idx] @ w, y[val_idx])
            cv_scores.append(val_error)
        
        avg_score = np.mean(cv_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_lambda = lambda_reg
    
    return best_lambda
```

## 🧠 **Memory Aids & Intuition**

### **Ridge vs Housebuilding Analogy**
```
OLS = Xây nhà không có building code
- Tự do complete, có thể unstable
- Perfect fit to current data
- Risk collapse với new data

Ridge = Xây nhà với safety regulations  
- Ít freedom hơn nhưng safer
- Small trade-off in current fit
- Much more stable với new conditions
```

### **Lambda như Volume Control**
```
λ = 0:    🔊🔊🔊 (Full volume, possible noise)
λ = 0.01: 🔊🔊   (Slight reduction)  
λ = 1.0:  🔊     (Moderate volume)
λ = 100:  🔈     (Very quiet)
λ = ∞:    🔇     (Muted)
```

### **Regularization Path Visualization**
```
No Reg:    w₁=5.2, w₂=-3.1, w₃=8.7  (Large weights)
Light:     w₁=4.8, w₂=-2.9, w₃=8.1  (Slight shrinkage)
Moderate:  w₁=3.2, w₂=-1.8, w₃=5.4  (Clear shrinkage)
Heavy:     w₁=0.8, w₂=-0.4, w₃=1.2  (Heavy shrinkage)
```

## 🔍 **Troubleshooting Guide**

### **Common Issues & Solutions**

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Still overfitting** | High val error vs train | Increase λ |
| **Underfitting** | Both train/val error high | Decrease λ |
| **Numerical instability** | Weird weight values | Check feature scaling |
| **Poor performance** | Ridge worse than OLS | Check if regularization needed |

### **Diagnostic Checks**
```python
# 1. Check conditioning improvement
κ_original = np.linalg.cond(X.T @ X)
κ_ridge = np.linalg.cond(X.T @ X + lambda_reg * np.eye(p))
print(f"Condition number: {κ_original:.2e} → {κ_ridge:.2e}")

# 2. Check effective degrees of freedom
eigenvals = np.linalg.eigvals(X.T @ X)
df_eff = np.sum(eigenvals / (eigenvals + lambda_reg))
print(f"Effective DF: {df_eff:.1f}/{p}")

# 3. Check weight shrinkage
w_ols = ols_fit(X, y)
w_ridge = ridge_fit(X, y, lambda_reg)
shrinkage = np.linalg.norm(w_ridge) / np.linalg.norm(w_ols)
print(f"Weight shrinkage: {shrinkage:.3f}")
```

### **Lambda Selection Guidelines**
```python
# Rule of thumb starting points
n, p = X.shape

if n >> p:  # Many samples
    lambda_start = 0.01
elif n ≈ p:  # Equal samples/features  
    lambda_start = 1.0
else:  # Few samples (n < p)
    lambda_start = 10.0

# Then use CV to refine
lambda_grid = lambda_start * np.logspace(-2, 2, 50)
```

## 📈 **Advanced Topics**

### **Bayesian Interpretation**
Ridge regression equivalent to MAP estimation với Gaussian prior:
```
w ~ N(0, σ²/λ I)
y|w ~ N(Xw, σ²I)
```

### **Kernel Ridge Regression**
For non-linear problems:
```
w* = X^T (XX^T + λI)^{-1} y
f(x_new) = k(x_new, X) (K + λI)^{-1} y
```

### **Group Ridge Regression**
Penalize groups of features together:
```
Penalty = λ Σⱼ ||wⱼ||₂
```

## 📖 **Connections to Other Methods**

### **Relationship to SVD**
```
X = UΣV^T  (SVD decomposition)
w_ridge = V D_λ U^T y
where D_λ = diag(σᵢ/(σᵢ² + λ))
```

### **Connection to PCA**
Ridge regression shrinks along principal components:
- High variance directions: Less shrinkage
- Low variance directions: More shrinkage

### **Limiting Cases**
```
λ → 0:   Ridge → OLS
λ → ∞:   Ridge → Zero weights
```

---

*"Ridge Regression: Add just enough constraint to stabilize the solution without losing too much fit"* ⚖️🏔️✨