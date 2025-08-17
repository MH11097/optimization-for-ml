# Gradient Descent Methods - Kiến Thức Toán Học & Setups

## 📚 **Lý Thuyết Toán Học**

### **Định Nghĩa Cơ Bản**
Gradient Descent là thuật toán tối ưu iterative để tìm minimum của function f(x).

**Công thức cập nhật:**
```
x_{k+1} = x_k - α_k ∇f(x_k)
```

Trong đó:
- `x_k`: điểm hiện tại
- `α_k`: learning rate (step size)
- `∇f(x_k)`: gradient tại x_k
- `x_{k+1}`: điểm tiếp theo

### **Gradient của MSE (Mean Squared Error)**
Cho bài toán linear regression: `f(w) = (1/2n) ||Xw - y||²`

**Gradient:**
```
∇f(w) = (1/n) X^T (Xw - y)
```

**Hessian:**
```
∇²f(w) = (1/n) X^T X
```

### **Điều Kiện Convergence**
1. **Lipschitz Continuity:** `||∇f(x) - ∇f(y)|| ≤ L||x - y||`
2. **Strong Convexity:** `f(y) ≥ f(x) + ∇f(x)^T(y-x) + (μ/2)||y-x||²`

**Convergence rate:** `O(1/k)` cho convex, `O(ρ^k)` cho strongly convex

## 🎯 **Các Setup và Ý Nghĩa**

### **1. Standard Setup**
```python
learning_rate = 0.01      # Moderate
max_iterations = 1000     # Sufficient
tolerance = 1e-6          # Good precision
```

**🧠 Cách nhớ:**
- "0.01 = 1% của gradient mỗi step"
- "Như đi bộ vừa phải, không vội không chậm"
- "1000 bước đủ cho hầu hết bài toán"

**⚖️ Trade-offs:**
- ✅ Ổn định, ít dao động
- ✅ Dễ tune, predictable
- ❌ Không nhanh nhất
- ❌ Có thể chậm cho số liệu lớn

### **2. Fast Setup**
```python
learning_rate = 0.1       # High
max_iterations = 500      # Fewer needed
tolerance = 1e-5          # Slightly relaxed
```

**🧠 Cách nhớ:**
- "0.1 = 10% của gradient, bước to"
- "Như chạy nhanh, có thể vượt đích"
- "500 bước vì hy vọng convergence nhanh"

**⚖️ Trade-offs:**
- ✅ Convergence rất nhanh
- ✅ Tốt cho experimentation
- ❌ Risk overshooting
- ❌ Có thể dao động

**⚠️ Warning Signs:**
- Cost tăng thay vì giảm
- Oscillations around minimum
- NaN values

### **3. Precise Setup**
```python
learning_rate = 0.001     # Low
max_iterations = 2000     # More needed
tolerance = 1e-8          # Very strict
```

**🧠 Cách nhớ:**
- "0.001 = 0.1% gradient, bước nhỏ"
- "Như walking meditation, từ từ chính xác"
- "2000 bước vì cần thời gian"

**⚖️ Trade-offs:**
- ✅ Highest precision
- ✅ Very stable
- ✅ Finds true minimum
- ❌ Slow training
- ❌ May be overkill

## 📈 **Convergence Analysis**

### **Linear Convergence Rate**
Cho strongly convex function:
```
f(x_k) - f* ≤ ρ^k (f(x_0) - f*)
```

Trong đó `ρ = (κ-1)/(κ+1)`, `κ = L/μ` (condition number)

**Condition number impact:**
- `κ = 1`: Perfect (ρ = 0)
- `κ = 10`: Good (ρ ≈ 0.82)
- `κ = 100`: Poor (ρ ≈ 0.98)

### **Optimal Learning Rate**
Cho quadratic function: `α* = 2/(L + μ)`

**Rule of thumb:**
- Start with `α = 1/L`
- If oscillating: decrease α
- If too slow: increase α

## 🔧 **Setup Selection Guide**

### **Theo Mục Đích**

| Mục đích | Setup | Lý do |
|----------|-------|-------|
| 🎓 **Học tập** | Standard | Hiểu rõ behavior, ít surprising |
| ⚡ **Prototyping** | Fast | Kết quả nhanh, iterate ideas |
| 🏭 **Production** | Precise | Accuracy quan trọng, có thời gian |
| 🧪 **Research** | Precise | Reproducible, high quality |

### **Theo Đặc Điểm Dataset**

| Dataset | Khuyến nghị | Lý do |
|---------|------------|-------|
| **Small (< 1K)** | Precise | Có thể afford slow training |
| **Medium (1K-10K)** | Standard | Balance tốt |
| **Large (> 10K)** | Fast hoặc SGD | Training time matters |

### **Theo Condition Number**

| Condition κ | Learning Rate | Strategy |
|-------------|---------------|-----------|
| **κ < 10** | 0.1 (Fast) | Well-conditioned, can go fast |
| **10 ≤ κ < 100** | 0.01 (Standard) | Moderate conditioning |
| **κ ≥ 100** | 0.001 (Precise) | Ill-conditioned, go slow |

## 🧠 **Memory Aids**

### **Learning Rate Intuition**
```
α = 0.001  →  "Cụ già đi bộ"     (chậm, chắc chắn)
α = 0.01   →  "Người bình thường" (vừa phải)
α = 0.1    →  "Chạy nhanh"       (nhanh, risk넘어짐)
α = 1.0    →  "Nhảy xa"          (probably overshoot)
```

### **Convergence Patterns**
```
📉 Good:     Cost giảm smooth
📊 OK:       Cost giảm với small oscillations  
🌊 Warning:  Cost oscillates around minimum
💥 Bad:      Cost explodes or NaN
```

### **Debugging Checklist**
1. **Cost tăng?** → Giảm learning rate
2. **Too slow?** → Tăng learning rate hoặc check convergence
3. **Plateaus?** → Check tolerance hoặc local minimum
4. **NaN values?** → Learning rate quá lớn

## 🎓 **Advanced Concepts**

### **Momentum Variant**
```python
v = β * v + α * gradient
w = w - v
```
- `β = 0.9`: "Nhớ 90% momentum cũ"
- Giúp vượt local minima
- Accelerate in consistent directions

### **Adaptive Learning Rate**
```python
α_t = α_0 / (1 + decay_rate * t)
```
- Start fast, slow down over time
- "Như brake khi gần đích"

### **Line Search**
Automatically find optimal step size:
```python
α_t = argmin_α f(x_t - α * ∇f(x_t))
```

## 🔍 **Troubleshooting Guide**

### **Common Issues**

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Overshooting** | Cost oscillates/increases | Decrease α |
| **Slow convergence** | Cost plateaus early | Increase α or check tolerance |
| **Poor conditioning** | Very slow, erratic | Use preconditioning |
| **Local minimum** | Stuck at suboptimal | Add momentum or restart |

### **Setup Debugging**
```python
# Quick test for learning rate
for α in [0.001, 0.01, 0.1]:
    run_few_iterations(α)
    if cost_decreased:
        print(f"α = {α} works")
```

## 📖 **Further Reading**

### **Key Papers**
1. **Cauchy (1847)**: Original gradient descent
2. **Polyak (1964)**: Momentum methods
3. **Nesterov (1983)**: Accelerated gradient descent

### **Modern Variations**
- **AdaGrad**: Adaptive per-parameter learning rates
- **Adam**: Adaptive moments estimation  
- **RMSprop**: Root mean square propagation

---

*"Gradient Descent: Follow the steepest downhill path to find the valley bottom"* 🏔️➡️🏞️