# Advanced Optimization Methods - Kiến Thức Toán Học & Setups

## 📚 **Tổng Quan Advanced Methods**

### **Phân Loại Methods**
Advanced optimization methods giải quyết các limitations của basic methods:

| Category | Methods | Main Innovation |
|----------|---------|-----------------|
| **Adaptive Learning Rate** | AdaGrad, RMSprop, Adam | Per-parameter learning rates |
| **Momentum Variants** | Heavy Ball, Nesterov, NADAM | Acceleration techniques |
| **Coordinate Methods** | Coordinate Descent, ADMM | Structured optimization |
| **Variance Reduction** | SVRG, SAGA, SARAH | Reduce stochastic noise |
| **Second-Order** | Natural Gradients, K-FAC | Better curvature information |
| **Meta-Learning** | MAML, Reptile | Learn to optimize |

### **Khi Nào Cần Advanced Methods**
- ✅ **Standard methods fail**: Convergence issues
- ✅ **Special structure**: Sparse, constrained, multi-objective
- ✅ **Scale challenges**: Very large or very small problems
- ✅ **Performance critical**: Need best possible results

## 🚀 **AdaGrad - Adaptive Gradient**

### **Core Idea**
Adapt learning rate cho từng parameter based on historical gradients:
```
α_t^{(i)} = α / √(G_t^{(i)} + ε)
```

**Update rule:**
```
G_t^{(i)} = G_{t-1}^{(i)} + (g_t^{(i)})²
w_t^{(i)} = w_{t-1}^{(i)} - α_t^{(i)} * g_t^{(i)}
```

### **Setup Configurations**

#### **Standard AdaGrad Setup**
```python
learning_rate = 0.01      # Higher than SGD
epsilon = 1e-8            # Numerical stability
accumulate_gradients = True # Track gradient squares
```

**🧠 Intuition:**
- "Parameters với large gradients get smaller learning rates"
- "Automatically adapts to feature importance"
- "Good for sparse features"

**⚖️ Trade-offs:**
- ✅ No manual LR tuning per parameter
- ✅ Great for sparse data
- ✅ Robust to feature scaling
- ❌ Learning rate monotonically decreases
- ❌ Can stop learning too early

#### **Modified AdaGrad Setup**
```python
learning_rate = 0.1       # Even higher initial
epsilon = 1e-6            # Less conservative
decay_factor = 0.99       # Forget old gradients slowly
```

## 🎯 **RMSprop - Root Mean Square Propagation**

### **Core Idea**
Fix AdaGrad's decreasing learning rate problem với exponential moving average:
```
v_t = β * v_{t-1} + (1-β) * g_t²
w_t = w_{t-1} - α * g_t / √(v_t + ε)
```

### **Setup Configurations**

#### **Standard RMSprop Setup**
```python
learning_rate = 0.001     # Standard rate
beta = 0.9                # Exponential decay factor
epsilon = 1e-8            # Numerical stability
```

**🧠 Intuition:**
- "β = 0.9 means remember 90% of recent gradient history"
- "Like AdaGrad but với forgetting mechanism"
- "Good middle ground between AdaGrad và SGD"

#### **Aggressive RMSprop Setup**
```python
learning_rate = 0.01      # Higher rate
beta = 0.99               # Longer memory
epsilon = 1e-6            # Less conservative
```

## ⚡ **Adam - Adaptive Moment Estimation**

### **Core Idea**
Combines momentum (first moment) với adaptive learning rates (second moment):
```
m_t = β₁ * m_{t-1} + (1-β₁) * g_t          # First moment
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²         # Second moment

m̂_t = m_t / (1 - β₁ᵗ)                      # Bias correction
v̂_t = v_t / (1 - β₂ᵗ)                      # Bias correction

w_t = w_{t-1} - α * m̂_t / (√v̂_t + ε)
```

### **Setup Configurations**

#### **Standard Adam Setup**
```python
learning_rate = 0.001     # Conservative default
beta1 = 0.9               # Momentum parameter
beta2 = 0.999             # RMSprop parameter
epsilon = 1e-8            # Numerical stability
```

**🧠 Intuition:**
- "β₁ = 0.9: Remember direction (momentum)"
- "β₂ = 0.999: Remember gradient magnitudes"
- "Bias correction prevents slow start"
- "Best of both worlds: momentum + adaptive LR"

#### **Fast Adam Setup**
```python
learning_rate = 0.01      # More aggressive
beta1 = 0.9               # Standard momentum
beta2 = 0.99              # Shorter adaptive memory
epsilon = 1e-6            # Less conservative
```

#### **Stable Adam Setup**
```python
learning_rate = 0.0001    # Very conservative
beta1 = 0.95              # More momentum
beta2 = 0.9999            # Longer adaptive memory
epsilon = 1e-10           # High precision
```

## 🌊 **Nesterov Accelerated Gradient (NAG)**

### **Core Idea**
"Look ahead" trước khi compute gradient:
```
v_t = μ * v_{t-1} + α * ∇f(w_t - μ * v_{t-1})
w_t = w_{t-1} - v_t
```

### **Setup Configurations**

#### **Standard NAG Setup**
```python
learning_rate = 0.01      # Standard rate
momentum = 0.9            # Strong momentum
```

**🧠 Intuition:**
- "Look ahead = check gradient at future position"
- "Prevents overshooting better than standard momentum"
- "Especially good cho convex functions"

## 🎲 **SVRG - Stochastic Variance Reduced Gradient**

### **Core Idea**
Reduce variance trong stochastic gradients bằng full gradient snapshots:
```
# Every m iterations
μ = (1/n) Σᵢ ∇fᵢ(w̃)  # Full gradient snapshot

# Regular iterations  
g_t = ∇fᵢ(w_t) - ∇fᵢ(w̃) + μ
w_t = w_{t-1} - α * g_t
```

### **Setup Configurations**

#### **Standard SVRG Setup**
```python
learning_rate = 0.1       # Can be larger due to variance reduction
snapshot_frequency = 100  # Compute full gradient every 100 iterations
```

## 🧮 **Method Comparison Matrix**

### **Convergence Rates**

| Method | Convex | Strongly Convex | Non-convex |
|--------|--------|-----------------|------------|
| **SGD** | O(1/√k) | O(1/k) | - |
| **AdaGrad** | O(1/√k) | O(log k/k) | - |
| **RMSprop** | O(1/√k) | O(1/k) | Good empirically |
| **Adam** | O(1/√k) | O(1/k) | Good empirically |
| **SVRG** | O(1/k) | O(exp(-k)) | - |

### **Memory Requirements**

| Method | Per Parameter | Total Memory |
|--------|---------------|--------------|
| **SGD** | O(1) | O(n) |
| **AdaGrad** | O(1) | O(n) |
| **RMSprop** | O(1) | O(n) |
| **Adam** | O(1) | O(2n) |
| **SVRG** | O(1) | O(n) |

### **Hyperparameter Sensitivity**

| Method | LR Sensitivity | Other Parameters |
|--------|----------------|------------------|
| **SGD** | High | Medium (momentum) |
| **AdaGrad** | Medium | Low |
| **RMSprop** | Medium | Medium (β) |
| **Adam** | Low | Low (β₁, β₂) |
| **SVRG** | Medium | Medium (m) |

## 🎯 **Method Selection Guide**

### **Theo Problem Type**

| Problem Type | Recommended | Why |
|--------------|-------------|-----|
| **Computer Vision** | Adam, SGD+momentum | Good với noisy gradients |
| **NLP** | Adam, AdaGrad | Sparse features common |
| **Tabular Data** | BFGS, Adam | Well-conditioned problems |
| **Reinforcement Learning** | PPO, TRPO | Policy optimization needs |
| **Large Scale** | SGD variants | Memory efficiency |

### **Theo Dataset Size**

| Dataset Size | Method | Reason |
|--------------|--------|--------|
| **Small (< 1K)** | BFGS, Newton | Can afford second-order |
| **Medium (1K-100K)** | Adam, RMSprop | Good balance |
| **Large (100K-1M)** | SGD, SVRG | Efficiency matters |
| **Very Large (> 1M)** | Distributed SGD | Scalability critical |

### **Theo Computational Budget**

| Budget | Method | Trade-off |
|--------|--------|-----------|
| **Low** | SGD | Simple, fast per iteration |
| **Medium** | Adam | Adaptive, reasonable overhead |
| **High** | SVRG, Natural Gradients | Best convergence |

## 🔧 **Implementation Best Practices**

### **Adam Implementation**
```python
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def update(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
        
        self.t += 1
        
        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update weights
        weights = weights - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return weights
```

### **Adaptive Learning Rate Scheduling**
```python
def adaptive_lr_schedule(optimizer, val_loss_history, patience=10):
    if len(val_loss_history) > patience:
        recent_losses = val_loss_history[-patience:]
        if all(recent_losses[i] >= recent_losses[i-1] 
               for i in range(1, len(recent_losses))):
            # No improvement for 'patience' epochs
            optimizer.lr *= 0.5
            print(f"Reducing learning rate to {optimizer.lr}")
```

## 🧠 **Memory Aids & Intuition**

### **Optimizer Family Tree**
```
SGD (Grandparent)
├── SGD + Momentum (Better direction memory)
│   └── Nesterov (Look-ahead momentum)
├── AdaGrad (Adaptive per-parameter rates)
│   └── RMSprop (Forget old gradients)
│       └── Adam (RMSprop + Momentum)
│           └── AdamW (Better weight decay)
└── Variance Reduction (SVRG, SAGA)
```

### **Optimizer Personalities**
```
SGD        = "Simple, reliable worker"
           - Does the job, no frills
           - Needs good supervision (LR tuning)

Adam       = "Smart, adaptable employee"  
           - Learns what works for each task
           - Generally reliable với minimal supervision

SVRG       = "Meticulous perfectionist"
           - Takes time to get full picture
           - Delivers high-quality results

AdaGrad    = "Enthusiastic starter who burns out"
           - Great initial progress
           - Slows down over time
```

### **Learning Rate Intuition**
```
SGD:      🎯 "One size fits all arrows"
AdaGrad:  🏹 "Arrows adapt to target distance"  
RMSprop:  🎯🔄 "Adaptive arrows với memory reset"
Adam:     🏹🧠 "Smart arrows với direction memory"
```

## 🔍 **Troubleshooting Advanced Methods**

### **Common Issues**

| Issue | Method | Solution |
|-------|--------|----------|
| **Slow convergence** | Adam | Increase LR or decrease β₂ |
| **Unstable training** | AdaGrad | Add more regularization |
| **Poor generalization** | All adaptive | Try SGD với momentum |
| **High memory usage** | Adam | Switch to AdaGrad |

### **Convergence Diagnostics**
```python
def diagnose_optimizer(loss_history, grad_norms):
    # Check for common patterns
    if np.std(loss_history[-50:]) > np.mean(loss_history[-50:]) * 0.1:
        print("Warning: High loss variance - consider reducing LR")
    
    if np.mean(grad_norms[-10:]) < 1e-6:
        print("Warning: Very small gradients - possible convergence")
    
    if len(loss_history) > 100:
        recent_improvement = loss_history[-100] - loss_history[-1]
        if recent_improvement < 1e-4:
            print("Warning: Minimal recent improvement")
```

## 📈 **Future Directions**

### **Emerging Methods**
- **Lookahead Optimizer**: Maintains two sets of weights
- **RAdam**: Rectified Adam với warm-up
- **AdaBound**: Transitions from Adam to SGD
- **LARS/LAMB**: Large batch optimization

### **Meta-Optimization**
- **Learning to optimize**: Neural networks that learn optimization
- **AutoML for optimizers**: Automatic optimizer selection
- **Federated optimization**: Distributed learning scenarios

---

*"Advanced Methods: Standing on the shoulders of giants to reach optimization excellence"* 🚀🧠⚡