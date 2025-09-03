# Nhật ký học Stochastic Methods - Khi Randomness trở thành Sức mạnh

*Hành trình khám phá thế giới optimization với dữ liệu lớn - Nơi mà noise becomes feature và scalability is king*

---

## Khởi đầu: Từ Deterministic đến Stochastic

Sau khi master các deterministic methods (GD, Newton, Quasi-Newton), tôi bước vào thế giới hoàn toàn khác: **Stochastic optimization**.

Ở đây, thay vì sử dụng toàn bộ dataset mỗi iteration, ta chỉ dùng một subset (mini-batch). Điều này tạo ra "noise" nhưng lại mở ra khả năng xử lý datasets khổng lồ!

**Core transformation:**
- **Batch gradient:** ∇f(θ) = (1/n)Σᵢ₌₁ⁿ ∇fᵢ(θ)
- **Mini-batch gradient:** ∇̂f(θ) = (1/|B|)Σᵢ∈B ∇fᵢ(θ)

Trong đó B ⊂ {1,2,...,n} là random subset, |B| << n.

**Key insight:** ∇̂f(θ) là unbiased estimator của ∇f(θ), nhưng có variance!

---

## Ngày 1: Understanding Mini-batch Effects

### Thí nghiệm 33: **33_setup_sgd_batch_1000.py**
*"Standard mini-batch - Finding the sweet spot"*

**Batch size analysis:**
- Full batch: n ≈ 32,000 (all training data)
- Mini-batch: 1,000 (khoảng 3% of data)
- Reduction: 32x less computation per iteration

**Mathematical expectation:**
- E[∇̂f(θ)] = ∇f(θ) ✓ (unbiased)
- Var[∇̂f(θ)] = (1/|B|)σ² (variance inversely proportional to batch size)

**Results:**
- 67 epochs to convergence
- Loss curve: noisy but descending trend  
- Much faster per epoch than full-batch
- Total time: significantly reduced

**First revelation:** Noise doesn't prevent convergence, it just makes path "wiggly"!

**Variance-bias trade-off:**
- Smaller batch → Higher variance, faster per iteration
- Larger batch → Lower variance, slower per iteration  

---

### Thí nghiệm 34: **34_setup_sgd_batch_1600.py**
*"Larger batch - Reducing the noise"*

**Hypothesis:** Batch size 1,600 (5% data) sẽ có less noise, faster convergence.

**Results:**
- 52 epochs (vs 67 với batch 1,000)
- Smoother loss curve
- Slightly slower per epoch

**Variance reduction confirmation:**
- Var[∇̂f] = σ²/1600 vs σ²/1000
- 37.5% variance reduction achieved
- Cleaner convergence path

**Learning:** Larger batches trade computation cho stability.

---

### Thí nghiệm 35: **35_setup_sgd_batch_3200.py**
*"Large batch - Approaching deterministic behavior"*

**Batch size:** 3,200 (10% of data)

**Results:**
- 38 epochs to convergence
- Very smooth loss curve  
- Approaches full-batch behavior
- Higher per-epoch cost

**Mathematical insight:**
- Central Limit Theorem in action
- √batch_size improvement in gradient estimation
- Diminishing returns on variance reduction

---

### Thí nghiệm 36: **36_setup_sgd_batch_6400.py**
*"Very large batch - The extreme case"*

**Batch size:** 6,400 (20% of data)

**Results:**
- 28 epochs to convergence
- Almost deterministic convergence
- Significant per-epoch computational cost

**Trade-off realization:**
- Beyond certain batch size, improvements marginal
- Computational cost increases linearly
- Memory requirements can become prohibitive

**Optimal batch size insight:** Usually around 32-512 cho most problems, tùy thuộc vào dataset size và hardware constraints.

---

## Ngày 2: Learning Rate Scheduling - Taming the Noise

### Thí nghiệm 37: **37_setup_sgd_linear_decay_batch_1000_lr_01.py**
*"Linear decay - Classical approach to stochastic convergence"*

**Motivation:** Fixed learning rate trong stochastic setting có thể không converge to exact minimum vì noise.

**Schedule:** αₖ = α₀/(k+1) = 0.1/(epoch+1)

**Theory behind:**
- Stochastic convergence requires Σαₖ = ∞, Σαₖ² < ∞
- Linear decay satisfies both conditions
- Early epochs: large steps for fast progress
- Late epochs: small steps for precision

**Results:**
- Start: lr=0.1, End: lr=0.001
- 45 epochs to convergence (vs 67 fixed lr)
- Smooth final convergence despite early noise

**A-ha moment:** Learning rate scheduling transforms noisy SGD into precise optimizer!

---

### Thí nghiệm 38: **38_setup_sgd_sqrt_decay_batch_1000_lr_01.py**
*"Square root decay - Gentler reduction"*

**Schedule:** αₖ = α₀/√(k+1) = 0.1/√(epoch+1)

**Comparison với linear:**
- Slower decay rate
- Maintains larger learning rates longer
- Still satisfies convergence conditions

**Results:**
- 42 epochs (slightly faster than linear!)
- Good balance between progress và precision
- Less aggressive than linear decay

**Insight:** Sometimes slower decay cho phép more exploration before settling.

---

## Ngày 3: Advanced SGD Variants

### Thí nghiệm 39: **39_setup_sgd_momentum_batch_1000_lr_01_mom_09.py**
*"Momentum meets stochastic - Smoothing the noise"*

**Stochastic momentum:**
- vₖ = βvₖ₋₁ + ∇̂f(θₖ)  (stochastic gradient)
- θₖ₊₁ = θₖ - αvₖ

**Why momentum helps stochastic:**
- Momentum vector averages recent gradients
- Natural noise reduction mechanism  
- Maintains direction despite gradient noise

**Results:**
- 34 epochs - significant improvement!
- Much smoother convergence path
- Momentum acts as "low-pass filter" for noise

**Mathematical beauty:** 
- E[vₖ] points toward true gradient direction
- Var[vₖ] reduced due to averaging effect
- Best of both worlds: speed + stability

---

### Thí nghiệm 40: **40_setup_sgd_exponential_decay_batch_1000_lr_01_gamma_095.py**
*"Exponential decay - Aggressive schedule"*

**Schedule:** αₖ = α₀ × γᵏ với γ = 0.95

**Characteristics:**
- Fast initial decay
- Aggressive reduction trong early epochs
- Risk of premature learning rate reduction

**Results:**
- 39 epochs
- Very fast initial progress
- Potential underfitting risk nếu decay too aggressive

**Lesson:** Exponential decay requires careful tuning của γ parameter.

---

### Thí nghiệm 41: **41_setup_sgd_backtracking_batch_1000_c1_0001.py**
*"Adaptive learning rate - Line search inspiration"*

**Stochastic line search challenge:**
- Traditional line search không applicable (noisy gradients)
- Need adaptive mechanism for stochastic setting

**Adaptive strategy:**
- Increase lr if loss decreases consistently
- Decrease lr if loss increases  
- Armijo-inspired condition: c₁ = 1e-4

**Results:**
- Variable learning rates: 0.05 → 0.12 → 0.08 → ...
- 31 epochs - excellent performance!
- Algorithm adapts to problem characteristics

**Innovation:** Bringing line search benefits to stochastic optimization!

---

## Deep Mathematical Insights từ Stochastic Journey

### 1. Variance-Bias Trade-off trong Mini-batch
**Gradient estimation error:**
- Bias: E[∇̂f - ∇f] = 0 (unbiased estimator)
- Variance: Var[∇̂f] = σ²/|B|
- MSE = Bias² + Variance = 0 + σ²/|B|

**Optimal batch size balances:**
- Computational cost: O(|B|)
- Estimation quality: O(1/√|B|)

### 2. Learning Rate Requirements for Convergence
**Robbins-Monro conditions:**
- Σₖ αₖ = ∞ (infinite total learning)
- Σₖ αₖ² < ∞ (decreasing step sizes)

**Common schedules:**
- αₖ = a/(k+b): satisfies both conditions
- αₖ = a×γᵏ: only satisfies if γ = 1 (constant)
- αₖ = a/√k: satisfies both conditions

### 3. Momentum trong Stochastic Setting
**Exponential moving average:**
vₖ = βvₖ₋₁ + (1-β)∇̂fₖ

**Effective averaging window:** ≈ 1/(1-β)
- β = 0.9 → averages ~10 recent gradients  
- β = 0.99 → averages ~100 recent gradients

### 4. Generalization Benefits của SGD
**Implicit regularization:**
- Noise in gradients prevents overfitting
- Random sampling provides regularization effect
- Often generalizes better than full-batch methods

---

## Practical Implementation Wisdom

### Batch Size Selection Guidelines:
- **Small datasets (n < 10K):** batch = 32-128
- **Medium datasets (n < 100K):** batch = 128-512  
- **Large datasets (n > 1M):** batch = 512-2048
- **Rule of thumb:** Start with √n, adjust based on hardware

### Learning Rate Scheduling Best Practices:
1. **Start high:** Initial lr should enable fast progress
2. **Decay gradually:** Premature decay hurts convergence
3. **Monitor validation:** Use validation loss for schedule adaptation
4. **Common schedules:**
   - Step decay: Reduce by factor every N epochs
   - Cosine decay: Smooth decrease following cosine curve
   - Polynomial decay: Gradual polynomial reduction

### Memory và Computational Considerations:
- **Forward pass:** O(|B| × model_size)
- **Backward pass:** O(|B| × model_size)  
- **Memory:** O(|B| + model_params)
- **Wall-clock time:** Often dominated by data loading!

---

## Kết luận Stochastic Methods Journey

### Revolutionary Realizations:
1. **Noise can be beneficial** - Helps escape local minima, provides regularization
2. **Scalability breakthrough** - Makes optimization feasible for massive datasets  
3. **Batch size = fundamental trade-off** - Computation vs gradient quality
4. **Scheduling is crucial** - Adaptive learning rates essential for convergence
5. **Momentum still works** - Even more important in stochastic setting
6. **Hardware considerations matter** - Algorithm choice depends on computational resources

### Performance Ranking (epochs to convergence):
1. **SGD + Adaptive LR** (setup 41) - 31 epochs, smartest adaptation
2. **SGD + Momentum** (setup 39) - 34 epochs, best noise handling
3. **SGD + Sqrt Decay** (setup 38) - 42 epochs, balanced approach  
4. **SGD + Linear Decay** (setup 37) - 45 epochs, classical method
5. **SGD + Exponential Decay** (setup 40) - 39 epochs, aggressive but effective
6. **Large Batch (6400)** (setup 36) - 28 epochs, but expensive per epoch
7. **Standard Mini-batch** (setup 33) - 67 epochs, baseline

### Algorithm Selection Strategy:

**Use Small Batches (32-256) when:**
- Limited memory/computational resources
- Want implicit regularization benefits  
- Need frequent updates for online learning
- Working với very large datasets

**Use Large Batches (512-2048) when:**
- Abundant computational resources
- Need stable convergence  
- Can parallelize efficiently
- Working với smaller, well-curated datasets

**Use Momentum when:**
- Dealing với noisy gradients
- Want accelerated convergence
- Function has consistent curvature
- Can afford extra memory for velocity

**Use Learning Rate Scheduling when:**
- Need precise convergence
- Working với non-convex objectives
- Have validation set for monitoring
- Want to minimize final loss value

### Deep Philosophical Insights:

**Embracing Randomness:** Stochastic methods teach us that randomness, properly harnessed, becomes a powerful optimization tool rather than an obstacle.

**Scale Changes Everything:** What works for small problems may fail for large ones. Stochastic methods are fundamentally different philosophy, not just technical variation.

**Hardware-Algorithm Co-design:** Modern ML requires thinking about algorithms và hardware together. Batch size isn't just mathematical choice - it's systems engineering decision.

### Final Wisdom:
Stochastic optimization represents the marriage of mathematical theory với practical necessity. In the age of big data, deterministic methods are luxury we can't afford. SGD and its variants have become the workhorses of modern machine learning, proving that sometimes "good enough" gradient estimates are better than perfect but computationally prohibitive ones.

**The Journey Continues:** From here, the path leads to even more sophisticated methods - Adam, AdaGrad, natural gradients, distributed optimization. But the foundation built through this journey - understanding the trade-offs between accuracy, speed, memory, và generalization - will guide all future explorations.

**Complete Journey Reflection:** From simple gradient descent to sophisticated stochastic methods, we've covered the full spectrum of optimization algorithms. Each family teaches unique lessons, and the best practitioners know when to use which tool. The art of optimization lies not in knowing one method perfectly, but in understanding the strengths và weaknesses of each approach and choosing wisely for each problem.