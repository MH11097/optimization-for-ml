# Nhật ký học Quasi-Newton - The Art of Hessian Approximation

*Hành trình khám phá những thuật toán "thông minh" - Lấy được second-order benefits mà không cần tính toán đắt đỏ*

---

## Khởi đầu: Từ Newton đến Quasi-Newton

Sau khi trải nghiệm sức mạnh của Newton methods, tôi đối mặt với một câu hỏi lớn: Liệu có cách nào để có được quadratic convergence mà không cần tính Hessian đắt đỏ?

Đây chính là lý do Quasi-Newton methods ra đời - một trong những breakthrough vĩ đại nhất của optimization!

**Core idea:** Thay vì tính H = ∇²f(x), ta xây dựng approximation Bₖ ≈ H qua các gradient observations.

**Secant equation - Linh hồn của Quasi-Newton:**
**Bₖ₊₁sₖ = yₖ**

Trong đó:
- sₖ = xₖ₊₁ - xₖ (step vector)
- yₖ = ∇f(xₖ₊₁) - ∇f(xₖ) (gradient change)
- Bₖ₊₁: Hessian approximation cho iteration tiếp theo

**Intuition:** Nếu function locally quadratic, thì Bsₖ = yₖ phải hold true!

---

## Ngày 1: BFGS - The Golden Standard

### Thí nghiệm 25: **25_setup_bfgs_ols.py**
*"Broyden-Fletcher-Goldfarb-Shanno - Khi 4 great minds cùng sáng tạo"*

**BFGS update formula - Mathematical masterpiece:**

Bₖ₊₁ = Bₖ - (BₖsₖsₖᵀBₖ)/(sₖᵀBₖsₖ) + (yₖyₖᵀ)/(yₖᵀsₖ)

**Breakdown:**
- Term 1: Bₖ (current approximation)
- Term 2: -(BₖsₖsₖᵀBₖ)/(sₖᵀBₖsₖ) (remove bad curvature)  
- Term 3: +(yₖyₖᵀ)/(yₖᵀsₖ) (add observed curvature)

**Lý thuyết đằng sau:**
- Satisfies secant equation: Bₖ₊₁sₖ = yₖ ✓
- Maintains positive definiteness (nếu yₖᵀsₖ > 0) ✓  
- Closest to Bₖ in Frobenius norm ✓
- Superlinear convergence rate ✓

**Kết quả:**
- 12 iterations (vs 3 for Newton, 78 for momentum GD)
- B matrix evolves: I → ... → approximates H⁻¹
- Step sizes adaptive: 0.05 → 0.12 → 0.08 → ...
- Superlinear convergence clearly visible!

**A-ha moment:** BFGS is learning the problem structure through gradient observations!

**Practical advantages over Newton:**
- No Hessian computation required
- No matrix inversion per iteration  
- O(n²) storage vs O(n²) computation per step
- Robust to function approximation errors

---

### Thí nghiệm 26: **26_setup_bfgs_ridge.py**
*"BFGS meets Ridge regularization"*

**Question:** How does regularization affect Quasi-Newton approximation?

**Mathematical impact:**
- Original: ∇f(θ) = 2X^T(Xθ - y)
- Ridge: ∇f(θ) = 2X^T(Xθ - y) + 2λθ
- Extra λθ term affects gradient differences yₖ

**Results:**
- 8 iterations (faster than pure BFGS!)
- Better-conditioned approximation Bₖ
- More stable convergence path

**Why faster:**
- Ridge regularization makes function more "quadratic-like"
- Better function → better Hessian approximation
- Better approximation → faster convergence

**Deep insight:** Regularization helps not just the original function, but also the approximation quality!

---

## Ngày 2: L-BFGS - Memory-Efficient Revolution

### Thí nghiệm 27: **27_setup_lr1_ols.py**
*"Limited memory BFGS - Solving the storage problem"*

**Memory challenge with full BFGS:**
- Stores full n×n matrix Bₖ
- For large n: prohibitive memory usage
- n=100,000 → need 40GB just for one matrix!

**L-BFGS breakthrough idea:**
- Don't store Bₖ explicitly
- Store only m recent (sₖ, yₖ) pairs
- Reconstruct Hₖ⁻¹∇f implicitly when needed

**Two-loop recursion - Computational magic:**
```
q = ∇fₖ
for i = k-1, k-2, ..., k-m:
    αᵢ = (sᵢᵀq)/(yᵢᵀsᵢ)  
    q = q - αᵢyᵢ
r = H₀q  # Initial Hessian approximation
for i = k-m, k-m+1, ..., k-1:
    β = (yᵢᵀr)/(yᵢᵀsᵢ)
    r = r + sᵢ(αᵢ - β)
return r  # This is Hₖ⁻¹∇fₖ
```

**Results với m=5:**
- 15 iterations (bit slower than full BFGS)
- Memory: O(mn) instead of O(n²)
- Still superlinear convergence!

**Trade-off analysis:**
- Full BFGS: Better approximation, more memory
- L-BFGS: Good approximation, scalable memory

**Breakthrough realization:** L-BFGS makes second-order methods practical for large-scale problems!

---

### Thí nghiệm 28: **28_setup_lbfgs_ols_m_10.py**
*"More memory = Better approximation?"*

**Hypothesis:** Increasing memory from 5 to 10 will improve convergence.

**Results:**
- 13 iterations (vs 15 với m=5)
- Better Hessian approximation quality
- Slightly more computation per iteration

**Memory vs accuracy trade-off:**
- m=5: Fast but rougher approximation
- m=10: Better approximation, still scalable
- m=20: Diminishing returns usually

**Practical guideline:** m=10-20 usually optimal for most problems.

---

### Thí nghiệm 29: **29_setup_lbfgs_ridge_m_5_reg_001.py**
*"L-BFGS + Ridge combo with memory efficiency"*

**Combination benefits:**
- Ridge: Better-conditioned problem
- L-BFGS: Memory-efficient second-order approximation
- Small memory (m=5): Maximum efficiency

**Results:**
- 11 iterations
- Excellent memory efficiency
- Good convergence rate
- Perfect for medium-large scale problems

**Sweet spot discovery:** This combination scales well while maintaining effectiveness.

---

## Ngày 3: Enhanced Line Search Strategies

### Thí nghiệm 30: **30_setup_bfgs_backtracking_ols_c1_0001.py**
*"BFGS với sophisticated line search"*

**Enhanced backtracking parameters:**
- Armijo c₁ = 1e-4 (sufficient decrease)
- Wolfe c₂ = 0.1 (curvature condition - more restrictive!)
- Backtrack ρ = 0.5 (aggressive reduction)
- Max iterations = 100 (thorough search)

**Why more restrictive curvature condition:**
- Standard: c₂ = 0.9 (loose curvature requirement)
- Here: c₂ = 0.1 (tight curvature requirement)
- Ensures better gradient information for BFGS update

**Results:**
- 10 iterations (faster than standard BFGS!)
- High-quality step sizes
- Better BFGS approximation due to good line search

**Mathematical insight:** High-quality line search → high-quality (sₖ, yₖ) pairs → better Hessian approximation!

**Line search quality affects:**
1. Immediate progress (obvious)
2. Future approximation quality (subtle but crucial!)

---

## Deep Dive: Mathematical Beauty of Quasi-Newton

### 1. Why BFGS Update Formula Works

**The optimization problem BFGS solves:**
```
min ||B - Bₖ||_F subject to:
- Bsₖ = yₖ (secant condition)
- B = Bᵀ (symmetry)
- B positive definite
```

**Solution via Lagrange multipliers leads to BFGS formula!**

### 2. Superlinear Convergence Theory

**Convergence rate:** ||xₖ₊₁ - x*|| ≤ εₖ||xₖ - x*||

Trong đó εₖ → 0, nghĩa là:
- Faster than linear: ρᵏ
- Slower than quadratic: Cεₖ²
- "Best of both worlds"

### 3. Curvature Information Learning

**BFGS learns function curvature through:**
- sₖ vectors: directions explored
- yₖ vectors: gradient changes observed  
- Ratio yₖᵀsₖ: directional curvature measure

**Intuitive analogy:** Like a blind person learning room shape by walking and feeling walls!

---

## Practical Implementation Insights

### Storage Requirements:
- **Full BFGS:** O(n²) for matrix Bₖ
- **L-BFGS:** O(mn) for m recent pairs
- **Gradient methods:** O(n) for gradient only

### Computational Cost per Iteration:
- **Newton:** O(n³) - Hessian inversion
- **Full BFGS:** O(n²) - Matrix-vector products
- **L-BFGS:** O(mn) - Two-loop recursion
- **Gradient:** O(n) - Gradient computation

### When Quasi-Newton Fails:
1. **Non-smooth functions:** Gradient differences unreliable
2. **Very ill-conditioned problems:** Poor approximation quality
3. **Noisy gradients:** Corrupted (sₖ, yₖ) information
4. **Non-convex with many local minima:** Approximation breaks down

---

## Kết luận Quasi-Newton Journey

### Revolutionary Discoveries:
1. **Hessian approximation is feasible** - Don't need exact second-order info
2. **Learning through observation** - Gradient changes reveal curvature
3. **Memory management breakthrough** - L-BFGS solves scalability  
4. **Line search quality matters** - Affects both progress và approximation
5. **Superlinear convergence achievable** - Without full Newton cost

### Performance Ranking:
1. **BFGS + Enhanced Line Search** (setup 30) - 10 iterations, best quality
2. **BFGS + Ridge** (setup 26) - 8 iterations, most stable  
3. **L-BFGS + Ridge** (setup 29) - 11 iterations, best scalability
4. **L-BFGS m=10** (setup 28) - 13 iterations, good balance
5. **Standard BFGS** (setup 25) - 12 iterations, baseline
6. **L-BFGS m=5** (setup 27) - 15 iterations, maximum efficiency

### Algorithm Selection Guide:

**Use Full BFGS when:**
- n < 1,000 (memory not issue)
- High accuracy required
- Computational budget allows O(n²) per iteration

**Use L-BFGS when:**
- n > 10,000 (memory critical)
- Good accuracy sufficient
- Scalability priority

**Avoid Quasi-Newton when:**
- n > 1,000,000 (even L-BFGS expensive)
- Very noisy gradients
- Non-smooth objectives
- Simple convex problems (GD might suffice)

### Key Mathematical Lessons:
- **Secant equation = Core principle** - Bsₖ = yₖ drives everything
- **Positive definiteness = Convergence guarantee** - Must maintain yₖᵀsₖ > 0
- **Approximation quality = Success predictor** - Good (sₖ, yₖ) → good Bₖ
- **Memory vs accuracy trade-off** - More history → better approximation

### Philosophical Insight:
Quasi-Newton represents one of optimization's greatest intellectual achievements: **turning expensive exact computation into cheap intelligent approximation**. It's the bridge between first-order simplicity và second-order power.

**Final wisdom:** In the hierarchy of optimization algorithms, Quasi-Newton occupies the sweet spot - sophisticated enough to handle real problems, efficient enough to scale, robust enough to trust.

**Next adventure:** Stochastic methods - Where randomness becomes a feature, not a bug, and we tackle datasets too large for batch methods!