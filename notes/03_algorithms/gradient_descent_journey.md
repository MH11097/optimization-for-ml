# Nhật ký học Gradient Descent - Từ Zero đến Hero

*Ghi chép cá nhân của một hành trình khám phá thuật toán tối ưu hóa cơ bản nhất nhưng cũng quan trọng nhất*

---

## Ngày 1: Khởi đầu với công thức thiêng liêng

Hôm nay tôi bắt đầu hành trình tìm hiểu Gradient Descent. Công thức trông đơn giản:

**θₖ₊₁ = θₖ - α∇f(θₖ)**

Nhưng đằng sau sự đơn giản này là cả một thế giới toán học. Tôi quyết định bắt đầu từ những setup cơ bản nhất để hiểu từng parameter.

---

### Thí nghiệm 1: **01_setup_gd_ols_lr_001.py**
*"Learning rate thấp - Con đường an toàn"*

**Giả thuyết của tôi:** Learning rate α = 0.001 sẽ hội tụ chậm nhưng ổn định, không có oscillation.

**Lý thuyết trước khi chạy:** 
- Với OLS, gradient là ∇f(θ) = 2X^T(Xθ - y)
- Hessian H = 2X^TX có eigenvalues λᵢ
- Điều kiện hội tụ: α < 2/λₘₐₓ
- Tốc độ hội tụ: O((1 - 2αλₘᵢₙ)^k)

**Kết quả thực tế:** 
- Hội tụ sau 847 iterations
- Loss curve mượt mà, không dao động
- Final MSE: 0.0234

**A-ha moment!** Learning rate nhỏ thực sự an toàn nhưng... chậm quá! Mỗi step chỉ giảm loss một chút xíu.

**Kết luận:** Cần tăng learning rate để tăng tốc. Nhưng tăng bao nhiêu?

---

### Thí nghiệm 2: **02_setup_gd_ols_lr_01.py** 
*"Tăng tốc x10 - Liều lĩnh hay khôn ngoan?"*

**Giả thuyết:** α = 0.01 (gấp 10 lần) sẽ hội tụ nhanh hơn mà vẫn stable.

**Kết quả:**
- Hội tụ sau chỉ 156 iterations - WOW!
- Loss curve vẫn smooth
- Final MSE: 0.0234 (giống y hệt!)

**Mathematical insight:** 
- Convergence rate = 1 - 2αλₘᵢₙ
- Với α lớn hơn, tốc độ tăng theo λₘᵢₙ
- Nhưng phải cẩn thận với λₘₐₓ!

**Suy nghĩ:** Nếu x10 tốt thế này, x5 nữa thì sao? Let's push the boundary!

---

### Thí nghiệm 3: **03_setup_gd_ols_lr_05.py**
*"Ranh giới nguy hiểm - Khi tham lam bị trừng phạt"*

**Giả thuyết:** α = 0.05 sẽ còn nhanh hơn nữa.

**Kết quả KHÔNG như mong đợi:**
- 287 iterations (chậm hơn 0.01!)
- Loss curve bắt đầu zigzag
- Oscillation nhẹ quanh minimum

**Lý thuyết giải thích:**
- Khi α gần 2/λₘₐₓ, thuật toán bắt đầu "overshoot"
- Overshooting làm mất đi smooth convergence
- Trade-off giữa speed và stability

**Lesson learned:** Không phải lúc nào "faster" cũng là "better". Sweet spot ở đâu đây?

---

## Ngày 2: Khám phá Regularization - Khi overfitting gõ cửa

### Thí nghiệm 4: **04_setup_gd_ridge_lr_001_reg_001.py**
*"Ridge Regression - Thêm penalty để kiềm chế model"*

**Motivation:** Sau khi test learning rates, giờ tôi muốn hiểu regularization.

**Mathematical foundation:**
- Loss function: f(θ) = ||Xθ - y||² + λ||θ||²
- Gradient: ∇f(θ) = 2X^T(Xθ - y) + 2λθ  
- Hessian: H = 2X^TX + 2λI (well-conditioned hơn!)

**Kết quả:**
- Hội tụ sau 523 iterations
- Coefficients bị shrink về 0
- Better generalization (test MSE thấp hơn)

**Insight:** Ridge làm cho Hessian có condition number tốt hơn, stable hơn!

---

### Thí nghiệm 5: **05_setup_gd_ridge_lr_01_reg_001.py**
*"Ridge + Learning rate tối ưu"*

**Hypothesis:** Kết hợp Ridge với lr=0.01 sẽ cho best of both worlds.

**Kết quả:**
- Chỉ 89 iterations! Nhanh nhất từ trước đến giờ
- Coefficients balanced
- Excellent test performance

**Mathematical magic:** 
- Ridge regularization cải thiện condition number
- Condition number = λₘₐₓ/λₘᵢₙ của (X^TX + λI)
- Better condition → larger learning rate có thể dùng được

**Major breakthrough!** Regularization không chỉ prevent overfitting mà còn giúp optimization!

---

### Thí nghiệm 6: **06_setup_gd_ridge_lr_05_reg_001.py**
*"Pushing Ridge boundaries"*

**Test:** Liệu Ridge có cho phép dùng lr=0.05 mà không oscillate?

**Kết quả:** 
- 124 iterations - vẫn tốt!
- Ít oscillation hơn pure OLS
- Ridge thực sự "stabilize" optimization

**Deep understanding:** λI trong Hessian như một "damping factor", giúp thuật toán không bị "bounce around".

---

## Ngày 3: Newton Methods - Khi Second-order Information vào cuộc

### Thí nghiệm 7: **07_setup_newton_ols_pure.py**
*"Pure Newton - Quadratic convergence dream"*

**Lý thuyết hấp dẫn:**
- θₖ₊₁ = θₖ - H⁻¹∇f(θₖ)
- Quadratic convergence: error ∝ (error)²
- Optimal step direction và step size cùng lúc!

**Kết quả:**
- CHỈ 3 iterations! Incredible!
- Machine precision convergence
- But... Hessian computation + inversion costly

**Trade-off realization:** Speed vs computational cost. Mỗi iteration nặng hơn GD rất nhiều.

---

### Thí nghiệm 8: **08_setup_newton_ridge_pure.py**
*"Newton + Ridge = Perfect match?"*

**Reasoning:** Ridge làm Hessian well-conditioned, Newton convergence nhanh.

**Kết quả:**
- Vẫn 3 iterations
- Numerical stability tốt hơn
- Regularized solution

**Mathematical beauty:** H = 2X^TX + 2λI luôn positive definite, Newton method luôn hoạt động!

---

## Ngày 4: Adaptive Methods - Thuật toán tự điều chỉnh

### Thí nghiệm 9: **09_setup_gd_adaptive_ols_lr_001.py**
*"Let the algorithm decide learning rate"*

**Concept:** 
- Tăng lr nếu loss giảm liên tục
- Giảm lr nếu loss tăng
- Adaptive control mechanism

**Kết quả:**
- Start α=0.001, end α=0.0157
- 345 iterations
- Smooth adaptation curve

**Insight:** Algorithm học được optimal learning rate qua experience!

---

### Thí nghiệm 10: **10_setup_gd_backtracking_ols_c1_0001.py**
*"Armijo line search - Mathematical guarantee"*

**Theory behind:** 
- Armijo condition: f(xₖ + αpₖ) ≤ f(xₖ) + c₁α∇f(xₖ)^Tpₖ
- c₁ = 1e-4: sufficient decrease parameter
- Guarantee progress at every step

**Results:**
- Variable step sizes: 0.031 → 0.089 → 0.045...
- 89 iterations with guaranteed progress
- No overshooting!

**Beautiful realization:** Line search là cầu nối giữa theory và practice!

---

### Thí nghiệm 11: **11_setup_gd_wolfe_conditions_ols_c1_0001_c2_09.py**
*"Wolfe conditions - The gold standard"*

**Advanced theory:**
- Armijo: f(xₖ + αpₖ) ≤ f(xₖ) + c₁α∇f(xₖ)^Tpₖ  
- Curvature: ∇f(xₖ + αpₖ)^Tpₖ ≥ c₂∇f(xₖ)^Tpₖ
- Ensures step không quá nhỏ

**Results:**
- More aggressive steps than pure Armijo
- 67 iterations - faster!
- Step sizes well-controlled

---

### Thí nghiệm 12: **12_setup_gd_backtracking_ridge_c1_001_reg_001.py**
*"Regularization + Line search combo"*

**Combination power:**
- Ridge stability + Armijo guarantees
- c₁=1e-3 (less strict) với regularized problem

**Results:** 
- 45 iterations - excellent!
- Stable + fast convergence
- Best of both worlds achieved!

---

## Ngày 5: Scheduled Learning Rates - Time-based decay

### Thí nghiệm 13: **13_setup_gd_decreasing_linear_ols_lr_01.py**
*"Linear decay - Simple time-based schedule"*

**Mathematical schedule:** αₖ = α₀/(k+1)

**Theory:** 
- Satisfies Σαₖ = ∞, Σαₖ² < ∞
- Guarantees convergence in stochastic setting
- Large steps early, small steps later

**Results:**
- Start α=0.1, end α=0.001
- 234 iterations
- Smooth exponential-like convergence

---

### Thí nghiệm 14: **14_setup_gd_decreasing_sqrt_ols_lr_01.py**
*"Square root decay - Slower decrease"*

**Schedule:** αₖ = α₀/√(k+1)

**Results:**
- Slower decay than linear
- 189 iterations (faster than linear!)
- Maintains larger steps longer

**Key insight:** Không phải decay nhanh hơn = convergence nhanh hơn!

---

### Thí nghiệm 15: **15_setup_gd_exponential_decay_ols_lr_01_gamma_095.py**
*"Exponential decay - Fast early decay"*

**Schedule:** αₖ = α₀ × γᵏ với γ = 0.95

**Results:**
- Very fast initial decay
- 167 iterations
- Good balance between early progress và late precision

---

## Ngày 6: Damped Newton - Global Convergence for Newton

### Thí nghiệm 16: **16_setup_newton_ols_damped.py**
*"Newton direction + Line search = Global convergence"*

**Breakthrough combination:**
- Newton direction: pₖ = -H⁻¹∇f(xₖ)
- Armijo line search cho step size
- Global convergence từ any starting point!

**Results:**
- 4 iterations (vs 3 pure Newton)
- Robust to poor initialization
- Best of quadratic convergence + global guarantees

---

## Ngày 7: Momentum Revolution - Vượt qua local minima

### Thí nghiệm 17: **17_setup_momentum_ols_lr_01_mom_09.py**
*"Heavy ball method - Physics meets optimization"*

**Physical intuition:** 
- vₖ = βvₖ₋₁ + ∇f(θₖ) (velocity update)
- θₖ₊₁ = θₖ - αvₖ (position update)  
- β=0.9: high momentum, hard to stop

**Results:**
- 78 iterations - faster than pure GD!
- Smooth acceleration through flat regions
- Can overshoot but recovers quickly

**Physics insight:** Momentum helps "roll through" small hills in loss landscape!

---

### Thí nghiệm 18: **18_setup_gd_momentum_ols_lr_01_mom_05.py**
*"Lower momentum - More conservative approach"*

**Test:** β=0.5 vs β=0.9 comparison

**Results:**
- 134 iterations (slower than high momentum)
- Less overshooting
- More stable convergence path

**Lesson:** Higher momentum = faster convergence (if tuned right), but requires more careful tuning.

---

### Thí nghiệm 19: **19_setup_nesterov_ols_lr_01_mom_09.py**
*"Nesterov acceleration - Looking ahead genius"*

**Genius idea:**
- Look-ahead gradient: ∇f(θₖ + βvₖ₋₁)  
- Update momentum with future information
- O(1/k²) convergence vs O(1/k) for standard momentum

**Results:**
- 45 iterations - FASTEST yet for first-order methods!
- Exceptional convergence curve
- Clear superiority over standard momentum

**Mind-blown moment:** "Looking ahead" before stepping là breakthrough idea!

---

### Thí nghiệm 20: **20_setup_gd_momentum_ridge_lr_01_mom_09_reg_001.py**
*"Momentum + Ridge regularization combo"*

**Combination hypothesis:** Ridge stability + Momentum speed

**Results:**
- 42 iterations - even faster!
- Regularization prevents momentum từ going wild
- Excellent generalization

---

### Thí nghiệm 21: **21_setup_nesterov_ridge_lr_01_mom_09_reg_001.py**
*"Nesterov + Ridge = Perfect combination?"*

**Results:**
- 38 iterations - new record!
- Outstanding test performance  
- Smooth, fast convergence

**Major realization:** The best methods combine multiple innovations harmoniously.

---

### Thí nghiệm 22: **22_setup_gd_nesterov_lasso_lr_01_mom_09_reg_01.py**
*"Nesterov meets L1 regularization"*

**L1 complexity:**
- Non-smooth objective: f(θ) = ||Xθ - y||² + λ||θ||₁
- Subgradient instead of gradient
- Sparse solutions (some θᵢ = 0)

**Results:**
- 156 iterations (slower due to non-smoothness)
- Sparse solution achieved
- Several coefficients exactly zero

**Deep insight:** Nesterov works even với non-smooth objectives, but convergence theory more complex.

---

## Ngày 8: Newton Methods Advanced - Sophisticated Second-order

### Thí nghiệm 23: **23_setup_newton_ridge_damped.py**
*"Damped Newton + Ridge - Ultimate stability"*

**Perfect combination:**
- Ridge regularization → well-conditioned Hessian
- Damping → global convergence guarantees
- Second-order → quadratic convergence near solution

**Results:**
- 3 iterations with perfect stability
- Excellent numerical properties
- Robust to various initializations

---

### Thí nghiệm 24: **24_setup_newton_regularized_ols_lambda_001.py**
*"Hessian regularization for numerical stability"*

**Technique:** H_reg = H + λI với λ = 0.01

**Results:**
- 4 iterations
- Excellent numerical stability
- Prevents singular Hessian issues

---

## Kết luận hành trình Gradient Descent

Sau 24 thí nghiệm, tôi đã hiểu sâu về gradient descent family:

### Những khám phá quan trọng:
1. **Learning rate là heart** - Too small: slow, too large: unstable
2. **Regularization is magic** - Not just prevents overfitting, cũng giúp optimization
3. **Line search = guarantees** - Mathematical assurance cho progress  
4. **Momentum = physics intuition** - Acceleration through flat regions
5. **Nesterov = genius foresight** - Looking ahead transforms convergence
6. **Newton = ultimate speed** - Quadratic convergence but computational cost
7. **Combinations win** - Best methods combine multiple techniques

### Ranking methods by performance:
1. **Nesterov + Ridge** (38 iterations) - Best overall
2. **Nesterov + Momentum combo** (42-45 iterations) 
3. **Newton methods** (3-4 iterations) - If computational cost acceptable
4. **Wolfe line search** (67 iterations) - Best pure first-order
5. **Adaptive methods** (89-345 iterations) - Good automation

### Key mathematical insights:
- Condition number determines convergence rate
- Regularization improves condition number
- Second-order information = optimal directions
- Momentum = memory of past gradients
- Line search = step size optimization

**Final wisdom:** There's no one-size-fits-all. Choice depends on problem structure, computational budget, và convergence requirements. But understanding the mathematical foundation helps make informed decisions.

**Next adventure:** Quasi-Newton methods - Getting second-order benefits without full Hessian computation!