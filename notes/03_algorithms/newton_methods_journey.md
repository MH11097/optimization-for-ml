# Nhật ký học Newton Methods - Khi Second-order Information thống trị

*Hành trình khám phá những thuật toán mạnh mẽ nhất trong tối ưu hóa - từ Pure Newton đến Advanced Regularization*

---

## Khởi đầu: Từ First-order đến Second-order

Sau khi đã master được Gradient Descent family, tôi bước vào thế giới của second-order optimization. Đây là nơi mà thông tin về curvature (độ cong) của loss function được sử dụng để tạo ra những bước nhảy "thông minh" hơn.

**Công thức thiêng liêng:**
**θₖ₊₁ = θₖ - H⁻¹∇f(θₖ)**

Trong đó H là Hessian matrix - ma trận đạo hàm bậc 2. Nhìn đơn giản nhưng đằng sau là cả một thế giới toán học phức tạp!

---

## Ngày 1: Pure Newton - Sức mạnh của Second-order

### Thí nghiệm 7: **07_setup_newton_ols_pure.py** 
*"Quadratic convergence dream - Liệu có thật như lý thuyết?"*

**Lý thuyết khiến tôi phấn khích:**
- Quadratic convergence: |εₖ₊₁| ≤ C|εₖ|²
- Nghĩa là error giảm theo bình phương của error hiện tại!
- Nếu error = 0.1, next step error ≤ 0.01
- Nếu error = 0.01, next step error ≤ 0.0001

**Mathematical foundation cho OLS:**
- f(θ) = ||Xθ - y||²
- ∇f(θ) = 2X^T(Xθ - y) 
- H = ∇²f(θ) = 2X^TX
- Newton step: θₖ₊₁ = θₖ - (X^TX)⁻¹X^T(Xθₖ - y)

**Kết quả khiến tôi choáng váng:**
- CHỈ 3 iterations!
- Loss: 156.7 → 0.067 → 2.3e-6 → machine precision
- Mỗi iteration giảm error hàng triệu lần!

**A-ha moment:** Đây chính là "Godlike convergence" mà textbook nói đến!

**Nhưng có một "but":**
- Mỗi iteration phải compute và invert Hessian
- Complexity: O(n³) cho matrix inversion
- Memory: O(n²) để store Hessian

**First lesson:** Newton = Speed demon but computational beast!

---

### Thí nghiệm 8: **08_setup_newton_ridge_pure.py**
*"Ridge regularization - Khi Hessian cần được tăng cường"*

**Motivation:** Pure Newton có thể fail nếu Hessian singular. Ridge có thể giải quyết?

**Mathematical enhancement:**
- f(θ) = ||Xθ - y||² + λ||θ||²
- ∇f(θ) = 2X^T(Xθ - y) + 2λθ
- H = 2X^TX + 2λI ← Key difference!

**Why this matters:**
- X^TX có thể singular (rank deficient)  
- Adding λI makes H = X^TX + λI always positive definite
- Guaranteed invertible Hessian!

**Kết quả:**
- Vẫn 3 iterations (tương tự pure Newton)
- Nhưng numerically stable hơn
- Coefficients được regularized (shrunk toward zero)
- Better generalization on test set

**Deep insight:** Ridge không chỉ prevent overfitting, mà còn "repair" singular Hessian matrix!

**Mathematical beauty:** 
- Condition number của H = (λₘₐₓ + λ)/(λₘᵢₙ + λ)
- Khi λ > 0, condition number luôn finite và reasonable
- Better-conditioned matrix → stable inversion

---

## Ngày 2: Damped Newton - Global Convergence Revolution

### Thí nghiệm 16: **16_setup_newton_ols_damped.py**
*"Khi Pure Newton gặp Line Search - Marriage made in heaven"*

**Problem với Pure Newton:** 
- Chỉ guarantees local convergence
- Nếu start point xa optimal, có thể diverge
- Full Newton step có thể quá aggressive

**Damped Newton solution:**
- θₖ₊₁ = θₖ - αₖH⁻¹∇f(θₖ)
- αₖ được chọn via Armijo line search
- Combines Newton direction với adaptive step size

**Lý thuyết line search:**
- Armijo condition: f(xₖ + αpₖ) ≤ f(xₖ) + c₁α∇f^Tpₖ
- c₁ = 1e-4: sufficient decrease parameter
- Backtracking: α = α × ρ until condition satisfied

**Kết quả:**
- 4 iterations (vs 3 pure Newton)
- BUT: Global convergence guarantee!
- Step sizes: 1.0 → 1.0 → 1.0 → 1.0 (full steps accepted!)
- Loss curve: smooth monotonic decrease

**Surprising discovery:** For well-conditioned OLS, full Newton steps usually satisfy Armijo condition!

**Why this works:**
- Newton direction pₖ = -H⁻¹∇f là descent direction  
- For quadratic functions, Newton step is optimal
- Line search mainly serves as "safety net"

**Key lesson:** Damping adds robustness với minimal cost for well-behaved problems.

---

### Thí nghiệm 23: **23_setup_newton_ridge_damped.py**
*"The holy trinity: Newton + Ridge + Damping"*

**Triple combination power:**
1. Ridge regularization → Well-conditioned Hessian
2. Newton direction → Quadratic convergence  
3. Line search → Global convergence guarantee

**Mathematical perfection:**
- H = 2X^TX + 2λI (always positive definite)
- pₖ = -H⁻¹∇f (optimal direction)
- αₖ via Armijo (guaranteed progress)

**Results:**
- 3 iterations (back to pure Newton speed!)
- Perfect numerical stability
- Excellent generalization
- Robust to various initializations

**Why so effective:**
- Ridge fixes any Hessian conditioning issues
- Well-conditioned Hessian → full steps accepted
- Line search becomes formality for quadratic problems

**Philosophical insight:** The best algorithms combine multiple good ideas synergistically.

---

## Ngày 3: Advanced Regularization Techniques

### Thí nghiệm 24: **24_setup_newton_regularized_ols_lambda_001.py**
*"Hessian regularization - Direct mathematical intervention"*

**Different từ Ridge regularization:**
- Ridge: Adds λ||θ||² to objective function
- Hessian reg: Adds λI directly to Hessian matrix
- H_regularized = H + λI

**Why this distinction matters:**
- Ridge affects both gradient và Hessian
- Hessian reg chỉ affects curvature information
- Pure numerical technique vs mathematical model change

**Results với λ = 0.01:**
- 4 iterations
- Excellent numerical stability  
- No shrinkage of coefficients (unlike Ridge)
- Pure computational regularization

**Mathematical understanding:**
- Eigenvalues của H become λᵢ + λ
- Condition number = (λₘₐₓ + λ)/(λₘᵢₙ + λ)
- Smaller condition number → better inversion

**When to use:**
- Hessian reg: Purely numerical issues
- Ridge reg: Want mathematical regularization + numerical stability

---

### Thí nghiệm 31: **31_setup_newton_backtracking_ols_c1_0001.py**
*"Fine-tuned line search parameters"*

**Experimenting với line search parameters:**
- Armijo c₁ = 1e-4 (standard choice)
- Backtrack ρ = 0.8 (moderate reduction)
- More careful step size selection

**Results:**
- 4 iterations
- Step sizes carefully controlled
- Robust convergence path

**Line search insights:**
- c₁ too small → accepts too many steps (less progress guarantee)
- c₁ too large → rejects good steps (slower convergence)
- c₁ = 1e-4 is well-tested sweet spot

---

### Thí nghiệm 32: **32_setup_newton_regularized_ridge_lambda_01_reg_001.py**
*"Dual regularization - Maximum stability approach"*

**Double regularization strategy:**
1. Ridge penalty: λ₁||θ||² in objective (λ₁ = 0.01)
2. Hessian regularization: λ₂I in Hessian (λ₂ = 0.1)

**Mathematical formulation:**
- f(θ) = ||Xθ - y||² + λ₁||θ||²
- H_effective = 2X^TX + 2λ₁I + λ₂I = 2X^TX + (2λ₁ + λ₂)I

**Why use both:**
- Ridge: Mathematical model regularization
- Hessian: Numerical computation regularization  
- Different purposes, complementary effects

**Results:**
- 4 iterations
- Maximum numerical stability
- Well-regularized solution
- Excellent condition number

**Trade-off analysis:**
- More stable than single regularization
- Slightly more hyperparameters to tune
- Excellent for ill-conditioned problems

---

## Deep Mathematical Insights từ Newton Journey

### 1. Convergence Rate Analysis
**Linear vs Quadratic convergence:**
- GD: |εₖ₊₁| ≤ ρ|εₖ| với ρ < 1
- Newton: |εₖ₊₁| ≤ C|εₖ|²

**Practical implications:**
- Newton: 10⁻² → 10⁻⁴ → 10⁻⁸ → 10⁻¹⁶
- GD: 10⁻² → 10⁻²·⁵ → 10⁻³ → 10⁻³·⁵

### 2. Computational Complexity
**Per iteration cost:**
- GD: O(n) (gradient computation)
- Newton: O(n³) (Hessian inversion)

**Total complexity:**
- GD: O(kn) với k iterations (k có thể rất lớn)
- Newton: O(jn³) với j iterations (j rất nhỏ)

**Break-even point:** Newton wins when n³ << k×n, tức n² << k

### 3. Condition Number Impact
**GD convergence rate:** 1 - 2λₘᵢₙ/λₘₐₓ = 1 - 2/κ
**Newton:** Không phụ thuộc condition number (locally)!

### 4. Regularization Effects
**Ridge regularization:**
- Condition number: κ → (λₘₐₓ + λ)/(λₘᵢₙ + λ)
- Always improves conditioning
- Shrinks coefficients toward zero

**Hessian regularization:**
- Pure numerical technique
- Improves inversion stability
- No model bias introduced

---

## Kết luận Newton Methods Journey

### Key Discoveries:
1. **Newton = Ultimate local optimizer** - Quadratic convergence là real
2. **Computational cost = Main limitation** - O(n³) per iteration
3. **Regularization = Numerical savior** - Fixes singular Hessian issues
4. **Damping = Global convergence** - Combines local optimality với global guarantees
5. **Best practice = Multiple techniques** - Newton + Ridge + Line search

### When to use Newton Methods:
✅ **Good for:**
- Small to medium problems (n < 10,000)  
- Well-conditioned problems
- When high precision needed
- Quadratic or near-quadratic objectives

❌ **Avoid for:**
- Large-scale problems (n > 100,000)
- Ill-conditioned problems without regularization
- Non-smooth objectives
- When gradient computation already expensive

### Ranking by effectiveness:
1. **Newton + Ridge + Damping** (setup 23) - Best overall
2. **Pure Newton + Ridge** (setup 8) - Simple và effective  
3. **Dual regularization** (setup 32) - Maximum stability
4. **Pure Newton** (setup 7) - When problem well-behaved

### Mathematical wisdom gained:
- Second-order information = Quantum leap in convergence speed
- Regularization serves dual purpose: math + numerics
- Line search = Insurance policy for global convergence
- Condition number = Key predictor of algorithm success

**Final insight:** Newton methods represent the pinnacle of local optimization. They achieve theoretical optimum convergence rate but at computational cost. The art is knowing when the trade-off is worthwhile.

**Next adventure:** Quasi-Newton methods - Can we get Newton benefits without full Hessian computation? The best of both worlds awaits!