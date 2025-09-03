# Hành trình phát triển thuật toán tối ưu hóa

## Tổng quan codebase hiện có

Dự án tối ưu hóa này có cấu trúc hoàn chỉnh với 4 họ thuật toán chính:

### 1. Gradient Descent Family (16 setup hiện có + 6 mới)
### 2. Newton Methods (4 setup hiện có + 3 mới) 
### 3. Quasi-Newton (3 setup hiện có + 3 mới)
### 4. Stochastic Gradient Descent (9 setup hiện có + 3 mới)

---

## 1. GRADIENT DESCENT - Từ cơ bản đến nâng cao

### Nền tảng toán học
**Công thức cơ bản:** θₖ₊₁ = θₖ - α∇f(θₖ)
- α: learning rate (tốc độ học)
- ∇f(θₖ): gradient tại điểm hiện tại
- Điều kiện hội tụ: Lipschitz continuity, strong convexity
- Tốc độ hội tụ: Linear O(ρᵏ) với ρ < 1

### Hành trình từng bước

#### Cấp độ 1: Gradient Descent cơ bản (Fixed Learning Rate)
**Mục tiêu:** Hiểu ảnh hưởng của learning rate

1. **setup_gd_ols_lr_001.py** - Learning rate nhỏ (0.001)
   - Hội tụ chậm nhưng ổn định
   - Phù hợp với dữ liệu có scale lớn
   
2. **setup_gd_ols_lr_01.py** - Learning rate trung bình (0.01)
   - Cân bằng giữa tốc độ và ổn định
   - Thường là điểm khởi đầu tốt
   
3. **setup_gd_ols_lr_05.py** - Learning rate cao (0.05)
   - Hội tụ nhanh nhưng có thể dao động
   - Risk overshoot minimum

**Insight toán học:** Learning rate phải < 2/L với L là Lipschitz constant của gradient

#### Cấp độ 2: Regularization (Ridge Regression)
**Mục tiêu:** Kiểm soát overfitting

4. **setup_gd_ridge_lr_001_reg_001.py** - Regularization nhỏ
   - L2 penalty: λ||θ||₂²
   - Shrinks weights về 0
   
5. **setup_gd_ridge_lr_01_reg_001.py** - Cân bằng lr và regularization
6. **setup_gd_ridge_lr_05_reg_001.py** - Learning rate cao + regularization

**Insight toán học:** Ridge làm Hessian well-conditioned hơn: H + λI

#### Cấp độ 3: Adaptive Learning Rates
**Mục tiêu:** Tự động điều chỉnh step size

7. **setup_gd_adaptive_ols_lr_001.py** - Adaptive step size
   - Tăng α nếu loss giảm liên tục
   - Giảm α nếu loss tăng
   
8. **setup_gd_backtracking_ols_c1_0001.py** - Backtracking line search
   - Armijo condition: f(xₖ + αpₖ) ≤ f(xₖ) + c₁α∇f(xₖ)ᵀpₖ
   - Đảm bảo sufficient decrease

**[MỚI] setup_gd_wolfe_conditions_ols_c1_0001_c2_09.py** - Wolfe conditions
   - Thêm curvature condition: ∇f(xₖ + αpₖ)ᵀpₖ ≥ c₂∇f(xₖ)ᵀpₖ
   - Đảm bảo step không quá nhỏ

**[MỚI] setup_gd_backtracking_ridge_c1_001_reg_001.py** - Backtracking + regularization

#### Cấp độ 4: Scheduled Learning Rates
**Mục tiêu:** Giảm dần learning rate theo thời gian

9. **setup_gd_decreasing_linear_ols_lr_01.py** - Linear decay
   - αₖ = α₀/(k+1)
   - Đảm bảo Σαₖ = ∞, Σαₖ² < ∞
   
10. **setup_gd_decreasing_sqrt_ols_lr_01.py** - Square root decay
    - αₖ = α₀/√(k+1)
    - Chậm hơn linear decay

**[MỚI] setup_gd_exponential_decay_ols_lr_01_gamma_095.py** - Exponential decay
    - αₖ = α₀ × γᵏ với γ = 0.95
    - Giảm nhanh ban đầu, chậm sau này

#### Cấp độ 5: Momentum Methods
**Mục tiêu:** Tăng tốc hội tụ và vượt qua local minima

11. **setup_momentum_ols_lr_01_mom_09.py** - Standard momentum
    - vₖ = βvₖ₋₁ + ∇f(θₖ)
    - θₖ₊₁ = θₖ - αvₖ
    
**[MỚI] setup_gd_momentum_ols_lr_01_mom_05.py** - Momentum thấp (0.5)
    - So sánh với momentum cao (0.9)
    
12. **setup_nesterov_ols_lr_01_mom_09.py** - Nesterov acceleration
    - "Look ahead": gradient tại θₖ + βvₖ₋₁
    - Tốc độ hội tụ O(1/k²) thay vì O(1/k)

**[MỚI] setup_gd_momentum_ridge_lr_01_mom_09_reg_001.py** - Momentum + regularization

13. **setup_nesterov_ridge_lr_01_mom_09_reg_001.py** - Nesterov + Ridge

**[MỚI] setup_gd_nesterov_lasso_lr_01_mom_09_reg_01.py** - Nesterov + L1 regularization

**Insight toán học:** Momentum như "quán tính", giúp vượt qua saddle points và local minima

---

## 2. NEWTON METHODS - Second-order optimization

### Nền tảng toán học
**Công thức:** θₖ₊₁ = θₖ - H⁻¹∇f(θₖ)
- H: Hessian matrix (ma trận đạo hàm bậc 2)
- Quadratic convergence khi gần nghiệm
- Yêu cầu H positive definite

### Hành trình từng bước

#### Cấp độ 1: Pure Newton
1. **setup_newton_ols_pure.py** - Newton thuần túy
   - H = 2XᵀX cho OLS
   - Hội tụ rất nhanh nếu H well-conditioned
   
2. **setup_newton_ridge_pure.py** - Newton + Ridge
   - H = 2XᵀX + 2λI
   - Regularization cải thiện condition number

#### Cấp độ 2: Damped Newton (Line Search)
3. **setup_newton_ols_damped.py** - Damped Newton
   - θₖ₊₁ = θₖ - αH⁻¹∇f(θₖ)
   - α từ line search, đảm bảo global convergence
   
4. **setup_newton_ridge_damped.py** - Damped Newton + Ridge

#### Cấp độ 3: Regularized Newton
**[MỚI] setup_newton_regularized_ols_lambda_001.py** - Modified Hessian
   - H + λI để ensure positive definiteness
   - λ = 0.001 cho numerical stability

**[MỚI] setup_newton_backtracking_ols_c1_0001.py** - Newton + backtracking
   - Kết hợp Newton direction với Armijo line search

**[MỚI] setup_newton_regularized_ridge_lambda_01_reg_001.py** - Dual regularization
   - Cả Hessian regularization và Ridge penalty

**Insight toán học:** Newton methods có thể diverge nếu Hessian không positive definite hoặc khởi tạo xa nghiệm.

---

## 3. QUASI-NEWTON - Approximating Second-order Information

### Nền tảng toán học
**Secant condition:** Bₖ₊₁sₖ = yₖ
- sₖ = θₖ₊₁ - θₖ (step vector)
- yₖ = ∇f(θₖ₊₁) - ∇f(θₖ) (gradient difference)
- Bₖ ≈ H (approximation of Hessian)

### Hành trình từng bước

#### Cấp độ 1: BFGS Standard
1. **setup_bfgs_ols.py** - BFGS cơ bản
   - Full matrix update
   - Superlinear convergence
   
2. **setup_bfgs_ridge.py** - BFGS + regularization

#### Cấp độ 2: L-BFGS (Memory Efficient)
3. **setup_lr1_ols.py** - Limited memory BFGS
   - Chỉ lưu m cặp (s,y) gần nhất
   - Suitable cho large-scale problems

#### Cấp độ 3: Advanced Quasi-Newton
**[MỚI] setup_lbfgs_ols_m_10.py** - L-BFGS với memory=10
   - Cân bằng giữa memory và convergence rate

**[MỚI] setup_lbfgs_ridge_m_5_reg_001.py** - L-BFGS + regularization
   - Memory nhỏ (5) cho efficiency

**[MỚI] setup_bfgs_backtracking_ols_c1_0001.py** - BFGS + line search
   - Improved global convergence properties

**Insight toán học:** Quasi-Newton methods maintain curvature information without computing expensive Hessian.

---

## 4. STOCHASTIC METHODS - Large-scale Optimization

### Nền tảng toán học
**Mini-batch gradient:** ∇f(θ) ≈ (1/|B|)Σᵢ∈B ∇fᵢ(θ)
- B: mini-batch indices
- Trade-off giữa computational cost và variance
- Convergence rate phụ thuộc learning rate schedule

### Hành trình từng bước

#### Cấp độ 1: Basic SGD với Different Batch Sizes
1. **setup_sgd_batch_1000.py** - Standard batch size
2. **setup_sgd_batch_1600.py** - Larger batch
3. **setup_sgd_batch_3200.py** - Large batch (gần Full-batch)
4. **setup_sgd_batch_6400.py** - Very large batch

**Insight:** Larger batch → Lower variance, Higher computational cost

#### Cấp độ 2: Learning Rate Scheduling
5. **setup_sgd_linear_decay_batch_1000_lr_01.py** - Linear decay
6. **setup_sgd_sqrt_decay_batch_1000_lr_01.py** - Square root decay

#### Cấp độ 3: Advanced SGD Variants
**[MỚI] setup_sgd_momentum_batch_1000_lr_01_mom_09.py** - SGD + momentum
   - Giảm variance của stochastic gradients

**[MỚI] setup_sgd_exponential_decay_batch_1000_lr_01_gamma_095.py** - Exponential decay
   - Fast initial decay, slow later convergence

**[MỚI] setup_sgd_backtracking_batch_1000_c1_0001.py** - SGD + approximate line search
   - Modified line search cho stochastic setting

---

## Hành trình thực hành - Thứ tự setup theo cấp độ

### GIAI ĐOẠN 1: CƠ BẢN (Setup 1-8)
**Mục tiêu:** Hiểu ảnh hưởng của learning rate và regularization

1. **setup_gd_ols_lr_001.py** - GD cơ bản, lr thấp
2. **setup_gd_ols_lr_01.py** - GD cơ bản, lr trung bình  
3. **setup_gd_ols_lr_05.py** - GD cơ bản, lr cao
4. **setup_gd_ridge_lr_001_reg_001.py** - Ridge + lr thấp
5. **setup_gd_ridge_lr_01_reg_001.py** - Ridge + lr trung bình
6. **setup_gd_ridge_lr_05_reg_001.py** - Ridge + lr cao
7. **setup_newton_ols_pure.py** - Newton cơ bản
8. **setup_newton_ridge_pure.py** - Newton + Ridge

### GIAI ĐOẠN 2: ADAPTIVE METHODS (Setup 9-16)
**Mục tiêu:** Tự động điều chỉnh step size

9. **setup_gd_adaptive_ols_lr_001.py** - Adaptive step size
10. **setup_gd_backtracking_ols_c1_0001.py** - Backtracking line search
11. **[MỚI] setup_gd_wolfe_conditions_ols_c1_0001_c2_09.py** - Wolfe conditions
12. **[MỚI] setup_gd_backtracking_ridge_c1_001_reg_001.py** - Backtracking + Ridge
13. **setup_gd_decreasing_linear_ols_lr_01.py** - Linear decay
14. **setup_gd_decreasing_sqrt_ols_lr_01.py** - Sqrt decay
15. **[MỚI] setup_gd_exponential_decay_ols_lr_01_gamma_095.py** - Exponential decay
16. **setup_newton_ols_damped.py** - Damped Newton

### GIAI ĐOẠN 3: MOMENTUM & ACCELERATION (Setup 17-24)
**Mục tiêu:** Tăng tốc hội tụ

17. **setup_momentum_ols_lr_01_mom_09.py** - Standard momentum
18. **[MỚI] setup_gd_momentum_ols_lr_01_mom_05.py** - Momentum thấp
19. **setup_nesterov_ols_lr_01_mom_09.py** - Nesterov acceleration
20. **[MỚI] setup_gd_momentum_ridge_lr_01_mom_09_reg_001.py** - Momentum + Ridge
21. **setup_nesterov_ridge_lr_01_mom_09_reg_001.py** - Nesterov + Ridge
22. **[MỚI] setup_gd_nesterov_lasso_lr_01_mom_09_reg_01.py** - Nesterov + L1
23. **setup_newton_ridge_damped.py** - Damped Newton + Ridge
24. **[MỚI] setup_newton_regularized_ols_lambda_001.py** - Regularized Newton

### GIAI ĐOẠN 4: QUASI-NEWTON (Setup 25-30)
**Mục tiêu:** Approximate second-order methods

25. **setup_bfgs_ols.py** - BFGS cơ bản
26. **setup_bfgs_ridge.py** - BFGS + Ridge
27. **setup_lr1_ols.py** - L-BFGS memory efficient
28. **[MỚI] setup_lbfgs_ols_m_10.py** - L-BFGS memory=10
29. **[MỚI] setup_lbfgs_ridge_m_5_reg_001.py** - L-BFGS + Ridge
30. **[MỚI] setup_bfgs_backtracking_ols_c1_0001.py** - BFGS + line search

### GIAI ĐOẠN 5: ADVANCED NEWTON (Setup 31-32)
**Mục tiêu:** Sophisticated second-order methods

31. **[MỚI] setup_newton_backtracking_ols_c1_0001.py** - Newton + backtracking
32. **[MỚI] setup_newton_regularized_ridge_lambda_01_reg_001.py** - Dual regularization

### GIAI ĐOẠN 6: STOCHASTIC METHODS (Setup 33-47)
**Mục tiêu:** Large-scale optimization

33. **setup_sgd_batch_1000.py** - SGD cơ bản
34. **setup_sgd_batch_1600.py** - SGD batch lớn hơn
35. **setup_sgd_batch_3200.py** - SGD large batch
36. **setup_sgd_batch_6400.py** - SGD very large batch
37. **setup_sgd_linear_decay_batch_1000_lr_01.py** - SGD linear decay
38. **setup_sgd_sqrt_decay_batch_1000_lr_01.py** - SGD sqrt decay
39. **[MỚI] setup_sgd_momentum_batch_1000_lr_01_mom_09.py** - SGD + momentum
40. **[MỚI] setup_sgd_exponential_decay_batch_1000_lr_01_gamma_095.py** - SGD exponential decay
41. **[MỚI] setup_sgd_backtracking_batch_1000_c1_0001.py** - SGD + line search

### Hướng dẫn thực hành theo từng giai đoạn:

**Giai đoạn 1-2 (Setup 1-16):** Chạy tuần tự để hiểu cơ bản
**Giai đoạn 3-4 (Setup 17-30):** So sánh performance với giai đoạn trước
**Giai đoạn 5-6 (Setup 31-41):** Advanced methods cho specific use cases

### Phân tích kết quả:
- **Convergence plots:** So sánh tốc độ hội tụ
- **Final loss:** Chất lượng nghiệm cuối cùng  
- **Computational time:** Efficiency comparison
- **Robustness:** Stability với different initializations

### Khi nào dùng thuật toán nào:

**Gradient Descent:**
- Small-medium datasets
- Well-conditioned problems
- Khi cần hiểu gradient flow

**Newton Methods:**
- Small datasets (n < 10,000)
- Well-conditioned Hessian
- Khi cần convergence rất nhanh

**Quasi-Newton:**
- Medium datasets
- Khi Hessian expensive to compute
- Good balance speed vs memory

**Stochastic Methods:**
- Large datasets (n > 100,000)
- Online learning scenarios
- Limited computational resources

---

## Tổng kết hành trình

**47 setup files** tạo thành một curriculum hoàn chỉnh từ cơ bản đến nâng cao:
- **22 Gradient Descent setups** - Nền tảng vững chắc
- **7 Newton setups** - Second-order precision  
- **6 Quasi-Newton setups** - Practical efficiency
- **12 Stochastic setups** - Scalable solutions

Mỗi setup được thiết kế để dạy một concept cụ thể, với mathematical foundation rõ ràng và practical insights có thể áp dụng vào real-world problems.