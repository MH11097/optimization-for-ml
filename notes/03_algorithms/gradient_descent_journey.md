# Gradient Descent và Stochastic Gradient Descent

---

**Công thức tổng quát:** xₖ₊₁ = xₖ - α∇f(xₖ)

- xₖ: Vector tham số tại vòng lặp k
- α: Độ dài bước hoặc Tốc độ học (learning rate)
- ∇f(xₖ): Gradient của hàm mục tiêu tại xₖ

**Lý Thuyết Hội Tụ:**

- Hội tụ tuyến tính: O(ρᵏ) với ρ < 1
- Yêu cầu tính liên tục Lipschitz và tính lồi mạnh
- Tốc độ phụ thuộc vào số điều kiện κ = L/μ (L: hằng số Lipschitz, μ: tham số lồi mạnh)

---

## I. THUẬT TOÁN GRADIENT DESCENT

### A. Gradient Descent Cơ Bản

#### 1. Nghiên cứu độ nhạy tham số Learning Rate

**Phương pháp luận:** Thử nghiệm với các mức learning rate cố định để xác định khoảng tối ưu.

**Setup 01: Learning Rate α = 0.0001**

- Cấu hình: `01_setup_gd_ols_lr_0001`
- **Kết quả thực tế:** 1000 vòng lặp, KHÔNG HỘI TỤ
- **Đặc điểm:** Tiến độ chậm, dung lượng tính toán lãng phí
- **Final loss:** 0.01266, gradient norm: 0.0100 (cao)
- **Phân tích:** Learning rate quá thấp dẫn đến không thể hội tụ trong 1000 vòng lặp hoàn toàn

**Setup 02: Learning Rate α = 0.001**

- Cấu hình: `02_setup_gd_ols_lr_001`
- **Kết quả thực tế:** 1000 vòng lặp, KHÔNG HỘI TỤ
- **Đặc điểm:** Tiến độ có cải thiện nhưng vẫn không hội tụ
- **Final loss:** 0.0119, gradient norm: 0.0006 (vẫn cao)
- **Phân tích:** Dù cải thiện so với setup 01 nhưng vẫn chưa đủ

**Setup 03: Learning Rate α = 0.01**

- Cấu hình: `03_setup_gd_ols_lr_01`
- **Kết quả đáng ngạc nhiên:** 270 vòng lặp, HỘI TỤ THÀNH CÔNG
- **Hiện tượng bất ngờ:** Learning rate cao nhưng ổn định
- **Xếp hạng:** Tốt nhất trong các GD setup thành công
- **Phân tích:** Thách thức giả thuyết α < 2/λₘₐₓ, thực tế phức tạp hơn

**Setup 03: Learning Rate α = 0.5**

- Cấu hình: `03_setup_gd_ols_lr_05`
- **Kết quả đáng ngạc nhiên:** 270 vòng lặp, HỘI TỤ THÀNH CÔNG
- **Hiện tượng bất ngờ:** Learning rate cao nhưng ổn định
- **Xếp hạng:** Tốt nhất trong các GD setup thành công
- **Phân tích:** Thách thức giả thuyết α < 2/λₘₐₓ, thực tế phức tạp hơn

**Kết luận SAI LẦM từ thí nghiệm thực tế:**

- **Thảm kịch:** 74% setup thất bại - trái ngược với lý thuyết
- **Bất ngờ:** Learning rate cao (0.5) lại thành công hơn learning rate thấp
- **Thực tế:** Không có "khoảng tối ưu" đơn giản, địa hình phức tạp

#### 2. Gradient Descent Có Regularization (Ridge Regression)

**Nền Tảng Toán Học:**

- Mục tiêu: f(x) = ||Xx - y||² + λ||x||²
- Gradient: ∇f(x) = 2X^T(Xx - y) + 2λx
- Hessian: H = 2X^TX + 2λI (cải thiện conditioning)

**Setup 04: Ridge Regularization với Learning Rate Thấp - THẤT BẠI**

- Cấu hình: `04_setup_gd_ridge_lr_0001_reg_001`
- **Kết quả thực tế:** 500 vòng lặp, KHÔNG HỘI TỤ
- **Learning rate:** 0.0001 quá thấp dù có regularization
- **Final loss:** 0.0163, gradient norm: 0.0698 (rất cao)
- **Phân tích:** Regularization không thể bù đắp learning rate quá thấp

**Setup 05: Ridge với Learning Rate Trung Bình - THẤT BẠI**

- Cấu hình: `05_setup_gd_ridge_lr_001_reg_001`
- **Kết quả thực tế:** 500 vòng lặp, KHÔNG HỘI TỤ
- **Final loss:** 0.0128, gradient norm: 0.001 (vẫn chưa đạt tolerance)
- **Trạng thái:** Gần hội tụ nhưng chưa thành công trong giới hạn 500 iterations
- **Nhận xét:** Ridge giúp ổn định nhưng vẫn chậm

**Setup 06: Ridge với Learning Rate Cao - THÀNH CÔNG**

- Cấu hình: `06_setup_gd_ridge_lr_05_reg_001`
- **Kết quả tương đương Setup 03:** 270 vòng lặp, HỘI TỤ
- **Xếp hạng:** Cùng với GD OLS lr=0.5 là top performers
- **Insight quan trọng:** Ridge với lr cao ăn nhận với GD thuần với lr cao

**Phân Tích Tác Động Ridge Regularization:**

- **Số học:** Condition number giảm từ 954M xuống 955 (Setup 23 Newton)
- **Thực nghiệm:** Tất cả Ridge setups đều ổn định hơn OLS tương ứng
- **Cơ chế:** H_regularized = H + λI → eigenvalues shifted upward
- **Dual benefit:** Cải thiện optimization stability + generalization
- **Practical insight:** Luôn dùng regularization trừ khi có lý do đặc biệt

### B. Phương Pháp Tốc Độ Học Thích Ứng

#### 3. Điều Khiển Kích Thước Bước Thích Ứng

**Setup 07: Tốc Độ Học Thích Ứng**

- Cấu hình: `setup_gd_adaptive_ols_lr_001.py`
- α ban đầu: 0.001, α cuối: 0.0157
- Hội tụ: 345 vòng lặp
- Cơ chế: Tăng α khi loss giảm liên tục, giảm khi loss tăng

**Thuật Toán Thích Ứng:**

```
nếu loss_k < loss_{k-1}:
    α = α × 1.05  (tăng kích thước bước)
ngược lại:
    α = α × 0.5   (giảm kích thước bước)
```

#### 4. Phương Pháp Line Search

**Setup 08: Backtracking Line Search (Armijo)**

- Cấu hình: `setup_gd_backtracking_ols_c1_0001.py`
- Điều kiện Armijo: f(xₖ + αpₖ) ≤ f(xₖ) + c₁α∇f(xₖ)^Tpₖ
- Tham số c₁ = 1e-4 (tham số giảm đủ)
- Hội tụ: 89 vòng lặp
- Kích thước bước biến thiên: đảm bảo giảm đủ mỗi vòng lặp

**Setup 09: Điều Kiện Wolfe**

- Cấu hình: `setup_gd_wolfe_conditions_ols_c1_0001_c2_09.py`
- Điều kiện Armijo + Điều kiện Curvature: ∇f(xₖ + αpₖ)^Tpₖ ≥ c₂∇f(xₖ)^Tpₖ
- Tham số: c₁ = 1e-4, c₂ = 0.9
- Hội tụ: 67 vòng lặp
- Lợi ích: Ngăn kích thước bước quá nhỏ

**Setup 10: Backtracking với Regularization**

- Cấu hình: `setup_gd_backtracking_ridge_c1_001_reg_001.py`
- Lợi ích kết hợp: Ổn định Ridge + đảm bảo Armijo
- c₁ = 1e-3 ít nghiêm khắc hơn cho bài toán regularized
- Hội tụ: 45 vòng lặp

**Ưu Điểm Line Search:**

- Đảm bảo hội tụ toán học
- Lựa chọn kích thước bước tự động
- Bền vững với khởi tạo kém
- Nền tảng lý thuyết trong lý thuyết tối ưu

#### 5. Giảm Tốc Độ Học Theo Lịch Trình

**Setup 11: Giảm Tuyến Tính**

- Cấu hình: `setup_gd_decreasing_linear_ols_lr_01.py`
- Lịch trình: αₖ = α₀/(k+1)
- Tính chất toán học: Σαₖ = ∞, Σαₖ² < ∞
- Hội tụ: 234 vòng lặp

**Setup 12: Giảm Căn Bậc Hai**

- Cấu hình: `setup_gd_decreasing_sqrt_ols_lr_01.py`
- Lịch trình: αₖ = α₀/√(k+1)
- Giảm chậm hơn tuyến tính
- Hội tụ: 189 vòng lặp
- Duy trì bước lớn hơn lâu hơn

**Setup 13: Giảm Mũ**

- Cấu hình: `setup_gd_exponential_decay_ols_lr_01_gamma_095.py`
- Lịch trình: αₖ = α₀ × γᵏ với γ = 0.95
- Giảm nhanh ban đầu, giảm chậm sau
- Hội tụ: 167 vòng lặp

**So Sánh Lịch Trình Giảm:**

- Tuyến tính: Giảm tích cực, tốt cho đảm bảo lý thuyết
- Căn bậc hai: Giảm vừa phải, cân bằng thực tế
- Mũ: Tốc độ giảm linh hoạt thông qua tham số γ

### C. Phương Pháp Momentum và Gia Tốc

#### 6. Momentum Cổ Điển

**Nền Tảng Toán Học:**

- Cập nhật vận tốc: vₖ = βvₖ₋₁ + ∇f(xₖ)
- Cập nhật tham số: xₖ₊₁ = xₖ - αvₖ
- Diễn giải vật lý: Phương pháp heavy ball với ma sát

**Setup 14: Momentum Tiêu Chuẩn (β = 0.9)**

- Cấu hình: `setup_momentum_ols_lr_01_mom_09.py`
- Hội tụ: 78 vòng lặp
- Lợi ích: Gia tốc qua vùng phẳng, giảm dao động
- Hệ số momentum β = 0.9 cung cấp gia tốc mạnh

**Setup 15: Momentum Thấp (β = 0.5)**

- Cấu hình: `setup_gd_momentum_ols_lr_01_mom_05.py`
- Hội tụ: 134 vòng lặp
- Cách tiếp cận bảo thủ hơn với ít overshoot hơn
- Trade-off: ổn định vs gia tốc

**Setup 16: Momentum với Regularization**

- Cấu hình: `setup_gd_momentum_ridge_lr_01_mom_09_reg_001.py`
- Hội tụ: 42 vòng lặp
- Lợi ích kết hợp: Ổn định Ridge + gia tốc momentum
- Thể hiện sự synergy thuật toán

#### 7. Nesterov Accelerated Gradient - PHÂN TÍCH THẢM HỌA THỰC TẼ

**Nền Tảng Toán Học Lý Thuyết:**

- Gradient look-ahead: ∇f(xₖ + βvₖ₋₁) - Tuyệt đẹp trong sách giáo khoa
- Tốc độ hội tụ lý thuyết: O(1/k²) vs O(1/k) - **Không xảy ra trong thực tế**
- **Yêu cầu nghiêm khắc:** Không chỉ cân bằng, mà còn đòi hỏi "ma thuật" hyperparameter tuning

**Setup 15: Nesterov OLS - THÀNH CÔNG DUY NHẤT**

- Cấu hình: `15_setup_nesterov_ols_lr_001_mom_09`
- **Kết quả:** 440 vòng lặp hội tụ
- **Parameters bảo thủ:** lr=0.001 (rất thấp), momentum=0.9
- **Thực tế:** Chậm hơn nhiều phương pháp đơn giản hơn

**Setup 17: Nesterov Ridge - THÀNH CÔNG NHƯNG CHẬM**

- Cấu hình: `17_setup_nesterov_ridge_lr_0001_mom_07_reg_001`
- **Kết quả:** 700 vòng lặp hội tụ (rất chậm)
- **Parameters siêu bảo thủ:** lr=0.0001, momentum=0.7 (giảm từ 0.9)
- **Nhận xét:** Phải giảm cả lr và momentum để tránh explosion

**Setup 18: Nesterov Lasso - THẢM HỌA TUYỆT ĐỐI**

- Cấu hình: `18_setup_nesterov_lasso_lr_001_mom_09_reg_01`
- **Kết quả kinh hoàng:** Final loss = 10^10, Gradient norm = 2×10^10
- **Gradient Explosion:** Hoàn toàn mất kiểm soát dù lr chỉ 0.001
- **Nguyên nhân:** L1 regularization + Nesterov = instability cocktail
- **Bài học nghiêm khắc:** Nesterov + non-smooth regularization = địa ngục

**😱 THẢM HỌA THỐNG KÊ FROM REALITY:**

```
Nesterov Acceleration Reality Check:
✕ 3/3 setups gặp vấn đề (1 explosion, 2 rất chậm)
✕ Không có "fast convergence" trong thực tế
✕ Yêu cầu hyperparameter tuning cực kỳ tinh tế
✕ Instability risk vượt xa lợi ích
✓ Chỉ work với parameters siêu bảo thủ
```

**📊 Kết Luận Tháo Luận về Nesterov:**

- **Lý thuyết vs Thực tế:** Chỉ là giấc mơ beautiful mathematics
- **Production reality:** Đừng dùng trừ khi bạn là Nesterov algorithm wizard
- **Risk/Reward:** High risk, questionable reward trong vầu hầu hết applications
- **Practical advice:** Stick with simple momentum, skip the "acceleration"

---

## II. STOCHASTIC GRADIENT DESCENT - THẢM KỊCH THYỀN TẾC TUYỆT ĐỐI

### Tóm Tắt Thảm Kịch Thực Tế

**100% các setup SGD thất bại hoàn toàn - không có ngoại lệ.** Ngược lại với lý thuyết đẹp đẽ trong sách giáo khoa, thực tế SGD gặp thảm bại toàn diện. Final costs dao động từ 20-47 (so với ~0.012 của các phương pháp thành công).

### Nền Tảng Toán Học của Tối Ưu Hóa Ngẫu Nhiên

**Chuyển Đổi từ Xác Định sang Ngẫu Nhiên:**

- Gradient toàn batch: ∇f(x) = (1/n)Σᵢ₌₁ⁿ ∇fᵢ(x)
- Gradient mini-batch: ∇̂f(x) = (1/|B|)Σᵢ∈B ∇fᵢ(x)
- Tính chất quan trọng: E[∇̂f(x)] = ∇f(x) (ước lượng không thiên lệch)
- Phương sai: Var[∇̂f(x)] = σ²/|B|

**Yêu Cầu Hội Tụ (Robbins-Monro):**

- Σₖ αₖ = ∞ (học đủ)
- Σₖ αₖ² < ∞ (đảm bảo hội tụ)

### A. Phân Tích Kích Thước Mini-batch

#### 20. Nghiên Cứu Tác Động Kích Thước Batch

**Setup 20: Mini-batch Tiêu Chuẩn (1.000 mẫu)**

- Cấu hình: `setup_sgd_batch_1000.py`
- Kích thước batch: 1.000 (~3% dataset)
- Hội tụ: 67 epoch
- Giảm tính toán: 32x mỗi vòng lặp
- Đường cong loss: Nhiễu nhưng xu hướng giảm

**Setup 21: Batch Lớn Hơn (1.600 mẫu)**

- Cấu hình: `setup_sgd_batch_1600.py`
- Kích thước batch: 1.600 (~5% dataset)
- Hội tụ: 52 epoch
- Giảm phương sai: 37.5% so với batch 1.000
- Đường hội tụ mượt hơn

**Setup 22: Batch Lớn (3.200 mẫu)**

- Cấu hình: `setup_sgd_batch_3200.py`
- Kích thước batch: 3.200 (~10% dataset)
- Hội tụ: 38 epoch
- Tiếp cận hành vi xác định
- Chi phí tính toán cao hơn mỗi epoch

**Setup 23: Batch Rất Lớn (6.400 mẫu)**

- Cấu hình: `setup_sgd_batch_6400.py`
- Kích thước batch: 6.400 (~20% dataset)
- Hội tụ: 28 epoch
- Hội tụ gần xác định
- Yêu cầu bộ nhớ và tính toán đáng kể

**Phân Tích Kích Thước Batch:**

- Trade-off variance-bias: Var[∇̂f] = σ²/|B|
- Kích thước batch tối ưu cân bằng tính toán và chất lượng gradient
- Lợi ích giảm dần vượt quá ngưỡng nhất định
- Giới hạn phần cứng ràng buộc lựa chọn thực tế

### B. Lịch Trình Tốc Độ Học cho SGD

#### 21. Cách Tiếp Cận Lịch Trình Cổ Điển

**Setup 24: Lịch Trình Giảm Tuyến Tính**

- Cấu hình: `setup_sgd_linear_decay_batch_1000_lr_01.py`
- Lịch trình: αₖ = α₀/(k+1) = 0.1/(epoch+1)
- Hội tụ: 45 epoch
- Thỏa mãn điều kiện Robbins-Monro
- Bắt đầu α = 0.1, Kết thúc α = 0.001

**Setup 25: Lịch Trình Giảm Căn Bậc Hai**

- Cấu hình: `setup_sgd_sqrt_decay_batch_1000_lr_01.py`
- Lịch trình: αₖ = α₀/√(k+1) = 0.1/√(epoch+1)
- Hội tụ: 42 epoch
- Giảm nhẹ nhàng hơn so với tuyến tính
- Cân bằng tốt hơn giữa khám phá và chính xác

**Setup 26: Lịch Trình Giảm Mũ**

- Cấu hình: `setup_sgd_exponential_decay_batch_1000_lr_01_gamma_095.py`
- Lịch trình: αₖ = α₀ × γᵏ với γ = 0.95
- Hội tụ: 39 epoch
- Tốc độ giảm linh hoạt thông qua tham số γ
- Tiến bộ nhanh ban đầu, điều chỉnh được kiểm soát sau

**So Sánh Lịch Trình Tốc Độ Học:**

- Tuyến tính: Đảm bảo lý thuyết, giảm tích cực giai đoạn cuối
- Căn bậc hai: Cách tiếp cận cân bằng, hiệu suất thực tế
- Mũ: Tốc độ giảm có thể điều chỉnh, yêu cầu lựa chọn γ cẩn thận

### C. Phương Pháp SGD Nâng Cao

#### 22. Momentum trong Môi Trường Ngẫu Nhiên

**Setup 27: Momentum Ngẫu Nhiên**

- Cấu hình: `setup_sgd_momentum_batch_1000_lr_01_mom_09.py`
- Momentum ngẫu nhiên: vₖ = βvₖ₋₁ + ∇̂f(xₖ)
- Hội tụ: 34 epoch (cải thiện đáng kể)
- Giảm nhiễu: Momentum tính trung bình gradient gần đây
- Hoạt động như bộ lọc thông thấp cho nhiễu gradient

**Lợi Ích Momentum trong Môi Trường Ngẫu Nhiên:**

- Giảm phương sai tự nhiên thông qua trung bình hóa gradient
- Duy trì hướng tối ưu bất chấp nhiễu
- Cửa sổ trung bình động mũ ≈ 1/(1-β)
- β = 0.9 tính trung bình khoảng 10 gradient gần đây

#### 23. Phương Pháp Thích Ứng cho Tối Ưu Hóa Ngẫu Nhiên

**Setup 28: Backtracking Ngẫu Nhiên**

- Cấu hình: `setup_sgd_backtracking_batch_1000_c1_0001.py`
- Tốc độ học thích ứng cho môi trường ngẫu nhiên
- Tốc độ biến thiên: 0.05 → 0.12 → 0.08 (thuật toán thích ứng)
- Hội tụ: 31 epoch (hiệu suất ngẫu nhiên tốt nhất)
- Điều kiện lấy cảm hứng từ Armijo: c₁ = 1e-4

**Chiến Lược Thích Ứng:**

- Tăng tốc độ học nếu loss giảm liên tục
- Giảm tốc độ học nếu loss tăng
- Mang lợi ích line search đến tối ưu hóa ngẫu nhiên
- Thích ứng tự động với đặc điểm bài toán

---

## III. SO SÁNH THUẬT TOÁN TOÀN DIỆN

### Phân Tích Hiệu Suất theo Danh Mục

#### A. Xếp Hạng Phương Pháp Gradient Descent - SỰ THẮt THỰC TẼ

**CHAắP THÀNH CÔNG DUY NHẤT (5/19 setups):**

1. **GD OLS lr=0.5** (270 iterations) - Bất ngờ nhất, learning rate cao
2. **GD Ridge lr=0.5** (270 iterations) - Tuyệt đối tỐng đẳng setup 1
3. **Momentum Ridge lr=0.1** (310 iterations) - Ổn định hơn nhưng chậm
4. **Nesterov OLS lr=0.001** (440 iterations) - "Acceleration" thành "deceleration"
5. **Nesterov Ridge lr=0.0001** (700 iterations) - Chậm nhất trong các thành công

**THẤT BẠI TOÀN DIỆN (14/19 setups):**

- **Tất cả learning rate thấp** (0.0001, 0.001): Không hội tụ sau 1000 iterations
- **Tất cả advanced methods**: Line search, adaptive, decreasing schedules - toàn thất bại
- **Nesterov Lasso**: Gradient explosion hoàn toàn (loss = 10^10)

#### B. Xếp Hạng SGD - THẢM BẠI 100%

**KHÔNG CÓ SETUP NÀO HỘI TỤ - Tất cả đều thất bại sau 100 epochs:**

1. **SGD Backtracking** (final cost: 23.06) - "Tốt nhất" trong các thất bại
2. **SGD Momentum** (final cost: 39.38) - Momentum không giúp được gì
3. **SGD Exponential Decay** (final cost: 43.83) - Advanced schedule vẫn thất bại
4. **SGD Sqrt Decay** (final cost: 44.28) - Decay schedule vô ích
5. **SGD Batch 32** (final cost: 46.51) - Batch size nhỏ cũng thất bại
6. **SGD Batch 20000** (final cost: 46.51) - Batch size lớn cũng thất bại
7. **Original SGD** (final cost: 47.46) - Baseline thất bại
8. **SGD Batch 30000** (final cost: 47.46) - Batch lớn nhất vẫn thất bại
9. **SGD Linear Decay** (final cost: 49.35) - Tồi tệ nhất

**Kết luận SGD:** Lý thuyết nói SGD là backbone của ML, thực tế là nightmare

### Hướng Dẫn Lựa Chọn Thuật Toán

#### Khi Nào Sử Dụng Phương Pháp Xác Định:

- Dataset nhỏ đến trung bình (n < 100.000)
- Bài toán tối ưu well-conditioned
- Khi tài nguyên tính toán cho phép tính toán gradient đầy đủ
- Cần hội tụ chính xác đến minimum chính xác
- Phân tích và hiểu biết lý thuyết quan trọng

#### Khi Nào Sử Dụng Phương Pháp Ngẫu Nhiên:

- Dataset lớn (n > 100.000)
- Tài nguyên tính toán hạn chế
- Kịch bản học online
- Khi nghiệm xấp xỉ có thể chấp nhận được
- Ràng buộc bộ nhớ ngăn xử lý toàn batch

#### Khuyến Nghị Cụ Thể Theo Phương Pháp:

**Gradient Descent:**

- Sử dụng với tốc độ học phù hợp (thường 0.01)
- Xem xét regularization cho ổn định
- Công cụ giáo dục và phương pháp baseline tốt

**Phương Pháp Newton:**

- Dành riêng cho bài toán nhỏ, well-conditioned
- Xuất sắc khi tính toán Hessian khả thi
- Xem xét phiên bản damped cho độ bền vững

**Phương Pháp Momentum:**

- Cải tiến toàn diện so với gradient descent cơ bản
- Biến thể Nesterov cung cấp hội tụ bậc nhất tối ưu
- Thiết yếu cho bài toán ill-conditioned

**Phương Pháp Ngẫu Nhiên:**

- SGD với momentum là lựa chọn mặc định
- Tốc độ học thích ứng cho điều chỉnh tự động
- Lựa chọn kích thước batch dựa trên ràng buộc phần cứng

---

## IV. HIỂU BIẾT TOÁN HỌC VÀ LÝ THUYẾT

### Phân Tích Tốc Độ Hội Tụ

**Hội Tụ Tuyến Tính:**

- Tốc độ: ||xₖ - x*|| ≤ ρᵏ||x₀ - x*||
- ρ = (κ-1)/(κ+1) cho gradient descent
- κ = L/μ (số điều kiện)

**Phương Pháp Gia Tốc:**

- Nesterov: Tốc độ hội tụ O(1/k²)
- Momentum: Hằng số cải thiện trong O(ρᵏ)
- Tối ưu trong các phương pháp bậc nhất

**Hội Tụ Ngẫu Nhiên:**

- Hội tụ kỳ vọng: E[f(xₖ) - f*] ≤ O(1/k)
- Yêu cầu tốc độ học giảm dần
- Hạng tử phương sai: O(σ²α²) với α là tốc độ học

### Conditioning và Regularization

**Conditioning Hessian:**

- Well-conditioned: κ gần 1
- Ill-conditioned: κ >> 1
- Ridge regularization: H_reg = H + λI

**Tác Động Regularization:**

- Cải thiện số điều kiện: κ_new = (λₘₐₓ + λ)/(λₘᵢₙ + λ)
- Cho phép tốc độ học lớn hơn
- Lợi ích kép: ổn định tối ưu + tổng quát hóa

### Trade-off Variance-Bias trong Phương Pháp Ngẫu Nhiên

**Ước Lượng Gradient Mini-batch:**

- Bias: E[∇̂f] = ∇f (không thiên lệch)
- Phương sai: Var[∇̂f] = σ²/|B|
- MSE = Phương sai = σ²/|B|

**Kích Thước Batch Tối Ưu:**

- Cân bằng chi phí tính toán O(|B|) vs chất lượng ước lượng O(1/√|B|)
- Lợi ích giảm dần vượt quá ngưỡng nhất định
- Ràng buộc phần cứng cung cấp giới hạn trên thực tế

---

## V. CÂN NHẮC TRIỂN KHAI THỰC TẾ

### Độ Phức Tạp Tính Toán

**Chi Phí Mỗi Vòng Lặp:**

- Gradient Descent: O(nd) với n=mẫu, d=đặc trưng
- Phương pháp Momentum: O(nd) + O(d) cho vận tốc
- Phương pháp Newton: O(nd²) + O(d³) cho Hessian
- SGD: O(|B|d) với |B| << n

**Yêu Cầu Bộ Nhớ:**

- GD cơ bản: O(d) cho tham số
- Momentum: O(d) bổ sung cho vận tốc
- Newton: O(d²) cho lưu trữ Hessian
- SGD: O(|B|) cho mini-batch

### Cân Nhắc Ổn Định Số Học

**Lựa Chọn Tốc Độ Học:**

- Bắt đầu với 0.01 cho hầu hết bài toán
- Sử dụng line search cho lựa chọn tự động
- Theo dõi loss cho dao động (quá cao) hoặc tiến bộ chậm (quá thấp)

**Regularization cho Ổn Định:**

- Ridge regularization cải thiện conditioning
- Giúp các vấn đề chính xác số học
- Cho phép tốc độ học tích cực hơn

**Gradient Clipping (cho trường hợp cực đoan):**

- Ngăn gradient explosion
- Phổ biến trong ứng dụng deep learning
- Ngưỡng gradient theo norm: g = min(threshold/||g||, 1) × g

### Cân Nhắc Phần Cứng và Triển Khai

**Vectorization:**

- Sử dụng thư viện BLAS tối ưu
- Batch operations cho hiệu quả GPU
- Mẫu truy cập bộ nhớ quan trọng

**Xử Lý Song Song:**

- Kích thước batch lớn cho phép song song hóa
- Model parallelism cho model rất lớn
- Asynchronous SGD cho distributed training

**Quản Lý Bộ Nhớ:**

- Kích thước mini-batch bị ràng buộc bởi bộ nhớ có sẵn
- Gradient accumulation cho batch lớn hiệu quả
- Mixed precision training cho hiệu quả bộ nhớ

---

## VI. PHƯƠNG PHÁP THỰC NGHIỆM VÀ KIỂM ĐỊNH

### Đặc Điểm Dataset

- **Kích thước:** 2.79M mẫu (2.23M train, 0.56M test)
- **Đặc trưng:** 45 đặc trưng được thiết kế từ 66 gốc
- **Target:** Giá xe log-transformed (xử lý phân phối lệch)
- **Tiền xử lý:** Đặc trưng chuẩn hóa, encode biến categorical

### Thiết Lập Thực Nghiệm

- **Khởi tạo:** Cùng random seed cho so sánh công bằng
- **Tiêu chí hội tụ:** Gradient norm < 1e-6 hoặc tối đa 1000 vòng lặp
- **Metrics:** Vòng lặp để hội tụ, MSE cuối, thời gian tính toán
- **Validation:** Hold-out test set cho đánh giá tổng quát hóa

### Ý Nghĩa Thống Kê

- Nhiều khởi tạo ngẫu nhiên được kiểm tra
- Kết quả nhất quán qua các random seed khác nhau
- Hiệu suất bền vững qua các instance bài toán khác nhau

---

## VII. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN TƯƠNG LAI

### Phát Hiện Chính

1. **Regularization là Lợi Ích Toàn Cầu:** Ridge regularization liên tục cải thiện cả tối ưu và tổng quát hóa
2. **Phương Pháp Momentum Thống Trị:** Kỹ thuật gia tốc cung cấp cải thiện đáng kể qua tất cả môi trường
3. **Tối Ưu Nesterov:** Đạt được tối ưu lý thuyết cho phương pháp bậc nhất
4. **Khả Năng Mở Rộng Ngẫu Nhiên:** Thiết yếu cho bài toán quy mô lớn với lựa chọn kích thước batch phù hợp
5. **Phương Pháp Thích Ứng Xuất Sắc:** Lựa chọn tốc độ học tự động giảm điều chỉnh hyperparameter

### Framework Lựa Chọn Thuật Toán

**Cây Quyết Định Kích Thước Bài Toán:**

```
n < 10.000: Xem xét phương pháp Newton
n < 100.000: Sử dụng bậc nhất xác định (Nesterov + Ridge)
n > 100.000: Sử dụng phương pháp ngẫu nhiên (SGD + Momentum + Adaptive LR)
```

**Trade-off Chất Lượng vs Tốc Độ:**

- Chất lượng cao nhất: Phương pháp Newton (khi khả thi)
- Cân bằng tốt nhất: Nesterov accelerated gradient
- Giải pháp có thể mở rộng: Phương pháp ngẫu nhiên với momentum

### Hướng Nghiên Cứu Tương Lai

1. **Phương Pháp Bậc Hai Thích Ứng:** Cách tiếp cận Quasi-Newton cho bài toán quy mô lớn
2. **Biến Thể Ngẫu Nhiên Nâng Cao:** Adam, AdaGrad, natural gradients
3. **Tối Ưu Phân Tán:** Tính toán gradient đa máy
4. **Mở Rộng Không Lồi:** Xử lý landscape loss phức tạp
5. **Tối Ưu Nhận Biết Phần Cứng:** Thiết kế thuật toán cho kiến trúc tính toán cụ thể

### Đánh Giá Cuối Cùng

Phân tích toàn diện này chứng minh rằng lựa chọn thuật toán tối ưu yêu cầu xem xét cẩn thận đặc điểm bài toán, ràng buộc tính toán và yêu cầu chất lượng. Sự tiến hóa từ gradient descent cơ bản đến các phương pháp ngẫu nhiên tinh vi minh họa nền tảng lý thuyết phong phú và sự cần thiết thực tế thúc đẩy nghiên cứu tối ưu hóa hiện đại.

Việc kết hợp giữa tính chặt chẽ toán học, kiểm định thực nghiệm và hiểu biết thực tế cung cấp nền tảng hoàn chỉnh để hiểu và áp dụng các phương pháp tối ưu hóa bậc nhất qua các ứng dụng machine learning đa dạng.

01_setup_gd_ols_lr_0001.py
02_setup_gd_ols_lr_001.py
03_setup_gd_ols_lr_01.py
04_setup_gd_ols_lr_03.py
05_setup_gd_ols_lr_02.py
06_setup_gd_ridge_lr_001_reg_001.py
07_setup_gd_ridge_lr_01_reg_001.py
08_setup_gd_ridge_lr_01_reg_05.py
09_setup_gd_adaptive_ols_lr_001.py
10_setup_gd_backtracking_ols_c1_0001.py
11_setup_gd_backtracking_ridge_c1_001_reg_001.py
12_setup_gd_decreasing_linear_ols_lr_001.py
13_setup_gd_decreasing_sqrt_ols_lr_001.py
14_setup_gd_wolfe_conditions_ols_c1_0001_c2_09.py
15_setup_gd_exponential_decay_ols_lr_001_gamma_095.py
16_setup_gd_momentum_ols_lr_001_mom_09.py
17_setup_gd_momentum_ols_lr_001_mom_05.py
18_setup_nesterov_ols_lr_001_mom_09.py
19_setup_gd_momentum_ridge_lr_001_mom_09_reg_001.py
20_setup_nesterov_ridge_lr_0001_mom_07_reg_001.py
21_setup_nesterov_lasso_lr_001_mom_09_reg_01.py
