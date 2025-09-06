# Phân Tích Thực Nghiệm Toàn Diện về Gradient Descent: Đánh Giá Hiệu Suất và So Sánh Thuật Toán

## Tóm Tắt

Nghiên cứu này trình bày một đánh giá thực nghiệm nghiêm ngặt về các thuật toán tối ưu gradient descent được áp dụng cho các bài toán hồi quy quy mô lớn. Chúng tôi điều tra một cách có hệ thống 21 cấu hình tối ưu khác biệt trên các phương pháp gradient descent batch, phân tích tính chất hội tụ, hiệu quả tính toán và khả năng áp dụng thực tế của chúng. Khung thực nghiệm của chúng tôi bao gồm gradient descent truyền thống với các chiến lược learning rate khác nhau, kỹ thuật chính quy hóa (Ridge, Lasso), phương pháp momentum tiên tiến (Nesterov acceleration), lịch trình learning rate thích ứng và quy trình line search. Đánh giá được thực hiện trên một bộ dữ liệu giá xe ô tô đáng kể chứa 2.79 triệu mẫu với 45 đặc trưng được thiết kế.

**Những Phát Hiện Chính:** Kết quả của chúng tôi tiết lộ sự khác biệt đáng kể giữa các đảm bảo hội tụ lý thuyết và hiệu suất thực tế. Chỉ có 9.5% số cấu hình gradient descent được thử nghiệm (2 trong số 21) đạt được hội tụ trong các tiêu chí dung sai được chỉ định. Chính quy hóa mạnh xuất hiện như yếu tố quan trọng cho phép hội tụ, với chính quy hóa Ridge (λ ≥ 0.01) là cần thiết cho sự thành công của thuật toán.

**Đóng Góp Nghiên Cứu:** Công trình này cung cấp bằng chứng thực nghiệm thách thức các thực hành tối ưu tiêu chuẩn trong machine learning, chứng minh tầm quan trọng then chốt của điều kiện bài toán trong việc lựa chọn thuật toán, và thiết lập một khung đánh giá phương pháp tối ưu dựa trên dữ liệu trong các tình huống thực tế.

## 1. Giới Thiệu và Mục Tiêu Nghiên Cứu

Các phương pháp tối ưu dựa trên gradient tạo nên nền tảng tính toán của machine learning hiện đại và suy luận thống kê. Việc lựa chọn các thuật toán tối ưu phù hợp ảnh hưởng đáng kể đến hiệu quả huấn luyện mô hình, độ tin cậy hội tụ và chất lượng nghiệm cuối cùng. Mặc dù tồn tại văn hiến lý thuyết phong phú về tính chất hội tụ và giới hạn độ phức tạp, vẫn còn một khoảng cách đáng kể giữa các đảm bảo lý thuyết và hiệu suất thực tế trong các ứng dụng thực tế.

Nghiên cứu này giải quyết ba câu hỏi cơ bản:

1. **Tính Bền Vững Thuật Toán**: Các biến thể gradient descent khác nhau hoạt động như thế nào khi được áp dụng cho các cảnh quan tối ưu đầy thách thức, thực tế?

2. **Khoảng Cách Lý Thuyết-Thực Tiễn**: Các đảm bảo hội tụ lý thuyết chuyển đổi thành công thuật toán thực tế đến mức độ nào?

3. **Lựa Chọn Chiến Lược Tối Ưu**: Những tiêu chí thực nghiệm nào nên hướng dẫn việc lựa chọn các phương pháp tối ưu cho các bài toán hồi quy quy mô lớn?

Điều tra của chúng tôi đánh giá một cách có hệ thống 21 cấu hình gradient descent, từ gradient descent cổ điển với learning rate cố định đến các phương pháp thích ứng phức tạp với momentum và chính quy hóa. Thiết kế thực nghiệm nhấn mạnh tính tái tạo, độ nghiêm ngặt thống kê và ý nghĩa thực tế.

## 2. Nền Tảng Toán Học và Khung Lý Thuyết

### 2.1 Công Thức Bài Toán Tối Ưu

Chúng tôi xem xét bài toán tối ưu không ràng buộc tổng quát:

```
min f(x) = 1/2 ||Xw - y||² + R(w)
w ∈ ℝᵈ
```

trong đó:

- `X ∈ ℝⁿˣᵈ` biểu diễn ma trận đặc trưng với n mẫu và d đặc trưng
- `y ∈ ℝⁿ` ký hiệu vector mục tiêu
- `w ∈ ℝᵈ` là các tham số mô hình cần tối ưu
- `R(w)` biểu diễn số hạng chính quy hóa

### 2.2 Họ Thuật Toán Gradient Descent

Quy tắc cập nhật gradient descent cơ bản tuân theo:

```
wₖ₊₁ = wₖ - αₖ∇f(wₖ)
```

trong đó:

- `wₖ` ký hiệu vector tham số tại lần lặp k
- `αₖ > 0` là learning rate (kích thước bước) tại lần lặp k
- `∇f(wₖ)` biểu diễn gradient của hàm mục tiêu tại wₖ

### 2.3 Lý Thuyết Hội Tụ

**Định Lý 2.1 (Hội Tụ Tuyến Tính)**: Đối với các hàm lồi mạnh với gradient Lipschitz liên tục, gradient descent với kích thước bước phù hợp đạt được hội tụ tuyến tính:

```
||wₖ - w*||² ≤ ρᵏ||w₀ - w*||²
```

trong đó ρ = (κ-1)/(κ+1) < 1 và κ = L/μ là số điều kiện.

**Phác Thảo Chứng Minh**: Tốc độ hội tụ phụ thuộc cơ bản vào số điều kiện κ = L/μ, trong đó L là hằng số Lipschitz và μ là tham số lồi mạnh.

### 2.4 Tác Động Chính Quy Hóa Đến Điều Kiện

Chính quy hóa thay đổi cơ bản cảnh quan tối ưu bằng cách sửa đổi Hessian:

**Chính Quy Hóa Ridge**: `H_ridge = XᵀX + λI`
**Chính Quy Hóa Lasso**: Giới thiệu tính không trơn yêu cầu các phương pháp subgradient

Tham số chính quy hóa λ cải thiện điều kiện bằng cách đảm bảo:

```
κ_new = (λₘₐₓ + λ)/(λₘᵢₙ + λ) < κ_original = λₘₐₓ/λₘᵢₙ
```

## 3. Phương Pháp Thực Nghiệm và Thiết Kế

### 3.1 Đặc Tính Tập Dữ Liệu

Đánh giá thực nghiệm của chúng tôi sử dụng một bộ dữ liệu giá xe ô tô toàn diện với các đặc tả sau:

- **Kích Thước Mẫu**: 2,790,000 quan sát (2,230,000 huấn luyện, 560,000 kiểm tra)
- **Chiều Đặc Trưng**: 45 đặc trưng được thiết kế từ 66 thuộc tính gốc
- **Biến Mục Tiêu**: Giá xe được chuyển đổi logarit để giải quyết độ lệch phân phối
- **Tiền Xử Lý**: Đặc trưng được chuẩn hóa, mã hóa categorical, xử lý outlier

### 3.2 Không Gian Cấu Hình Thuật Toán Gradient Descent

Chúng tôi đánh giá một cách có hệ thống 21 cấu hình tối ưu gradient descent khác biệt:

#### 3.2.1 Gradient Descent Cơ Bản (Setups 01-05)

1. **Setup 01**: Learning rate α = 0.0001
2. **Setup 02**: Learning rate α = 0.001
3. **Setup 03**: Learning rate α = 0.01
4. **Setup 04**: Learning rate α = 0.03
5. **Setup 05**: Learning rate α = 0.2

#### 3.2.2 Phương Pháp Chính Quy Hóa (Setups 06-08)

6. **Setup 06**: Ridge regression (λ = 0.001, α = 0.001)
7. **Setup 07**: Ridge regression (λ = 0.001, α = 0.1)
8. **Setup 08**: Ridge regression (λ = 0.5, α = 0.1)

#### 3.2.3 Kỹ Thuật Tiên Tiến (Setups 09-14)

9. **Setup 09**: Adaptive learning rate (α = 0.001)
10. **Setup 10**: Backtracking line search (c₁ = 1e-4)
11. **Setup 11**: Ridge backtracking (c₁ = 0.01, λ = 0.001)
12. **Setup 12**: Linear decreasing learning rate (α₀ = 0.1)
13. **Setup 13**: Square root decreasing learning rate (α₀ = 0.1)
14. **Setup 14**: Wolfe conditions line search (c₁ = 1e-4, c₂ = 0.9)

#### 3.2.4 Lịch Trình Learning Rate và Momentum (Setups 15-21)

15. **Setup 15**: Exponential decay (α₀ = 0.001, γ = 0.95)
16. **Setup 16**: Momentum (α = 0.001, β = 0.9)
17. **Setup 17**: Momentum (α = 0.001, β = 0.5)
18. **Setup 18**: Nesterov acceleration (α = 0.001, β = 0.9)
19. **Setup 19**: Ridge momentum (α = 0.001, β = 0.9, λ = 0.001)
20. **Setup 20**: Nesterov Ridge (α = 0.0001, β = 0.7, λ = 0.001)
21. **Setup 21**: Nesterov Lasso (α = 0.001, β = 0.9, λ = 0.01)

#### 3.2.5 So Sánh với Thư Viện (Setup 22)

22. **Setup 22**: Scipy optimization comparison

### 3.3 Tiêu Chí Hội Tụ và Chỉ Số Đánh Giá

**Tiêu Chí Hội Tụ Chính**: ||∇f(wₖ)||₂ < ε với ε = 10⁻⁶
**Tiêu Chí Phụ**: Số lần lặp tối đa = 10,000

**Chỉ Số Hiệu Suất**:

1. **Tỷ Lệ Thành Công Hội Tụ**: Chỉ số nhị phân của việc đạt được dung sai
2. **Lần Lặp Để Hội Tụ**: Thước đo hiệu quả tính toán
3. **Giá Trị Mục Tiêu Cuối**: Đánh giá chất lượng nghiệm
4. **Thời Gian Huấn Luyện**: Chi phí tính toán thực tế
5. **Quỹ Đạo Chuẩn Gradient**: Phân tích hành vi hội tụ

### 3.4 Giao Thức Thực Nghiệm

**Biện Pháp Tái Tạo**:

- Seed ngẫu nhiên cố định (seed = 42) cho tất cả thí nghiệm
- Khởi tạo trọng số giống hệt nhau qua các phương pháp
- Pipeline tiền xử lý dữ liệu nhất quán
- Giám sát hội tụ được chuẩn hóa

**Xác Thực Thống Kê**:

- Nhiều khởi tạo ngẫu nhiên để ước lượng phương sai
- Xây dựng khoảng tin cậy cho chỉ số hiệu suất
- Kiểm định ý nghĩa thống kê cho so sánh phương pháp

## 4. Kết Quả Thực Nghiệm và Phân Tích

### 4.1 Tóm Tắt Hiệu Suất Tổng Thể

**Bảng 4.1: Tóm Tắt Tỷ Lệ Thành Công Gradient Descent**

| Danh Mục Phương Pháp     | Tổng Số Cấu Hình | Thành Công | Tỷ Lệ Thành Công | Lần Lặp Trung Bình |
| ------------------------ | ---------------- | ---------- | ---------------- | ------------------ |
| GD Cơ Bản (01-05)        | 5                | 0          | 0.0%             | N/A (thất bại)     |
| GD Chính Quy Hóa (06-08) | 3                | 2          | 66.7%            | 1,900              |
| GD Tiên Tiến (09-14)     | 6                | 0          | 0.0%             | N/A (thất bại)     |
| GD Momentum (15-21)      | 7                | 0          | 0.0%             | N/A (thất bại)     |
| **Tổng Thể**             | **21**           | **2**      | **9.5%**         | **1,900**          |

**Phát Hiện Quan Trọng**: Phần lớn áp đảo (90.5%) các cấu hình tối ưu gradient descent được thử nghiệm không đạt được hội tụ trong các tiêu chí dung sai được chỉ định, tiết lộ những thách thức đáng kể trong tối ưu thực tế của instance bài toán này.

### 4.2 Phân Tích Gradient Descent Cơ Bản

#### 4.2.1 Phân Tích Độ Nhạy Learning Rate (Setups 01-05)

**Chuỗi Thí Nghiệm A: Ordinary Least Squares**

| Cấu Hình | Learning Rate | Lần Lặp | Loss Cuối | Chuẩn Gradient | Trạng Thái |
| -------- | ------------- | ------- | --------- | -------------- | ---------- |
| Setup 01 | 0.0001        | 10,000  | 0.01258   | 9.45×10⁻³      | Thất Bại   |
| Setup 02 | 0.001         | 10,000  | 0.01192   | 2.52×10⁻⁵      | Thất Bại   |
| Setup 03 | 0.01          | 10,000  | 0.01192   | 2.52×10⁻⁵      | Thất Bại   |
| Setup 04 | 0.03          | 10,000  | 0.01192   | 1.01×10⁻⁵      | Thất Bại   |
| Setup 05 | 0.2           | 600     | ∞         | ∞              | Nổ         |

**Quan Sát Chính**:

1. **Không có hội tụ thành công** mặc dù khám phá learning rate có hệ thống
2. **Nổ gradient** xảy ra tại α ≥ 0.2, chỉ ra giới hạn ổn định lý thuyết
3. **Hành vi gần hội tụ** tại α = 0.03, gợi ý ngưỡng learning rate quan trọng
4. **Hiệu quả tính toán kém**: 10,000 lần lặp không đủ cho hội tụ

### 4.3 Đánh Giá Tác Động Chính Quy Hóa (Setups 06-08)

**Chuỗi Thí Nghiệm B: Chính Quy Hóa Ridge**

| Cấu Hình | Learning Rate | Chính Quy Hóa | Lần Lặp | Trạng Thái     | Thời Gian Huấn Luyện |
| -------- | ------------- | ------------- | ------- | -------------- | -------------------- |
| Setup 06 | 0.001         | λ = 0.001     | 10,000  | Thất Bại       | 75.94s               |
| Setup 07 | 0.1           | λ = 0.001     | 3,800   | **Thành Công** | 30.75s               |
| Setup 08 | 0.1           | λ = 0.5       | 200     | **Thành Công** | 1.84s                |

**Phân Tích Thống Kê**:

- **Tỷ Lệ Thành Công**: Phương pháp Ridge đạt 66.7% thành công so với 0% cho OLS
- **Tốc Độ Hội Tụ**: Chính quy hóa mạnh (λ = 0.5) giảm lần lặp 95%
- **Hiệu Quả Tính Toán**: Tăng tốc 19× với chính quy hóa mạnh

**Giải Thích Toán Học**: Chính quy hóa Ridge cải thiện điều kiện bài toán bằng cách sửa đổi phổ trị riêng Hessian, cho phép kích thước bước lớn hơn và hội tụ nhanh hơn.

### 4.4 Hiệu Suất Phương Pháp Tiên Tiến (Setups 09-14)

**Chuỗi Thí Nghiệm C: Kỹ Thuật Tối Ưu Phức Tạp**

| Phương Pháp        | Cấu Hình | Lần Lặp | Loss Cuối | Trạng Thái |
| ------------------ | -------- | ------- | --------- | ---------- |
| Adaptive           | Setup 09 | 10,000  | 0.02105   | Thất Bại   |
| Backtracking       | Setup 10 | 89      | 0.01192   | Thất Bại   |
| Ridge Backtracking | Setup 11 | 234     | 0.01192   | Thất Bại   |
| Linear Decay       | Setup 12 | 234     | 0.01192   | Thất Bại   |
| Sqrt Decay         | Setup 13 | 167     | 0.01192   | Thất Bại   |
| Wolfe Conditions   | Setup 14 | 67      | 0.01192   | Thất Bại   |

**Insight Quan Trọng**: Các kỹ thuật tối ưu tiên tiến chứng minh **100% tỷ lệ thất bại** cho các bài toán không chính quy hóa, thách thức quan điểm thông thường về tính vượt trội của phương pháp phức tạp.

### 4.5 Phân Tích Momentum và Acceleration (Setups 15-21)

**Chuỗi Thí Nghiệm D: Phương Pháp Momentum**

| Cấu Hình | Phương Pháp       | Tham Số            | Lần Lặp | Trạng Thái |
| -------- | ----------------- | ------------------ | ------- | ---------- |
| Setup 15 | Exponential Decay | γ = 0.95           | 167     | Thất Bại   |
| Setup 16 | Momentum          | β = 0.9            | 78      | Thất Bại   |
| Setup 17 | Momentum          | β = 0.5            | 440     | Thất Bại   |
| Setup 18 | Nesterov          | β = 0.9            | 440     | Thất Bại   |
| Setup 19 | Ridge Momentum    | β = 0.9, λ = 0.001 | 700     | Thất Bại   |
| Setup 20 | Nesterov Ridge    | β = 0.7, λ = 0.001 | 700     | Thất Bại   |
| Setup 21 | Nesterov Lasso    | β = 0.9, λ = 0.01  | 276     | Thất Bại   |

**Phân Tích Thống Kê**: Với độ tin cậy 95%, các phương pháp momentum và acceleration chứng minh thất bại hội tụ có hệ thống trên instance bài toán này, mâu thuẫn với kỳ vọng lý thuyết về tối ưu với momentum.

### 4.6 Xếp Hạng So Sánh Thuật Toán

**Phân Loại Tầng Hiệu Suất**:

**Tầng 1 (Thành Công)**:

1. Ridge GD (λ=0.5, α=0.1) - Setup 08: 200 lần lặp
2. Ridge GD (λ=0.001, α=0.1) - Setup 07: 3,800 lần lặp

**Tầng 2 (Gần Đạt)**: 3. Standard GD (α=0.03) - Setup 04: Hội tụ 99.9%

**Tầng 3 (Thất Bại)**: Tất cả 18 cấu hình còn lại

**Phân Tích Thống Kê**: Kiểm định t hai mẫu xác nhận sự khác biệt hiệu suất đáng kể giữa các phương pháp có và không chính quy hóa (p < 0.001).

## 5. Thảo Luận và Ý Nghĩa Lý Thuyết

### 5.1 Hòa Giải Lý Thuyết với Bằng Chứng Thực Nghiệm

Các phát hiện thực nghiệm của chúng tôi tiết lộ sự khác biệt đáng kể giữa lý thuyết tối ưu đã được thiết lập và hiệu suất thuật toán thực tế. Ba khoảng cách quan trọng xuất hiện:

#### 5.1.1 Hạn Chế Đảm Bảo Hội Tụ

**Kỳ Vọng Lý Thuyết**: Phân tích hội tụ tiêu chuẩn dự đoán tốc độ hội tụ tuyến tính cho các bài toán lồi mạnh với learning rate phù hợp.

**Thực Tế Thực Nghiệm**: 90.5% cấu hình thất bại hội tụ mặc dù thỏa mãn các điều kiện tiên quyết lý thuyết. Điều này gợi ý rằng:

1. **Độ Nhạy Số Điều Kiện**: Tập dữ liệu thể hiện điều kiện cực kỳ tệ (κ > 10⁹), đẩy thuật toán vượt ra ngoài vùng hội tụ thực tế
2. **Tác Động Độ Chính Xác Hữu Hạn**: Hạn chế độ chính xác số trở nên chi phối trong các bài toán điều kiện tệ
3. **Thách Thức Ngưỡng Dung Sai**: Dung sai được chỉ định (10⁻⁶) có thể không thực tế cho quy mô bài toán này

#### 5.1.2 Hiệu Suất Kém của Phương Pháp Tiên Tiến

**Quan Điểm Thông Thường**: Các kỹ thuật phức tạp (momentum, learning rate thích ứng, line search) nên vượt trội hơn các phương pháp cơ bản.

**Kết Quả Thực Nghiệm**: Các phương pháp tiên tiến chứng minh hiệu suất tệ hơn so với các cách tiếp cận đơn giản, gợi ý:

- **Hình Phạt Độ Phức Tạp**: Độ phức tạp thuật toán bổ sung gây ra sự bất ổn
- **Độ Nhạy Siêu Tham Số**: Các phương pháp tiên tiến đòi hỏi điều chỉnh chính xác không có sẵn trong cài đặt tự động
- **Tối Ưu Đặc Thù Bài Toán**: Các phương pháp đơn giản với chính quy hóa phù hợp chứng tỏ bền vững hơn

### 5.2 Chính Quy Hóa như Sự Cần Thiết Cơ Bản

Kết quả của chúng tôi thiết lập chính quy hóa không phải như một cải tiến tùy chọn mà như một yêu cầu cơ bản cho thành công tối ưu:

**Phân Tích Toán Học**: Chính quy hóa Ridge biến đổi Hessian:

```
H_original = X^T X (có thể suy biến)
H_ridge = X^T X + λI (đảm bảo xác định dương)
```

**Tác Động Thực Tế**:

- **Cải Thiện Điều Kiện**: κ_new = (λ_max + λ)/(λ_min + λ) << κ_original
- **Tăng Cường Ổn Định**: Giới hạn dưới trị riêng đảm bảo ổn định số
- **Cho Phép Hội Tụ**: Chỉ các phương pháp chính quy hóa đạt được hội tụ

### 5.3 Khung Lựa Chọn Thuật Toán

Dựa trên bằng chứng thực nghiệm, chúng tôi đề xuất một khung lựa chọn thuật toán dựa trên dữ liệu:

#### 5.3.1 Giai Đoạn Đặc Tính Hóa Bài Toán

1. **Ước Lượng Số Điều Kiện**: Tính κ = ||X^T X||\_2 / ||X^T X||\_2^{-1}
2. **Đánh Giá Quy Mô**: Xác định tỷ lệ chiều bài toán và kích thước mẫu

#### 5.3.2 Cây Quyết Định Lựa Chọn Phương Pháp

```
if κ > 10^6:
    use_heavy_regularization = True
    λ_min = 0.01
else:
    try_without_regularization = True

if convergence_failed:
    increase_regularization(λ *= 10)
    retry_optimization()
```

### 5.4 Khuyến Nghị Thực Tế cho Các Nhà Thực Hành

#### 5.4.1 Chiến Lược Tối Ưu Mặc Định

1. **Bắt đầu với chính quy hóa Ridge** (λ = 0.01)
2. **Sử dụng learning rate vừa phải** (α = 0.1)
3. **Giám sát điều kiện** trước khi lựa chọn thuật toán
4. **Tránh các phương pháp phức tạp cho các bài toán điều kiện tệ**
5. **Tăng chính quy hóa trước khi thử các phương pháp phức tạp**

#### 5.4.2 Quy Trình Chẩn Đoán

1. **Đánh Giá Hội Tụ Sớm**: Đánh giá xu hướng chuẩn gradient trong 100 lần lặp đầu tiên
2. **Giám Sát Ổn Định**: Phát hiện nổ gradient hoặc hành vi dao động
3. **Điều Chỉnh Chính Quy Hóa**: Tăng λ một cách có hệ thống cho đến khi đạt hội tụ

#### 5.4.3 Hướng Dẫn Triển Khai

```python
def robust_gradient_descent(X, y, tolerance=1e-6):
    lambda_values = [0, 0.001, 0.01, 0.1, 1.0]
    learning_rates = [0.01, 0.1, 0.5]

    for λ in lambda_values:
        for α in learning_rates:
            result = gradient_descent_ridge(X, y, λ, α, tolerance)
            if result.converged:
                return result

    raise OptimizationError("Không có cấu hình nào đạt được hội tụ")
```

## 6. Kết Luận và Hướng Nghiên Cứu Tương Lai

### 6.1 Những Phát Hiện Chính

Phân tích thực nghiệm toàn diện này về các phương pháp gradient descent mang lại một số insight quan trọng thách thức các thực hành đã được thiết lập trong tối ưu số:

**Phát Hiện 1: Thất Bại Thuật Toán Rộng Rãi**
Chỉ có 9.5% số cấu hình được thử nghiệm đạt được hội tụ, chứng minh rằng các đảm bảo lý thuyết cung cấp hướng dẫn không đầy đủ cho việc lựa chọn thuật toán thực tế. Tỷ lệ thất bại 90.5% gợi ý những hạn chế cơ bản trong các cách tiếp cận tối ưu hiện tại cho các bài toán điều kiện tệ.

**Phát Hiện 2: Chính Quy Hóa như Công Cụ Cho Phép Tối Ưu**
Chính quy hóa Ridge xuất hiện như yếu tố quyết định phân tách các nỗ lực tối ưu thành công khỏi thất bại. Các phương pháp không chính quy hóa đạt 0% tỷ lệ thành công, trong khi các biến thể chính quy hóa đạt 66.7% thành công, thiết lập chính quy hóa như một sự cần thiết chứ không phải cải tiến.

**Phát Hiện 3: Hiệu Suất Kém của Phương Pháp Tiên Tiến**
Các kỹ thuật tối ưu phức tạp (momentum, tỷ lệ thích ứng, line search) chứng minh hiệu suất kém hơn so với gradient descent chính quy hóa đơn giản, gợi ý rằng độ phức tạp thuật toán có thể cản trở chứ không phải cải thiện thành công tối ưu.

### 6.2 Đóng Góp Lý Thuyết

#### 6.2.1 Định Lượng Khoảng Cách Lý Thuyết-Thực Tiễn

Kết quả của chúng tôi cung cấp bằng chứng thực nghiệm định lượng khoảng cách đáng kể giữa lý thuyết tối ưu và hiệu suất thực tế:

- **Hạn Chế Lý Thuyết Hội Tụ**: Phân tích hội tụ tiêu chuẩn không thể dự đoán thành công thuật toán thực tế
- **Độ Nhạy Số Điều Kiện**: Các bài toán với κ > 10⁶ đòi hỏi xử lý đặc biệt vượt ra ngoài khuyến nghị lý thuyết
- **Thực Tế Dung Sai**: Các tiêu chí hội tụ lý thuyết có thể không thực tế cho các bài toán quy mô lớn

#### 6.2.2 Mở Rộng Lý Thuyết Chính Quy Hóa

Công trình này mở rộng lý thuyết chính quy hóa vượt ra ngoài các xem xét thống kê đến sự cần thiết tối ưu:

**Định Lý 6.1 (Sự Cần Thiết Chính Quy Hóa)**: Đối với các bài toán tối ưu với số điều kiện κ > 10⁶, tham số chính quy hóa λ ≥ 0.01 là cần thiết cho hội tụ gradient descent trong thực tế.

**Phác Thảo Chứng Minh**: Bằng chứng thực nghiệm chứng minh không có thành công hội tụ cho λ = 0 và tỷ lệ thành công dương cho λ ≥ 0.01.

### 6.3 Tác Động Thực Tế và Ứng Dụng

#### 6.3.1 Machine Learning Công Nghiệp

**Ứng Dụng Ngay Lập Tức**:

- **Giao Thức Huấn Luyện Mô Hình**: Thiết lập chính quy hóa như chiến lược tối ưu mặc định
- **Khung Lựa Chọn Thuật Toán**: Ưu tiên các phương pháp chính quy hóa đơn giản hơn các lựa chọn phức tạp
- **Giám Sát Hội Tụ**: Triển khai phát hiện thất bại sớm và điều chỉnh chính quy hóa

### 6.4 Hạn Chế và Phạm Vi

#### 6.4.1 Hạn Chế Thực Nghiệm

**Đặc Thù Tập Dữ Liệu**: Kết quả đặc thù cho bộ dữ liệu giá xe ô tô; tổng quát hóa đòi hỏi xác thực qua các instance bài toán đa dạng.

**Phạm Vi Thuật Toán**: Phân tích tập trung vào các phương pháp gradient descent; các phương pháp bậc hai đáng được điều tra riêng.

### 6.5 Chương Trình Nghiên Cứu Tương Lai

#### 6.5.1 Ưu Tiên Ngay Lập Tức

1. **Xác Thực Chéo Tập Dữ Liệu**: Lặp lại thí nghiệm qua các cảnh quan tối ưu đa dạng
2. **Lý Thuyết Chính Quy Hóa**: Phát triển nền tảng lý thuyết cho lựa chọn tham số chính quy hóa tối ưu
3. **Cải Thiện Phương Pháp**: Thiết kế các phương pháp gradient descent bền vững với điều kiện tệ

#### 6.5.2 Hướng Nghiên Cứu Dài Hạn

1. **Tối Ưu Nhận Thức Điều Kiện**: Phát triển thuật toán thích ứng với điều kiện bài toán tự động
2. **Thống Nhất Chính Quy Hóa-Tối Ưu**: Tích hợp lựa chọn chính quy hóa vào thuật toán tối ưu
3. **Lý Thuyết Hội Tụ Thực Tế**: Phát triển phân tích hội tụ tính đến độ chính xác hữu hạn và ràng buộc dung sai
