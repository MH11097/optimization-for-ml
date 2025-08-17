# Thuật Toán Tối Ưu - Thử Nghiệm Các Setup

Mỗi thuật toán được tổ chức trong folder riêng với nhiều setup khác nhau để bạn thử nghiệm và so sánh.

## 📁 Cấu Trúc Thư Mục

```
03_algorithms/
├── gradient_descent/           # Gradient Descent với các setup
│   ├── standard_setup.py      # Setup chuẩn (lr=0.01)
│   ├── fast_setup.py          # Setup nhanh (lr=0.1)
│   └── precise_setup.py       # Setup chính xác (lr=0.001)
├── newton_method/             # Newton Method với các setup
├── stochastic_gd/             # SGD với các setup
├── ridge_regression/          # Ridge với các setup
└── advanced_methods/          # Các phương pháp nâng cao
```

## 🚀 Cách Sử Dụng

### Bước 1: Đảm bảo dữ liệu đã được xử lý
```bash
python src/02_preprocessing.py
```

### Bước 2: Chọn thuật toán và setup muốn thử
```bash
# Gradient Descent - Setup chuẩn
python src/03_algorithms/gradient_descent/standard_setup.py

# Gradient Descent - Setup nhanh
python src/03_algorithms/gradient_descent/fast_setup.py

# Gradient Descent - Setup chính xác
python src/03_algorithms/gradient_descent/precise_setup.py
```

### Bước 3: Xem kết quả
Mỗi setup sẽ tạo folder riêng trong `data/03_algorithms/` với:
- Kết quả đánh giá (JSON)
- Biểu đồ phân tích
- Lịch sử training
- Weights đã học

## 📊 So Sánh Các Setup

### Gradient Descent

| Setup | Learning Rate | Đặc điểm | Khi nào dùng |
|-------|---------------|----------|--------------|
| **Standard** | 0.01 | Ổn định, an toàn | Học tập, sản xuất |
| **Fast** | 0.1 | Nhanh, có thể dao động | Thử nghiệm nhanh |
| **Precise** | 0.001 | Chậm, rất chính xác | Nghiên cứu, precision cao |

### Cách Chọn Setup

**🎯 Cho người mới học:**
- Bắt đầu với `standard_setup.py`
- Hiểu được cách hoạt động cơ bản
- Ít rủi ro, kết quả ổn định

**⚡ Khi cần kết quả nhanh:**
- Dùng `fast_setup.py`
- Chấp nhận một ít trade-off về stability
- Tốt cho prototyping

**🎯 Khi cần precision tối đa:**
- Dùng `precise_setup.py`
- Có thời gian training lâu
- Ứng dụng production quan trọng

## 🔍 Phân Tích Kết Quả

Mỗi setup sẽ cho bạn:

### 1. Metrics Cơ Bản
- **MSE**: Mean Squared Error (càng nhỏ càng tốt)
- **R²**: R-squared (càng gần 1 càng tốt)
- **Training Time**: Thời gian training

### 2. Phân Tích Chi Tiết
- **Convergence curves**: Xem thuật toán hội tụ như thế nào
- **Gradient norms**: Theo dõi gradient giảm
- **Predictions vs Actual**: Xem độ chính xác
- **Residuals**: Phân tích lỗi

### 3. Đặc Điểm Setup
- **Pros/Cons**: Ưu nhược điểm
- **Recommendations**: Khi nào nên dùng
- **Stability analysis**: Độ ổn định

## 🧪 Thử Nghiệm Tự Do

### Experiment Workflow
1. **Chạy tất cả setup của 1 thuật toán**
2. **So sánh metrics và visualization**
3. **Chọn setup phù hợp với mục tiêu**
4. **Ghi chú lại insights**

### Ví dụ thử nghiệm:
```bash
# Thử tất cả Gradient Descent setups
python src/03_algorithms/gradient_descent/standard_setup.py
python src/03_algorithms/gradient_descent/fast_setup.py
python src/03_algorithms/gradient_descent/precise_setup.py

# So sánh kết quả trong data/03_algorithms/gradient_descent/
```

## 📝 Ghi Chú Thử Nghiệm

Khi thử nghiệm, hãy ghi chú:

### Quan sát Quan Trọng
- Setup nào cho kết quả tốt nhất?
- Trade-off giữa speed vs accuracy?
- Stability của từng setup?
- Phù hợp với mục tiêu của bạn?

### Template Ghi Chú
```
=== THỬ NGHIỆM GRADIENT DESCENT ===
Date: [ngày]
Dataset: [tên dataset]

Results:
- Standard Setup: MSE = [x], Time = [y]s
- Fast Setup: MSE = [x], Time = [y]s  
- Precise Setup: MSE = [x], Time = [y]s

Best Setup: [tên setup]
Reason: [lý do tại sao tốt nhất]
Notes: [quan sát khác]
```

## 🎓 Học Hỏi Từ Experiments

### Insights Quan Trọng
1. **Learning Rate Impact**: Xem ảnh hưởng của LR
2. **Convergence Patterns**: Hiểu cách thuật toán hội tụ
3. **Speed vs Accuracy**: Trade-off quan trọng
4. **Stability**: Khi nào setup ổn định

### Câu Hỏi Thú Vị
- Setup nào tốt nhất cho dataset này?
- Tại sao fast setup lại nhanh hơn?
- Khi nào precise setup không cần thiết?
- Làm thế nào để tune parameters tốt hơn?

## 🔧 Customization

Bạn có thể modify parameters trong từng file:
- Thay đổi learning rate
- Điều chỉnh max iterations  
- Thử tolerance khác
- Thêm analysis mới

Ví dụ:
```python
# Trong standard_setup.py, thay đổi:
learning_rate = 0.05  # Thay vì 0.01
max_iterations = 1500  # Thay vì 1000
```

Chúc bạn thử nghiệm vui vẻ! 🚀