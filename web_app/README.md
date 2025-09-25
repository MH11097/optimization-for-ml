# Optimization Algorithm Visualizer

Ứng dụng web tương tác để trực quan hóa các thuật toán tối ưu hóa với các tham số khác nhau.

## Tính năng

- **Chọn thuật toán**: Hỗ trợ gradient descent, Newton method, quasi-Newton, stochastic GD, và subgradient
- **Thanh trượt tương tác**: Điều chỉnh các tham số như learning rate, regularization, momentum
- **Biểu đồ 2D**:
  - Đồ thị hội tụ loss
  - Biểu đồ gradient norm
  - So sánh hiệu suất
- **Biểu đồ 3D**:
  - Không gian tham số
  - Bề mặt hội tụ
- **Chế độ so sánh**: So sánh nhiều setup cùng lúc

## Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
cd path/to/optimization-for-ml
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
cd web_app
python run_app.py
```

Hoặc chạy trực tiếp:

```bash
python app.py
```

### 3. Truy cập ứng dụng

Mở trình duyệt và truy cập: http://localhost:5000

## Cấu trúc dữ liệu

Ứng dụng đọc dữ liệu từ thư mục `data/03_algorithms/` với cấu trúc:

```
data/03_algorithms/
├── gradient_descent/
│   ├── 01a_setup_gd_ols_lr_0001/
│   │   ├── results.json
│   │   ├── training_history.csv
│   │   └── ...
│   └── ...
├── newton_method/
├── quasi_newton/
├── stochastic_gd/
└── subgradient/
```

Mỗi setup cần có:

- `results.json`: Kết quả training và metadata
- `training_history.csv`: Lịch sử training theo iteration

## Sử dụng

### 1. Chọn thuật toán

- Chọn thuật toán từ dropdown "Algorithm"
- Hệ thống sẽ tự động load các setup có sẵn

### 2. Chọn nhóm tham số

- Chọn nhóm tham số từ dropdown "Parameter Group"
- Ví dụ: Fixed LR, Scheduled LR, Momentum, etc.

### 3. Điều chỉnh tham số

- Sử dụng các thanh trượt để thay đổi giá trị tham số
- Biểu đồ sẽ cập nhật tự động

### 4. Chuyển đổi chế độ hiển thị

- **2D Charts**: Biểu đồ đường cơ bản
- **3D Charts**: Trực quan hóa không gian 3D

### 5. So sánh setup

- Bật "Comparison Mode"
- Thêm các setup vào danh sách so sánh
- Xem biểu đồ so sánh nhiều setup

## API Endpoints

- `GET /api/algorithms` - Danh sách thuật toán
- `GET /api/algorithms/{algorithm}/setups` - Setups của thuật toán
- `GET /api/algorithms/{algorithm}/parameter-ranges` - Phạm vi tham số
- `GET /api/algorithms/{algorithm}/grouped-setups` - Setups theo nhóm
- `GET /api/algorithms/{algorithm}/setup-by-params` - Tìm setup theo tham số
- `GET /api/setup/{path}/history` - Lịch sử training
- `GET /api/comparison` - So sánh nhiều setup

## Cấu trúc files

```
web_app/
├── app.py                  # Flask application chính
├── data_loader.py          # Module load và parse dữ liệu
├── run_app.py             # Script khởi động
├── requirements_web.txt    # Dependencies bổ sung
├── templates/
│   ├── base.html          # Template cơ sở
│   └── index.html         # Trang chính
├── static/
│   ├── css/
│   │   └── style.css      # CSS styling
│   └── js/
│       └── app.js         # JavaScript logic
└── README.md              # Tài liệu này
```

## Troubleshooting

### Lỗi "No algorithms found"

- Kiểm tra thư mục `data/03_algorithms/` có tồn tại
- Đảm bảo có ít nhất một thư mục thuật toán

### Lỗi "No setup found matching parameters"

- Các tham số có thể không khớp chính xác
- Kiểm tra file `results.json` có đúng format

### Charts không hiển thị

- Kiểm tra file `training_history.csv` có tồn tại
- Đảm bảo có cột `iteration`, `loss`, `gradient_norm`

## Tùy chỉnh

### Thêm thuật toán mới

1. Tạo thư mục trong `data/03_algorithms/`
2. Đảm bảo mỗi setup có `results.json` và `training_history.csv`
3. Ứng dụng sẽ tự động detect

### Thêm tham số mới

1. Cập nhật `_parse_setup_name()` trong `data_loader.py`
2. Thêm pattern regex cho tham số mới

### Tùy chỉnh biểu đồ

1. Chỉnh sửa functions trong `app.js`
2. Sử dụng Plotly.js API để tạo biểu đồ mới
