# Dự Án Tối Ưu Thuật Toán - Dự Đoán Giá Xe Cũ

**Framework Python đơn giản để so sánh các thuật toán tối ưu trên bài toán dự đoán giá xe cũ.**

## 🚀 Khởi Chạy Nhanh

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy từng bước
python src/01_eda.py
python src/02_preprocessing.py
python src/algorithms/gradient_descent/standard_setup.py
python src/algorithms/algorithm_comparator.py --list
```

## 📁 Cấu Trúc Dự Án

```
data/                           # Dữ liệu theo workflow số
├── 00_raw/                    # Dữ liệu gốc
├── 01_eda/                    # Kết quả phân tích dữ liệu  
├── 02_processed/              # Dữ liệu đã xử lý
├── 03_algorithms/             # Kết quả các thuật toán
│   ├── gradient_descent/
│   ├── newton_method/
│   ├── ridge_regression/
│   └── stochastic_gd/
└── 04_comparison/             # So sánh cuối cùng

src/                           # Code theo workflow số
├── 01_eda.py                 # Phân tích dữ liệu
├── 02_preprocessing.py       # Xử lý dữ liệu
├── algorithms/               # Các thuật toán
│   ├── gradient_descent/     # Gradient Descent với nhiều setup
│   ├── newton_method/        # Newton Method
│   ├── ridge_regression/     # Ridge Regression
│   ├── stochastic_gd/        # Stochastic GD
│   ├── advanced_methods/     # Các phương pháp nâng cao
│   └── algorithm_comparator.py  # Tool so sánh tổng hợp
└── utils/                    # Tiện ích chung
```

## 🎯 Các Thuật Toán Có Sẵn

### **Gradient Descent**
- `standard_setup.py` - Learning rate 0.01, ổn định
- `fast_setup.py` - Learning rate 0.1, nhanh
- `precise_setup.py` - Learning rate 0.001, chính xác
- `medium_setup.py` - Learning rate 0.05, cân bằng
- `slow_setup.py` - Learning rate 0.005, từ từ

### **Newton Method**
- `standard_setup.py` - Setup chuẩn với regularization

### **Proximal GD** 
- `standard_setup.py` - Cho L1 regularization (Lasso)

### **Subgradient Methods**
- `standard_setup.py` - Cho non-smooth optimization

### **Ridge Regression**
- Regularized linear regression

### **Stochastic GD**
- Online learning với mini-batches

### **Advanced Methods**
- Adam, RMSprop, BFGS và các phương pháp nâng cao

## 🔄 Cách Sử Dụng

### **Bước 1: Phân Tích Dữ Liệu**
```bash
python src/01_eda.py
```
- **Input**: `data/00_raw/used_cars_data.csv`
- **Output**: `data/01_eda/` (biểu đồ, phân tích correlation, thống kê)

### **Bước 2: Xử Lý Dữ Liệu**
```bash
python src/02_preprocessing.py
```
- **Input**: Dữ liệu gốc
- **Output**: `data/02_processed/` (train/test data đã clean)

### **Bước 3: Chạy Thuật Toán**
```bash
# Chạy từng thuật toán
python src/algorithms/gradient_descent/standard_setup.py
python src/algorithms/newton_method/standard_setup.py

# Hoặc chạy nhiều setup gradient descent
python src/algorithms/gradient_descent/fast_setup.py
python src/algorithms/gradient_descent/precise_setup.py
```

### **Bước 4: So Sánh Kết Quả**
```bash
# Xem các kết quả có sẵn
python src/algorithms/algorithm_comparator.py --list

# So sánh 2 thuật toán cụ thể
python src/algorithms/algorithm_comparator.py compare gradient_descent/standard newton_method/standard

# So sánh tất cả setup của gradient descent
python src/algorithms/algorithm_comparator.py analyze gradient_descent

# Chế độ interactive
python src/algorithms/algorithm_comparator.py --interactive

# Tạo báo cáo toàn diện
python src/algorithms/algorithm_comparator.py report --all
```

## 📊 Tính Năng So Sánh

### **Algorithm Comparator Tool**
File `algorithm_comparator.py` tích hợp tất cả tính năng:

- ✅ **Load kết quả** từ `/data/03_algorithms/` 
- ✅ **So sánh metrics** (MSE, R², thời gian training)
- ✅ **Visualization** (6 loại biểu đồ)
- ✅ **Convergence analysis** từ training history
- ✅ **Radar charts** cho so sánh đa chiều
- ✅ **Interactive selection** mode
- ✅ **Báo cáo comprehensive** 

### **Metrics Được So Sánh**
- **Test MSE** - Mean Squared Error (càng nhỏ càng tốt)
- **R² Score** - Coefficient of determination (càng gần 1 càng tốt)  
- **Training Time** - Thời gian training (giây)
- **Convergence** - Số iterations để hội tụ
- **MAPE** - Mean Absolute Percentage Error

## 🎨 Visualization

Mỗi thuật toán tạo ra:
- **Training curves** - Đường hội tụ của cost function
- **Predictions vs Actual** - So sánh dự đoán với thực tế
- **Residual plots** - Phân tích sai số
- **Performance comparison** - So sánh giữa các thuật toán

## 📈 Kết Quả Mỗi Thuật Toán

Mỗi setup tạo folder riêng chứa:
```
data/03_algorithms/gradient_descent/standard_setup/
├── results.json           # Metrics và metadata
├── training_history.csv   # Lịch sử training
├── weights.npy           # Weights đã train
└── standard_setup_results.png  # Visualization
```

## 💡 Hướng Dẫn Thêm Thuật Toán Mới

1. **Copy setup có sẵn:**
```bash
cp src/algorithms/gradient_descent/standard_setup.py src/algorithms/my_algorithm/my_setup.py
```

2. **Sửa implementation trong file mới**

3. **Chạy và test:**
```bash
python src/algorithms/my_algorithm/my_setup.py
```

4. **So sánh với tool:**
```bash
python src/algorithms/algorithm_comparator.py compare gradient_descent/standard my_algorithm/my_setup
```

## 🔧 Development

### **Requirements Chính**
- Python 3.8+
- numpy, pandas, matplotlib, seaborn
- sklearn (cho comparison)
- pathlib, json (built-in)

### **Thiết Kế Đơn Giản**
- **Workflow số** - Dễ theo dõi pipeline
- **Scripts độc lập** - Mỗi bước chạy riêng được
- **No complex classes** - Chỉ functions đơn giản
- **Standardized output** - Format nhất quán để so sánh

### **Memory Optimized**
- Chunked data loading
- Efficient numpy operations
- Reasonable memory footprint

## 📋 Dataset: Xe Cũ Mỹ

**3 triệu records** từ CarGurus.com với 66 features:
- **Giá xe** (target variable)
- **Thông số kỹ thuật** (engine, transmission, fuel economy)
- **Tình trạng** (mileage, accidents, owner count)
- **Thông tin địa lý** (city, dealer info)

## 🎯 Mục Đích Dự Án

- **Học thuật toán tối ưu** qua ví dụ thực tế
- **So sánh performance** các phương pháp khác nhau  
- **Hiểu trade-offs** giữa speed, accuracy, complexity
- **Thực hành** implementation từ scratch vs libraries

---

**Dự án này tập trung vào việc hiểu rõ các thuật toán tối ưu qua bài toán dự đoán giá xe cũ đơn giản và dễ theo dõi.**