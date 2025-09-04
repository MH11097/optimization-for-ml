# Model Usage Guide

## Tổng quan

Đã tái cấu trúc tất cả thuật toán optimization thành **model classes độc lập** với **setup scripts** riêng biệt. Mỗi thuật toán được tổ chức trong folder riêng với cấu trúc nhất quán.

## Cấu trúc mới

```
src/algorithms/
├── gradient_descent/
│   ├── gradient_descent_model.py        # Class GradientDescentModel
│   ├── momentum_gd_model.py             # Class MomentumGDModel
│   ├── setup_gd_ols_lr_01.py           # Fast OLS (lr=0.1)
│   ├── setup_gd_ols_lr_001.py          # Slow OLS (lr=0.01)
│   └── setup_gd_ridge_lr_001.py        # Ridge regression
├── newton_method/
│   ├── newton_model.py                  # Class NewtonModel
│   ├── damped_newton_model.py           # Class DampedNewtonModel
│   ├── setup_newton_ols_pure.py        # Pure Newton
│   └── setup_newton_ols_damped.py      # Damped Newton với line search
├── stochastic_gd/
│   ├── sgd_model.py                     # Class SGDModel
│   ├── setup_sgd_batch1.py             # SGD batch_size=1
│   └── setup_sgd_batch32.py            # Mini-batch SGD batch_size=32
├── quasi_newton/
│   ├── quasi_newton_model.py            # Class QuasiNewtonModel (BFGS)
│   └── setup_bfgs_ols.py               # BFGS for OLS
├── proximal_gd/
│   ├── proximal_gd_model.py             # Class ProximalGDModel
│   ├── setup_proximal_lasso.py         # Proximal GD for Lasso
│   └── setup_proximal_elastic.py       # Proximal GD for Elastic Net
└── algorithm_comparator.py              # Tool so sánh tất cả algorithms
```

## Cách sử dụng Model Classes

### 1. Gradient Descent

```python
from src.algorithms.gradient_descent.gradient_descent_model import GradientDescentModel

# Khởi tạo model
model = GradientDescentModel(
    ham_loss='ols',           # 'ols', 'ridge', 'lasso'
    learning_rate=0.01,
    so_lan_thu=10000,
    diem_dung=1e-6,
    regularization=0.01       # cho Ridge/Lasso
)

# Huấn luyện
results = model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá
metrics = model.evaluate(X_test, y_test)

# Lưu kết quả
model.save_results('my_experiment')

# Tạo biểu đồ
model.plot_results(X_test, y_test, 'my_experiment')
```

### 2. Newton Method

```python
from src.algorithms.newton_method.newton_model import NewtonModel

# Pure Newton Method
model = NewtonModel(
    ham_loss='ols',
    regularization=0.0,       # Không regularization cho OLS
    so_lan_thu=50,
    diem_dung=1e-10,
    numerical_regularization=1e-8  # Chỉ cho numerical stability
)

# Damped Newton Method
from src.algorithms.newton_method.damped_newton_model import DampedNewtonModel

model = DampedNewtonModel(
    ham_loss='ols',
    so_lan_thu=10000,
    diem_dung=1e-8,
    armijo_c1=1e-4,
    backtrack_rho=0.8
)
```

### 3. Stochastic Gradient Descent

```python
from src.algorithms.stochastic_gd.sgd_model import SGDModel

# Pure SGD
model = SGDModel(
    learning_rate=0.01,
    so_epochs=100,
    batch_size=1,         # Pure SGD
    random_state=42
)

# Mini-batch SGD
model = SGDModel(
    learning_rate=0.01,
    so_epochs=100,
    batch_size=32,        # Mini-batch
    random_state=42
)
```

## Cách chạy Experiments

### Chạy setup script đơn lẻ:

```bash
# Gradient Descent
python src/algorithms/gradient_descent/setup_ols_01.py
python src/algorithms/gradient_descent/setup_ridge_001.py

# Newton Method
python src/algorithms/newton_method/setup_pure_newton_ols.py
python src/algorithms/newton_method/setup_damped_newton_ols.py

# SGD
python src/algorithms/stochastic_gd/setup_original_sgd.py
python src/algorithms/stochastic_gd/setup_batch_32_sgd.py
```

### So sánh và phân tích tất cả algorithms:

```bash
# Chạy tool so sánh tổng hợp (thay thế cho run_experiments.py)
python src/algorithms/algorithm_comparator.py
```

Tool này sẽ:

- Thu thập kết quả từ tất cả experiments đã chạy
- Tạo bảng so sánh chi tiết
- Tạo các biểu đồ phân tích performance
- Tạo báo cáo HTML tổng hợp
- Lưu kết quả vào `data/04_comparison/`

## Đặc điểm chính

### 1. Interface nhất quán

Tất cả model classes đều có:

- `__init__()`: Khởi tạo với tham số
- `fit()`: Huấn luyện model
- `predict()`: Dự đoán
- `evaluate()`: Đánh giá trên test set
- `save_results()`: Lưu kết quả với tên file
- `plot_results()`: Tạo visualization với tên file

### 2. Tham số linh hoạt

- Mỗi model có tham số phù hợp với thuật toán
- Hỗ trợ nhiều loss functions (OLS, Ridge, Lasso)
- Tham số `ten_file` để tổ chức output

### 3. Kết quả được tổ chức

```
data/03_algorithms/[algorithm]/[ten_file]/
├── results.json           # Thông tin chi tiết
├── training_history.csv   # Lịch sử training
├── convergence_analysis.png
├── predictions_vs_actual.png
└── optimization_trajectory.png (nếu có)
```

### 4. So sánh và phân tích tổng hợp

- Tool `algorithm_comparator.py` phân tích tất cả algorithms
- Tạo báo cáo HTML với visualizations
- Export CSV để phân tích thêm
- So sánh performance, convergence, training time

## Lợi ích

1. **Đơn giản**: Chỉ cần khởi tạo class với tham số và gọi `fit()`
2. **Linh hoạt**: Dễ thay đổi tham số, thêm experiments mới
3. **Tổ chức**: Mỗi thuật toán/experiment có folder riêng
4. **Nhất quán**: Interface giống nhau cho tất cả algorithms
5. **Tái sử dụng**: Utils dùng chung, code không trùng lặp

## Test

Chạy test để verify các model hoạt động đúng:

```bash
python test_models_simple.py
```

## Ví dụ hoàn chỉnh

```python
# Import
from src.algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
from src.utils.data_process_utils import load_du_lieu

# Load data
X_train, X_test, y_train, y_test = load_du_lieu()

# Khởi tạo và huấn luyện nhiều models
models = {
    'fast_gd': GradientDescentModel(ham_loss='ols', learning_rate=0.1),
    'slow_gd': GradientDescentModel(ham_loss='ols', learning_rate=0.01),
    'ridge_gd': GradientDescentModel(ham_loss='ridge', learning_rate=0.01, regularization=0.01)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    results[name] = model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save_results(name)
    model.plot_results(X_test, y_test, name)

print("All experiments completed!")
```
