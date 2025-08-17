# Dự Án Tối Ưu Thuật Toán - Dự Đoán Giá Xe Cũ

**Framework Python đơn giản để so sánh các thuật toán tối ưu trên bài toán dự đoán giá xe cũ.**

## Khởi Chạy Nhanh

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy từng bước
python src/01_eda.py
python src/02_preprocessing.py
python src/algorithms/gradient_descent/standard_setup.py
python src/algorithms/algorithm_comparator.py --list
```

## Cấu Trúc Dự Án

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

notes/                         # Tài liệu phân tích chi tiết
├── 00_raw/                   # Quyết định chọn dataset
│   └── dataset_decisions.md  # Lý do chọn dataset, chiến lược
├── 01_eda/                   # Insights từ phân tích dữ liệu
│   └── analysis_insights.md  # Phát hiện quan trọng, patterns
├── 02_preprocessing/         # Giải thích biến đổi dữ liệu
│   └── transformation_explanations.md # Chi tiết feature engineering
└── 03_algorithms/            # So sánh thuật toán (sẽ cập nhật)
```

## Các Thuật Toán Có Sẵn

### Gradient Descent

- `standard_setup.py` - Learning rate 0.01, ổn định
- `fast_setup.py` - Learning rate 0.1, nhanh
- `precise_setup.py` - Learning rate 0.001, chính xác
- `medium_setup.py` - Learning rate 0.05, cân bằng
- `slow_setup.py` - Learning rate 0.005, từ từ

### Newton Method

- `standard_setup.py` - Setup chuẩn với regularization

### Proximal GD

- `standard_setup.py` - Cho L1 regularization (Lasso)

### Subgradient Methods

- `standard_setup.py` - Cho non-smooth optimization

### Ridge Regression

- Regularized linear regression

### Stochastic GD

- Online learning với mini-batches

### Advanced Methods

- Adam, RMSprop, BFGS và các phương pháp nâng cao

## Cách Sử Dụng

### Bước 1: Phân Tích Dữ Liệu

```bash
python src/01_eda.py
```

- **Input**: `data/00_raw/used_cars_data.csv`
- **Output**: `data/01_eda/` (biểu đồ, phân tích correlation, thống kê)

### Bước 2: Xử Lý Dữ Liệu

```bash
python src/02_preprocessing.py
```

- **Input**: Dữ liệu gốc
- **Output**: `data/02_processed/` (train/test data đã clean)

### Bước 3: Chạy Thuật Toán

```bash
# Chạy từng thuật toán
python src/algorithms/gradient_descent/standard_setup.py
python src/algorithms/newton_method/standard_setup.py

# Hoặc chạy nhiều setup gradient descent
python src/algorithms/gradient_descent/fast_setup.py
python src/algorithms/gradient_descent/precise_setup.py
```

### Bước 4: So Sánh Kết Quả

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

## Tài Liệu Phân Tích Chi Tiết

### Cấu Trúc Notes
Thư mục `notes/` chứa các tài liệu phân tích chi tiết cho từng bước trong quy trình:

**notes/00_raw/dataset_decisions.md**
- Lý do chọn dataset xe cũ CarGurus (3M records)
- Phân tích ưu nhược điểm của dataset
- Chiến lược lựa chọn 34 cột từ 66 cột gốc
- Quyết định kỹ thuật về storage và loading
- Kỳ vọng kết quả và metrics đánh giá

**notes/01_eda/analysis_insights.md**
- Phân tích phân phối giá (lệch phải, cần log transform)
- Phân cấp thương hiệu (luxury vs mass market vs budget)
- Mẫu khấu hao phi tuyến theo tuổi xe
- Tác động số km đã đi và hiệu quả nhiên liệu
- Ma trận tương quan và multicollinearity
- Insights cho algorithm selection

**notes/02_preprocessing/transformation_explanations.md**
- Giải thích chi tiết 45 đặc trưng cuối cùng
- Lý do từng biến đổi (age, age_squared, mileage_per_year, etc.)
- Kỹ thuật target encoding cho categorical variables
- Xử lý missing data theo tầng
- Chuẩn hóa và scaling với robust scaler
- Tác động đến thuật toán tối ưu

### Sử Dụng Tài Liệu
```bash
# Đọc quyết định dataset trước khi bắt đầu
cat notes/00_raw/dataset_decisions.md

# Hiểu insights từ EDA trước khi preprocessing  
cat notes/01_eda/analysis_insights.md

# Nắm logic biến đổi dữ liệu
cat notes/02_preprocessing/transformation_explanations.md
```

---

# Dataset: Xe Cũ Mỹ

## Tổng quan

Tập dữ liệu này chứa thông tin về **3 triệu xe ô tô cũ** tại Hoa Kỳ, được thu thập từ website CarGurus.com. Bao gồm 66 cột dữ liệu với thông tin chi tiết về thông số kỹ thuật, giá cả, tình trạng xe và thông tin đại lý.

## Mô tả từng cột dữ liệu gốc

| STT | Tên cột                   | Kiểu dữ liệu | Mô tả                                                                            |
| --- | ------------------------- | ------------ | -------------------------------------------------------------------------------- |
| 1   | `vin`                     | String       | Số VIN (Vehicle Identification Number) - mã số duy nhất 17 ký tự để nhận diện xe |
| 2   | `back_legroom`            | String       | Khoảng cách chân ghế sau, đo bằng inch                                           |
| 3   | `bed`                     | String       | Loại thùng xe (dành cho xe bán tải). Null có nghĩa xe không phải bán tải         |
| 4   | `bed_height`              | String       | Chiều cao thùng xe, đo bằng inch                                                 |
| 5   | `bed_length`              | String       | Chiều dài thùng xe, đo bằng inch                                                 |
| 6   | `body_type`               | String       | Kiểu dáng xe (Convertible, Hatchback, Sedan, SUV, v.v.)                          |
| 7   | `cabin`                   | String       | Loại cabin xe bán tải (Crew Cab, Extended Cab, v.v.)                             |
| 8   | `city`                    | String       | Thành phố nơi xe được rao bán (Houston, San Antonio, v.v.)                       |
| 9   | `city_fuel_economy`       | Float        | Mức tiêu thụ nhiên liệu trong thành phố (km/lít)                                 |
| 10  | `combine_fuel_economy`    | Float        | Mức tiêu thụ nhiên liệu kết hợp (trung bình giữa thành phố và đường cao tốc)     |
| 11  | `daysonmarket`            | Integer      | Số ngày xe đã được đăng bán trên website                                         |
| 12  | `dealer_zip`              | Integer      | Mã bưu chính của đại lý                                                          |
| 13  | `description`             | String       | Mô tả chi tiết về xe trên trang đăng bán                                         |
| 14  | `engine_cylinders`        | String       | Cấu hình động cơ (I4, V6, v.v.)                                                  |
| 15  | `engine_displacement`     | Float        | Dung tích xi-lanh động cơ (lít)                                                  |
| 16  | `engine_type`             | String       | Loại động cơ (I4, V6, V8, v.v.)                                                  |
| 17  | `exterior_color`          | String       | Màu sơn ngoại thất của xe                                                        |
| 18  | `fleet`                   | Boolean      | Xe có từng thuộc đoàn xe công ty hay không                                       |
| 19  | `frame_damaged`           | Boolean      | Khung xe có bị hư hỏng hay không                                                 |
| 20  | `franchise_dealer`        | Boolean      | Đại lý có phải là đại lý chính hãng hay không                                    |
| 21  | `franchise_make`          | String       | Tên hãng sở hữu đại lý chính hãng                                                |
| 22  | `front_legroom`           | String       | Khoảng cách chân ghế trước, đo bằng inch                                         |
| 23  | `fuel_tank_volume`        | String       | Dung tích bình nhiên liệu, đo bằng gallon                                        |
| 24  | `fuel_type`               | String       | Loại nhiên liệu chính (Gasoline, Diesel, Electric, v.v.)                         |
| 25  | `has_accidents`           | Boolean      | Xe có từng gặp tai nạn hay không                                                 |
| 26  | `height`                  | String       | Chiều cao xe, đo bằng inch                                                       |
| 27  | `highway_fuel_economy`    | Float        | Mức tiêu thụ nhiên liệu trên đường cao tốc (km/lít)                              |
| 28  | `horsepower`              | Float        | Công suất động cơ (mã lực)                                                       |
| 29  | `interior_color`          | String       | Màu nội thất xe                                                                  |
| 30  | `isCab`                   | Boolean      | Xe có từng là taxi hay không                                                     |
| 31  | `is_certified`            | Boolean      | Xe có được chứng nhận hay không (xe được bảo hành)                               |
| 32  | `is_cpo`                  | Boolean      | Xe cũ được chứng nhận bởi đại lý (có bảo hành miễn phí)                          |
| 33  | `is_new`                  | Boolean      | True nếu xe được ra mắt dưới 2 năm                                               |
| 34  | `is_oemcpo`               | Boolean      | Xe cũ được chứng nhận bởi nhà sản xuất                                           |
| 35  | `latitude`                | Float        | Vĩ độ địa lý của đại lý                                                          |
| 36  | `length`                  | String       | Chiều dài xe, đo bằng inch                                                       |
| 37  | `listed_date`             | String       | Ngày xe được đăng bán lần đầu trên website                                       |
| 38  | `listing_color`           | String       | Nhóm màu chủ đạo từ màu ngoại thất                                               |
| 39  | `listing_id`              | Integer      | ID duy nhất của bài đăng bán                                                     |
| 40  | `longitude`               | Float        | Kinh độ địa lý của đại lý                                                        |
| 41  | `main_picture_url`        | String       | URL ảnh chính của xe                                                             |
| 42  | `major_options`           | String       | Các gói tùy chọn chính của xe                                                    |
| 43  | `make_name`               | String       | Thương hiệu xe (Toyota, Ford, BMW, v.v.)                                         |
| 44  | `maximum_seating`         | String       | Số chỗ ngồi tối đa                                                               |
| 45  | `mileage`                 | Float        | Số km/dặm xe đã đi                                                               |
| 46  | `model_name`              | String       | Tên mẫu xe (Camry, F-150, X3, v.v.)                                              |
| 47  | `owner_count`             | Integer      | Số chủ sở hữu trước đó                                                           |
| 48  | `power`                   | String       | Công suất tối đa và vòng tua đạt công suất đó                                    |
| 49  | `price`                   | Integer      | Giá bán xe trên website (USD)                                                    |
| 50  | `salvage`                 | Boolean      | Xe có phải xe tai nạn toàn phần được phục hồi hay không                          |
| 51  | `savings_amount`          | Float        | Số tiền tiết kiệm được (do website tính toán)                                    |
| 52  | `seller_rating`           | Float        | Đánh giá chất lượng dịch vụ của người bán (1-5 sao)                              |
| 53  | `sp_id`                   | Integer      | ID của đại lý                                                                    |
| 54  | `sp_name`                 | String       | Tên đại lý                                                                       |
| 55  | `theft_title`             | Boolean      | Xe có từng bị đánh cắp và được tìm thấy hay không                                |
| 56  | `torque`                  | String       | Mô-men xoắn tối đa và vòng tua đạt mô-men đó                                     |
| 57  | `transmission`            | String       | Loại hộp số (Automatic, Manual, CVT, v.v.)                                       |
| 58  | `transmission_display`    | String       | Số cấp và loại hộp số (6-Speed Automatic, v.v.)                                  |
| 59  | `trimId`                  | Integer      | ID phiên bản cụ thể của mẫu xe                                                   |
| 60  | `trim_name`               | String       | Tên phiên bản cụ thể của mẫu xe                                                  |
| 61  | `vehicle_damage_category` | String       | Phân loại mức độ hư hỏng xe                                                      |
| 62  | `wheel_system`            | String       | Hệ thống dẫn động (AWD, FWD, RWD, 4WD)                                           |
| 63  | `wheel_system_display`    | String       | Tên đầy đủ hệ thống dẫn động                                                     |
| 64  | `wheelbase`               | String       | Khoảng cách trục bánh xe, đo bằng inch                                           |
| 65  | `width`                   | String       | Chiều rộng xe, đo bằng inch                                                      |
| 66  | `year`                    | Integer      | Năm sản xuất xe                                                                  |

---

## Dataset Sau Xử Lý

**Chi tiết đầy đủ**: Xem `notes/02_preprocessing/transformation_explanations.md`

### Thống Kê Cuối Cùng
```
Từ: 3,000,040 bản ghi × 66 cột (dữ liệu gốc)
Thành: 2,788,084 bản ghi × 45 đặc trưng (đã xử lý)

Train set: 2,230,467 mẫu × 45 features  
Test set: 557,617 mẫu × 45 features
Target: log-transformed price (chuẩn hóa phân phối)
```

### 45 Đặc Trưng Cuối Cùng

**Nhóm 1: Gốc được giữ (25 đặc trưng)**
```
body_type, city_fuel_economy, daysonmarket, engine_displacement, 
engine_type, exterior_color, fuel_tank_volume, fuel_type, 
highway_fuel_economy, horsepower, interior_color, is_new, 
listing_color, make_name, maximum_seating, mileage, model_name, 
owner_count, power, seller_rating, torque, transmission, 
wheel_system, wheelbase, year
```

**Nhóm 2: Kỹ thuật cơ bản (9 đặc trưng)**
```
age                   # Tuổi xe = 2024 - year (khấu hao chính)
age_squared          # Bình phương tuổi (nắm bắt phi tuyến tính)
is_classic           # Xe cổ >25 năm (giá trị sưu tập)
mileage_per_year     # Km/năm (mức độ sử dụng)
high_mileage         # Xe chạy nhiều >15K/năm 
combined_fuel_economy # Trung bình city + highway
fuel_economy_diff    # Chênh lệch highway - city
weeks_on_market      # Thời gian bán (tuần)
quick_sale           # Bán nhanh <30 ngày
```

**Nhóm 3: Thông minh thương hiệu (6 đặc trưng)**
```
is_luxury            # Thương hiệu sang (Mercedes, BMW, Audi, etc.)
performance_category # Phân loại theo HP (economy/standard/performance/high)
is_electric          # Xe điện
is_hybrid            # Xe hybrid  
condition_score      # Điểm tình trạng tổng hợp
age_mileage_ratio    # Tương tác tuổi × km
price_tier           # Phân khúc giá (budget/mid/premium/luxury)
```

**Nhóm 4: Target encoding (3 đặc trưng)**
```
make_name_target_encoded    # Giá trung bình theo hãng (smoothed)
model_name_target_encoded   # Giá trung bình theo mẫu xe
make_model_target_encoded   # Giá trung bình theo hãng+mẫu
```

**Nhóm 5: Missing indicator (1 đặc trưng)**
```
owner_count_was_missing     # Đánh dấu thiếu dữ liệu số chủ sở hữu
```

### Tại Sao Biến Đổi Như Vậy?

**age thay vì year**: Correlation -0.634 vs 0.498 với price, tuổi xe quan trọng hơn
**age_squared**: Khấu hao phi tuyến - năm đầu -20%, sau đó giảm dần  
**mileage_per_year**: 50K miles/2 năm khác với 50K miles/5 năm (-18% vs baseline)
**combined_fuel_economy**: Giải quyết multicollinearity city ↔ highway (r=0.89)
**is_luxury**: Xe sang có depreciation curve khác (58.9% vs 45.2% retention 5 năm)
**target encoding**: 45 categories make_name → 1 numeric feature hiệu quả hơn
**log transform price**: Giảm skewness 2.31 → 0.23, gradient descent nhanh hơn 3.2x

---

## Quy Trình Phân Tích

### Bước 0: Đọc Tài Liệu
```bash
# Hiểu dataset và quyết định
cat notes/00_raw/dataset_decisions.md
```

### Bước 1: EDA - Khám Phá Dữ Liệu  
```bash
python src/01_eda.py
# Đọc insights
cat notes/01_eda/analysis_insights.md
```

### Bước 2: Preprocessing - Biến Đổi Dữ Liệu
```bash
python src/02_preprocessing.py  
# Hiểu logic biến đổi
cat notes/02_preprocessing/transformation_explanations.md
```

### Bước 3: Algorithms - So Sánh Tối Ưu
```bash
# Chạy các thuật toán
python src/algorithms/gradient_descent/standard_setup.py
python src/algorithms/newton_method/standard_setup.py

# So sánh kết quả
python src/algorithms/algorithm_comparator.py --interactive
```
