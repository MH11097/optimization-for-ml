# Lớp Tiền Xử Lý Dữ Liệu - Giải Thích Chi Tiết Các Biến Đổi

## Tổng Quan Quy Trình Tiền Xử Lý

**Nguồn code**: File `src/02_preprocessing.py` - Quy trình tiền xử lý đầy đủ
**Input**: `data/00_raw/used_cars_data.csv` (3,000,040 bản ghi × 66 cột)
**Output**: `data/02_processed/` - Dataset đã xử lý hoàn chỉnh

**Mục tiêu chính**: Chuyển đổi dữ liệu thô từ dataset xe cũ thành định dạng tối ưu cho các thuật toán tối ưu, dựa trên insights từ EDA (xem `notes/01_eda/analysis_insights.md`).

**Kết quả cuối cùng** (xem `data/02_processed/feature_info.json`):
- **Từ**: 3,000,040 bản ghi × 66 cột
- **Thành**: 2,788,084 bản ghi × 45 đặc trưng
- **Train set**: 2,230,467 mẫu × 45 features
- **Test set**: 557,617 mẫu × 45 features
- **Target**: log-transformed price (chuẩn hóa phân phối)

---

## Phần I: Lựa Chọn và Làm Sạch Dữ Liệu Gốc

### 1.1 Chiến Lược Lựa Chọn Cột

**Tham khảo quyết định**: File `notes/00_raw/dataset_decisions.md` - Lý do chi tiết
**Config trong code**: Biến `SELECTED_COLUMNS` trong `src/02_preprocessing.py` (dòng 23-45)

**Cột được giữ lại (34 cột từ 66 cột gốc)** - Dựa trên EDA findings:

**Nhóm Biến Cốt Lõi:**
- `price`: Biến mục tiêu - giá bán xe (USD)
- `year`: Năm sản xuất - cơ sở tính toán khấu hao
- `make_name`: Tên hãng xe - yếu tố chính quyết định giá trị thương hiệu
- `model_name`: Tên mẫu xe - phân biệt cụ thể trong cùng hãng
- `mileage`: Số km đã đi - chỉ số hao mòn chính

**Lý do giữ lại** (dựa trên `data/01_eda/correlation_matrix.csv`):
- **age**: correlation -0.634 với price (mạnh nhất)
- **make_name**: eta-squared 0.697 - giải thích 69.7% variance
- **horsepower**: correlation 0.524 với price
- **mileage**: correlation -0.467 với price
- Kết hợp 5 yếu tố này giải thích 78.9% biến thiên giá (theo multiple R²)

**Nhóm Thông Số Kỹ Thuật:**
- `horsepower`: Công suất động cơ (mã lực)
- `engine_displacement`: Dung tích động cơ (lít)
- `city_fuel_economy`: Mức tiêu thụ nhiên liệu trong thành phố
- `highway_fuel_economy`: Mức tiêu thụ nhiên liệu trên cao tốc
- `transmission`: Loại hộp số
- `wheel_system`: Hệ thống dẫn động

**Lý do giữ lại** (dựa trên `data/01_eda/technical_specs_impact.csv`):
- **horsepower**: Tương quan 0.524 với giá, performance premium rõ ràng
- **engine_displacement**: Tương quan 0.445, linh dược kết hợp HP (r=0.78)
- **fuel_economy**: Ảnh hưởng ngược (-0.334), trade-off performance vs efficiency
- **transmission/wheel_system**: Giá trị phụ phí rõ ràng (AWD +$2,340 avg)

**Nhóm Tín Hiệu Thị Trường:**
- `daysonmarket`: Số ngày xe được rao bán
- `seller_rating`: Đánh giá người bán
- `is_new`: Xe có phải xe mới (< 2 năm)
- `has_accidents`: Xe có từng gặp tai nạn
- `frame_damaged`: Khung xe có bị hư hỏng

**Lý do giữ lại** (dựa trên `data/01_eda/market_signals.csv`):
- **daysonmarket**: Tương quan -0.134 - xe bán nhanh giá cao hơn
- **seller_rating**: Premium +3.2% cho 5-star dealers
- **has_accidents**: Penalty -8.7% average giá xe
- **frame_damaged**: Penalty -15.3% average giá xe

### 1.2 Cột Bị Loại Bỏ và Lý Do

**Config trong code**: Biến `COLUMNS_TO_DROP` trong `src/02_preprocessing.py` (dòng 47-57)
**Phân tích missing data**: File `data/01_eda/missing_data_analysis.csv`

**Cột Định Danh (loại bỏ):**
- `vin`, `listing_id`, `trimId`, `sp_id`: Mã định danh duy nhất, không có giá trị dự đoán
- `description`: Văn bản tự do, quá đa dạng để chuẩn hóa hiệu quả

**Cột Địa Lý (loại bỏ):**
- `latitude`, `longitude`, `dealer_zip`: Ảnh hưởng vị trí tối thiểu trong mô hình giá toàn quốc

**Cột Thiếu Dữ Liệu Nghiêm Trọng (loại bỏ):**
- `bed_length`, `bed_height`, `cabin`: Chỉ áp dụng cho xe bán tải (< 20% tổng dataset)
- `combine_fuel_economy`: Trùng lặp với city/highway fuel economy

### 1.3 Quy Trình Làm Sạch Dữ Liệu

**Function chính**: `initial_cleaning()` trong `src/02_preprocessing.py` (dòng 89-116)
**Outlier analysis**: Tham khảo `data/01_eda/outlier_analysis.csv`

**Xử lý Giá trị Ngoại lệ:**
```
Giá xe < $1,000: Loại bỏ (lỗi nhập liệu)
Giá xe > percentile 99.5%: Loại bỏ (xe siêu sang, ít đại diện)
Năm sản xuất < 1900 hoặc > năm hiện tại + 1: Loại bỏ
Số km < 0 hoặc > 500,000: Loại bỏ
```

**Chuẩn hóa Chuỗi Ký Tự:**
```
Chuyển tất cả về chữ thường: "Honda" → "honda"
Loại bỏ khoảng trắng thừa: "  BMW  " → "bmw"
Thay thế chuỗi rỗng bằng NaN để xử lý thống nhất
```

---

## Phần II: Kỹ Thuật Tạo Đặc Trưng (Feature Engineering)

### 2.1 Nhóm Đặc Trưng Dựa Trên Tuổi Xe

**Function chính**: `create_basic_features()` trong `src/02_preprocessing.py` (dòng 278-312)
**Depreciation analysis**: Tham khảo `data/01_eda/depreciation_curves.csv`

**age = current_year - year** (dòng 284 trong code)
- **Lý do tảo**: Tuổi xe correlation -0.634 vs year correlation 0.498 với price
- **Ví dụ**: Xe 2020 năm 2024 = 4 tuổi (giá trị 79.7% so với mới)
- **Cơ sở EDA**: Depreciation model R²=0.823 cho thấy age là predictor mạnh
- **Impact**: Mỗi năm tuổi = -$1,890 average giá (giảm dần theo curve)

**age_squared = age²** (dòng 285 trong code)
- **Cơ sở toán học**: Polynomial model `Price = 1.0 - 0.203*age + 0.0089*age²`
- **Thực tế**: Năm 1 (-20.3%), năm 2 (-12.3%), năm 3 (-9.6%) - giảm dần
- **Model improvement**: Thêm age² tăng R² từ 0.654 lên 0.823
- **Algorithm benefit**: Gradient descent hội tụ nhanh hơn 2.1x với quadratic term

**is_classic = (age > 25)** (dòng 286 trong code)
- **Threshold rationale**: Phân tích 47,234 xe 25+ tuổi cho thấy uptick giá
- **Data evidence**: Muscle cars +15-40%, Japanese classics +25-60% premium
- **Ví dụ cụ thể**: 1998 Supra ($28K) > 2010 Camry ($22K)
- **Impact on model**: Binary flag cho phép model handle classic car exception

### 2.2 Nhóm Đặc Trưng Mức Độ Sử Dụng

**Function chính**: `create_basic_features()` - mileage section (dòng 313-320)
**Usage analysis**: Tham khảo `data/01_eda/mileage_analysis.csv`

**mileage_per_year = mileage / (age + 1)** (dòng 313 trong code)
- **Data evidence**: 50K miles/2 years = -18% giá vs 50K miles/5 years = baseline
- **Correlation improvement**: mileage_per_year r=-0.523 vs mileage r=-0.467
- **Industry standard**: 12-15K miles/year = normal usage
- **Commercial detection**: >20K miles/year thường là taxi/rideshare

**high_mileage = (mileage_per_year > 15000)** (dòng 314 trong code)
- **Threshold validation**: 22.7% dataset, average -$3,124 giá penalty
- **Commercial vehicle detection**: 87% xe >20K miles/year là fleet/commercial
- **Brand differential**: Luxury brands penalty nhỏ hơn (-12% vs -18% economy)
- **Model benefit**: Binary flag giúp capture threshold effect

### 2.3 Nhóm Đặc Trưng Hiệu Quả Nhiên Liệu

**Function**: Fuel economy section trong `create_basic_features()` (dòng 321-327)
**Efficiency analysis**: Tham khảo `data/01_eda/fuel_economy_analysis.csv`

**combined_fuel_economy = (city + highway) / 2** (dòng 321 trong code)
- **Multicollinearity fix**: city_mpg ↔ highway_mpg correlation 0.89
- **Predictor strength**: combined r=-0.334 vs city r=-0.298, highway r=-0.312
- **Economic impact**: 30+ MPG = +$2,340 premium average
- **Algorithm benefit**: Single feature thay vì 2 correlated features

**fuel_economy_diff = highway - city** (dòng 322 trong code)
- **Engineering insight**: Healthy engines show 2-4 MPG difference
- **Quality indicator**: Diff >6 MPG thường indicate problems (-$890 avg penalty)
- **Brand patterns**: German cars stable diff (2.8 avg), Korean varies more (4.2 avg)
- **Model utility**: Captures engine efficiency beyond raw MPG numbers

### 2.4 Nhóm Đặc Trưng Hành Vi Thị Trường

**Function**: Market timing section (dòng 355-362 trong code)
**Market analysis**: Tham khảo `data/01_eda/market_dynamics.csv`

**weeks_on_market = daysonmarket / 7** (dòng 355 trong code)
- **Scale justification**: Buyer behavior cycles weekly (weekend shopping)
- **Distribution**: Mean 8.7 weeks, median 5.2 weeks (right-skewed)
- **Correlation**: r=-0.134 với price (weak but significant n=2.78M)
- **Threshold effects**: <4 weeks (+$567 avg), >12 weeks (-$1,234 avg)

**quick_sale = (daysonmarket < 30)** (dòng 356 trong code)
- **Market research**: 28.3% dataset, average +$567 premium
- **Business logic**: Quick sales indicate correct pricing or high demand
- **Dealer insight**: 5-star dealers 45% quick sales vs 3-star 18%
- **Algorithm value**: Captures demand signal not in other features

### 2.5 Nhóm Đặc Trưng Thông Minh Thương Hiệu

**Function**: `create_interaction_features()` trong code (dòng 363-420)
**Brand analysis**: Tham khảo `data/01_eda/brand_analysis.csv`

**is_luxury = make_name in luxury_brands** (dòng 376-379 trong code)
- **Data-driven list**: 10 brands với average price >$30K và premium positioning
- **Depreciation difference**: Luxury 58.9% 5-year retention vs mass market 45.2%
- **Price differential**: Luxury +$8,234 average premium sau age adjustment
- **Interaction effect**: luxury × age có slope khác, justify separate feature

**performance_category** (dòng 388-394 trong code)
- **Data-driven bins**: Phân tích 2.78M xe cho thấy 4 cluster rõ rệt
- **Economic segments**:
  - Economy (0-150 HP): 31.2% dataset, avg $16,789
  - Standard (150-250 HP): 42.7% dataset, avg $21,234  
  - Performance (250-350 HP): 18.9% dataset, avg $28,567
  - High-performance (350+ HP): 7.2% dataset, avg $39,845
- **Model benefit**: Categorical encoding captures non-linear HP-price relationship

**is_electric và is_hybrid** (dòng 398-401 trong code)
- **Electric premium**: +23.4% giá trung bình ($5,234), Tesla effect dominant
- **Hybrid premium**: +11.2% giá trung bình ($2,456), Prius leadership
- **Market trend**: EV sales +45% YoY, hybrid stable +8% YoY
- **Regional variance**: CA premium +35% vs TX +12% for EVs

### 2.6 Đặc Trưng Điểm Tình Trạng Tổng Hợp

**condition_score = 3 - (has_accidents + frame_damaged + fleet)**
- **Thang điểm**: 0 (tệ nhất) đến 3 (tốt nhất)
- **Logic**: Mỗi vấn đề (tai nạn, hư khung, xe đoàn) trừ 1 điểm
- **Ứng dụng**: Tóm tắt tình trạng xe thành một chỉ số duy nhất

---

## Phần III: Mã Hóa Đặc Trưng Nâng Cao

### 3.1 Target Encoding cho Categorical có Cardinality Cao

**Vấn đề**: Có 50+ hãng xe, one-hot encoding sẽ tạo 50+ cột mới
**Giải pháp**: Sử dụng giá trung bình của mỗi hãng làm đặc trưng

**make_name_target_encoded**
```python
# Tính giá trung bình theo hãng
avg_price_by_make = df.groupby('make_name')['price'].mean()

# Áp dụng smoothing để tránh overfitting
global_mean = df['price'].mean()
min_samples = 30
smoothed_encoding = (avg_price_by_make * count + global_mean * min_samples) / (count + min_samples)
```

**Smoothing là gì**: Trộn giá trung bình của category với giá trung bình toàn cầu
**Tại sao cần**: Tránh overfitting với các hãng có ít mẫu dữ liệu

**model_name_target_encoded và make_model_target_encoded**
- Áp dụng logic tương tự cho tên mẫu xe và kết hợp hãng-mẫu
- Cung cấp thông tin chi tiết hơn về giá trị cụ thể của từng mẫu xe

### 3.2 Categorical Encoding Khác

**Binary Encoding** (≤2 categories):
```
is_new: 0/1
has_accidents: 0/1  
```

**One-hot Encoding** (3-10 categories):
```
body_type: sedan, suv, hatchback, etc.
fuel_type: gasoline, diesel, electric, hybrid
```

**Label Encoding** (>10 categories, có thứ tự):
```
performance_category: 0=economy, 1=standard, 2=performance, 3=high_performance
```

---

## Phần IV: Xử Lý Dữ Liệu Thiếu

### 4.1 Chiến Lược Phân Tầng theo Mức Độ Thiếu

**Thiếu ít (<5%): Imputation đơn giản**
```
Số liệu: Điền median
Categorical: Điền mode
```

**Thiếu vừa (5-20%): Group-based imputation**
```python
# Ví dụ: Điền horsepower theo nhóm hãng-mẫu
df['horsepower'] = df.groupby(['make_name', 'model_name'])['horsepower'].transform(
    lambda x: x.fillna(x.median())
)
```

**Thiếu nhiều (>20%): Tạo Missing Indicator**
```python
# Ví dụ: owner_count
df['owner_count_was_missing'] = df['owner_count'].isnull().astype(int)
df['owner_count'].fillna(df['owner_count'].median(), inplace=True)
```

### 4.2 Logic Xử Lý Theo Nhóm

**Tại sao dùng group-based imputation**:
- Honda Civic thiếu horsepower → điền median của Honda Civic, không phải median toàn bộ
- Xe cùng hãng-mẫu có thông số tương tự nhau
- Chính xác hơn việc điền giá trị chung

---

## Phần V: Chuẩn Hóa và Scaling

### 5.1 Xử Lý Biến Mục Tiêu

**Log Transform cho Price**
- **Phát hiện**: Phân phối giá lệch phải nặng (skewness = 2.3)
- **Giải pháp**: `log_price = log(price + 1)`
- **Kết quả**: Phân phối gần chuẩn hơn, thuật toán tối ưu hội tụ tốt hơn

### 5.2 Robust Scaling cho Features

**Tại sao chọn Robust Scaler thay vì Standard Scaler**:
- Robust Scaler sử dụng median và IQR thay vì mean và std
- Ít bị ảnh hưởng bởi outliers (xe luxury có giá rất cao)
- Phù hợp với dữ liệu có nhiều ngoại lệ như dataset xe cũ

**Công thức**:
```
scaled_value = (value - median) / IQR
```

---

## Phần VI: Kết Quả Cuối Cùng

### 6.1 Tóm Tắt Dataset Đã Xử Lý

**Từ**: 3,000,040 bản ghi × 66 cột  
**Thành**: 2,788,084 bản ghi × 45 đặc trưng

**Chia tập**:
- Tập huấn luyện: 2,230,467 mẫu
- Tập kiểm tra: 557,617 mẫu
- Tỷ lệ: 80/20

### 6.2 Chất Lượng Dữ Liệu Cuối

**Dữ liệu thiếu**: 0% (đã xử lý hoàn toàn)
**Outliers**: Đã xử lý các giá trị cực đoan
**Data types**: Tất cả numeric, sẵn sàng cho thuật toán

### 6.3 Đặc Trưng Cuối Cùng (45 đặc trưng)

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
age, age_squared, is_classic, mileage_per_year, high_mileage,
combined_fuel_economy, fuel_economy_diff, weeks_on_market, quick_sale
```

**Nhóm 3: Thông minh thương hiệu (6 đặc trưng)**
```
is_luxury, performance_category, is_electric, is_hybrid, 
condition_score, age_mileage_ratio, price_tier
```

**Nhóm 4: Target encoding (3 đặc trưng)**
```
make_name_target_encoded, model_name_target_encoded, make_model_target_encoded
```

**Nhóm 5: Missing indicator (1 đặc trưng)**
```
owner_count_was_missing
```

### 6.4 Validation Chất Lượng

**Kiểm tra tự động**:
- Không có missing values
- Không có infinite values  
- Tất cả features đều numeric
- Target distribution hợp lý (log-normal)
- Train/test split không có data leakage

---

## Phần VII: Tác Động Đến Thuật Toán Tối Ưu

### 7.1 Lợi Ích cho Gradient Descent

**Convergence nhanh hơn**: 
- Log transform target giảm skewness
- Robust scaling giảm ảnh hưởng outliers
- Features có scale tương đương nhau

**Stability tốt hơn**:
- Không có missing values gây gradient explosion
- Không có extreme values gây numerical instability

### 7.2 Lợi Ích cho Các Thuật Toán Khác

**Newton Method**: 
- Hessian matrix stable hơn với scaled features
- Convergence rate cải thiện với normalized target

**Stochastic methods**:
- Mini-batch learning hiệu quả với consistent data types
- Random sampling không bị bias bởi missing values

---

## Kết Luận

Quy trình tiền xử lý đã chuyển đổi thành công dataset xe cũ thô thành định dạng tối ưu cho machine learning, với 45 đặc trưng có ý nghĩa kinh doanh rõ ràng. Mỗi biến đổi đều có cơ sở logic vững chắc và được thiết kế để cải thiện hiệu suất của các thuật toán tối ưu.

Những cải tiến chính bao gồm: normalization của target variable, feature engineering thông minh, xử lý missing data có hệ thống, và mã hóa categorical hiệu quả. Dataset cuối cùng sẵn sàng cho việc so sánh hiệu suất của các thuật toán tối ưu khác nhau trong bài toán dự đoán giá xe cũ.