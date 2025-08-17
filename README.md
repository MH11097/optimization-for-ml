# Dự Án Tối Ưu Thuật Toán - Dự Đoán Giá Xe Cũ

**Framework Python đơn giản để so sánh các thuật toán tối ưu trên bài toán dự đoán giá xe cũ.**

## 🚀 Khởi Chạy Nhanh

```bash
# Cài đặt dependencies
pip install -r requirements.txt

## 📁 Cấu Trúc Dự Án

```

data/ # Dữ liệu theo workflow số
├── 00_raw/ # Dữ liệu gốc
├── 01_eda/ # Kết quả phân tích dữ liệu  
├── 02_processed/ # Dữ liệu đã xử lý
├── 03_algorithms/ # Kết quả các thuật toán
│ ├── gradient_descent/
│ ├── newton_method/
│ ├── ridge_regression/
│ └── stochastic_gd/
└── 04_comparison/ # So sánh cuối cùng

src/ # Code theo workflow số
├── 01_eda.py # Phân tích dữ liệu
├── 02_preprocessing.py # Xử lý dữ liệu
├── algorithms/ # Các thuật toán
│ ├── gradient_descent/ # Gradient Descent với nhiều setup
│ ├── newton_method/ # Newton Method
│ ├── ridge_regression/ # Ridge Regression
│ ├── stochastic_gd/ # Stochastic GD
│ ├── advanced_methods/ # Các phương pháp nâng cao
│ └── algorithm_comparator.py # Tool so sánh tổng hợp
└── utils/ # Tiện ích chung

````

## 🔄 Cách Sử Dụng

### **Bước 1: Phân Tích Dữ Liệu**
```bash
python src/01_eda.py
````

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

# 📋 Dataset: Xe Cũ Mỹ

## 📊 Tổng quan

Tập dữ liệu này chứa thông tin về **3 triệu xe ô tô cũ** tại Hoa Kỳ, được thu thập từ website CarGurus.com. Bao gồm 66 cột dữ liệu với thông tin chi tiết về thông số kỹ thuật, giá cả, tình trạng xe và thông tin đại lý.

## 📋 Mô tả từng cột dữ liệu

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
