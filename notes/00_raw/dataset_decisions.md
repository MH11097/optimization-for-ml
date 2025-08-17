# Lớp Dữ Liệu Thô - Quyết Định Chọn Dataset và Lý Do

## Tổng Quan Dataset

**Tên Dataset**: Used Cars Dataset (Dữ liệu xe cũ Mỹ)  
**Nguồn**: CarGurus.com - Nền tảng mua bán xe cũ hàng đầu Mỹ  
**Quy mô**: Khoảng 3 triệu bản ghi, 66 đặc trưng

---

## Tại Sao Chọn Dataset Này?

### Những Ưu Điểm Chính

1. **Quy mô phù hợp cho việc kiểm thử thuật toán tối ưu**

   - Với hơn 3 triệu bản ghi, dataset này đủ lớn để đánh giá hiệu suất của các thuật toán

2. **Tính ứng dụng thương mại cao**

   - Đây là bài toán định giá thực tế, có thể áp dụng trực tiếp trong kinh doanh
   - Các chỉ số đánh giá rõ ràng: MAE, MAPE có ý nghĩa kinh doanh trực tiếp

3. **Đa dạng về loại đặc trưng**

   - Kết hợp cả dữ liệu phân loại (thương hiệu, mẫu xe) và số liệu (giá, số km đã đi)
   - Thông số kỹ thuật (công suất, mức tiêu thụ nhiên liệu)
   - Tín hiệu thị trường (số ngày rao bán, đánh giá người bán)
   - Chỉ số tình trạng xe (tai nạn, hư hỏng)

4. **Cấu trúc dữ liệu có hệ thống**
   - Schema nhất quán từ một nguồn duy nhất (CarGurus)
   - Mẫu dữ liệu bị thiếu hợp lý và có thể dự đoán được
   - Biến mục tiêu rõ ràng (giá bán)

---

## Chiến Lược Lựa Chọn Cột Dữ Liệu

### Biến Mục Tiêu: price

```
Khoảng giá trị: $1,000 - $100,000+ USD
Phân phối: Lệch phải (nhiều xe giá thấp, ít xe sang)
Quyết định: Sẽ cần biến đổi logarit để chuẩn hóa
```

### Các Biến Dự Đoán Cốt Lõi (Bắt buộc có)

```
year         → Yếu tố tuổi xe (khấu hao chính)
make_name    → Giá trị thương hiệu (chiếm 60-70% sự biến thiên giá)
model_name   → Sự khác biệt giữa các mẫu xe cụ thể
mileage      → Chỉ số sử dụng và hao mòn
horsepower   → Thông số hiệu suất
body_type    → Loại thân xe
```

### Thông Số Kỹ Thuật (Quan Trọng)

```
engine_displacement  → Dung tích động cơ = công suất + ý nghĩa thuế
fuel_economy        → Yếu tố chi phí vận hành
transmission        → Sở thích số sàn vs số tự động
wheel_system        → Phụ phí AWD, tiêu chuẩn FWD
```

### Tín Hiệu Thị Trường (Có Giá Trị)

```
daysonmarket    → Chỉ số cầu
seller_rating   → Tín hiệu tin tưởng/chất lượng
is_new         → Danh mục đặc biệt
has_accidents  → Phạt do tình trạng xe
```

### Các Cột Bị Loại Bỏ - Lý Do

```
Số VIN          → ID duy nhất, không có giá trị dự đoán
Tọa độ GPS      → Ảnh hưởng vị trí tối thiểu đến định giá
Văn bản tự do   → Mô tả quá đa dạng, khó chuẩn hóa
Trường URL      → Không có giá trị kinh doanh
ID nội bộ       → Đặc thù nền tảng, không thể chuyển giao
```

---

## Đánh Giá Chất Lượng Dữ Liệu

### Điểm Mạnh Được Phát Hiện

- **Các đặc trưng cốt lõi đầy đủ**: giá, năm, hãng, mẫu xe đều có đầy đủ
- **Mẫu thiếu dữ liệu hợp lý**: các tính năng cao cấp thiếu ở xe bình dân là logic
- **Định dạng nhất quán**: được chuẩn hóa từ một nền tảng duy nhất
- **Không có lỗi nhập liệu rõ ràng**: các khoảng giá trị hợp lý

### Vấn Đề Được Xác Định

- **Khoảng 35% thiếu dữ liệu ở các tính năng tùy chọn** (bed_length, cabin)
- **Không nhất quán chuỗi ký tự**: "Honda" vs "HONDA" vs "honda"
- **Ngoại lệ**: xe $1, xe sang $500K+
- **Danh sách trùng lặp**: cùng một xe được đăng nhiều lần

---

## Kết Quả Mong Đợi

### Chỉ Số Kinh Doanh Cần Theo Dõi

```
MAE (Sai Số Tuyệt Đối Trung Bình): Mục tiêu < $2,000
MAPE (Sai Số Phần Trăm Tuyệt Đối Trung Bình): Mục tiêu < 15%
Điểm R²: Mục tiêu > 0.85
```

### Chỉ Số Hiệu Suất Thuật Toán

```
Thời Gian Huấn Luyện: So sánh giữa các thuật toán
Sử Dụng Bộ Nhớ: Theo dõi mức tiêu thụ RAM đỉnh
Tốc Độ Hội Tụ: Số lần lặp đến giá trị tối ưu
Khả Năng Mở Rộng: Hiệu suất theo kích thước dữ liệu
```

### Thông Tin Chi Tiết Về Tầm Quan Trọng Đặc Trưng Mong Đợi

```
Các biến dự đoán hàng đầu: năm, tên hãng, số km, công suất
Hiệu ứng tương tác: thương hiệu sang × tuổi xe khác nhau
Động lực thị trường: mối quan hệ ngày rao bán vs giá
```

---

## Các Bước Tiếp Theo

1. **Phân Tích Khám Phá Dữ Liệu (01_eda.py)**: Tìm hiểu sâu về phân phối, tương quan
2. **Tiền Xử Lý (02_preprocessing.py)**: Làm sạch, thiết kế đặc trưng
3. **Mô Hình Hóa**: Kiểm tra các thuật toán tối ưu một cách có hệ thống
