# Các biến thể của Newton trong tối ưu hóa lồi

## 1. Bài toán tối ưu lồi

Trong tối ưu hóa lồi, ta thường muốn giải:

\[
\min_x f(x)
\]

với \(f(x)\) là hàm **lồi, khả vi 2 lần**.

Điều kiện cực tiểu:  
\[
\nabla f(x^_) = 0, \quad H(x^_) \succeq 0
\]

Giải trực tiếp hệ \(\nabla f(x)=0\) thường khó, nên ta cần **phương pháp lặp**: bắt đầu từ một điểm ban đầu \(x_0\), rồi cập nhật dần \(x_1, x_2, \dots\) cho đến khi hội tụ.

---

## 2. Newton’s Method

### Ý tưởng

- Xấp xỉ \(f(x)\) bằng Taylor bậc 2 quanh \(x_k\):  
  \[
  f(x) \approx f(x_k) + \nabla f(x_k)^T (x-x_k) + \tfrac{1}{2}(x-x_k)^T H(x_k)(x-x_k)
  \]
- Tối ưu hóa xấp xỉ này cho ta bước Newton:  
  \[
  x\_{k+1} = x_k - H(x_k)^{-1}\nabla f(x_k)
  \]

### Đặc điểm quan trọng

- Hội tụ bậc 2 (rất nhanh khi gần nghiệm).
- Sử dụng cả gradient (hướng đi) và Hessian (độ cong).
- Tốn chi phí \(O(n^3)\) do phải nghịch đảo Hessian.

### Minh họa trực quan

- Gradient Descent: chỉ biết hướng dốc để đi xuống.
- Newton: xây parabol quanh điểm hiện tại, nhảy ngay về cực tiểu của parabol đó.

---

## 3. Damped Newton

### Ý tưởng

- Newton chuẩn có thể nhảy quá xa nếu ở xa nghiệm.
- Thêm hệ số bước \(\alpha\):  
  \[
  x\_{k+1} = x_k - \alpha H(x_k)^{-1}\nabla f(x_k)
  \]
- \(\alpha\) chọn qua line search hoặc backtracking để đảm bảo giảm hàm.

### Đặc điểm quan trọng

- Ổn định toàn cục hơn Newton chuẩn.
- Vẫn giữ hội tụ nhanh khi gần nghiệm.

### Minh họa trực quan

- Newton chuẩn: nhảy thẳng về cực tiểu gần đúng.
- Damped Newton: đi cẩn thận hơn, chọn bước phù hợp để chắc chắn giảm hàm.

---

## 4. Quasi-Newton (BFGS)

### Ý tưởng

- Newton cần Hessian → quá tốn kém.
- Quasi-Newton: thay vì tính Hessian, **ước lượng dần từ gradient**.
- Sử dụng secant condition:  
  \[
  B*{k+1}s_k = y_k
  \]  
  với \(s_k = x*{k+1}-x*k\), \(y_k = \nabla f(x*{k+1})-\nabla f(x_k)\).

### Cách phát triển

- BFGS cập nhật trực tiếp nghịch đảo Hessian xấp xỉ \(H*k\):  
  \[
  H*{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T
  \]  
  với \(\rho_k = 1/(y_k^T s_k)\).

### Đặc điểm quan trọng

- Không cần Hessian thật.
- Hội tụ nhanh gần như Newton.
- Bộ nhớ \(O(n^2)\).

### Minh họa trực quan

- Mỗi bước giống một “thí nghiệm”: từ bước dịch chuyển và thay đổi gradient, ta điều chỉnh dần giả-Hessian.

---

## 5. Limited-memory BFGS (L-BFGS)

### Ý tưởng

- BFGS cần lưu ma trận \(n \times n\) → không khả thi khi \(n\) lớn.
- L-BFGS: chỉ lưu một số cặp gần đây \((s_k, y_k)\), dùng công thức đệ quy để tính hướng đi.

### Đặc điểm quan trọng

- Bộ nhớ chỉ \(O(mn)\), với \(m\) nhỏ (5–20).
- Phù hợp cho bài toán cực lớn (machine learning, NLP).
- Hội tụ chậm hơn BFGS một chút, nhưng khả thi với dữ liệu lớn.

### Minh họa trực quan

- Giống như chỉ ghi nhớ vài đoạn đường gần nhất thay vì lưu toàn bộ bản đồ.
- Vẫn đủ thông tin để tìm hướng hợp lý.

---

## 6. So sánh bốn thuật toán

| Thuật toán    | Bộ nhớ     | Tốc độ hội tụ       | Tính toán mỗi bước              | Quy mô ứng dụng            |
| ------------- | ---------- | ------------------- | ------------------------------- | -------------------------- |
| Newton        | \(O(n^2)\) | Rất nhanh (bậc 2)   | Nghịch đảo Hessian (\(O(n^3)\)) | Bài toán nhỏ – vừa         |
| Damped Newton | \(O(n^2)\) | Nhanh, ổn định hơn  | Newton + line search            | Bài toán nhỏ – vừa         |
| BFGS          | \(O(n^2)\) | Quasi-Newton, nhanh | Cập nhật ma trận                | Bài toán vừa (n vài nghìn) |
| L-BFGS        | \(O(mn)\)  | Tốt, nhẹ            | Cập nhật vectơ                  | Bài toán rất lớn           |

---

## 7. Kết luận

- Newton: cơ bản, nhanh nhưng tốn kém.
- Damped Newton: thêm ổn định khi xa nghiệm.
- BFGS: tiết kiệm hơn, không cần Hessian, hội tụ nhanh.
- L-BFGS: mở rộng cho dữ liệu cực lớn, chuẩn mực trong machine learning truyền thống.
