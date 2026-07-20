# 🚦 Hệ Thống Trích Xuất & Đếm Lưu Lượng Giao Thông (Night Vision)

<img width="1845" height="959" alt="Chụp màn hình từ 2026-07-20 18-57-29" src="https://github.com/user-attachments/assets/fd9f2b19-b91f-428e-b99f-a281405dd088" />

Một ứng dụng Web chuyên nghiệp được xây dựng bằng **Streamlit**, kết hợp sức mạnh của **YOLOv8** và thuật toán theo dõi quỹ đạo **DeepSORT** để phát hiện, theo dõi và đếm số lượng phương tiện giao thông trong điều kiện thiếu sáng (ban đêm).

## 🌟 Tính Năng Nổi Bật

- **Giao Diện Web Trực Quan:** Tải video trực tiếp lên trình duyệt và nhận kết quả realtime, không cần chạy qua terminal cục bộ.
- **Tracking Ổn Định (DeepSORT):** Giảm thiểu tối đa hiện tượng mất ID hoặc đếm đúp nhờ cơ chế `max_age` và trích xuất đặc trưng ngoại hình bằng MobileNet.
- **Thuật Toán Giao Cắt Chính Xác:** Sử dụng phép toán tích có hướng (Cross Product - `ccw`) để kiểm tra hướng cắt của vector quỹ đạo xe với Vạch Ảo (Virtual Line), đảm bảo nhận diện chính xác xe đi IN hay OUT.
- **Tối Ưu Bộ Nhớ:** Tích hợp cơ chế Garbage Collection, tự động dọn dẹp lịch sử tọa độ (`track_history`) của các xe đã đi khuất khỏi màn hình sau mỗi 100 frames để chống tràn RAM.

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ:** Python 3.10+
- **Frontend/Web Framework:** Streamlit
- **Computer Vision:** OpenCV (`cv2`), Ultralytics YOLOv8
- **Object Tracking:** `deep_sort_realtime`
- **Toán học & Ma trận:** NumPy

## 🚀 Hướng Dẫn Cài Đặt (Local Environment)

Khuyến nghị sử dụng **Anaconda** để quản lý môi trường ảo, tránh xung đột thư viện.

**1. Clone kho lưu trữ này về máy:**
```bash
git clone [https://github.com/hoanghoang203205/Traffic_Tracking.git](https://github.com/hoanghoang203205/Traffic_Tracking.git)
cd Traffic_Tracking
```

**2. Tạo môi trường ảo và kích hoạt:**
```bash
conda create -n tracking_env python=3.10 -y
conda activate tracking_env
```

**3. Cài đặt các thư viện yêu cầu:**
*(Lưu ý: Đảm bảo bạn đã cài đặt PyTorch phiên bản hỗ trợ CUDA nếu muốn chạy trên GPU).*
```bash
pip install streamlit opencv-python numpy ultralytics deep-sort-realtime
```

## 🎯 Hướng Dẫn Sử Dụng

Sau khi cài đặt xong, bạn khởi động Web App bằng lệnh sau tại thư mục gốc của dự án:

```bash
streamlit run app.py
```
- Trình duyệt sẽ tự động mở địa chỉ `http://localhost:8501`.
- Nhấn vào thanh sidebar bên trái để **Tải lên một video giao thông (.mp4, .avi)**.
- Hệ thống sẽ tự động xử lý và thống kê lưu lượng xe hiển thị ngay trên màn hình.

## 🧠 Cấu Trúc Luồng Dữ Liệu (The Data Bridge)

Dự án này xử lý trực tiếp vấn đề bất đồng bộ định dạng dữ liệu giữa YOLOv8 (GPU Tensors) và DeepSORT (CPU NumPy). 
Dữ liệu bounding box `[x1, y1, x2, y2]` từ mạng nơ-ron được bóc tách và chuyển đổi toán học sang định dạng `([left, top, width, height], confidence, class_id)` để nạp vào bộ lọc Kalman Filter của DeepSORT một cách mượt mà.

---
*Dự án được phát triển nhằm mục đích nghiên cứu và ứng dụng Computer Vision vào hệ thống giao thông thông minh.*
