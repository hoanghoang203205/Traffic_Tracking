import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN WEB CƠ BẢN
# ==========================================
st.set_page_config(page_title="Hệ Thống Đếm Xe Ban Đêm", page_icon="🚦", layout="wide")
st.title("🚦 Hệ Thống Trích Xuất & Đếm Lưu Lượng Giao Thông")
st.markdown("Dự án sử dụng **YOLOv8** và **DeepSORT** để theo dõi quỹ đạo phương tiện trong điều kiện thiếu sáng.")


# ==========================================
# 2. KHAI BÁO CÁC HÀM TOÁN HỌC (Giữ nguyên)
# ==========================================
def ccw(A, B, C):
    """Kiểm tra xem điểm C nằm bên Trái (True) hay Phải (False) của vector AB."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def check_intersect(A, B, C, D):
    """Kiểm tra đoạn thẳng AB (Vạch kẻ) có cắt đoạn CD (Quỹ đạo xe) hay không."""
    return ccw(C, D, A) != ccw(C, D, B) and ccw(A, B, C) != ccw(A, B, D)


# Cấu hình Vạch ảo (Virtual Line)
line_A = (100, 350)
line_B = (600, 350)

# ==========================================
# 3. TẠO TÍNH NĂNG UPLOAD VIDEO CHO WEB
# ==========================================
st.sidebar.header("Tải Video Lên")
uploaded_file = st.sidebar.file_uploader("Chọn file video (.mp4, .avi)", type=['mp4', 'avi'])

if uploaded_file is not None:
    # Lấy file từ Web lưu tạm xuống ổ cứng để OpenCV có thể đọc được
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.sidebar.success("Tải video thành công! Hệ thống đang xử lý...")

    # Khởi tạo các vùng trống trên Web để đẩy dữ liệu vào sau
    col1, col2 = st.columns(2)
    metric_in = col1.empty()
    metric_out = col2.empty()
    stframe = st.empty()  # Khung chứa Video hiển thị

    # Nút bấm để dừng hệ thống đang chạy
    stop_button = st.sidebar.button("Dừng Xử Lý")

    # ==========================================
    # 4. KHỞI TẠO BIẾN VÀ MODEL
    # ==========================================
    track_history = {}
    counted_ids = set()
    count_in = 0
    count_out = 0
    class_names = {0: 'Bike', 1: 'Car', 2: 'Bus', 3: 'Truck'}


    # Load Model (Nên dùng cache trong thực tế để web không load lại model nhiều lần)
    @st.cache_resource
    def load_model_and_tracker():
        model = YOLO("runs/detect/Vehicle_Tracking/Night_Weather_Model5/weights/best.pt")
        tracker = DeepSort(
            max_age=30, n_init=3, nms_max_overlap=1.0,
            max_cosine_distance=0.2, embedder="mobilenet", half=True
        )
        return model, tracker


    model, tracker = load_model_and_tracker()
    cap = cv2.VideoCapture(video_path)

    # ==========================================
    # 5. VÒNG LẶP XỬ LÝ CHÍNH (PIPELINE CỦA BẠN)
    # ==========================================
    while cap.isOpened() and not stop_button:
        success, frame = cap.read()
        if not success:
            st.success("Đã xử lý xong toàn bộ video!")
            break

        # --- BƯỚC A: CHẠY YOLOV8 ---
        results = model.predict(frame, conf=0.3, verbose=False)[0]

        # --- BƯỚC B: THE DATA BRIDGE ---
        raw_detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            w, h = x2 - x1, y2 - y1
            raw_detections.append(([x1, y1, w, h], conf, cls_id))

        # --- BƯỚC C: TRACKING UPDATE ---
        tracks = tracker.update_tracks(raw_detections, frame=frame)

        # --- BƯỚC D: LẤY ID VÀ VẼ THỦ CÔNG ---
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bx1, by1, bx2, by2 = map(int, ltrb)

            # Tính tọa độ điểm D
            cx = int((bx1 + bx2) / 2)
            cy = int(by2)
            diem_D = (cx, cy)

            # Tính điểm C
            diem_C = track_history.get(track_id)

            if diem_C is not None and track_id not in counted_ids:
                if check_intersect(line_A, line_B, diem_C, diem_D):
                    if ccw(line_A, line_B, diem_C) == True:
                        count_in += 1
                    else:
                        count_out += 1
                    counted_ids.add(track_id)

            track_history[track_id] = diem_D

            # Phân loại và Vẽ Bounding Box
            class_id = track.get_det_class()
            if class_id is None:
                class_name = "Predicting"
            else:
                class_name = class_names.get(class_id, f"Unknown_{class_id}")
            label = f"{class_name} ID:{track_id}"

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.rectangle(frame, (bx1, by1 - 20), (bx1 + 140, by1), (0, 255, 0), -1)
            cv2.putText(frame, label, (bx1 + 5, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if diem_C is not None:
                cv2.line(frame, diem_C, diem_D, (0, 255, 255), 2)

        # Vẽ Vạch ảo
        cv2.line(frame, line_A, line_B, (0, 0, 255), 3)

        # Quản lý bộ nhớ
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_count % 100 == 0:
            active_ids = set([track.track_id for track in tracks if track.is_confirmed()])
            track_history = {tid: pos for tid, pos in track_history.items() if tid in active_ids}

        # --- ĐẨY DỮ LIỆU LÊN GIAO DIỆN WEB ---
        # 1. Cập nhật bảng số
        metric_in.metric(label="Tổng Xe Đi Vào (IN)", value=count_in)
        metric_out.metric(label="Tổng Xe Đi Ra (OUT)", value=count_out)

        # 2. Đẩy frame đã vẽ lên Web (Quan trọng: Channels="BGR" để Web không bị ám xanh)
        stframe.image(frame, channels="BGR", use_container_width=True)

    # Giải phóng bộ nhớ sau khi chạy xong
    cap.release()
    try:
        os.remove(video_path)  # Xóa file tạm sau khi dùng xong
    except:
        pass

else:
    st.info("Vui lòng tải lên một video từ cột bên trái để bắt đầu hệ thống.")