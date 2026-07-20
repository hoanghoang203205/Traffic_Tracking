import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def ccw(A, B, C):
    """
    Kiểm tra xem điểm C nằm bên Trái (True) hay Phải (False) của vector AB.
    Tọa độ dạng tuple: (x, y)
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_intersect(A, B, C, D):
    """
    Kiểm tra đoạn thẳng AB (Vạch kẻ) có cắt đoạn CD (Quỹ đạo xe) hay không.
    """
    return ccw(C, D, A) != ccw(C, D, B) and ccw(A, B, C) != ccw(A, B, D)


# 1. Cấu hình Vạch ảo (Virtual Line)
line_A = (100, 350)
line_B = (600, 350)

# 2. Khởi tạo "Bộ nhớ đệm" (State Management)
track_history = {}  # Dictionary (Tọa độ quá khứ C)
counted_ids = set() # Set (Khóa ID đã đếm)

# 3. Biến tổng (Analytics)
count_in = 0
count_out = 0
class_names = {0: 'Bike', 1: 'Car', 2: 'Bus', 3: 'Truck'}
# Khởi tạo YOLO và DeepSORT Tracker ở đây...

# 1. Khởi tạo mô hình YOLOv8n đã train
model = YOLO("runs/detect/Vehicle_Tracking/Night_Weather_Model5/weights/best.pt")

# 2. Khởi tạo bộ theo dõi DeepSORT thủ công
# max_age: Số frame tối đa giữ lại vết của xe khi bị khuất
# embedder: Sử dụng mobilenet siêu nhẹ để trích xuất đặc trưng ngoại hình nhằm bảo đảm FPS
tracker = DeepSort(
    max_age=30,                 # Nhớ ID tối đa 30 frame khi xe bị che khuất. Sau 30 frame thi dung track, xoa ID
    n_init=3,                   # Yêu cầu xuất hiện liên tục 3 frame mới cấp ID
    nms_max_overlap=1.0,        # Neu IOU cua 2 bbox > 1. Coi nhu la 1 object
    max_cosine_distance=0.2,    # Khoảng cách cosine giữa 2 vector đặc trưng, > 0.2 thì coi là 2 object khác nhau
    embedder="mobilenet",       # Mạng CNN siêu nhẹ dùng để trích xuất đặc trưng màu sắc/hình khối
    half=True                   # Sử dụng FP16 để tối ưu hóa VRAM trên RTX 3050
)

# 3. Đọc luồng Video
video_path = "Video_1.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- BƯỚC A: CHẠY YOLOV8 ĐỂ LẤY KHUNG HÌNH THÔ (DETECTION ONLY) ---
    results = model.predict(frame,
                            conf=0.3,           # Bbox that has Confidence score > 0.3 is kept.
                            verbose=False)[0]   # predict() returns a list, [0] extracts the current frame's result

    # --- BƯỚC B: TỰ TẠO CẦU NỐI DỮ LIỆU (THE DATA BRIDGE) ---
    raw_detections = []
    for box in results.boxes:
        # Lấy tọa độ dạng [x1, y1, x2, y2] từ GPU chuyển về CPU dưới dạng Numpy

        # [0] dùng để bóc vỏ Tensor từ (1, 4) thành (4,) trước khi đẩy về CPU và Numpy
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # [0] trích xuất trực tiếp giá trị số từ Tensor (1,)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        # TOÁN HỌC BIẾN ĐỔI: Chuyển sang [left, top, width, height] theo yêu cầu của DeepSORT
        w = x2 - x1
        h = y2 - y1
        left = x1
        top = y1

        # Đóng gói đúng định dạng chuẩn của Tracker: ([ltwh], confidence, class_id)
        raw_detections.append(([left, top, w, h], conf, cls_id))

    # --- BƯỚC C: CẬP NHẬT TRẠNG THÁI VÀ GHÉP CẶP ID (TRACKING UPDATE) ---
    # Truyền trực tiếp danh sách vừa đóng gói và frame ảnh hiện tại vào Tracker
    tracks = tracker.update_tracks(raw_detections, frame=frame)

    # --- BƯỚC D: LẤY ID ĐÃ GHÉP CẶP VÀ VẼ THỦ CÔNG ---
    for track in tracks:
        # Nếu Tracker chưa xác nhận được vết di chuyển đủ tin cậy thì bỏ qua

        if not track.is_confirmed():   #Bỏ qua các track chưa tồn tại đủ số frame (n_init=3) để lọc nhiễu
            continue

        # 2. ĐỊNH DANH (ID): Lấy số báo danh duy nhất (bất biến qua các frame nhờ cơ chế max_age)
        track_id = track.track_id



        # Lấy tọa độ hộp đã được bộ lọc Kalman Filter tối ưu [left, top, right, bottom]
        ltrb = track.to_ltrb()
        bx1, by1, bx2, by2 = map(int, ltrb)

        # Tính tọa độ Tâm đáy của Bounding Box (Điểm bánh xe chạm đường)
        cx = int((bx1 + bx2) / 2)
        cy = int(by2)
        diem_D = (cx, cy)

        # Rút tọa độ Quá khứ từ Dictionary (Đây chính là Điểm C)
        diem_C = track_history.get(track_id)

        # Nếu xe này đã tồn tại từ frame trước (Có điểm C)
        # VÀ nó chưa từng bị đếm (chưa nằm trong counted_ids)
        if diem_C is not None and track_id not in counted_ids:

            # Xét giao cắt giữa Vạch (A, B) và Quỹ đạo xe (C, D)
            if check_intersect(line_A, line_B, diem_C, diem_D):

                # Nếu cắt -> Xét hướng In/Out dựa vào vị trí của C
                if ccw(line_A, line_B, diem_C) == True:
                    count_in += 1
                    print(f"[IN] Xe {track_id} vừa đi vào!")
                else:
                    count_out += 1
                    print(f"[OUT] Xe {track_id} vừa đi ra!")

                # Khóa ID này lại, các frame sau có cắt nữa cũng không đếm
                counted_ids.add(track_id)

        # Cập nhật vị trí Hiện tại (D) thành Quá khứ (C) cho vòng lặp sau
        track_history[track_id] = diem_D




        # 3. PHÂN LOẠI (Class): Lấy nhãn vật thể dạng số nguyên (Passthrough từ YOLO, vd: 2=car, 3=moto)
        class_id = track.get_det_class()  # Lấy lại mã lớp (0: Bike, 1: Car,...)
        if class_id is None:
            class_name = "Predicting"
        else:
            class_name = class_names.get(class_id, f"Unknown_{class_id}")
        label = f"{class_name} ID:{track_id}"

        # Vẽ Bounding Box bằng OpenCV
        cv2.rectangle(frame,         # Ma trận ảnh gốc đang bị ghi đè màu trực tiếp (in-place)
                      (bx1, by1),    # Điểm neo Top-Left (dữ liệu mượt từ to_ltrb)
                      (bx2, by2),    # Điểm neo Bottom-Right
                      (0, 255, 0),   # Chuẩn màu BGR của OpenCV
                                2)   # Độ dày nét vẽ (pixel)


        # Vẽ nền nhãn
        cv2.rectangle(frame, (bx1, by1 - 20), (bx1 + 140, by1), (0, 255, 0), -1)
        # Ghi chữ ID và Class lên hình
        cv2.putText(
             frame,                      #: Ghi đè trực tiếp lên ma trận ảnh hiện tại
             label,                      #: Chuỗi string cần hiển thị (ví dụ: "ID: 15")
             (bx1+5, by1-5),             #: Điểm neo (Bottom-Left của text). Đẩy lùi vào 5px và nhô lên 5px để không đè vạch Bbox
             cv2.FONT_HERSHEY_SIMPLEX,   #: Font không chân, nét đơn tối ưu cho tốc độ OpenCV
             0.5,                        #: Hệ số scale kích thước chữ
             (0, 0, 0),                  #: Màu BGR -> Đen
             1,                          #: Độ dày nét chữ (1 pixel)
             cv2.LINE_AA,                #: Anti-Aliasing (Khử răng cưa) giúp chữ hiển thị mượt trên video
        )
        if diem_C is not None:
            cv2.line(frame, diem_C, diem_D, (0, 255, 255), 2)
    # Vẽ Vạch ảo
    cv2.line(frame, line_A, line_B, (0, 0, 255), 3)

    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_count % 100 == 0:
        # Lấy danh sách các ID ĐANG CÒN TRÊN MÀN HÌNH hiện tại
        active_ids = set([track.track_id for track in tracks if track.is_confirmed()])

        # Lọc track_history: Chỉ giữ lại tọa độ của những xe đang active
        track_history = {tid: pos for tid, pos in track_history.items() if tid in active_ids}



    # Vẽ Bảng thống kê In/Out
    cv2.putText(frame, f"IN: {count_in}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(frame, f"OUT: {count_out}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Custom Integrated Tracker Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()