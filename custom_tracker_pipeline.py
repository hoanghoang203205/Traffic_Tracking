import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def ccw(A, B, C):
    """Check if point C is on the left (True) or right (False) of vector AB."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_intersect(A, B, C, D):
    """Check if line segment AB intersects with CD."""
    return ccw(C, D, A) != ccw(C, D, B) and ccw(A, B, C) != ccw(A, B, D)

# --- Configuration & State Management ---
line_A = (100, 350)
line_B = (600, 350)

track_history = {}  
counted_ids = set() 
count_in = 0
count_out = 0
class_names = {0: 'Bike', 1: 'Car', 2: 'Bus', 3: 'Truck'}

# --- Initialize Models ---
model = YOLO("runs/detect/Vehicle_Tracking/Night_Weather_Model5/weights/best.pt")
tracker = DeepSort(
    max_age=30, n_init=3, nms_max_overlap=1.0, 
    max_cosine_distance=0.2, embedder="mobilenet", half=True
)

cap = cv2.VideoCapture("Video_1.mp4")

# --- Main Pipeline ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Detection (YOLOv8)
    results = model.predict(frame, conf=0.3, verbose=False)[0]

    raw_detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        w, h = x2 - x1, y2 - y1
        raw_detections.append(([x1, y1, w, h], conf, cls_id))

    # 2. Tracking (DeepSORT)
    tracks = tracker.update_tracks(raw_detections, frame=frame)

    # 3. Processing & Analytics
    for track in tracks:
        if not track.is_confirmed():   
            continue

        track_id = track.track_id
        bx1, by1, bx2, by2 = map(int, track.to_ltrb())
        
        # Calculate center bottom of bounding box
        cx, cy = int((bx1 + bx2) / 2), int(by2)
        diem_D = (cx, cy)
        diem_C = track_history.get(track_id)

        # Intersection logic
        if diem_C is not None and track_id not in counted_ids:
            if check_intersect(line_A, line_B, diem_C, diem_D):
                if ccw(line_A, line_B, diem_C):
                    count_in += 1
                    print(f"[IN] Vehicle {track_id} entered.")
                else:
                    count_out += 1
                    print(f"[OUT] Vehicle {track_id} exited.")
                counted_ids.add(track_id)

        track_history[track_id] = diem_D

        # 4. Visualization
        class_id = track.get_det_class()
        class_name = class_names.get(class_id, f"Unknown_{class_id}") if class_id is not None else "Predicting"
        label = f"{class_name} ID:{track_id}"

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.rectangle(frame, (bx1, by1 - 20), (bx1 + 140, by1), (0, 255, 0), -1)
        cv2.putText(frame, label, (bx1+5, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        if diem_C is not None:
            cv2.line(frame, diem_C, diem_D, (0, 255, 255), 2)

    cv2.line(frame, line_A, line_B, (0, 0, 255), 3)

    # Memory cleanup (Garbage Collection)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 100 == 0:
        active_ids = {track.track_id for track in tracks if track.is_confirmed()}
        track_history = {tid: pos for tid, pos in track_history.items() if tid in active_ids}

    # Draw HUD
    cv2.putText(frame, f"IN: {count_in}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(frame, f"OUT: {count_out}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow("Custom Integrated Tracker Pipeline", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
