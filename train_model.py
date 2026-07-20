from ultralytics import YOLO

if __name__ == '__main__':
    # Tải file trọng số siêu nhẹ của YOLOv8 làm nền tảng
    model = YOLO('yolov8n.pt')

    # Bắt đầu huấn luyện với bộ Master Dataset
    results = model.train(
        data='Master_Dataset/data.yaml',
        epochs=50,          # Chạy 50 vòng là đủ để model "ngộ" ra các đặc trưng ban đêm/mưa
        imgsz=640,          # Kích thước ảnh chuẩn của YOLO
        batch=16,           # Phù hợp với card 4GB VRAM
        device=0,           # Chạy trên GPU
        project='Vehicle_Tracking', # Tên thư mục lưu kết quả
        name='Night_Weather_Model'  # Tên phiên bản model
    )
