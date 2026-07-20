import os
import shutil

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ MAPPING
# ==========================================
# Thay bằng tên thư mục thực tế máy bạn vừa tải về
DIR_WEATHER = "Adverse-Weather-1"
DIR_NIGHT = "vehicles-night-2"

DIR_MASTER = "Master_Dataset"

# Bản đồ quy đổi: {ID cũ : ID mới chuẩn}. Đặt -1 nếu muốn xóa nhãn đó.
MAP_WEATHER = {0: 1, 1: 0, 2: 2, 3: -1, 4: 1, 5: 3}

# LƯU Ý: Bạn hãy soi lại ảnh bộ Night để sửa dictionary này cho đúng thực tế nhé!
MAP_NIGHT = {0: 0, 1: 1, 2: 2, 3: 3}


# ==========================================
# HÀM XỬ LÝ LÕI
# ==========================================
def process_dataset(src_dir, dest_dir, label_map, prefix=""):
    for split in ['train', 'valid', 'test']:
        src_img_dir = os.path.join(src_dir, split, 'images')
        src_lbl_dir = os.path.join(src_dir, split, 'labels')

        dest_img_dir = os.path.join(dest_dir, split, 'images')
        dest_lbl_dir = os.path.join(dest_dir, split, 'labels')

        # Tạo thư mục đích nếu chưa có
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)

        if not os.path.exists(src_img_dir): continue

        for img_name in os.listdir(src_img_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            # Thêm prefix vào tên file để tránh 2 bộ dữ liệu có ảnh trùng tên nhau
            new_name = prefix + img_name
            lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
            new_lbl_name = prefix + lbl_name

            src_img_path = os.path.join(src_img_dir, img_name)
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
            dest_img_path = os.path.join(dest_img_dir, new_name)
            dest_lbl_path = os.path.join(dest_lbl_dir, new_lbl_name)

            # Kiểm tra xem có file label không
            if not os.path.exists(src_lbl_path): continue

            # Đọc file label cũ, quy đổi ID, ghi ra file label mới
            valid_labels = []
            with open(src_lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    old_id = int(parts[0])

                    if old_id in label_map:
                        new_id = label_map[old_id]
                        if new_id != -1:  # Chỉ giữ lại ID hợp lệ
                            parts[0] = str(new_id)
                            valid_labels.append(" ".join(parts))

            # Chỉ copy ảnh và tạo txt nếu bức ảnh đó còn nhãn (sau khi lọc)
            if len(valid_labels) > 0:
                shutil.copy(src_img_path, dest_img_path)
                with open(dest_lbl_path, 'w') as f:
                    f.write("\n".join(valid_labels) + "\n")


# ==========================================
# THỰC THI
# ==========================================
print("Bắt đầu gộp bộ Adverse Weather...")
process_dataset(DIR_WEATHER, DIR_MASTER, MAP_WEATHER, prefix="weather_")

print("Bắt đầu gộp bộ Vehicles Night...")
process_dataset(DIR_NIGHT, DIR_MASTER, MAP_NIGHT, prefix="night_")

# Tạo file data.yaml cho bộ Master
yaml_content = f"""
path: ../{DIR_MASTER} # Đường dẫn tương đối từ vị trí chạy file train
train: train/images
val: valid/images
test: test/images

nc: 4
names: ['Bike', 'Car', 'Bus', 'Truck']
"""

with open(os.path.join(DIR_MASTER, "data.yaml"), 'w') as f:
    f.write(yaml_content.strip())

print("Hoàn tất! Bộ dữ liệu vàng đã sẵn sàng trong thư mục Master_Dataset.")