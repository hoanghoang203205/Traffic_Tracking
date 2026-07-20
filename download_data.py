from roboflow import Roboflow

# 1. Khai báo API Key của bạn
rf = Roboflow(api_key="XOpAON6BtGKUKSpBCP2T")

print("Đang tải bộ dữ liệu thời tiết xấu (Adverse Weather)...")
# Thông số bóc tách từ link 1 của bạn
project_weather = rf.workspace("home-gpd4c").project("adverse-weather-pafty")
# Version thường là 1, nếu trên web báo version khác thì bạn đổi số 1 thành số đó
dataset_weather = project_weather.version(1).download("yolov8")

print("\nĐang tải bộ dữ liệu ban đêm (Vehicles Night)...")
# Thông số bóc tách từ link 2 của bạn
project_night = rf.workspace("kevin-pham").project("vehicles-night-dpkue")
dataset_night = project_night.version(2).download("yolov8") # Lưu ý: Bộ này tác giả đang để version chuẩn là 2

print("\nĐã tải xong toàn bộ dữ liệu!")