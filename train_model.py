from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('yolov8n.pt')


    results = model.train(
        data='Master_Dataset/data.yaml',
        epochs=50,          
        imgsz=640,         
        batch=16,          
        device=0,           
        project='Vehicle_Tracking', 
        name='Night_Weather_Model'  
    )
