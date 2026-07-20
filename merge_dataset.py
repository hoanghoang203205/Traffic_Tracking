import os
import shutil


DIR_WEATHER = "Adverse-Weather-1"
DIR_NIGHT = "vehicles-night-2"

DIR_MASTER = "Master_Dataset"


MAP_WEATHER = {0: 1, 1: 0, 2: 2, 3: -1, 4: 1, 5: 3}


MAP_NIGHT = {0: 0, 1: 1, 2: 2, 3: 3}



def process_dataset(src_dir, dest_dir, label_map, prefix=""):
    for split in ['train', 'valid', 'test']:
        src_img_dir = os.path.join(src_dir, split, 'images')
        src_lbl_dir = os.path.join(src_dir, split, 'labels')

        dest_img_dir = os.path.join(dest_dir, split, 'images')
        dest_lbl_dir = os.path.join(dest_dir, split, 'labels')

        
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)

        if not os.path.exists(src_img_dir): continue

        for img_name in os.listdir(src_img_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            
            new_name = prefix + img_name
            lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
            new_lbl_name = prefix + lbl_name

            src_img_path = os.path.join(src_img_dir, img_name)
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
            dest_img_path = os.path.join(dest_img_dir, new_name)
            dest_lbl_path = os.path.join(dest_lbl_dir, new_lbl_name)

           
            if not os.path.exists(src_lbl_path): continue

            
            valid_labels = []
            with open(src_lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    old_id = int(parts[0])

                    if old_id in label_map:
                        new_id = label_map[old_id]
                        if new_id != -1:  
                            parts[0] = str(new_id)
                            valid_labels.append(" ".join(parts))

            
            if len(valid_labels) > 0:
                shutil.copy(src_img_path, dest_img_path)
                with open(dest_lbl_path, 'w') as f:
                    f.write("\n".join(valid_labels) + "\n")




process_dataset(DIR_WEATHER, DIR_MASTER, MAP_WEATHER, prefix="weather_")


process_dataset(DIR_NIGHT, DIR_MASTER, MAP_NIGHT, prefix="night_")


yaml_content = f"""
path: ../{DIR_MASTER} 
train: train/images
val: valid/images
test: test/images

nc: 4
names: ['Bike', 'Car', 'Bus', 'Truck']
"""

with open(os.path.join(DIR_MASTER, "data.yaml"), 'w') as f:
    f.write(yaml_content.strip())

