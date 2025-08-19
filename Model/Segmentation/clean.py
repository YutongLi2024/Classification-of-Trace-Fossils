import os
import shutil
import random


base_dir = "." 
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

output_dirs = {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2
}


for split in output_dirs:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)


all_images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(all_images)


total = len(all_images)
start = 0
for split, ratio in output_dirs.items():
    end = start + int(total * ratio)
    split_images = all_images[start:end]
    for img in split_images:
        img_path = os.path.join(images_dir, img)
        img_dest = os.path.join(images_dir, split, img)

        label_file = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        label_dest = os.path.join(labels_dir, split, label_file)

        
        shutil.move(img_path, img_dest)

        if os.path.exists(label_path):
            shutil.move(label_path, label_dest)
        else:
            print(f"[警告] 未找到对应标签: {label_file}")
    start = end

