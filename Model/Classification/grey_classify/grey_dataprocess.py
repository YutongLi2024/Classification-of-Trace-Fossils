import shutil
import cv2
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent # 当前脚本所在目录: D:\Desktop\UCL_Individual_Project\Approach1\Classify\
INPUT_DIR = BASE_DIR / "norm_class" / "Extra_ROI_Image_224_augmented_yolo_style"
OUTPUT_DIR = BASE_DIR / "norm_class" / "Grey_Extra_ROI_Image_224_augmented_yolo_style"


if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_path in INPUT_DIR.rglob("*.jpg"):  # 递归遍历所有子文件夹下的jpg
    rel_path = img_path.relative_to(INPUT_DIR)
    save_path = OUTPUT_DIR / rel_path
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 创建对应子目录
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(save_path), gray_3c)