import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

CROP_SIZE = 224
# 📁 当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # D:\Desktop\UCL_Individual_Project\Approach1
print("BASE_DIR", BASE_DIR)
# ✅ 相对路径配置
ROI_DIR = BASE_DIR / "Classify" / "norm_class" / f"Extra_ROI_Image_224_0611_005853_epoch480.pt"
LABEL_ROOT= BASE_DIR / "Classify" / "labels_classify"
OUTPUT_ROOT = BASE_DIR / "Classify" / "norm_class" / f"Extra_ROI_Image_{CROP_SIZE}"


CLASS_MAP = {
    0: "Octopodichnus_didactylus",
    1: "Paleohelcura_dunbari",
    2: "Paleohelcura_lyonsensis",
    3: "Octopodichnus_minor",
    4: "Octopodichnus_raymondi",
    5: "Paleohelcura_tridactyla",
    6: "Mesichnium_benjamini",
    7: "Triavestigia_niningeri",
}


def get_label_from_txt(txt_path):
    with open(txt_path, "r") as f:
        for line in f:
            if line.strip():
                class_id = int(line.strip().split()[0])
                return CLASS_MAP.get(class_id, "unknown")
    return None


def collect_all_image_to_roi():
    # 返回: {原图名: [(roi_path, label名)]}
    img_to_rois = defaultdict(list)
    label_dirs = [LABEL_ROOT / sub for sub in ["train", "val", "test"]]
    for label_dir in label_dirs:
        for txt_file in label_dir.glob("*.txt"):
            base_name = txt_file.stem
            label = get_label_from_txt(txt_file)
            if not label or label == "unknown":
                continue
            # 所有属于该原图的 ROI
            for roi_img in ROI_DIR.glob(f"{base_name}_*.jpg"):
                img_to_rois[base_name].append((roi_img, label))
    return img_to_rois


def split_image_to_rois(img_to_rois, ratio):
    img_list = list(img_to_rois.keys())
    random.shuffle(img_list)
    n_train = int(len(img_list) * ratio[0])
    train_imgs = set(img_list[:n_train])
    val_imgs = set(img_list[n_train:])
    print(f"总原图数: {len(img_list)}，train: {len(train_imgs)}，val: {len(val_imgs)}")
    return train_imgs, val_imgs


def save_split_by_img(img_to_rois, train_imgs, val_imgs):
    stats = {"train": defaultdict(int), "val": defaultdict(int)}
    for split, img_set in [("train", train_imgs), ("val", val_imgs)]:
        for base_name in img_set:
            for roi_path, label in img_to_rois[base_name]:
                out_dir = OUTPUT_ROOT / split / label
                out_dir.mkdir(parents=True, exist_ok=True)
                dst = out_dir / roi_path.name
                shutil.copy2(roi_path, dst)
                stats[split][label] += 1

    # 输出统计结果
    for split in ["train", "val"]:
        print(f"\n🔎 {split}集类别数量统计：")
        for label, count in sorted(stats[split].items()):
            print(f"    {label}: {count}")

if __name__ == "__main__":
    random.seed(42)
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    img_to_rois = collect_all_image_to_roi()
    train_imgs, val_imgs = split_image_to_rois(img_to_rois, ratio=(0.7, 0.3))
    save_split_by_img(img_to_rois, train_imgs, val_imgs)
    print("\n🎉 数据集已按原图划分完成！")
