import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

CROP_SIZE = 224
# Base directory of the project (e.g., D:\Desktop\UCL_Individual_Project\Approach1)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print("BASE_DIR", BASE_DIR)

# Relative path configuration
ROI_DIR = BASE_DIR / "Classify" / "norm_class" / f"Extra_ROI_Image_224_0611_005853_epoch480.pt"
LABEL_ROOT = BASE_DIR / "Classify" / "labels_classify"
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


def get_label_from_txt(txt_path: Path):
    """Return the first mapped class label from a YOLO-style txt file, or None/unknown."""
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            class_id = int(s.split()[0])
            return CLASS_MAP.get(class_id, "unknown")
    return None


def collect_all_image_to_roi():
    """
    Build a mapping:
        { base_image_stem: [(roi_image_path, class_label), ...] }
    Labels are gathered from train/val/test label folders; ROIs are matched by stem.
    """
    img_to_rois = defaultdict(list)
    label_dirs = [LABEL_ROOT / sub for sub in ["train", "val", "test"]]
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        for txt_file in label_dir.glob("*.txt"):
            base_name = txt_file.stem
            label = get_label_from_txt(txt_file)
            if not label or label == "unknown":
                continue
            # Collect all ROI images that belong to this original image
            for roi_img in ROI_DIR.glob(f"{base_name}_*.jpg"):
                img_to_rois[base_name].append((roi_img, label))
    return img_to_rois


def split_image_to_rois(img_to_rois, ratio=(0.6, 0.2, 0.2)):
    """
    Split by original image (not by ROI) to avoid leakage across splits.
    ratio := (train, val, test) and must sum to 1.0.
    """
    assert len(ratio) == 3, "Ratio must be a 3-tuple: (train, val, test)."
    total = sum(ratio)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    img_list = list(img_to_rois.keys())
    random.shuffle(img_list)

    n_total = len(img_list)
    n_train = int(n_total * ratio[0])
    n_val = int(n_total * ratio[1])
    n_test = n_total - n_train - n_val  # ensure full coverage

    train_imgs = set(img_list[:n_train])
    val_imgs = set(img_list[n_train:n_train + n_val])
    test_imgs = set(img_list[n_train + n_val:])

    print(f"Total original images: {n_total}, train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(test_imgs)}")
    return train_imgs, val_imgs, test_imgs


def save_split_by_img(img_to_rois, train_imgs, val_imgs, test_imgs):
    """
    Save ROI images grouped by split and class label:
        OUTPUT_ROOT/{train|val|test}/{label}/<roi_file>
    Also print per-split class counts.
    """
    stats = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    for split, img_set in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        for base_name in img_set:
            for roi_path, label in img_to_rois[base_name]:
                out_dir = OUTPUT_ROOT / split / label
                out_dir.mkdir(parents=True, exist_ok=True)
                dst = out_dir / roi_path.name
                shutil.copy2(roi_path, dst)
                stats[split][label] += 1

    # Summary per split
    for split in ["train", "val", "test"]:
        print(f"\n{split} split class counts:")
        for label, count in sorted(stats[split].items()):
            print(f"    {label}: {count}")


if __name__ == "__main__":
    random.seed(42)

    # Recreate output root
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    img_to_rois = collect_all_image_to_roi()
    train_imgs, val_imgs, test_imgs = split_image_to_rois(img_to_rois, ratio=(0.6, 0.2, 0.2))
    save_split_by_img(img_to_rois, train_imgs, val_imgs, test_imgs)
    print("\nDataset has been split by original images and saved to:", OUTPUT_ROOT)
