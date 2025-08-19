import os
import re
import cv2
import shutil
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Configuration (paths consistent with your original) ----------------
BASE_DIR = Path(__file__).resolve().parent.parent  # e.g. D:\Desktop\UCL_Individual_Project\Approach1

SEG_MODEL_PATH = BASE_DIR / "Segmentation" / "runs" / "segment" / "yolov8s_0611_005853" / "weights" / "epoch480.pt"
CLS_MODEL_PATH = BASE_DIR / "Classify" / "yolo_cls_weights" / "yolov8s-cls_grey_0612_012918" / "weights" / "best.pt"

# Expect images to be under: Seg_and_Cls/NewImage/<split>/ (train|val|test)
IMAGE_ROOT = BASE_DIR / "Seg_and_Cls" / "NewImage"

# Extract timestamp from classifier weights path for output naming
_match = re.search(r'yolov8s-cls_grey_([\d_]+)', str(CLS_MODEL_PATH))
cls_weight_timestamp = _match.group(1) if _match else "timestamp_not_found"

EXTRACTION_MODE = "mask"  # "bbox" or "mask"
CROP_SIZE = 224
OUTPUT_ROOT = BASE_DIR / "Seg_and_Cls" / "outputs" / f"NewImage_{EXTRACTION_MODE}_{CROP_SIZE}_{cls_weight_timestamp}"
OUTPUT_CSV = "Predict.csv"

META_XLSX_PATH = BASE_DIR / "Seg_and_Cls" / "Specimen list_new.xlsx"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ---------------- Utilities ----------------
def resize_and_pad(image, target_size: int):
    """
    Resize then pad to a square (target_size x target_size).
    Rule A: if max dimension >= target_size/2 -> scale so max dim == target_size
    Rule B: otherwise -> scale so max dim == 320
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim >= target_size / 2:
        scale = target_size / max_dim
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 320 / max_dim
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    new_h, new_w = resized.shape[:2]
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return padded


def extract_region(image, mask, mode: str):
    """
    Extract ROI using the mask's bounding rectangle, optionally applying the mask itself,
    then resize+pad to CROP_SIZE.
    """
    x, y, w, h = cv2.boundingRect(mask)
    if mode == "mask":
        masked = cv2.bitwise_and(image, image, mask=mask)
        roi = masked[y:y + h, x:x + w]
    else:
        roi = image[y:y + h, x:x + w]
    return resize_and_pad(roi, CROP_SIZE)


def annotate_image(img, bbox, label, colour=(0, 255, 0)):
    """Draw a rectangle and a label on the image (for full-image visualisation)."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colour, 2)


def load_class_names(cls_model):
    """Return class names ordered by index from a YOLO classifier model."""
    if isinstance(cls_model.names, dict):
        return [cls_model.names[i] for i in sorted(cls_model.names.keys())]
    return list(cls_model.names)


def load_ground_truth(meta_path: Path):
    """
    Load ground truth mapping from Excel.
    Returns: { SPECIMEN_ID (no spaces, uppercased) -> 'Genus_species' }
    """
    try:
        print("Loading ground-truth metadata (Excel)...")
        df = pd.read_excel(meta_path)
        df["label_gt"] = df["Ichnogenus"].str.strip() + "_" + df["Ichnospecies"].str.strip()
        df["Specimen number"] = df["Specimen number"].str.replace(" ", "").str.upper()
        lookup = pd.Series(df["label_gt"].values, index=df["Specimen number"]).to_dict()
        print("Ground-truth metadata loaded.")
        return lookup
    except FileNotFoundError:
        print(f"Warning: metadata file not found: {meta_path}")
        return {}


def iter_images(split_dir: Path):
    """Yield image paths under the given split directory."""
    if not split_dir.exists():
        return
    for p in split_dir.rglob("*"):
        if p.suffix in IMAGE_EXTS:
            yield p


def auto_create_split_if_requested(src_root: Path, dst_root: Path, ratio=(0.6, 0.2, 0.2)):
    """
    Optionally create a 60:20:20 split from images directly under `src_root` (no subfolders required).
    The split is created under `dst_root/train|val|test`. This function is NO-OP unless explicitly called.
    Split is filename-based and does not infer labels.
    """
    all_imgs = [p for p in src_root.rglob("*") if p.suffix in IMAGE_EXTS]
    if not all_imgs:
        print(f"No images found for auto-split under: {src_root}")
        return

    random.shuffle(all_imgs)
    n = len(all_imgs)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    n_test = n - n_train - n_val
    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:]
    }

    if dst_root.exists():
        shutil.rmtree(dst_root)
    for sp in ["train", "val", "test"]:
        (dst_root / sp).mkdir(parents=True, exist_ok=True)
        for p in splits[sp]:
            shutil.copy2(p, dst_root / sp / p.name)

    print(f"Auto-split completed under: {dst_root}")
    print(f"Counts → train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")


# ---------------- Main prediction & evaluation ----------------
def run_prediction(split: str):
    """
    Run segmentation + classification on images under IMAGE_ROOT/<split>/.
    Saves per-image prediction text and annotated preview to OUTPUT_ROOT/<split>/.
    Returns the DataFrame of specimen-level results (Specimen ID, GT, Prediction, Correct).
    """
    split_dir = IMAGE_ROOT / split
    if not split_dir.exists():
        print(f"Error: split directory does not exist: {split_dir}")
        return pd.DataFrame()

    out_dir = OUTPUT_ROOT / split
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    seg_model = YOLO(SEG_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)
    class_names = load_class_names(cls_model)

    # GT lookup (if available)
    gt_lookup = load_ground_truth(META_XLSX_PATH)

    records = []

    image_paths = list(iter_images(split_dir))
    if not image_paths:
        print(f"No images found under: {split_dir}")
        return pd.DataFrame()

    for img_path in tqdm(image_paths, desc=f"Processing images ({split})"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: failed to read image: {img_path}")
            continue

        annotated = image.copy()
        pred_txt_path = out_dir / f"{img_path.stem}.txt"

        # Direct classification (whole image)
        pred = cls_model.predict(image, verbose=False)[0]
        probs = pred.probs.data.cpu().numpy()
        top_idx = int(np.argmax(probs))
        label = class_names[top_idx]
        conf = float(probs[top_idx])

        # Save per-image prediction
        with open(pred_txt_path, "w") as f_out:
            f_out.write(f"{label} {conf:.4f}\n")

        # Optional visual overlay on the entire image frame
        annotate_image(annotated, (0, 0, image.shape[1], image.shape[0]), label)
        out_img = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_img), annotated)

        # Try to parse specimen ID like "ABC 123" from the filename start
        m = re.match(r"^([A-Z]+ [A-Z0-9]+)", img_path.stem)
        specimen_id = m.group(1).replace(" ", "").upper() if m else None

        gt_label = gt_lookup.get(specimen_id, "Unknown") if specimen_id else "Unknown"
        correct = (label == gt_label) if gt_label != "Unknown" else False

        records.append({
            "Split": split,
            "Filename": img_path.name,
            "Specimen ID": specimen_id if specimen_id else "",
            "Ground Truth": gt_label,
            "Prediction": label,
            "Confidence": conf,
            "Correct": correct
        })

    df = pd.DataFrame(records)
    # Save CSV per split
    csv_path = out_dir / OUTPUT_CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved per-image predictions to: {csv_path}")

    # Accuracy summary for this split (if GT available)
    if not df.empty and "Correct" in df.columns:
        acc = df["Correct"].mean() if df["Correct"].notna().any() else float("nan")
        print(f"Accuracy on '{split}': {acc:.2%}" if not np.isnan(acc) else f"Accuracy on '{split}': N/A")

        # Simple bar plot: correct vs incorrect
        cnt = df["Correct"].value_counts()
        labels = ["Incorrect", "Correct"] if False in cnt.index else ["Correct"]
        values = [cnt.get(False, 0), cnt.get(True, 0)]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.title(f"Classification Accuracy ({split}): {acc:.2%}" if not np.isnan(acc) else f"Classification Accuracy ({split})")
        plt.ylabel("Number of Predictions")
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy_bar_plot.png")
        plt.close()

        # Append an accuracy line to the CSV file
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write("\n")
            if not np.isnan(acc):
                f.write(f"The classification accuracy on '{split}' is {acc:.2%}\n")
            else:
                f.write(f"The classification accuracy on '{split}' is N/A (no ground truth matched)\n")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="New images prediction with optional 60:20:20 split and per-split evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to run: train, val, or test.")
    parser.add_argument("--auto_split", action="store_true",
                        help="If set, create a 60:20:20 split under 'Seg_and_Cls/NewImage_split' from flat images under 'Seg_and_Cls/NewImage'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for auto-split.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Optionally create a 60:20:20 split from flat images under NewImage/
    if args.auto_split:
        src = IMAGE_ROOT           # expects images directly under this root (no split folders)
        dst = BASE_DIR / "Seg_and_Cls" / "NewImage_split"
        auto_create_split_if_requested(src_root=src, dst_root=dst, ratio=(0.6, 0.2, 0.2))
        # Switch to the new split root
        global IMAGE_ROOT
        IMAGE_ROOT = dst

    # Ensure the selected split exists
    split_dir = IMAGE_ROOT / args.split
    if not split_dir.exists():
        print(f"Error: expected directory does not exist: {split_dir}")
        print("Tip: create 'train', 'val', 'test' subfolders under the images root, "
              "or run with --auto_split to generate a 60:20:20 split.")
        return

    # Run prediction + evaluation on the requested split
    _ = run_prediction(args.split)


if __name__ == "__main__":
    main()
