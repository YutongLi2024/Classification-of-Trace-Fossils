import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

# 📁 当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent.parent  # D:\Desktop\UCL_Individual_Project\Approach1

# ✅ 相对路径配置
SEG_MODEL_PATH = BASE_DIR / "Segmentation" / "runs" / "segment" / "yolov8s_0611_005853" / "weights" / "epoch480.pt"
CLS_MODEL_PATH = BASE_DIR / "Classify" / "yolo_cls_weights" /"yolov8s-cls_grey_0612_012918" / "weights" / "best.pt" # best
IMAGE_DIR      = BASE_DIR / "Seg_and_Cls" /"NewImage"
image_name_path = "_".join(IMAGE_DIR.parts[-2:])

match = re.search(r'yolov8s-cls_grey_([\d_]+)', str(CLS_MODEL_PATH))  # 匹配 "yolov8s-cls_" 后面跟着的日期时间字符串 (例如 0602_122422)
cls_weight_timestamp = match.group(1)
EXTRACTION_MODE = "mask"  # "bbox" or "mask"
CROP_SIZE = 224
OUTPUT_DIR     = BASE_DIR / "Seg_and_Cls" / "outputs" / f"NewImage_{image_name_path}_{EXTRACTION_MODE}_{CROP_SIZE}_{cls_weight_timestamp}"
OUTPUT_CSV = "Predict.csv"
# Meta_xlsx_path=BASE_DIR /"Seg_and_Cls" / "Specimen list.xlsx"
Meta_xlsx_path=BASE_DIR /"Seg_and_Cls" / "Specimen list_new.xlsx"


def resize_and_pad(image, target_size):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    # 规则1、2、3：最大边都 >= target_size/2
    if max_dim >= target_size / 2:
        scale = target_size / max_dim
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        new_h, new_w = resized.shape[:2]
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
        return padded

    # 规则4：太小的区域，最大边缩放到 320，再填充到 640x640
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


def extract_region(image, mask, mode):
    if mode == "mask":
        x, y, w, h = cv2.boundingRect(mask)
        masked = cv2.bitwise_and(image, image, mask=mask)
        roi = masked[y:y+h, x:x+w]
        return resize_and_pad(roi, CROP_SIZE)
    else:  # bbox + padding
        x, y, w, h = cv2.boundingRect(mask)
        roi = image[y:y+h, x:x+w]
        return resize_and_pad(roi, CROP_SIZE)


def annotate_image(img, bbox, label, color=(0,255,0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载分类模型
    cls_model = YOLO(CLS_MODEL_PATH)
    class_names = cls_model.names

    image_paths = list(Path(IMAGE_DIR).rglob("*.*")) 

    for img_path in tqdm(image_paths, desc="处理图像"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ 无法读取图像: {img_path}")
            continue

        annotated = image.copy()
        pred_txt_path = Path(OUTPUT_DIR) / f"{img_path.stem}.txt"

        # 直接分类
        pred = cls_model.predict(image, verbose=False)[0]
        probs = pred.probs.data.cpu().numpy()
        top_idx = int(np.argmax(probs))
        label = class_names[top_idx]
        conf = float(probs[top_idx])

        # 保存分类结果
        with open(pred_txt_path, "w") as f_out:
            f_out.write(f"{label} {conf:.4f}\n")

        # 可选：画整个图像的边框和标签
        annotate_image(annotated, (0, 0, image.shape[1], image.shape[0]), label)
        out_path = Path(OUTPUT_DIR) / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), annotated)


def evaluate_predictions(Meta_xlsx_path, pred_dir, output_csv):
    # 加载元数据
    df = pd.read_excel(Meta_xlsx_path)
    df["label_gt"] = df["Ichnogenus"].str.strip() + "_" + df["Ichnospecies"].str.strip()
    df["Specimen number"] = df["Specimen number"].str.replace(" ", "").str.upper()

    results = []
    pred_dir = Path(pred_dir)
    for pred_txt in pred_dir.glob("*.txt"):
        # 用正则表达式提取编号
        match = re.match(r"^([A-Z]+ [A-Z0-9]+)", pred_txt.stem)
        if not match:
            print(f"❌ 无法从文件名中提取 specimen 编号: {pred_txt.name}")
            continue
        specimen_id = match.group(1).replace(" ", "").upper()

        gt_row = df[df["Specimen number"] == specimen_id]
        if gt_row.empty:
            continue
        gt_label = gt_row.iloc[0]["label_gt"]
        with open(pred_txt, "r") as f:
            pred_label = f.readline().strip().split()[0]
        results.append({
            "Specimen ID": specimen_id,
            "Ground Truth": gt_label,
            "Prediction": pred_label,
            "Correct": pred_label == gt_label
        })

    # 保存 CSV 文件
    results_df = pd.DataFrame(results)
    print("📋 当前结果 DataFrame 列名：", results_df.columns.tolist())
    print("📦 当前结果数据条数：", len(results_df))
    print(results_df.head())  # 查看前几条记录
    csv_path = pred_dir / output_csv
    results_df.to_csv(csv_path, index=False)

    # 输出准确率
    acc = results_df["Correct"].mean()
    print(f"\n📊 分类准确率: {acc:.2%}，详情见: {csv_path}")
    # ✅ 将准确率写入 CSV 文件末尾一行（追加模式）
    with open(csv_path, "a") as f:
        f.write("\n")  # 空行
        f.write(f"The classification Accuracy is {acc:.2%}\n")  

    # 生成柱状图
    count = results_df["Correct"].value_counts()
    labels = ["Incorrect", "Correct"] if False in count.index else ["Correct"]
    values = [count.get(False, 0), count.get(True, 0)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["red", "green"])
    plt.title(f"Classification Accuracy: {acc:.2%}")
    plt.ylabel("Number of Predictions")
    plt.tight_layout()
    plt.savefig(pred_dir / "accuracy_bar_plot.png")
    plt.close()



if __name__ == "__main__":
    main()
    # 分类评估
    evaluate_predictions(
        Meta_xlsx_path=Meta_xlsx_path,
        pred_dir=OUTPUT_DIR,
        output_csv= OUTPUT_CSV
    )