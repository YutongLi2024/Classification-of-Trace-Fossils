import os
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import re

# --- 配置部分 (保持不变) ---
# 📁 当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ 相对路径配置
SEG_MODEL_PATH = BASE_DIR / "Segmentation" / "runs" / "segment" / "yolov8s_0611_005853" / "weights" / "epoch480.pt"
CLS_MODEL_PATH = BASE_DIR / "Classify" / "yolo_cls_weights" / "yolov8s-cls_0612_005808" / "weights" / "best.pt"
IMAGE_DIR      = BASE_DIR / "Segmentation" / "images"
image_name_path = "_".join(IMAGE_DIR.parts[-2:])

match = re.search(r'yolov8s-cls_([\d_]+)', str(CLS_MODEL_PATH))
cls_weight_timestamp = match.group(1) if match else "timestamp_not_found"
EXTRACTION_MODE = "mask"
CROP_SIZE = 224
OUTPUT_DIR     = BASE_DIR / "Seg_and_Cls" / "outputs" / f"{EXTRACTION_MODE}_{CROP_SIZE}_{cls_weight_timestamp}"
# --- 修改点: 定义新的、包含所有信息的CSV文件名 ---
OUTPUT_RICH_CSV = "prediction_with_probabilities.csv"
Meta_xlsx_path = BASE_DIR / "Seg_and_Cls" / "Specimen list.xlsx"


def resize_and_pad(image, target_size):
    # (此函数无需改动)
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


def extract_region(image, mask, mode):
    # (此函数无需改动)
    if mode == "mask":
        x, y, w, h = cv2.boundingRect(mask)
        masked = cv2.bitwise_and(image, image, mask=mask)
        roi = masked[y:y+h, x:x+w]
        return resize_and_pad(roi, CROP_SIZE)
    else:
        x, y, w, h = cv2.boundingRect(mask)
        roi = image[y:y+h, x:x+w]
        return resize_and_pad(roi, CROP_SIZE)


# 这是您预测脚本的 main 函数的最终修正版

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    seg_model = YOLO(SEG_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)

    # --- 核心修改点: 正确地从模型中提取有序的类别名称 ---
    # model.names 是一个字典 {0: 'name_A', 1: 'name_B'}
    # 我们需要按索引顺序 (0, 1, 2...) 排列的名称列表
    if isinstance(cls_model.names, dict):
        class_names = [cls_model.names[i] for i in sorted(cls_model.names.keys())]
    else: # 兼容旧版本可能返回列表的情况
        class_names = cls_model.names
    print(f"✅ 已正确加载类别名称 (共 {len(class_names)} 个): {class_names}")

    # --- 加载并处理真值数据 (逻辑保持不变) ---
    print("🔄 正在加载并预处理真值数据 (Ground Truth)...")
    try:
        gt_df = pd.read_excel(Meta_xlsx_path)
        gt_df["label_gt"] = gt_df["Ichnogenus"].str.strip() + "_" + gt_df["Ichnospecies"].str.strip()
        gt_df["Specimen number"] = gt_df["Specimen number"].str.replace(" ", "").str.upper()
        gt_lookup = pd.Series(gt_df.label_gt.values, index=gt_df["Specimen number"]).to_dict()
        print("✅ 真值数据加载完毕。")
    except FileNotFoundError:
        print(f"❌ 错误: 未找到元数据文件 {Meta_xlsx_path}。")
        gt_lookup = {}

    results_list = []
    
    image_paths = list(Path(IMAGE_DIR).rglob("*.*")) 
    for img_path in tqdm(image_paths, desc="🧠 正在处理和预测图像"):
        # ... (图像读取、分割、ROI提取等代码保持完全不变) ...
        image = cv2.imread(str(img_path))
        if image is None: continue

        result = seg_model.predict(image, verbose=False)[0]
        masks = result.masks
        if masks is None or len(masks) == 0: continue
        
        masks_data = masks.data.cpu().numpy()
        conf_scores = result.boxes.conf.cpu().numpy()
        areas = np.array([np.sum(mask > 0.5) for mask in masks_data])
        score_area = conf_scores * areas
        max_idx = int(np.argmax(score_area))

        mask_raw = masks_data[max_idx]
        resized_mask = cv2.resize(mask_raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary_mask = (resized_mask * 255).astype(np.uint8)
        
        roi = extract_region(image, binary_mask, mode=EXTRACTION_MODE)
        if roi is None or roi.size == 0: continue
        
        # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # roi_gray_3c = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        
        # --- 分类预测 ---
        # pred = cls_model.predict(roi_gray_3c, verbose=False)[0]
        pred = cls_model.predict(roi, verbose=False)[0]
        
        probs_vector = pred.probs.data.cpu().numpy()
        top_idx = int(np.argmax(probs_vector))
        predicted_label = class_names[top_idx]

        # --- 整合信息 (逻辑保持不变) ---
        match = re.match(r"^([A-Z]+ [A-Z0-9]+)", img_path.stem)
        if not match: continue
        specimen_id = match.group(1).replace(" ", "").upper()
        
        ground_truth_label = gt_lookup.get(specimen_id, "Unknown")
        
        current_result = {
            "Specimen ID": specimen_id,
            "Ground Truth": ground_truth_label,
            "Prediction": predicted_label,
            "Correct": predicted_label == ground_truth_label
        }
        
        # --- 这里的循环现在会使用正确的类别名称来创建列标题 ---
        for i, class_name in enumerate(class_names):
            current_result[f"Prob_{class_name}"] = probs_vector[i]
            
        results_list.append(current_result)

    # --- 保存最终的"富"CSV文件 (添加了float_format) ---
    print("\n✅ 所有图像处理完毕，正在生成最终的CSV文件...")
    if not results_list:
        print("⚠️ 未生成任何预测结果。")
        return
        
    final_results_df = pd.DataFrame(results_list)
    
    prob_cols = [f"Prob_{name}" for name in class_names]
    column_order = ["Specimen ID", "Ground Truth", "Prediction", "Correct"] + prob_cols
    final_results_df = final_results_df[column_order]
    
    output_csv_path = OUTPUT_DIR / OUTPUT_RICH_CSV
    # 使用 float_format 来控制小数位数
    final_results_df.to_csv(output_csv_path, index=False, float_format='%.3f')
    
    print(f"\n🎉 成功！已将包含完整概率的预测结果保存至:\n{output_csv_path}")
    print("\n📋 CSV文件预览:")
    print(final_results_df.head())

    accuracy = final_results_df["Correct"].mean()
    print(f"\n📊 总体准确率: {accuracy:.2%}")



# --- 修改点: 旧的 evaluate_predictions 函数已不再需要，可以完全删除 ---
if __name__ == "__main__":
    main()
    # 注意：这里不再调用 evaluate_predictions 函数
    # 您现在应该使用另一个独立的评估脚本来分析新生成的 "prediction_with_probabilities.csv" 文件