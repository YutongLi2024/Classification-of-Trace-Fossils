import os
import argparse
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

# 📁 当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent.parent # D:\Desktop\UCL_Individual_Project\Approach1\Classify\
YOLO_CLS_WEIGHTS_PATH = BASE_DIR / "yolo_cls_weights" / "yolov8s-cls"
DATA_DIR = BASE_DIR / "norm_class" / "Genus_Extra_ROI_Image_224"
# 默认配置
DEFAULT_CFG = {
    "data_dir": DATA_DIR,
    "model_name": YOLO_CLS_WEIGHTS_PATH,
    "input_size": 640,
    "epochs": 600,
    "batch": 64,
    "lr0": 0.01,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "save_period": 0,
    "augment": True,
    "cos_lr": False,
    "device": "0",
    "workers": 8,
    "patience": 0,
}

def prepare_data(data_dir):
    """验证是否存在 train 和 val 子目录"""
    for split in ["train", "val"]:
    # for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"缺少目录: {split_dir}")
        if len(os.listdir(split_dir)) == 0:
            raise ValueError(f"{split} 目录下没有类别子文件夹")

def train(args):
    """训练模型"""
    exp_name = f"{args.model_name}_Genus_Colour_{datetime.now().strftime('%m%d_%H%M%S')}"
    model = YOLO(f"{args.model_name}.pt")
    results = model.train(
        data=args.data_dir,
        epochs=args.epochs,
        imgsz=args.input_size,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        augment=args.augment,
        cos_lr=args.cos_lr,
        device=args.device,
        workers=args.workers,
        name=exp_name,
        exist_ok=True,
        patience=0,
        auto_augment="randaugment",
    )

    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"✅ 训练完成，模型保存在: {best_pt}")
    return best_pt

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 分类训练脚本")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CFG["data_dir"])
    parser.add_argument("--model_name", type=str, default=DEFAULT_CFG["model_name"], choices=["yolov8n-cls", "yolov8s-cls", "yolov8m-cls"])
    parser.add_argument("--input_size", type=int, default=DEFAULT_CFG["input_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULT_CFG["batch"])
    parser.add_argument("--lr0", type=float, default=DEFAULT_CFG["lr0"])
    parser.add_argument("--lrf", type=float, default=DEFAULT_CFG["lrf"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CFG["weight_decay"])
    parser.add_argument("--augment", action="store_true", default=DEFAULT_CFG["augment"])
    parser.add_argument("--cos_lr", action="store_true", default=DEFAULT_CFG["cos_lr"])
    parser.add_argument("--device", type=str, default=DEFAULT_CFG["device"])
    parser.add_argument("--workers", type=int, default=DEFAULT_CFG["workers"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prepare_data(args.data_dir)
    best_model = train(args)
