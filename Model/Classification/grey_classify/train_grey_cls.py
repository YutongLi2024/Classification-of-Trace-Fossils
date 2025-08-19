import os
import argparse
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_CLS_WEIGHTS_PATH = BASE_DIR / "yolo_cls_weights" / "yolov8s-cls"
DATA_DIR = BASE_DIR / "grey_class" / "Grey_Genus_Extra_ROI_Image_224"

# Default configuration
DEFAULT_CFG = {
    "data_dir": DATA_DIR,
    "model_name": YOLO_CLS_WEIGHTS_PATH,  # path stem without ".pt"
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

def prepare_data(data_dir: str) -> None:
    """Verify that 'train', 'val' and 'test' subfolders exist and contain class subfolders."""
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")
        # Ensure at least one class folder exists under this split
        class_dirs = [p for p in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, p))]
        if len(class_dirs) == 0:
            raise ValueError(f"No class subfolders found under: {split_dir}")

def train(args) -> str:
    """Train a YOLOv8 classification model."""
    exp_name = f"{Path(str(args.model_name)).name}_Genus_Grey_{datetime.now().strftime('%m%d_%H%M%S')}"
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
        patience=0,                 # keep early stopping disabled to match your original logic
        auto_augment="randaugment",
    )

    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"Training completed. Best model saved to: {best_pt}")
    return best_pt

def evaluate(model_path: str, data_dir: str, input_size: int, batch: int, device: str, workers: int, split: str):
    """Evaluate a trained model on a specific split ('val' or 'test')."""
    print(f"\nEvaluating on '{split}' split ...")
    model = YOLO(model_path)
    results = model.val(
        data=data_dir,
        imgsz=input_size,
        batch=batch,
        device=device,
        workers=workers,
        split=split,  # 'train' | 'val' | 'test'
    )

    # Best-effort metric summary across Ultralytics versions
    metrics = getattr(results, "metrics", None)
    if metrics is not None:
        top1 = getattr(metrics, "top1", None)
        top5 = getattr(metrics, "top5", None)
        if top1 is not None:
            print(f"Top-1 accuracy: {top1:.4f}")
        if top5 is not None:
            print(f"Top-5 accuracy: {top5:.4f}")

    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict) and results_dict:
        print("Metrics summary:")
        for k, v in results_dict.items():
            try:
                print(f"  {k}: {float(v):.6f}")
            except Exception:
                print(f"  {k}: {v}")

    print(f"Evaluation on '{split}' completed.")
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 classification training script")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_CFG["data_dir"]))
    # Accept either a model family name (e.g., 'yolov8s-cls') or a path stem to custom weights
    parser.add_argument("--model_name", type=str, default=str(DEFAULT_CFG["model_name"]))
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

    # Evaluate on validation and test splits
    evaluate(
        model_path=best_model,
        data_dir=args.data_dir,
        input_size=args.input_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="val",
    )
    evaluate(
        model_path=best_model,
        data_dir=args.data_dir,
        input_size=args.input_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="test",
    )
