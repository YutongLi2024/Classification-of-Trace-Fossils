import os
import argparse
from datetime import datetime
from ultralytics import YOLO


def prepare_data(data_dir: str) -> None:
    """Verify that required subfolders exist and contain class subfolders."""
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")
        # Ensure there is at least one class subfolder
        class_dirs = [p for p in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, p))]
        if len(class_dirs) == 0:
            raise ValueError(f"No class subfolders found under: {split_dir}")


def train(args) -> str:
    """Train a YOLOv8 classification model."""
    exp_name = f"{args.model_name}_Genus_Colour_{datetime.now().strftime('%m%d_%H%M%S')}"
    model = YOLO(f"{args.model_name}.pt")

    results = model.train(
        data=args.data_dir,          # expects train/ val/ test/ subfolders
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
        patience=0,                  # keep early-stop disabled as in your original
        auto_augment="randaugment",
    )

    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"✅ Training completed. Best model saved to: {best_pt}")
    return best_pt


def evaluate(model_path: str, data_dir: str, input_size: int, batch: int, device: str, workers: int, split: str = "test"):
    """Evaluate a trained model on a specific split ('val' or 'test')."""
    model = YOLO(model_path)
    print(f"\n--- 🔎 Evaluating on '{split}' split ---")
    results = model.val(
        data=data_dir,
        imgsz=input_size,
        batch=batch,
        device=device,
        workers=workers,
        split=split,  # Ultralytics supports 'train' | 'val' | 'test'
    )

    # Best-effort summary printouts across Ultralytics versions
    metrics = getattr(results, "metrics", None)
    if metrics is not None:
        # Try common classification metrics if present
        top1 = getattr(metrics, "top1", None)
        top5 = getattr(metrics, "top5", None)
        if top1 is not None:
            print(f"Top-1 accuracy: {top1:.4f}")
        if top5 is not None:
            print(f"Top-5 accuracy: {top5:.4f}")

    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict) and len(results_dict) > 0:
        print("Metrics summary:")
        for k, v in results_dict.items():
            try:
                print(f"  {k}: {float(v):.6f}")
            except Exception:
                print(f"  {k}: {v}")

    print(f"✅ Evaluation on '{split}' completed.")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 classification training script")
    # You can define DEFAULT_CFG elsewhere; falling back to sensible defaults if not present
    DEFAULT_CFG = {
        "data_dir": "./data",
        "model_name": "yolov8n-cls",
        "input_size": 224,
        "epochs": 100,
        "batch": 32,
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 5e-4,
        "augment": True,
        "cos_lr": True,
        "device": "0",      # GPU id or 'cpu'
        "workers": 8,
    }

    parser.add_argument("--data_dir", type=str, default=DEFAULT_CFG["data_dir"])
    parser.add_argument("--model_name", type=str, default=DEFAULT_CFG["model_name"],
                        choices=["yolov8n-cls", "yolov8s-cls", "yolov8m-cls"])
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

    # Optional: evaluate on the validation set for a clean report
    evaluate(
        model_path=best_model,
        data_dir=args.data_dir,
        input_size=args.input_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="val",
    )

    # Required: evaluate on the test set
    evaluate(
        model_path=best_model,
        data_dir=args.data_dir,
        input_size=args.input_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="test",
    )
