import os
from ultralytics import YOLO
import csv
import datetime

# Global configuration
CURRENT_DIR = os.getcwd()
DATA_YAML = os.path.abspath(os.path.join(CURRENT_DIR, 'data.yaml'))
# BASE_MODEL = "yolo_weights/yolov8s-seg.pt"  # Base model path (alternative form)
BASE_MODEL = os.path.join(CURRENT_DIR, 'yolo_weights', 'yolov8s-seg.pt')
TRAIN_NAME = f"segmentation_run_yolov8s_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"  # Training run directory name
TEST_IMAGES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, 'images', 'test'))  # Test images directory


def train_model():
    """Train an instance segmentation model."""
    model = YOLO(BASE_MODEL)
    
    # Training parameters
    train_args = {
        "data": DATA_YAML,
        "task": "segment",
        "epochs": 500,
        "imgsz": 640,
        "batch": 64,
        "name": TRAIN_NAME,
        "save": True,
        "save_period": 20,
        "val": True,
        "exist_ok": True,   
        "patience": 0,
        "device": 0         # Use GPU index 0
    }
    
    # Start training
    _results = model.train(**train_args)
    
    # Return path to the best model
    best_pt = os.path.join("runs", "segment", TRAIN_NAME, "weights", "best.pt")
    return os.path.abspath(best_pt)


def test_model(model_weights, save_dir="predict_results—517"):
    """Evaluate the model on the test split and save visualisations and metrics."""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the provided model weights
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model file not found: {model_weights}")
    model = YOLO(model_weights)

    # Evaluate on the test set defined in data.yaml
    print("\nComputing test-set metrics...")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        task="segment",
        name="segmentation_eval"
    )
    print("\nTest-set evaluation results:")
    print(f"mAP50-95: {metrics.box.map:.2f}")
    print(f"mAP50: {metrics.box.map50:.2f}")
    print(f"mAP75: {metrics.box.map75:.2f}")
    
    # Generate prediction visualisations on the test images
    print("\nGenerating prediction visualisations...")
    predict_args = {
        "source": TEST_IMAGES_DIR,
        "conf": 0.3,
        "save": True,
        "show_conf": True,
        "show_labels": True,
        "project": ".",        # Root directory for saving
        "name": save_dir,      # Subdirectory name (e.g., predict_results—517)
        "exist_ok": True
    }
    model.predict(**predict_args)
    print(f"\nPredictions saved to: {os.path.abspath(save_dir)}")
    
    # Save evaluation metrics to CSV
    results_path = os.path.join(save_dir, "evaluation_metrics.csv")
    # Ensure the directory exists (defensive; it should already exist)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["mAP50-95", f"{metrics.box.map:.4f}"])
        writer.writerow(["mAP50", f"{metrics.box.map50:.4f}"])
        writer.writerow(["mAP75", f"{metrics.box.map75:.4f}"])

    print(f"\nTest-set metrics saved to: {results_path}")


def main(train=True, test=True):
    """Entry point to train and/or test the segmentation model."""
    best_model = None
    
    if train:
        print("Starting training...")
        best_model = train_model()
        print(f"\nTraining completed. Best model saved to: {best_model}")
    
    if test:
        if not best_model:
            # Support manually specifying an existing model path
            custom_model_path = r"D:\Desktop\UCL_Individual_Project\Approach1\Segmentation\runs\segment\segmentation_run\weights\best.pt"
            if os.path.exists(custom_model_path):
                best_model = custom_model_path
            else:
                default_model = os.path.join("runs", "segment", TRAIN_NAME, "weights", "best.pt")
                if not os.path.exists(default_model):
                    raise FileNotFoundError(
                        "No trained model was found. Please check one of these paths:\n"
                        f"1. {custom_model_path}\n"
                        f"2. {default_model}"
                    )
                best_model = default_model
        
        print("\nStarting test...")
        test_model(best_model)


if __name__ == "__main__":
    main(train=True, test=False)

