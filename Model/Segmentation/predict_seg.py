import os
from ultralytics import YOLO
import csv
from glob import glob
import datetime


CURRENT_DIR = os.getcwd()
DATA_YAML = os.path.abspath(os.path.join(CURRENT_DIR, 'data.yaml'))
TEST_IMAGES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, 'images', 'test'))
WEIGHTS_DIR = os.path.join(CURRENT_DIR, 'runs', 'segment', 'segmentation_run_yolov8s', 'weights')
RESULTS_DIR = f"outputs_{datetime.now().strftime('%m%d_%H%M%S')}"

def evaluate_model(model_path, epoch_num):
    
    save_dir = os.path.join(RESULTS_DIR, f"epoch_{epoch_num}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n evaluate Epoch {epoch_num} model...")
    model = YOLO(model_path)
    
 
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        task="segment",
    )
    
    
    predict_args = {
        "source": TEST_IMAGES_DIR,
        "conf": 0.3,
        "save": True,
        "show_conf": True,
        "show_labels": True,
        "project": ".",
        "name": save_dir,
        "exist_ok": True
    }
    model.predict(**predict_args)
    
    
    results_path = os.path.join(save_dir, "metrics.csv")
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Metric", "Value"])
        writer.writerow([epoch_num, "mAP50-95", f"{metrics.box.map:.4f}"])
        writer.writerow([epoch_num, "mAP50", f"{metrics.box.map50:.4f}"])
        writer.writerow([epoch_num, "mAP75", f"{metrics.box.map75:.4f}"])
    
    return {
        "epoch": epoch_num,
        "mAP50-95": metrics.box.map,
        "mAP50": metrics.box.map50,
        "mAP75": metrics.box.map75
    }

def main():
    weight_files = glob(os.path.join(WEIGHTS_DIR, "epoch*.pt"))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    
   
    for weight_file in sorted(weight_files):
        epoch_num = int(os.path.basename(weight_file).replace("epoch", "").replace(".pt", ""))
        result = evaluate_model(weight_file, epoch_num)
        all_results.append(result)
    
    
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(summary_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "mAP50-95", "mAP50", "mAP75"])
        for result in all_results:
            writer.writerow([
                result["epoch"],
                f"{result['mAP50-95']:.4f}",
                f"{result['mAP50']:.4f}",
                f"{result['mAP75']:.4f}"
            ])
    
    
    best_model = max(all_results, key=lambda x: x["mAP50-95"])
    print("\nEvaluation completed!")
    print(f"Best model is from Epoch {best_model['epoch']}:")
    print(f"mAP50-95: {best_model['mAP50-95']:.4f}")
    print(f"mAP50: {best_model['mAP50']:.4f}")
    print(f"mAP75: {best_model['mAP75']:.4f}")
    print(f"\nDetailed results have been saved to the {RESULTS_DIR} directory")

if __name__ == "__main__":
    main()