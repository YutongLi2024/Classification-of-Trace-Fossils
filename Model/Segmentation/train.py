import os
from ultralytics import YOLO
import csv
import datetime

# 全局配置
CURRENT_DIR = os.getcwd()
DATA_YAML = os.path.abspath(os.path.join(CURRENT_DIR, 'data.yaml'))
# BASE_MODEL = "yolo_weights/yolov8s-seg.pt"  # 基础模型路径
BASE_MODEL = os.path.join(CURRENT_DIR, 'yolo_weights', 'yolov8s-seg.pt')
TRAIN_NAME = f"segmentation_run_yolov8s_{datetime.now().strftime('%m%d_%H%M%S')}"  # 训练结果目录名
TEST_IMAGES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, 'images', 'test'))  # 测试图片路径

def train_model():
    """训练实例分割模型"""
    # 初始化模型
    model = YOLO(BASE_MODEL)
    
    # 训练参数
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
        "exist_ok": True,  # ✅ 允许覆盖已有目录
        "patience": 0,
        "device": 0  # 使用GPU
    }
    
    # 开始训练
    results = model.train(**train_args)
    
    # 返回最佳模型路径
    best_pt = os.path.join("runs", "segment", TRAIN_NAME, "weights", "best.pt")
    return os.path.abspath(best_pt)

def test_model(model_weights, save_dir="predict_results—517"):
    """测试并可视化模型"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载最佳模型（修改这里↓↓↓）
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"模型文件不存在: {model_weights}")
    model = YOLO(model_weights)  # 使用传入的模型路径

    
    # 在测试集上评估指标
    print("\n正在计算测试集指标...")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        task="segment",
        name="segmentation_eval"
    )
    print("\n测试集评估结果：")
    print(f"mAP50-95: {metrics.box.map:.2f}")
    print(f"mAP50: {metrics.box.map50:.2f}")
    print(f"mAP75: {metrics.box.map75:.2f}")
    
    # 对测试图片进行预测可视化
    print("\n正在生成预测可视化...")
    predict_args = {
        "source": TEST_IMAGES_DIR,
        "conf": 0.3,
        "save": True,
        "show_conf": True,
        "show_labels": True,
        "project": ".",         # 设置保存根目录
        "name": save_dir,       # 目录名（例如 predict_results）
        "exist_ok": True
    }
    model.predict(**predict_args)
    print(f"\n预测结果已保存至：{os.path.abspath(save_dir)}")
    
    
    # 保存评估指标到 CSV 文件
    results_path = os.path.join(save_dir, "evaluation_metrics.csv")
    # 创建保存目录
    if not os.path.exists(model_weights):
        os.makedirs(save_dir, exist_ok=True)
    
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["mAP50-95", f"{metrics.box.map:.4f}"])
        writer.writerow(["mAP50", f"{metrics.box.map50:.4f}"])
        writer.writerow(["mAP75", f"{metrics.box.map75:.4f}"])

    print(f"\n📄 测试集评估指标已保存到: {results_path}")


def main(train=True, test=True):
    """主函数"""
    best_model = None
    
    if train:
        print("开始训练...")
        best_model = train_model()
        print(f"\n训练完成，最佳模型已保存至：{best_model}")
    
    if test:
        if not best_model:
            # 修改这里↓↓↓ 支持手动指定已有模型路径
            custom_model_path = r"D:\Desktop\UCL_Individual_Project\Approach1\Segmentation\runs\segment\segmentation_run\weights\best.pt"
            if os.path.exists(custom_model_path):
                best_model = custom_model_path
            else:
                default_model = os.path.join("runs", "segment", TRAIN_NAME, "weights", "best.pt")
                if not os.path.exists(default_model):
                    raise FileNotFoundError(f"未找到训练好的模型，请检查路径：\n1. {custom_model_path}\n2. {default_model}")
                best_model = default_model
        
        print("\n开始测试...")
        test_model(best_model)

if __name__ == "__main__":
    # 用法示例：
    # 只训练：main(train=True, test=False)
    # 只测试：main(train=False, test=True)
    # 同时训练测试：main()
    main(train=True, test=False)
    # main(train=False, test=True)