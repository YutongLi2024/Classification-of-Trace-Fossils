import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# 随机种子
random.seed(42)

# 现有已划分的数据目录（合并来源）
original_root = Path("dataset_classify")  # 包含 train/val/test
# 输出目录
target_root = Path("dataset_classify_split")
splits = ["train", "val", "test"]
split_ratio = [0.6, 0.2, 0.2]
image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# 清空输出目录
if target_root.exists():
    shutil.rmtree(target_root)
for split in splits:
    (target_root / split).mkdir(parents=True, exist_ok=True)

# 收集所有图片路径：按类别分类
all_images = defaultdict(list)

for split in ["train", "val", "test"]:
    split_path = original_root / split
    if not split_path.exists():
        continue
    for class_dir in split_path.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in class_dir.iterdir():
            if img_path.suffix in image_exts:
                all_images[class_name].append(img_path)

# 重新划分并保存
for class_name, image_paths in all_images.items():
    random.shuffle(image_paths)
    total = len(image_paths)
    n_train = max(1, int(total * split_ratio[0]))
    n_val = max(1, int(total * split_ratio[1]))
    n_test = total - n_train - n_val

    split_map = {
        "train": image_paths[:n_train],
        "val": image_paths[n_train:n_train + n_val],
        "test": image_paths[n_train + n_val:]
    }

    for split in splits:
        split_dir = target_root / split / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_map[split]:
            shutil.copy2(img_path, split_dir / img_path.name)

    print(f"✅ {class_name}: 总数 {total} → train {len(split_map['train'])}, val {len(split_map['val'])}, test {len(split_map['test'])}")

print("\n🎉 重划分完成，输出目录为 dataset_classify_split/")


# 汇总统计信息
stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

for split in splits:
    split_path = target_root / split
    for class_dir in split_path.iterdir():
        if not class_dir.is_dir():
            continue
        count = len([f for f in class_dir.iterdir() if f.suffix in image_exts])
        stats[class_dir.name][split] = count

# 排序类别名称
classes = sorted(stats.keys())
train_counts = [stats[c]["train"] for c in classes]
val_counts   = [stats[c]["val"] for c in classes]
test_counts  = [stats[c]["test"] for c in classes]

# 画图
x = range(len(classes))
bar_width = 0.25

plt.figure(figsize=(12, 6))
plt.bar([i - bar_width for i in x], train_counts, width=bar_width, label="Train")
plt.bar(x, val_counts, width=bar_width, label="Val")
plt.bar([i + bar_width for i in x], test_counts, width=bar_width, label="Test")

plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Image Count per Class (Train / Val / Test)")
plt.xticks(x, classes, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig("dataset_distribution.png")
plt.show()