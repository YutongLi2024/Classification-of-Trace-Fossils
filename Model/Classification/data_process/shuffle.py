import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)

# Existing dataset root containing train/val/test (to be re-split)
original_root = Path("dataset_classify")
# Output root
target_root = Path("dataset_classify_split")
splits = ["train", "val", "test"]
split_ratio = [0.6, 0.2, 0.2]
image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Clean output directory
if target_root.exists():
    shutil.rmtree(target_root)
for split in splits:
    (target_root / split).mkdir(parents=True, exist_ok=True)

# Collect all image paths grouped by class
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

# Re-split per class and copy files
for class_name, image_paths in all_images.items():
    random.shuffle(image_paths)
    total = len(image_paths)
    n_train = max(1, int(total * split_ratio[0]))
    n_val = max(1, int(total * split_ratio[1]))
    n_test = total - n_train - n_val  # allocate remainder to test

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

    print(f"{class_name}: total {total} → train {len(split_map['train'])}, val {len(split_map['val'])}, test {len(split_map['test'])}")

print("\nRe-splitting completed. Output directory: dataset_classify_split/")

# Aggregate split statistics
stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

for split in splits:
    split_path = target_root / split
    if not split_path.exists():
        continue
    for class_dir in split_path.iterdir():
        if not class_dir.is_dir():
            continue
        count = len([f for f in class_dir.iterdir() if f.suffix in image_exts])
        stats[class_dir.name][split] = count

# Prepare data for plotting
classes = sorted(stats.keys())
train_counts = [stats[c]["train"] for c in classes]
val_counts   = [stats[c]["val"] for c in classes]
test_counts  = [stats[c]["test"] for c in classes]

# Plot distribution
x = range(len(classes))
bar_width = 0.25

plt.figure(figsize=(12, 6))
plt.bar([i - bar_width for i in x], train_counts, width=bar_width, label="Train")
plt.bar(x, val_counts, width=bar_width, label="Val")
plt.bar([i + bar_width for i in x], test_counts, width=bar_width, label="Test")

plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Image Count per Class (Train / Val / Test)")
plt.xticks(list(x), classes, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.savefig("dataset_distribution.png")
plt.show()
