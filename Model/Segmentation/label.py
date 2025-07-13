import os

def process_label_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        # 将第一个数字（类别ID）替换为 "0"，其余保持不变
        new_line = "0 " + " ".join(parts[1:])
        new_lines.append(new_line)

    with open(file_path, 'w') as f:
        f.write("\n".join(new_lines))

def process_all_labels(root_dir):
    for split in ["train", "val", "test"]:
        label_dir = os.path.join(root_dir, split)
        for filename in os.listdir(label_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(label_dir, filename)
                process_label_file(file_path)
                print(f"✅ 已处理：{file_path}")

# 调用入口
if __name__ == "__main__":
    label_root = "labels_seg"  # 替换为你的实际路径（相对或绝对）
    process_all_labels(label_root)
