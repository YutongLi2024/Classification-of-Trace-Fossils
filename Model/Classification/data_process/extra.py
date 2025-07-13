import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

class RegionExtractor:
    def __init__(self):
        self.raw_images_dir = Path("D:/Desktop/UCL_Individual_Project/Approach1/Segmentation/images")
        self.labels_dir = Path("labels_classify")
        self.output_dir = Path("dataset_classify")
        
        # 配置参数
        self.padding_ratio = 0  # 区域扩展比例
        self.min_area_ratio = 0.001  # 最小区域面积比例（相对于原图）
        self.debug_mode = False  # 调试模式显示处理过程

    def parse_label(self, label_line):
        """解析单行标签数据"""
        parts = list(map(float, label_line.strip().split()))
        if len(parts) < 3 or len(parts) % 2 != 1:
            raise ValueError(f"Invalid label format: {label_line}")
        
        class_id = int(parts[0])
        points = [(x, y) for x, y in zip(parts[1::2], parts[2::2])]
        return class_id, points

    def normalize_to_pixels(self, points, img_w, img_h):
        """将归一化坐标转换为像素坐标"""
        return [(int(x * img_w), int(y * img_h)) for (x, y) in points]

    def calculate_roi(self, points, img_w, img_h):
        """计算包含多边形的最小矩形区域"""
        pts_array = np.array(points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts_array)
        
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)
        return (
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(img_w, x + w + pad_x),
            min(img_h, y + h + pad_y)
        )

    def extract_region(self, img, points):
        """提取多边形区域"""
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts_array = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, pts_array, 255)
        masked = cv2.bitwise_and(img, img, mask=mask)
        x1, y1, x2, y2 = self.calculate_roi(points, img.shape[1], img.shape[0])
        return masked[y1:y2, x1:x2]

    def resize_and_pad_to_square(self, image, target_size=640):
        """根据规则缩放并填充图像，使其变为 target_size x target_size"""
        h, w = image.shape[:2]
        max_dim = max(h, w)

        # 规则1、2、3：最大边都 >= target_size/2
        if max_dim >= target_size / 2:
            scale = target_size / max_dim
            resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            new_h, new_w = resized.shape[:2]
            top = (target_size - new_h) // 2
            bottom = target_size - new_h - top
            left = (target_size - new_w) // 2
            right = target_size - new_w - left
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
            return padded

        # 规则4：太小的区域，最大边缩放到 320，再填充到 640x640
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


    def process_split(self, split):
        """处理单个split的数据"""
        print(f"\n🔍 正在处理 {split} 数据集...")
        
        label_dir = self.labels_dir / split
        image_dir = self.raw_images_dir / split
        output_dir = self.output_dir / split
        print("image_dir", image_dir)
    
        label_files = list(label_dir.glob("*.txt"))
        progress_bar = tqdm(label_files, desc=f"处理 {split} 数据")
        
        for label_file in progress_bar:
            with open(label_file, "r") as f:
                lines = f.readlines()
            
            img_name = label_file.stem
            img_path = None
            if not img_path:
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    matches = list(image_dir.rglob(f"{img_name}{ext}"))
                    if matches:
                        img_path = matches[0]  # 取第一个匹配的
                        break
                    
            if not img_path:
                # print(f"\n⚠️ 图片未找到: {img_name}")
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                # print(f"\n❌ 无法读取图片: {img_path}")
                continue
                
            img_h, img_w = img.shape[:2]
            
            for i, line in enumerate(lines):
                try:
                    class_id, norm_points = self.parse_label(line)
                    points = self.normalize_to_pixels(norm_points, img_w, img_h)
                    
                    area = cv2.contourArea(np.array(points))
                    if area < (img_w * img_h * self.min_area_ratio):
                        print(f"\n⚠️ 忽略过小区域: {label_file} (面积: {area}px)")
                        continue
                        
                    cropped = self.extract_region(img, points)
                    cropped = self.resize_and_pad_to_square(cropped, target_size=640) # 新添加填充规则

                    
                    class_name = self.get_class_name(class_id)
                    save_dir = output_dir / class_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # save_path = save_dir / f"{img_name}_{i}.png"
                    if i == 0:
                        save_path = save_dir / f"{img_name}.jpg"
                    else:
                        save_path = save_dir / f"{img_name}_{i}.jpg"

                    cv2.imwrite(str(save_path), cropped)
                    
                        
                except Exception as e:
                    print(f"\n❌ 处理失败: {label_file} - {str(e)}")

    def get_class_name(self, class_id):
        """映射类别ID到名称"""
        class_map = {
            0: "Mesichnium_benjamini",
            1: "Octopodichnus_didactylus",
            2: "Paleohelcura_dunbari",
            3: "Paleohelcura_lyonsensis",
            4: "Octopodichnus_minor",
            5: "Triavestigia_niningeri",
            6: "Octopodichnus_raymondi",
            7: "Paleohelcura_tridactyla"
        }
        return class_map.get(class_id, "unknown")

    def run(self):
        """主运行函数"""
        # self.prepare_split_labels()
        print("🚀 开始提取标注区域...")
        
        for split in ["train", "val", "test"]:
            if not (self.labels_dir / split).exists():
                raise FileNotFoundError(f"缺少标签目录: {self.labels_dir/split}")
     
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir()
        
        for split in ["train", "val", "test"]:
            self.process_split(split)
            
        print("\n🎉 处理完成！输出目录结构：")
        print(f"输出路径: {self.output_dir.absolute()}")

    def prepare_split_labels(self):
        """根据 raw_images_dir 的 train/val/test 拆分标签到对应子目录"""
        flat_label_dir = self.labels_dir  # 所有 txt 的根目录
        
        for split in ["train", "val", "test"]:
            split_img_dir = self.raw_images_dir / split
            split_label_dir = flat_label_dir / split
            split_label_dir.mkdir(parents=True, exist_ok=True)
            
            if not split_img_dir.exists():
                print(f"❌ {split} 图片目录不存在！")
                continue
                
            # 统计图片文件
            all_images = []
            for ext in [".jpg", ".jpeg", ".png"]:
                images = list(split_img_dir.rglob(f"*{ext}"))
                all_images.extend(images)
                print(f"找到 {len(images)} 个 {ext} 文件")
            
            print(f"总共找到 {len(all_images)} 个图片文件")
            if all_images:
                print(f"示例图片文件: {all_images[:3]}")
            
            copied_count = 0
            for ext in [".jpg", ".jpeg", ".png"]:
                for image_path in split_img_dir.rglob(f"*{ext}"):
                    img_name = image_path.stem
                    label_file = flat_label_dir / f"{img_name}.txt"
                    if label_file.exists():
                        shutil.move(label_file, split_label_dir / label_file.name)
                        copied_count += 1
                    else:
                        print(f"⚠️ 标签文件不存在: {label_file.name}")
            
            print(f"{split} 数据集共拷贝了 {copied_count} 个标签文件")
            
            # 检查拷贝结果
            copied_labels = list(split_label_dir.glob("*.txt"))
            print(f"{split} 子目录中现在有 {len(copied_labels)} 个标签文件")

if __name__ == "__main__":
    extractor = RegionExtractor()
    
    extractor.padding_ratio = 0  # 增加区域周围的padding
    extractor.min_area_ratio = 0.001  # 过滤掉小于原图面积2%的区域
    extractor.run()
