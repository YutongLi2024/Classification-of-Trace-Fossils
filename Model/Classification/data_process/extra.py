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
        
        # Configuration parameters
        self.padding_ratio = 0            # Proportional padding around the polygon ROI
        self.min_area_ratio = 0.001       # Minimum polygon area ratio relative to the full image
        self.debug_mode = False           # If True, show intermediate processing steps

    def parse_label(self, label_line):
        """Parse a single label line in 'class_id x1 y1 x2 y2 ...' format (normalised)."""
        parts = list(map(float, label_line.strip().split()))
        if len(parts) < 3 or len(parts) % 2 != 1:
            raise ValueError(f"Invalid label format: {label_line}")
        
        class_id = int(parts[0])
        points = [(x, y) for x, y in zip(parts[1::2], parts[2::2])]
        return class_id, points

    def normalize_to_pixels(self, points, img_w, img_h):
        """Convert normalised coordinates (0–1) to pixel coordinates."""
        return [(int(x * img_w), int(y * img_h)) for (x, y) in points]

    def calculate_roi(self, points, img_w, img_h):
        """Compute the smallest axis-aligned rectangle enclosing the polygon, with optional padding."""
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
        """Extract the polygon region using a mask, then crop to the padded bounding rectangle."""
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts_array = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, pts_array, 255)
        masked = cv2.bitwise_and(img, img, mask=mask)
        x1, y1, x2, y2 = self.calculate_roi(points, img.shape[1], img.shape[0])
        return masked[y1:y2, x1:x2]

    def resize_and_pad_to_square(self, image, target_size=640):
        """
        Resize following the rules:
        - If the max dimension >= target_size/2, scale so max dim == target_size, then pad to square.
        - Otherwise, scale so max dim == 320, then pad to target_size x target_size.
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if max_dim >= target_size / 2:
            scale = target_size / max_dim
            resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            scale = 320 / max_dim
            resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        new_h, new_w = resized.shape[:2]
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )
        return padded

    def process_split(self, split):
        """Process a single split (train/val/test)."""
        print(f"\nProcessing split: {split}")
        
        label_dir = self.labels_dir / split
        image_dir = self.raw_images_dir / split
        output_dir = self.output_dir / split
        print("Image directory:", image_dir)
    
        label_files = list(label_dir.glob("*.txt"))
        progress_bar = tqdm(label_files, desc=f"Processing {split}")

        for label_file in progress_bar:
            with open(label_file, "r") as f:
                lines = f.readlines()
            
            img_name = label_file.stem
            img_path = None

            # Attempt to locate the corresponding image file by extension
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                matches = list(image_dir.rglob(f"{img_name}{ext}"))
                if matches:
                    img_path = matches[0]  # use the first match
                    break
                    
            if not img_path:
                # Image missing for this label; skip quietly
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                # Unreadable image; skip
                continue
                
            img_h, img_w = img.shape[:2]
            
            for i, line in enumerate(lines):
                try:
                    class_id, norm_points = self.parse_label(line)
                    points = self.normalize_to_pixels(norm_points, img_w, img_h)
                    
                    area = cv2.contourArea(np.array(points))
                    if area < (img_w * img_h * self.min_area_ratio):
                        print(f"Skipping small region: {label_file} (area: {area:.2f} px)")
                        continue
                        
                    cropped = self.extract_region(img, points)
                    cropped = self.resize_and_pad_to_square(cropped, target_size=640)

                    class_name = self.get_class_name(class_id)
                    save_dir = output_dir / class_name
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # Save as base.jpg for the first region, base_1.jpg, base_2.jpg, ... for subsequent ones
                    if i == 0:
                        save_path = save_dir / f"{img_name}.jpg"
                    else:
                        save_path = save_dir / f"{img_name}_{i}.jpg"

                    cv2.imwrite(str(save_path), cropped)
                    
                except Exception as e:
                    print(f"Failed to process: {label_file} - {str(e)}")

    def get_class_name(self, class_id):
        """Map class IDs to class names."""
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
        """Main entrypoint to extract labelled regions for train/val/test splits."""
        print("Starting region extraction...")
        
        # Ensure split label directories exist
        for split in ["train", "val", "test"]:
            if not (self.labels_dir / split).exists():
                raise FileNotFoundError(f"Missing label directory: {self.labels_dir / split}")
     
        # Recreate output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir()
        
        # Process each split
        for split in ["train", "val", "test"]:
            self.process_split(split)
            
        print("\nCompleted. Output directory structure:")
        print(f"Output root: {self.output_dir.absolute()}")

    def prepare_split_labels(self):
        """
        Split flat labels into train/val/test subfolders based on the presence of
        images under raw_images_dir/train|val|test. Only moves labels that have
        a matching image stem.
        """
        flat_label_dir = self.labels_dir
        
        for split in ["train", "val", "test"]:
            split_img_dir = self.raw_images_dir / split
            split_label_dir = flat_label_dir / split
            split_label_dir.mkdir(parents=True, exist_ok=True)
            
            if not split_img_dir.exists():
                print(f"{split} image directory does not exist.")
                continue
                
            # Find images (by extension)
            all_images = []
            for ext in [".jpg", ".jpeg", ".png"]:
                images = list(split_img_dir.rglob(f"*{ext}"))
                all_images.extend(images)
                print(f"Found {len(images)} files with extension {ext}")
            
            print(f"Total images found: {len(all_images)}")
            if all_images:
                print(f"Sample image files: {all_images[:3]}")
            
            moved_count = 0
            for ext in [".jpg", ".jpeg", ".png"]:
                for image_path in split_img_dir.rglob(f"*{ext}"):
                    img_name = image_path.stem
                    label_file = flat_label_dir / f"{img_name}.txt"
                    if label_file.exists():
                        shutil.move(label_file, split_label_dir / label_file.name)
                        moved_count += 1
                    else:
                        print(f"Label file not found: {label_file.name}")
            
            print(f"Moved {moved_count} label files into '{split}'")
            
            # Verify results
            copied_labels = list(split_label_dir.glob("*.txt"))
            print(f"'{split}' now contains {len(copied_labels)} label files")

if __name__ == "__main__":
    extractor = RegionExtractor()
    
    # Tweakable parameters
    extractor.padding_ratio = 0          # Increase padding around the ROI if desired
    extractor.min_area_ratio = 0.001     # Filter out regions smaller than this ratio of the image area
    extractor.run()
