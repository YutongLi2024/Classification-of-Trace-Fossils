# Classification-of-Trace-Fossils

A reproducible deep learning pipeline for the segmentation and hierarchical classification of trace fossils, designed for both colour and grayscale imaging data. This repository accompanies is my dissertation the study "[xxxx]" and aims to promote objective, quantitative ichnotaxonomy and transparent research practices.

## Project Overview

This repository contains code, scripts, and processed datasets for:

- Instance segmentation of trace fossil images
- Hierarchical classification (genus and species level) using YOLO-based models
- Support for both colour and grayscale datasets
- Model evaluation, including F1-Score, Recall, Precision, ROC curves and confusion matrices

## Repository Structure

```text


Classification-of-Trace-Fossils/
│
├── Datasets/
│   ├── Colour_Mask_Data.zip      # Annotated colour dataset (images, masks)
│   └── Grey_Mask_Data.zip        # Annotated grayscale dataset
│
├── Model/
│   └── Classification/
│       ├── colour_classify/
│       │   └── train_cls.py      # Training script for colour-based classifier
│       ├── grey_classify/
│       │   ├── grey_dataprocess.py
│       │   └── train_grey_cls.py # Training script for grayscale classifier
│       └── data_process/
│           ├── extra.py
│           ├── genus_process.py
│           ├── shuffle_roi.py
│           └── shuffle.py        # Data processing and augmentation scripts
│
├── Prediction/
│   ├── segment_and_classify_grey.py   # Inference pipeline for grayscale data
│   └── segment_and_classify.py        # Inference pipeline for colour data
│
├── Segmentation/
│   ├── clean.py
│   ├── label.py
│   ├── predict_seg.py
│   └── train.py                  # Segmentation model training and utility scripts
│
└── README.md
```


## Run Program

### 1. Clone this repository

```bash
git clone https://github.com/YutongLi2024/Classification-of-Trace-Fossils.git
cd Classification-of-Trace-Fossils
```

### 2. Prepare your environment

```bash
pip install -r requirements.txt
```

### 3. Data Setup

* Unzip the datasets in `Datasets/` as needed.
* Update file paths in scripts if necessary.

### 4. Training & Inference

* **Segmentation:**
  Train instance segmentation model (see `Segmentation/train.py`), label masks, and predict segments.
* **Classification:**
  Train classification models using `Model/Classification/colour_classify/train_cls.py` (for colour) or `grey_classify/train_grey_cls.py` (for grayscale).
* **Prediction & Evaluation:**
  Use the scripts in `Prediction/` for segmentation and classification. Evaluation code will output confusion matrices, ROC curves, and performance metrics.

### 5. Reproducibility

All code, pre-processing, and model scripts are provided to ensure full reproducibility of the main results. For detailed usage, see inline comments in scripts.


