# Crop Disease Classification with PyTorch and Computer Vision

This project implements a clean, minimal, and fully reproducible machine‑learning pipeline for classifying tomato leaf diseases using the PlantVillage dataset.  
The goal is to demonstrate professional ML engineering practices: clear structure, reproducible data preparation, modular code, and step‑by‑step pipeline validation.

Specifially, the project includes:
 - PyTorch training pipeline
 - Data preprocessing and augmentation
 - Evaluation metrics (accuracy, F1)
 - ONNX export for deployment


---

## 📁 Project Structure

```
├── data/
│   ├── raw/                # Downloaded dataset (extracted)
│   └── processed/
│       └── train           # Training data
│       └── val             # Validation data
├── scripts/
│   └── split_tomato_dataset.sh   # Robust dataset preparation script
├── src/
│   ├── train.py            # Training logic
│   └── ...                 # Additional modules added over time
├── requirements.txt
└── README.md
```

---

## 🔧 Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .crop-disease-venv
source .crop-disease-venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🌐 Kaggle CLI Setup (Legacy API Token Method)

The current Kaggle CLI still requires the legacy authentication method.

1. Go to: **https://www.kaggle.com/** and create an account (necessary for creating an API token and downloading the dataset)

2. Go to: **https://www.kaggle.com/settings** and click:
   **Create Legacy API Key**
   This downloads a file named
   ```kaggle.json```
3. Move it to the correct location:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/<Download_Directory>/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Test authentification:
   ```bash
   kaggle datasets list -s plant
   ```
If you see dataset results, authentication is working.

## 📥 Download the PlantVillage Dataset

Inside the project root:

```bash
mkdir -p data/raw
kaggle datasets download -d emmarex/plantdisease -p data/raw
```

Extract it:

```bash
unzip data/raw/plantdisease.zip -d data/raw/plant_disease_labeled_images
```

Optionally, the zipped folder can now be deleted to avoid unnecesary memory allocation:

```bash
rm data/raw/plantdisease.zip
```

## 🍅 Prepare the Tomato Subset (train/val split)

This project includes a robust Bash script that:
 - handles nested folder structures
 - handles filenames with spaces
 - handles ```.jpg```, ```.JPG```, ```.jpeg```, ```.JPEG```, ```.png```, etc.
 - splits into train/val sets (currently set to a 4:1 ratio)
 - preserves class names

Run it:

```bash
chmod +x scripts/split_tomato_dataset.sh
./scripts/split_tomato_dataset.sh
```

After running, you should have:

```
data/processed/train/<class>/
data/processed/val/<class>/
```


# 🧪 Testing the Data Pipeline

Before training any model, verify that the dataset and DataLoader work correctly.

Run:

```bash
python src/train.py
```

Expected output:
 - number of classes detected
 - example class names
 - a batch of images and labels
 - tensor shapes like:

```
Images: torch.Size([32, 3, 224, 224])
Labels: torch.Size([32])
```

This confirms:
 - dataset structure is correct
 - transforms work
 - DataLoader works
 - pipeline is ready for model training
