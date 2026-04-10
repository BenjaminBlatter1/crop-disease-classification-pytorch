# Crop Disease Classification with PyTorch and Computer Vision

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Kaggle CLI Setup](#kaggle-cli-setup)
- [Pipeline Verification](#pipeline-verification)
- [License](#license)

## Overview
This project implements a clean, minimal, and fully reproducible machine‑learning pipeline for classifying tomato leaf diseases using the PlantVillage dataset.  
The goal is to demonstrate professional ML engineering practices: clear structure, reproducible data preparation, modular code, and step‑by‑step pipeline validation.

The project includes:
- A PyTorch-based training pipeline
- Dataset preprocessing and splitting
- A simple CNN model for classification
- Built‑in pipeline checks to ensure the setup is correct before training


## Project Structure

```
├── data/
│   ├── raw/                # Downloaded dataset (extracted)
│   └── processed/
│       └── train           # Training data
│       └── val             # Validation data
├── scripts/
│   └── split_tomato_dataset.sh   # Robust dataset preparation script
├── src/
│   ├── models/
│       ├── convolutional_neural_network.py # Simple CNN model
│   ├── train.py            # Pipeline verification
│   └── ...
├── requirements.txt
└── README.md
```

## Virtual Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .crop-disease-venv
source .crop-disease-venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Kaggle CLI Setup

The current Kaggle CLI still requires the legacy authentication method.

1. Go to: **https://www.kaggle.com/** and create an account (necessary for creating an API token and downloading the dataset)

2. Navigate to: **https://www.kaggle.com/settings**
3. Under *API*, click **Create Legacy API Key**
   This downloads a file named ```kaggle.json```
4. Move it to the correct location:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/<Download_Directory>/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
5. Test authentification:
   ```bash
   kaggle datasets list -s plant
   ```
If you see dataset results, authentication is working.

## Download the PlantVillage Dataset

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

## Prepare the Tomato Subset (train/val split)

As of now, training the CNN will be limited on images of tomato leaves.

This project includes a robust Bash script that:
 - focusses solely on tomato leaf images
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

## Pipeline Verification

Before training, the project provides built‑in checks to ensure everything is correctly set up.

Run all checks:

```bash
python src/train.py
```

Or run a specific check:

```bash
python src/train.py --test project_structure # Project structure check
python src/train.py --test dataset           # Dataset & DataLoader check
python src/train.py --test model             # Model forward-pass check
```

**What the checks validate**
 - Project structure: Ensures required directories exist.
 - Dataset & DataLoader: Confirms the processed dataset loads correctly and batches are valid.
 - Model sanity check: Builds the CNN model and runs a forward pass to verify output shapes.

These checks ensure the pipeline is stable and ready for training.

 ## License 
 This project is released under the MIT License, a permissive open‑source license that allows reuse, modification, and distribution with minimal restrictions.
