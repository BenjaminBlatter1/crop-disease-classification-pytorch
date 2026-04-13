# Crop Disease Classification with PyTorch and Computer Vision

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Kaggle CLI Setup](#kaggle-cli-setup)
- [Download the PlantVillage Dataset](#download-the-plantvillage-dataset)
- [Prepare the Tomato Subset (train/val split)](#prepare-the-tomato-subset-trainval-split)
- [Pipeline Verification](#pipeline-verification)
- [Model Training](#model-training)
  - [Data Augmentation](#data-augmentation)
    - [What augmentation is applied?](#what-augmentation-is-applied)
    - [Why is validation not augmented?](#why-is-validation-not-augmented)
    - [How to switch augmentation mode?](#how-to-switch-augmentation-mode)
    - [Transform pipeline](#transform-pipeline)
  - [Training Output and Logging](#training-output-and-logging)
  - [Single‑Image Inference](#single-image-inference)
- [Results](#results)
  - [Training and Validation Accuracy](#training-and-validation-accuracy)
  - [Training and Validation Loss](#training-and-validation-loss)
  - [Normalized Confusion Matrix](#normalized-confusion-matrix)
  - [Performance Summary](#performance-summary)
- [Future Work](#future-work)
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
├── data/
│   ├── raw/         # Downloaded dataset
│   └── processed/
│       ├── train/   # Training split
│       └── val/     # Validation split
│
├── results/
│   ├── plots/                # Accuracy/loss curves + confusion matrix
│   └── model_checkpoint.pth  # Saved model weights (after training)
│
├── scripts/
│   └── split_tomato_dataset.sh  # Tomato-only dataset preparation
│
├── src/
│   ├── evaluation/
│   │   └── confusion_matrix.py              # Normalized confusion matrix generation
│   ├── models/
│   │   └── convolutional_neural_network.py  # Simple CNN model
│   ├── visualization/
│   │   └── plot_metrics.py                  # Accuracy/loss curve plotting
│   ├── config.py                            # Centralized configuration settings
│   ├── dataset.py                           # Custom dataset loader
│   ├── inference.py                         # Single-image inference script
│   ├── model.py                             # ResNet‑18 model factory
│   └── train.py                             # Full training pipeline + tests
│
├── README.md
└── requirements.txt

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
python src/train.py --test all
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

## Model Training
Once the dataset and pipeline checks pass, you can start training the Convolutional Neural Network model.

Train with default 5 epochs:
```bash
python src/train.py --train
```

Train with a custom number of epochs:
```bash
python src/train.py --train --epochs <desired_number_of_epochs>
```

### Data Augmentation
Data augmentation improves the model’s robustness by simulating natural variation in real tomato leaf images. Agricultural imagery often varies in lighting, orientation, and color, and augmentation helps the model generalize better to these conditions.

#### What augmentation is applied?
The training pipeline applies light, biologically plausible transformations:
 - random horizontal flips
 - small rotations (±15°)
 - mild color jitter (brightness, contrast, saturation)

These augmentations increase dataset diversity without distorting the underlying leaf structure.

#### Why is validation not augmented?
Validation images remain clean and deterministic. This ensures that validation accuracy reflects true model performance rather than random augmentation noise. Augmenting validation data would make metrics unstable and non‑comparable across runs.

#### How to switch augmentation mode?
Augmentation is controlled through the global configuration:

```python
Config.use_augmentation = False
```

or via the command‑line interface:

```bash
python src/train.py --train --augment
```

If ```--augment``` is omitted, augmentation defaults to the value defined in Config.

#### Transform pipeline
The training script constructs the transform pipelines dynamically:

```python
train_transform, val_transform = get_transforms()
```

```get_transforms()``` returns a tuple of two ```torchvision.transforms.Compose``` objects: one for training (augmented or clean) and one for validation (always clean).

### Training Output and Logging
During training, the script displays live progress bars for both training and validation epochs.
All training information (loss, accuracy, epoch summaries) is also written to:

```
results/training.log
```

This makes the training process transparent and easy to debug.

### Single Image Inference
After training, you can run inference on any tomato leaf image:

```bash
python src/train.py --infer --image <path_to_image>
```

This loads the saved model checkpoint from results/model_checkpoint.pth and prints the predicted class.

## Results
Training the **SimpleCNN** model for **20 epochs** produced a set of evaluation artifacts that illustrate how the model learned over time and how well it generalizes to unseen tomato leaf images. These artifacts are stored under ```results/plots/``` and include accuracy and loss curves as well as a normalized confusion matrix.

### Training and Validation Accuracy
The model shows a smooth and stable learning trajectory across 20 epochs. Accuracy improves rapidly during the first five epochs (from **53.8% → 86.2%**), then continues to rise steadily. Validation accuracy remains consistently high throughout training and reaches a final value of **95.97%**.

This pattern indicates strong generalization and no signs of overfitting.

![Accuracy Curve](results/plots/accuracy_curve.png)

### Training and Validation Loss
Training loss decreases from **1.32 → 0.17**, while validation loss follows a similar trend, ending at **0.14**. The close alignment between the two curves demonstrates that the model maintains stable generalization and avoids divergence or instability.

![Loss Curve](results/plots/loss_curve.png)

### Normalized Confusion Matrix
The confusion matrix highlights strong per‑class performance across all ten tomato leaf disease categories. Most classes achieve **>90%** normalized accuracy. The model performs particularly well on:
 - Healthy
 - Curl virus
 - Spider mites
 - Target spot

Misclassifications occur primarily between visually similar fungal diseases (e.g., **Early blight** vs. **Late blight**). This pattern is expected and highlights where more advanced architectures or transfer learning can provide significant improvements.

![Confusion Matrix](results/plots/confusion_matrix.png)

### Performance Summary
 - **Final Training Accuracy:** 94.01%
 - **Final Validation Accuracy:** 95.97%
 - **Final Training Loss:** 0.1695
 - **Final Validation Loss:** 0.1362
 - **Strong per‑class performance** with minimal confusion between disease categories
 - **Stable convergence** with no overfitting
 - **Robust evaluation suite** including accuracy/loss curves and a normalized confusion matrix

These results demonstrate that even a lightweight CNN can achieve strong performance on the tomato leaf disease dataset, though the gap between training and validation accuracy suggests that more expressive models (e.g., ResNet18, MobileNetV2) will likely yield further gains.

## Future Work
Several improvements can further enhance model performance and robustness:
 - Transfer learning using pretrained CNN backbones
 - Hyperparameter tuning (learning rate, batch size, optimizer)
 - Larger or deeper model architectures
 - Test‑set evaluation and cross‑validation
 - Exporting the model for mobile or edge deployment

## License 
This project is released under the MIT License, a permissive open‑source license that allows reuse, modification, and distribution with minimal restrictions.
