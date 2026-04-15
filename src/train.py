"""
Training and diagnostic utilities for the tomato leaf disease classification project.

This module provides:
- project structure validation
- dataset loading and inspection
- model forward-pass testing
- full training pipeline with device selection
- optional data augmentation during training
- generation of training and validation metric plots (loss and accuracy)
- normalized confusion matrix computation
- model architecture selection (SimpleCNN, pretrained ResNet-18 etc.)
- layer-by-layer model summary and parameter count logging

Augmentation is controlled via Config.use_augmentation or the --augment flag.
Validation transforms remain deterministic to ensure consistent evaluation.

Model architecture is selected at training time via --model-architecture.
Pretrained architectures (e.g. ResNet-18) are fine-tuned on the tomato leaf dataset.

The module is designed to be executed as a script:

    # Run all diagnostic checks
    python -m src.train --test all

    # Train the model (default: 5 epochs)
    python -m src.train --train

    # Train with augmentation enabled explicitly
    python -m src.train --train --augment yes

    # Train with a custom number of epochs
    python -m src.train --train --epochs <desired_number_of_epochs>

    # Train with a custom model architecture
    python -m src.train --train --model-architecture {simplecnn, resnet18, ...}

It expects the dataset to be located under data/processed/ with the standard ImageFolder structure.
"""


import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary

import logging
from tqdm import tqdm

# When executing the module as a script (python src/train.py)
# ensure the project root is on sys.path and set the package name so
# relative imports below work. Preferred usage is: `python -m src.train`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

# Project specific imports (package-relative)
from .config import Config
from .evaluation.confusion_matrix import plot_confusion_matrix
from .models.convolutional_neural_network import SimpleCNN
from .models.model_factory import create_model
from .utils.device import get_best_device
from .visualization.plot_metrics import plot_training_curves

# Ensure results directory exists before configuring file logging
Path("results").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("results/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation transform pipelines.

    When Config.use_augmentation is True, the training pipeline applies light
    augmentation (horizontal flips, small rotations, color jitter). Validation
    transforms remain deterministic to ensure consistent evaluation.

    Returns:
        tuple: (training_transform, validation_transform)
    """
    if Config.use_augmentation:
        training_transform = transforms.Compose([
            transforms.Resize(Config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
        ])
    else:
        training_transform = transforms.Compose([
            transforms.Resize(Config.image_size),
            transforms.ToTensor(),
        ])

    validation_transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.ToTensor(),
    ])

    return training_transform, validation_transform

# ---------------------------------------------------------
# Project structure validation
# ---------------------------------------------------------
def test_project_structure() -> bool:
    """
    Validate that the expected project directory structure exists.

    Required directories:
        - data/raw
        - data/processed
        - scripts
        - src/models

    Returns:
        bool: True if all required directories exist, False otherwise.
    """
    
    logger.info("\n=== Project Structure Test ===")

    required_dirs = [
        "data/raw",
        "data/processed",
        "scripts",
        "src/models",
    ]

    missing = [d for d in required_dirs if not os.path.exists(d)]

    if missing:
        logger.error("Missing required directories:")
        for m in missing:
            logger.error(f"   - {m}")
        return False

    logger.info("Project structure looks good.")
    return True

# ---------------------------------------------------------
# Dataset + DataLoader test
# ---------------------------------------------------------
def test_dataset() -> bool:
    """
    Verify that the processed dataset can be loaded and batched.

    This function:
        - checks that data/processed/train exists
        - loads the dataset using ImageFolder
        - applies basic preprocessing transforms
        - loads a single batch to confirm DataLoader functionality

    Returns:
        bool: True if dataset loading and batching succeed, False otherwise.
    """
    
    logger.info("\n=== DataLoader Test ===")

    data_dir = "data/processed/train"

    if not os.path.exists(data_dir):
        logger.error("Processed dataset not found. Run split_tomato_dataset.sh first.")
        return False

    _, transform = get_transforms()

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)

    logger.info(f"Found {len(dataset.classes)} classes: {dataset.classes}")

    images, labels = next(iter(loader))

    logger.info("Batch loaded successfully.")
    logger.info(f"Images: {images.shape}")
    logger.info(f"Labels: {labels.shape}")
    logger.info("Dataset test looks good.")

    return True

# ---------------------------------------------------------
# Model definition + forward pass test
# ---------------------------------------------------------
def test_model() -> bool:
    """
    Validate that the SimpleCNN model can perform a forward pass.

    This function:
        - loads a batch from the training dataset
        - initializes the CNN with the correct number of classes
        - prints the model architecture
        - performs a forward pass
        - verifies output shape correctness

    Returns:
        bool: True if the forward pass succeeds, False otherwise.
    """
    
    logger.info("\n=== Model Forward Pass Test ===")

    data_dir = "data/processed/train"

    _, transform = get_transforms()

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)

    images, labels = next(iter(loader))
    num_classes = len(dataset.classes)

    logger.info(f"Using {num_classes} classes.")

    # Initialize your CNN
    model = SimpleCNN(num_classes)

    logger.info(f"Model architecture: {model}")
    logger.info(f"Input batch shape: {images.shape}")
    
    outputs = model(images)
    logger.info(f"Output batch shape: {outputs.shape}")

    assert outputs.shape == (images.size(0), num_classes)
    logger.info("Forward pass successful.")

    return True

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_model(epochs: int, model_architecture: str) -> bool:
    """
    Train the specified model architecture for a given number of epochs.

    The function:
        - loads training and validation datasets
        - applies preprocessing transforms, including optional data augmentation
          when Config.use_augmentation is enabled
        - selects the best available compute device and moves the model to it
        - instantiates the chosen architecture (SimpleCNN, pretrained ResNet-18, etc.)
        - logs a full layer-by-layer model summary and total parameter count
        - trains the model using cross-entropy loss and the Adam optimizer
        - evaluates on the validation set after each epoch
        - records training/validation loss and accuracy metrics
        - generates and saves training curves (loss and accuracy) to results/plots/
        - computes and saves a normalized confusion matrix for the final epoch
        - saves a model checkpoint containing weights, class names, and metadata

    Args:
        epochs (int): Number of full training epochs.
        model_architecture (str): Name of the model architecture to instantiate.

    Returns:
        bool: True when training completes successfully.
    """

    logger.info("\n=== Training Pipeline ===")

    list_of_training_accuracies = []
    list_of_training_losses = []
    list_of_validation_accuracies = []
    list_of_validation_losses = []

    device = get_best_device()
    logger.info(f"Using device: {device}")

    # Load transforms (with or without augmentation)
    training_transform, validation_transform = get_transforms()

    # Datasets
    training_dataset = datasets.ImageFolder("data/processed/train", transform=training_transform)
    validation_dataset = datasets.ImageFolder("data/processed/val", transform=validation_transform)

    training_loader = DataLoader(training_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    num_classes = len(training_dataset.classes)
    model = create_model(model_architecture, num_classes).to(device)
    
    logger.info(f"Selected architecture: {model_architecture}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    try:
        summary_str = summary(
            model,
            input_size=(3, Config.image_size, Config.image_size)
        )
        logger.info("\n" + str(summary_str))
    except Exception as e:
        logger.warning(f"Model summary unavailable: {e}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        logger.info(f"--- Epoch {epoch}/{epochs} ---")

        # Training
        model.train()
        training_loss = 0.0
        training_correct = 0
        training_total = 0

        for images, labels in tqdm(training_loader, desc=f"Training Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # 1. Forward pass
            outputs = model(images)

            # 2. Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * images.size(0)

            # 3. Compute predictions
            _, predictions_from_training = outputs.max(1)

            # 4. Accumulate accuracy
            training_correct += predictions_from_training.eq(labels).sum().item()
            training_total += labels.size(0)

        training_accuracy = training_correct / training_total
        training_loss /= training_total

        # Validation
        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        # Reset lists so confusion matrix is based on the final epoch only
        all_predictions_from_validation = []
        all_labels_from_validation = []

        with torch.no_grad():
            for images, labels in tqdm(validation_loader, desc=f"Validation Epoch {epoch}", leave=False):
                images, labels = images.to(device), labels.to(device)

                # 1. Forward pass
                outputs = model(images)

                # 2. Compute loss
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * images.size(0)

                # 3. Compute predictions
                _, predictions_from_validation = outputs.max(1)

                # 4. Accumulate accuracy
                validation_correct += predictions_from_validation.eq(labels).sum().item()
                validation_total += labels.size(0)

                # 5. Store predictions for confusion matrix
                all_predictions_from_validation.extend(predictions_from_validation.cpu().numpy())
                all_labels_from_validation.extend(labels.cpu().numpy())

        validation_accuracy = validation_correct / validation_total
        validation_loss /= validation_total

        list_of_training_accuracies.append(training_accuracy)
        list_of_training_losses.append(training_loss)
        list_of_validation_accuracies.append(validation_accuracy)
        list_of_validation_losses.append(validation_loss)

        logger.info(f"Training Loss: {training_loss:.2f} | Training Accuracy: {training_accuracy:.4f}")
        logger.info(f"Validation Loss: {validation_loss:.2f} | Validation Accuracy: {validation_accuracy:.4f}")

    # Plot curves
    plot_training_curves(
        epochs,
        list_of_training_accuracies,
        list_of_training_losses,
        list_of_validation_accuracies,
        list_of_validation_losses
    )

    # Confusion matrix
    class_names = training_dataset.classes
    plot_confusion_matrix(all_labels_from_validation, all_predictions_from_validation, class_names)
    
    # Save generated state model
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_names": class_names,
    }, "results/model_checkpoint.pth")


    
    # Save training meta data
    with open("results/training_metadata.txt", "w") as training_metadata:
        training_metadata.write(f"Total number of epochs: {epochs}\n")
        training_metadata.write(f"Final training accuracy: {training_accuracy:.4f}\n")
        training_metadata.write(f"Final training loss: {training_loss:.2f}\n")
        training_metadata.write(f"Final validation accuracy: {validation_accuracy:.4f}\n")
        training_metadata.write(f"Final validation loss: {validation_loss:.2f}\n")

    logger.info("Training complete.")
    return True

# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def main():
    """
    Command-line interface for running diagnostics and model training.

    Supported modes:
        --test project_structure    Validate directory layout
        --test dataset              Validate dataset loading
        --test model                Validate model forward pass
        --test all                  Run all diagnostic checks

        --train                                         Run the full training pipeline
        --epochs N                                      Override default epoch count (default: 5)
        --augment {yes,no}                              Explicitly enable or disable data augmentation.
        --model-architecture {simplecnn,resnet18, ...}  Select model architecture.

    Behavior:
        - If --test is provided, the corresponding diagnostic is executed.
        - If --train is provided, the training pipeline is executed.
        - If --augment is set to "yes", the training pipeline uses augmented transforms.
        - If --augment is set to "no", the training pipeline uses deterministic transforms.
        - If --model-architecture is provided, the selected architecture is used for training.
        - If --train is used without --epochs, training defaults to 5 epochs.
        - If no mode is selected, a help message is printed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        type=str,
        default="None",
        choices=["project_structure", "dataset", "model", "all"],
        help="Run a specific pipeline check or all together."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the full training pipeline."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train for (default: 5)."
    )

    parser.add_argument(
        "--augment",
        type=str,
        choices=["yes", "no"],
        default="no",
        help="Enable or disable augmentation."
    )

    parser.add_argument(
        "--model-architecture",
        type=str,
        choices=["simplecnn", "resnet18"],
        default="simplecnn",
        help="Select model architecture."
    )

    args = parser.parse_args()
    
    Config.use_augmentation = args.augment == "yes"

    if args.test == "all":
        test_project_structure()
        test_dataset()
        test_model()
    elif args.test == "project_structure":
        test_project_structure()
    elif args.test == "dataset":
        test_dataset()
    elif args.test == "model":
        test_model()
    elif args.train:
        train_model(args.epochs, args.model_architecture)
    else:
        logger.info("No mode selected. Use --test or --train.")

if __name__ == "__main__":
    main()
