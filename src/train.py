"""
Training and diagnostic utilities for the tomato leaf disease classification project.

This module provides:
- project structure validation
- dataset loading and inspection
- model forward-pass testing
- full training pipeline with device selection
- generation of training and validation metric plots (loss and accuracy)
- command-line interface for running tests or training

The module is designed to be executed as a script:

    python src/train.py --test all
    python src/train.py --train
    python src/train.py --train --epochs <desired_number_of_epochs>

It expects the dataset to be located under data/processed/ with the standard ImageFolder structure.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.convolutional_neural_network import SimpleCNN

# ---------------------------------------------------------
# Utility: Change color to green
# ---------------------------------------------------------
def green(text) -> str:
    return f"\033[32m{text}\033[0m"

# ---------------------------------------------------------
# Utility: Change color to red
# ---------------------------------------------------------
def red(text) -> str:
    return f"\033[91m{text}\033[0m"

# ---------------------------------------------------------
# Utility: device selection
# ---------------------------------------------------------
def get_best_device() -> torch.device:
    """
    Select the best available compute device.

    The function checks for GPU backends in the following order:
    1. NVIDIA CUDA
    2. Intel XPU
    3. AMD ROCm
    4. Apple Metal (MPS)
    5. CPU fallback

    Returns:
        torch.device: The most capable available device.
    """
    
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

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
    
    print("\n=== Project Structure Test ===")

    required_dirs = [
        "data/raw",
        "data/processed",
        "scripts",
        "src/models",
    ]

    missing = [d for d in required_dirs if not os.path.exists(d)]

    if missing:
        print(red("\nMissing required directories:"))
        for m in missing:
            print("   -", m)
        return False

    print(green("\nProject structure looks good."))
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
    
    print("\n=== DataLoader Test ===")

    data_dir = "data/processed/train"

    if not os.path.exists(data_dir):
        print(red("\nProcessed dataset not found. Run split_tomato_dataset.sh first."))
        return False

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Found {len(dataset.classes)} classes:")
    print(dataset.classes)

    images, labels = next(iter(loader))

    print("Batch loaded successfully.")
    print("Images:", images.shape)
    print("Labels:", labels.shape)

    print(green("\nDataset test looks good."))

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
    
    print("\n=== Model Forward Pass Test ===")

    data_dir = "data/processed/train"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    images, labels = next(iter(loader))
    num_classes = len(dataset.classes)

    print(f"Using {num_classes} classes.")

    # Initialize your CNN
    model = SimpleCNN(num_classes)

    print("\nModel architecture:")
    print(model)

    print("\nInput batch shape:", images.shape)
    outputs = model(images)
    print("Output batch shape:", outputs.shape)

    assert outputs.shape == (images.size(0), num_classes)
    print(green("\nForward pass successful."))

    return True

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_model(epochs) -> bool:
    """
    Train the SimpleCNN model for a specified number of epochs.

    The function:
        - loads training and validation datasets
        - applies preprocessing transforms
        - selects the best available compute device
        - trains the model using cross-entropy loss and Adam optimizer
        - evaluates on the validation set after each epoch
        - records training/validation loss and accuracy metrics
        - generates and saves training curves (loss and accuracy) to results/plots/
        - computes and saves a normalized confusion matrix and per-class accuracy metrics

    Args:
        epochs (int): Number of full training epochs.

    Returns:
        bool: True when training completes successfully.
    """

    print("\n=== Training Pipeline ===")

    list_of_training_accuracies = []
    list_of_training_losses = []
    list_of_validation_accuracies = []
    list_of_validation_losses = []

    device = get_best_device()
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Datasets
    training_dataset = datasets.ImageFolder("data/processed/train", transform=transform)
    validation_dataset = datasets.ImageFolder("data/processed/val", transform=transform)

    training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    num_classes = len(training_dataset.classes)
    model = SimpleCNN(num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Training
        model.train()
        training_loss = 0.0
        training_correct = 0
        training_total = 0

        for images, labels in training_loader:
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
            for images, labels in validation_loader:
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

        print(f"Training Loss: {training_loss:.4f} | Training Accuracy: {training_accuracy:.4f}")
        print(f"Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.4f}")

    # Plot curves
    from visualization.plot_metrics import plot_training_curves
    plot_training_curves(
        epochs,
        list_of_training_accuracies,
        list_of_training_losses,
        list_of_validation_accuracies,
        list_of_validation_losses
    )

    # Confusion matrix
    from evaluation.confusion_matrix import plot_confusion_matrix
    class_names = training_dataset.classes
    plot_confusion_matrix(all_labels_from_validation, all_predictions_from_validation, class_names)
    
    # Save generated state model
    torch.save(model.state_dict(), "results/model_checkpoint.pth")
    
    # Save training meta data
    with open("results/training_metadata.txt", "w") as training_metadata:
        training_metadata.write(f"epochs={epochs}\n")
        training_metadata.write(f"final_training_accuracy={training_accuracy}\n")
        training_metadata.write(f"final_validation_accuracy={validation_accuracy}\n")


    print(green("\nTraining complete."))
    return True

# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def main():
    """
    Command-line interface for running tests or training.

    Supported modes:
        --test project_structure   Validate directory layout
        --test dataset             Validate dataset loading
        --test model               Validate model forward pass
        --test all                 Run all tests
        --train                    Run the training pipeline
        --epochs N                 Override default epoch count (default: 5)

    Behavior:
        - If --train is provided, training is executed.
        - If --train is provided without --epochs, training defaults to 5 epochs.
        - If --test is provided, the corresponding diagnostic is executed.
        - If no mode is selected, a help message is printed.
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--test", 
        type=str, 
        default="None",
        choices=["project_structure", "dataset", "model", "all"],
        help="Run a specific pipeline check or all together")
    
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Run the full training pipeline")
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train for (default: 5)."
    )
    args = parser.parse_args()

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
        train_model(args.epochs)
    else:
        print("No mode selected. Use --test or --train.")

if __name__ == "__main__":
    main()
