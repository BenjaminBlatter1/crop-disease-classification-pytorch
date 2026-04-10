import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.convolutional_neural_network import SimpleCNN


# ---------------------------------------------------------
# Project structure validation
# ---------------------------------------------------------
def test_project_structure():
    print("\n=== Project Structure Test ===")

    required_dirs = [
        "data/raw",
        "data/processed",
        "scripts",
        "src/models",
    ]

    missing = [d for d in required_dirs if not os.path.exists(d)]

    if missing:
        print("❌ Missing required directories:")
        for m in missing:
            print("   -", m)
        return False

    print("✅ Project structure looks good.")
    return True


# ---------------------------------------------------------
# Dataset + DataLoader test
# ---------------------------------------------------------
def test_dataset():
    print("\n=== DataLoader Test ===")

    data_dir = "data/processed/train"

    if not os.path.exists(data_dir):
        print("❌ Processed dataset not found. Run split_tomato_dataset.sh first.")
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

    print("✅ Dataset test looks good.")

    return True


# ---------------------------------------------------------
# Model definition + forward pass test
# ---------------------------------------------------------
def test_model():
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
    print("✅ Forward pass successful.")

    return True


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        choices=["project_structure", "dataset", "model", "all"],
                        help="Which test to run")
    args = parser.parse_args()

    if args.test in ("project_structure", "all"):
        test_project_structure()

    if args.test in ("dataset", "all"):
        test_dataset()

    if args.test in ("model", "all"):
        test_model()


if __name__ == "__main__":
    main()
