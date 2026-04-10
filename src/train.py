import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.convolutional_neural_network import SimpleCNN

# ---------------------------------------------------------
# Utility: Change color to green
# ---------------------------------------------------------
def green(text):
    return f"\033[32m{text}\033[0m"

# ---------------------------------------------------------
# Utility: Change color to red
# ---------------------------------------------------------
def red(text):
    return f"\033[91m{text}\033[0m"

# ---------------------------------------------------------
# Utility: device selection
# ---------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print(red("\nMissing required directories:"))
        for m in missing:
            print("   -", m)
        return False

    print(green("\nProject structure looks good."))
    return True


# ---------------------------------------------------------
# Dataset + DataLoader test
# ---------------------------------------------------------
def test_dataset():
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
    print(green("\nForward pass successful."))

    return True


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_model():
    print("\n=== Training Pipeline ===")

    device = get_device()
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

    epochs = 5

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Training
        model.train()
        training_loss = 0
        correct = 0
        total = 0

        for images, labels in training_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        training_accuracy = correct / total
        training_loss /= total

        # Validation
        model.eval()
        validation_loss = 0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                validation_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                validation_correct += preds.eq(labels).sum().item()
                validation_total += labels.size(0)

        validation_accuracy = validation_correct / validation_total
        validation_loss /= validation_total

        print(f"Training Loss: {training_loss:.4f} | Training Accuracy: {training_accuracy:.4f}")
        print(f"Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.4f}")

    print(green("\nTraining complete."))
    return True


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="None",
                        choices=["project_structure", "dataset", "model", "all"],
                        help="Run a specific pipeline check or all together")
    parser.add_argument("--train", action="store_true",
                        help="Run the full training pipeline")
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
        train_model()
    else:
        print("No mode selected. Use --test or --train.")

if __name__ == "__main__":
    main()
