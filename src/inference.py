"""
Single-image inference script for the tomato leaf disease classification project.

This module loads a trained model checkpoint, applies the same preprocessing
pipeline used during training, performs inference on a single input image, and
prints the predicted class label.

Usage:
    python src/inference.py --image path/to/image.jpg
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from config import Config
from model import create_model


def load_image(image_path: str):
    """
    Load and preprocess a single image for inference.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, C, H, W).
    """
    transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


def load_class_names(train_dir: str):
    """
    Load class names based on the folder structure used during training.

    Args:
        train_dir (str): Path to the training dataset directory.

    Returns:
        list[str]: Alphabetically sorted class names.
    """
    root = Path(train_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    return classes


def run_inference(image_path: str):
    """
    Run inference on a single image and print the predicted class.

    Args:
        image_path (str): Path to the input image.
    """
    device = torch.device(Config.device)

    # Load class names
    class_names = load_class_names(Config.train_dir)

    # Load model
    model = create_model(num_classes=len(class_names))
    checkpoint_path = Path("results/model_checkpoint.pth")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Train the model first using:\n"
            "    python src/train.py --train"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = load_image(image_path).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = outputs.max(1)

    predicted_class = class_names[predicted_idx.item()]
    print(f"Predicted class: {predicted_class}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    args = parser.parse_args()

    run_inference(args.image)


if __name__ == "__main__":
    main()
