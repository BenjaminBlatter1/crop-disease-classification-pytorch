"""
Single‑image inference utilities for the crop disease classification project.

This module provides:
- A CLI entry point for running inference using PyTorch checkpoints,
  TorchScript models, or ONNX models.
- Architecture-aware checkpoint loading using the project's model factory.
- A reusable `predict_image()` function for programmatic inference.
- Deployment-ready TorchScript and ONNX inference paths.

The design separates:
- `run_inference()` — high-level CLI workflow for checkpoint inference
- `predict_image()` — reusable inference primitive for PyTorch models
- `predict_image_onnx()` — reusable inference primitive for ONNX models
"""

import argparse
from pathlib import Path
import logging

import torch
from torchvision import transforms
from PIL import Image
import onnxruntime
import numpy as np

# When executing the module as a script (python src/inference.py)
# ensure the project root is on sys.path and set the package name so
# package-relative imports below work. Preferred usage is: `python -m src.inference`.
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import Config
from .utils.device import get_best_device
from .models.model_factory import create_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess a single image for inference.

    Returns:
        torch.Tensor: Tensor of shape (1, 3, H, W) ready for model input.
    """
    transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension

def load_class_names(train_dir: str) -> list[str]:
    """
    Load class names from the provided training directory structure.

    Returns:
        list[str]: Alphabetically sorted class names.
    """
    root = Path(train_dir)
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    return class_names

def load_checkpoint_model(checkpoint_path: str):
    """
    Load a trained model checkpoint for inference.

    The checkpoint must contain:
        - "model_architecture": architecture name for model_factory
        - "num_classes": number of output classes
        - "model_state_dict": trained weights
        - "class_names": optional list of class names

    Returns:
        (model, class_names)
    """
    
    device = get_best_device()
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train a model first using: python -m src.train"
        )

    state = torch.load(checkpoint_path, map_location="cpu")

    model_architecture = state.get("model_architecture")
    if model_architecture is None:
        raise KeyError("Checkpoint missing 'model_architecture'.")

    num_classes = state.get("num_classes")
    if num_classes is None:
        raise KeyError("Checkpoint missing 'num_classes'.")

    model_state = state.get("model_state_dict")
    if model_state is None:
        raise KeyError("Checkpoint missing 'model_state_dict'.")

    class_names = state.get("class_names") or load_class_names(Config.train_dir)

    model = create_model(model_architecture, num_classes=num_classes)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    try:
        model.class_names = class_names
    except Exception:
        pass

    return model, class_names


# ---------------------------------------------------------------------------
# PyTorch inference
# ---------------------------------------------------------------------------

def predict_image(model: torch.nn.Module, image_path: str) -> tuple[str, float]:
    """
    Run inference on a single image using a PyTorch model.

    Returns:
        (label, confidence)
    """
    device = next(model.parameters()).device
    image = load_image(image_path).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)

    class_names = getattr(model, "class_names", None)
    if class_names is None:
        class_names = load_class_names(Config.train_dir)

    return class_names[idx.item()], conf.item()


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def predict_image_onnx(session: onnxruntime.InferenceSession, image_path: str) -> tuple[str, float]:
    """
    Run inference using an ONNX model.

    Returns:
        (label, confidence)
    """
    
    # Prepare input
    tensor = load_image(image_path)
    arr = tensor.cpu().numpy().astype(np.float32)

    # Determine input name
    try:
        input_name = session.get_inputs()[0].name
    except Exception:
        input_name = "input"

    outputs = session.run(None, {input_name: arr})
    logits = outputs[0]

    # Compute softmax in numpy
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    idx = int(np.argmax(probs, axis=1)[0])
    conf = float(np.max(probs, axis=1)[0])

    class_names = load_class_names(Config.train_dir)
    return class_names[idx], conf


# ---------------------------------------------------------------------------
# TorchScript / ONNX loaders
# ---------------------------------------------------------------------------

def load_torchscript(path: str) -> torch.nn.Module:
    """
    Load a TorchScript model for deployment inference.
    """
    device = get_best_device()
    model = torch.jit.load(path, map_location="cpu")
    model.to(device)
    model.eval()
    return model

def load_onnx(path: str) -> onnxruntime.InferenceSession:
    """
    Load an ONNX model for deployment inference.
    """
    return onnxruntime.InferenceSession(path)

# ---------------------------------------------------------------------------
# CLI workflow
# ---------------------------------------------------------------------------
def run_inference(image_path: str, checkpoint_path: str) -> None:
    """
    High‑level inference workflow for PyTorch checkpoints.
    """
    model, _ = load_checkpoint_model(checkpoint_path)
    label, conf = predict_image(model, image_path)
    logger.info(f"Predicted class: {label} ({conf:.4f})")

def main():
    """
    CLI entry point for single‑image inference.

    Supports:
        --model-type checkpoint   (PyTorch .pth)
        --model-type torchscript  (.pt)
        --model-type onnx         (.onnx)
    """
    parser = argparse.ArgumentParser(description="Run inference on a single image.")

    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--model-type", type=str, choices=["checkpoint", "torchscript", "onnx"], required=True)
    parser.add_argument("--model-path", type=str, required=True, help="Path to model file.")

    args = parser.parse_args()

    if args.model_type == "checkpoint":
        run_inference(args.image, args.model_path)
        return

    if args.model_type == "torchscript":
        model = load_torchscript(args.model_path)
        label, conf = predict_image(model, args.image)
        logger.info(f"Predicted class: {label} ({conf:.4f})")
        return

    if args.model_type == "onnx":
        session = load_onnx(args.model_path)
        label, conf = predict_image_onnx(session, args.image)
        logger.info(f"Predicted class: {label} ({conf:.4f})")
        return


if __name__ == "__main__":
    main()
