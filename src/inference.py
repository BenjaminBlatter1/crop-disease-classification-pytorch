"""
Single-image inference utilities for the tomato leaf disease classification project.

This module provides:
- A CLI entry point for running inference using the trained PyTorch checkpoint.
- A reusable `predict_image()` function for programmatic inference.
- Support for TorchScript and ONNX models for deployment scenarios.

The design keeps `run_inference()` as the high-level CLI workflow
and `predict_image()` as the reusable inference primitive.
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
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import Config
from .utils.device import get_best_device
from .models.convolutional_neural_network import SimpleCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess a single image for inference.

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

def load_class_names(train_dir: str) -> list[str]:
    """
    Load class names based on the folder structure used during training.

    Returns:
        list[str]: Alphabetically sorted class names.
    """
    root = Path(train_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    return classes

def load_checkpoint_model(checkpoint_path: str) -> torch.nn.Module:
    """
    Load the trained SimpleCNN checkpoint for inference.

    The model is always loaded on CPU first and then moved to the best
    available device (CUDA, XPU, ROCm, MPS, or CPU) as determined by
    `get_best_device()`.
    """
    device = get_best_device()

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Train the model first using:\n"
            "    python -m src.train --train"
        )

    state = torch.load(checkpoint_path, map_location="cpu")

    # Prefer class names stored in the checkpoint; fall back to scanning the
    # training directory if not present.
    class_names = state.get("class_names") or load_class_names(Config.train_dir)

    num_classes = state.get("num_classes")
    if num_classes is None:
        num_classes = len(class_names)

    model = SimpleCNN(num_classes=num_classes)

    model_state = state.get("model_state_dict")
    if model_state is None:
        raise KeyError("Checkpoint is missing 'model_state_dict'. Cannot load model.")

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # TorchScript ScriptModule objects are immutable and do not allow adding new attributes.
    # We try to attach class names for convenience, but fall back gracefully if the model
    # does not support attribute assignment.
    try:
        model.class_names = class_names
    except Exception:
        pass
    return model


def _get_class_names_from_checkpoint(checkpoint_path: str = "results/model_checkpoint.pth") -> list | None:
    """Attempt to load `class_names` from a saved checkpoint file.

    Returns the list of class names if present, otherwise `None`.
    """
    cp = Path(checkpoint_path)
    if not cp.exists():
        return None
    try:
        state = torch.load(cp, map_location="cpu")
        names = state.get("class_names")
        if names:
            return names
    except Exception:
        return None
    return None

def predict_image(model: torch.nn.Module, image_path: str) -> tuple[str, float]:
    """
    Run inference on a single image and return the predicted class and confidence.

    Returns:
        tuple[str, float]: Predicted class label and confidence score in [0, 1].
    """
    
    device = next(model.parameters()).device
    image = load_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = probs.max(1)

    # Prefer class names attached to the model, then checkpoint, then dataset folders
    class_names = getattr(model, "class_names", None)
    if class_names is None:
        class_names = _get_class_names_from_checkpoint() or load_class_names(Config.train_dir)

    return class_names[idx.item()], conf.item()


def predict_image_onnx(session: onnxruntime.InferenceSession, image_path: str) -> tuple[str, float]:
    """Run inference using an ONNX session and return (label, confidence).

    The function expects the ONNX model to accept an input named 'input' or to
    use the first input name declared in the model. The preprocessing mirrors
    `load_image()` (resize + ToTensor), and outputs are interpreted as logits.
    """
    # Prepare input
    tensor = load_image(image_path)
    arr = tensor.detach().cpu().numpy().astype(np.float32)

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

    class_names = _get_class_names_from_checkpoint() or load_class_names(Config.train_dir)
    return class_names[idx], conf

def load_torchscript(torch_script_file_path: str) -> torch.nn.Module:
    """
    Load a TorchScript model for deployment inference.

    The model is loaded on CPU and then moved to the best available device.
    """
    device = get_best_device()
    model = torch.jit.load(torch_script_file_path, map_location="cpu")
    model.to(device)
    model.eval()
    return model

def load_onnx(onnx_file_path: str):
    """
    Load an ONNX model for deployment inference.

    Returns:
        onnxruntime.InferenceSession: ONNX inference session.
    """
    
    return onnxruntime.InferenceSession(onnx_file_path)

def run_inference(image_path: str, checkpoint_path: str) -> None:
    """
    High-level CLI inference workflow using the PyTorch checkpoint.
    """

    model = load_checkpoint_model(checkpoint_path)
    label, conf = predict_image(model, image_path)
    logger.info(f"Predicted class: {label} ({conf:.4f})")

def main():
    """
    Entry point for single-image inference.

    This function parses command-line arguments, selects the inference backend
    specified by `--model-type` (checkpoint, torchscript, or onnx), loads the
    corresponding model or session, and prints the predicted class label and
    confidence score for the given image. TorchScript and ONNX modes require
    `--model-path`, while checkpoint inference uses the default training
    checkpoint stored in `results/model_checkpoint.pth`.
    """

    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image."
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["checkpoint", "torchscript", "onnx"],
        required=True,
        help="Type of model to use for inference."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model file (.pth, .pt, or .onnx)."
    )
    
    args = parser.parse_args()

    if args.model_type == "checkpoint":
        if not args.model_path:
            raise ValueError("Trained model checkpoint inference requires --model-path.")
        run_inference(args.image, args.model_path)
        return

    if args.model_type == "torchscript":
        if not args.model_path:
            raise ValueError("TorchScript inference requires --model-path.")
        model = load_torchscript(args.model_path)
        label, conf = predict_image(model, args.image)
        logger.info(f"Predicted class: {label} ({conf:.4f})")
        return

    if args.model_type == "onnx":
        if not args.model_path:
            raise ValueError("ONNX inference requires --model-path.")
        session = load_onnx(args.model_path)
        label, conf = predict_image_onnx(session, args.image)
        logger.info(f"Predicted class: {label} ({conf:.4f})")
        return



if __name__ == "__main__":
    main()
