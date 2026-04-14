"""
Utilities for exporting trained models to deployment formats.

This module provides functions to load a trained SimpleCNN checkpoint
and export it to TorchScript (.pt) and ONNX (.onnx) formats. These
formats are suitable for deployment on edge devices or integration
into production systems.
"""

from pathlib import Path

import torch
from .convolutional_neural_network import SimpleCNN


def load_checkpoint(model_path: str) -> torch.nn.Module:
    """
    Load a trained SimpleCNN model from a checkpoint for export.

    The checkpoint is always loaded on CPU to ensure stable TorchScript
    and ONNX export, regardless of the hardware used during training.

    The checkpoint must contain:
        - "model_state_dict": state dict of the SimpleCNN
        - "num_classes": number of output classes

    Args:
        model_path: Path to the .pth checkpoint file.

    Returns:
        torch.nn.Module: SimpleCNN model with loaded weights in eval mode.

    Raises:
        KeyError: If required keys are missing from the checkpoint.
        FileNotFoundError: If the checkpoint file does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    state = torch.load(model_path, map_location="cpu")

    num_classes = state.get("num_classes")
    if num_classes is None:
        raise KeyError("Checkpoint is missing 'num_classes'. Cannot export model.")

    model_state = state.get("model_state_dict")
    if model_state is None:
        raise KeyError("Checkpoint is missing 'model_state_dict'. Cannot export model.")

    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(model_state)
    model.eval()
    return model


def export_torchscript(model_path: str, output_path: str) -> None:
    """Export a trained model to TorchScript format."""
    model = load_checkpoint(model_path)
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, output_path)


def export_onnx(model_path: str, output_path: str) -> None:
    """Export a trained model to ONNX format using a dummy input."""
    model = load_checkpoint(model_path)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
