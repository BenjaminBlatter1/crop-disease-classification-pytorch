"""
Model export utilities for deployment.

This module loads a trained model checkpoint (of any model architecture 
supportedby the project's model factory) and exports it to TorchScript (.pt) 
and ONNX (.onnx) formats. Export always runs on CPU to ensure deterministic
and hardware-agnostic deployment artifacts.

The checkpoint must contain:
    - "model_architecture": string identifying the architecture
    - "num_classes": number of output classes
    - "model_state_dict": trained model weights
"""

from pathlib import Path
import torch

from src.models.model_factory import create_model


def load_checkpoint(model_path: str) -> torch.nn.Module:
    """
    Load a trained model from a checkpoint for export.

    Always loads on CPU for stable TorchScript and ONNX export.

    Args:
        model_path: Path to the .pth checkpoint file.

    Returns:
        torch.nn.Module: Model instance with loaded weights in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If required keys are missing from the checkpoint.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    state = torch.load(model_path, map_location="cpu")

    model_architecture = state.get("model_architecture")
    if model_architecture is None:
        raise KeyError("Checkpoint missing 'model_architecture'. Cannot export.")

    num_classes = state.get("num_classes")
    if num_classes is None:
        raise KeyError("Checkpoint missing 'num_classes'. Cannot export.")

    model_state = state.get("model_state_dict")
    if model_state is None:
        raise KeyError("Checkpoint missing 'model_state_dict'. Cannot export.")

    model = create_model(model_architecture, num_classes=num_classes)
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
