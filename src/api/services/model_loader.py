"""
Unified model loading and device selection for the inference API.

This module loads PyTorch checkpoints, TorchScript models, or ONNX models,
selects the appropriate compute device, and performs an optional warmup pass
to ensure stable latency on first inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
import onnxruntime as ort

from src.utils.device import get_best_device
from src.models.model_factory import create_model
from src.api.dependencies.config import Settings

logger = logging.getLogger("api")

class InferenceModel:
    """
    Unified interface for PyTorch, TorchScript, and ONNX inference.

    Attributes:
        model: The loaded model object (PyTorch or ONNX Runtime session).
        device: The compute device used for inference.
        model_type: One of {"checkpoint", "torchscript", "onnx"}.
    """

    def __init__(
        self,
        model: Any,
        device: torch.device,
        model_type: Literal["checkpoint", "torchscript", "onnx"],
    ) -> None:
        self.model = model
        self.device = device
        self.model_type = model_type

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a single preprocessed tensor.

        Args:
            tensor: A preprocessed image tensor of shape (1, 3, H, W).

        Returns:
            torch.Tensor: Raw model logits.
        """
        if self.model_type in {"checkpoint", "torchscript"}:
            tensor = tensor.to(self.device)
            with torch.no_grad():
                return self.model(tensor)

        if self.model_type == "onnx":
            ort_inputs = {self.model.get_inputs()[0].name: tensor.cpu().numpy()}
            ort_outs = self.model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0])

        raise ValueError(f"Unsupported model type: {self.model_type}")


def load_model(settings: Settings) -> InferenceModel:
    """
    Load a model based on the configured model path and device settings.

    Supports:
        - PyTorch checkpoints (.pth)
        - TorchScript models (.pt)
        - ONNX models (.onnx)

    Args:
        settings: Application configuration.

    Returns:
        InferenceModel: A unified inference wrapper.
    """

    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # --- Device selection ---
    if settings.device == "auto":
        device = get_best_device()
    else:
        device = torch.device(settings.device)

    logger.info(f"Using device: {device}")

    suffix = model_path.suffix.lower()

    # --- Load checkpoint ---
    if suffix == ".pth":
        checkpoint = torch.load(model_path, map_location=device)

        model = create_model(
            checkpoint["model_architecture"],
            num_classes=len(checkpoint["class_names"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        inference_model = InferenceModel(model, device, "checkpoint")

    # --- Load TorchScript ---
    elif suffix == ".pt":
        model = torch.jit.load(model_path, map_location=device)
        model.eval()

        inference_model = InferenceModel(model, device, "torchscript")

    # --- Load ONNX ---
    elif suffix == ".onnx":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=providers)

        inference_model = InferenceModel(session, device, "onnx")

    else:
        raise ValueError(f"Unsupported model format: {suffix}")

    # --- Warmup pass ---
    try:
        logger.info("Running warmup pass...")
        dummy = torch.randn(1, 3, 224, 224)
        inference_model.predict(dummy)
        logger.info("Warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

    return inference_model
