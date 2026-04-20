"""
Inference service for running preprocessing and model prediction.

This module contains all logic related to:
- preprocessing
- model inference
- probability extraction
- label mapping
"""

from __future__ import annotations
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def run_inference(
    img: Image.Image,
    model: Any,
    preprocess: Any,
    class_names: list[str] | None,
) -> dict:
    """
    Run preprocessing and model inference on a single image.

    Args:
        img (Image.Image): A decoded RGB image.
        model: Loaded model instance from app.state.MODEL.
        preprocess: Preprocessing function from app.state.PREPROCESS.
        class_names (list[str] | None): Optional list of class labels.

    Returns:
        dict: {
            "label": str,
            "confidence": float
        }
    """

    # Preprocess → (1, C, H, W)
    tensor = preprocess(img).unsqueeze(0)

    # Model forward → logits (1, num_classes)
    logits = model.predict(tensor)

    # Convert to tensor if model returns numpy
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)

    # Ensure shape is (1, C)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    # Softmax → probabilities (1, C)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Top prediction
    top_idx = int(probs.argmax())
    confidence = float(probs[top_idx])

    # Map to class name
    label = class_names[top_idx] if class_names else str(top_idx)

    return {"label": label, "confidence": confidence}
