"""
Image preprocessing pipeline for inference.

Applies the same transforms used during validation in training.
"""

from __future__ import annotations
from typing import Callable
from PIL import Image
import torchvision.transforms as T
import torch

def get_preprocess_function() -> Callable[[Image.Image], torch.Tensor]:
    """Return the preprocessing function used for inference."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], # Established ImageNet statistics
                    std=[0.229, 0.224, 0.225]), # Established ImageNet statistics
    ])
