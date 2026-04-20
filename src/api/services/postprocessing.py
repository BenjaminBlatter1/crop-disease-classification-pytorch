"""
Postprocessing utilities for converting model outputs into class predictions.
"""

from __future__ import annotations
import torch
from typing import List, Tuple

def postprocess(logits: torch.Tensor, class_names: List[str]) -> Tuple[str, float]:
    """Return (predicted_class, confidence)."""
    
    probs = torch.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)
    return class_names[idx.item()], conf.item()
