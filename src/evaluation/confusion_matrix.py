"""
Confusion matrix utilities for evaluating classification performance.

This module provides functions to:
- compute a normalized confusion matrix from model predictions
- visualize the matrix using a heatmap
- save the resulting plot to results/confusion_matrix.png

The confusion matrix helps identify which classes are frequently confused
and provides per-class insight beyond overall accuracy.

Short labels:
    The module supports both curated short labels for known tomato disease
    classes and dynamic fallback shortening for any unknown class names.
    This ensures readable axis labels even if the dataset changes or new
    classes are introduced.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Curated short labels for known tomato disease classes.
# These provide clean, human-readable names for the confusion matrix.
SHORT_LABELS = {
    "Tomato_Bacterial_spot": "Bacterial spot",
    "Tomato_Early_blight": "Early blight",
    "Tomato_Late_blight": "Late blight",
    "Tomato_Leaf_Mold": "Leaf mold",
    "Tomato_Septoria_leaf_spot": "Septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider mites",
    "Tomato_Target_Spot": "Target spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus": "TYLCV",
    "Tomato_Tomato_mosaic_virus": "TMV",
    "Tomato_healthy": "Healthy"
}

def to_short_label(original_name: str) -> str:
    """
    Convert a long dataset class name into a short, readable label.

    The function first checks whether a curated short label exists in
    SHORT_LABELS. If not, it applies a dynamic fallback strategy:

    - Remove the redundant 'Tomato_' prefix.
    - Split the remaining name by underscores.
    - Keep the last one or two tokens, which typically contain the
      meaningful disease name.
    - Capitalize the result for readability.

    Args:
        original_name (str): The full class name from the dataset.

    Returns:
        str: A human-readable short label suitable for axis display.
    """
    
    if original_name in SHORT_LABELS:
        return SHORT_LABELS[original_name]

    # Dynamic fallback shortening
    cleaned = original_name.replace("Tomato_", "")
    tokens = cleaned.split("_")

    # Keep last 1–2 tokens depending on length
    if len(tokens) >= 2:
        label = " ".join(tokens[-2:])
    else:
        label = tokens[0]

    return label.capitalize()

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path="results/plots/confusion_matrix.png"
):
    """
    Generate and save a normalized confusion matrix plot.

    The function computes a confusion matrix, normalizes it per true class,
    applies short-label conversion for readability, and visualizes the
    matrix using a seaborn heatmap. The resulting plot is saved to the
    specified location, and the output directory is created if necessary.

    Args:
        y_true (list or array): Ground-truth class indices.
        y_pred (list or array): Predicted class indices.
        class_names (list[str]): Class names in index order.
        save_path (str): Output file path for the PNG plot.
    """

    # Ensure output directory exists
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    # Convert long class names to short labels
    short_names = [to_short_label(name) for name in class_names]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize rows (true classes) with safe division (avoid divide-by-zero)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1
    cm_normalized = cm.astype(float) / denom

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names
    )

    plt.xlabel("Predicted Disease")
    plt.ylabel("Actual Disease")
    plt.title("Tomato Disease Classification - Normalized Confusion Matrix based on SimpleCNN")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
