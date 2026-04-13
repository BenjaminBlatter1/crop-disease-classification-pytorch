"""
Confusion matrix utilities for evaluating classification performance.

This module provides functions to:
- compute a normalized confusion matrix from model predictions
- visualize the matrix using a heatmap
- save the resulting plot to results/confusion_matrix.png

The confusion matrix helps identify which classes are frequently confused
and provides per-class insight beyond overall accuracy.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="results/confusion_matrix.png"):
    """
    Generate and save a normalized confusion matrix plot.

    Args:
        y_true (list or array): Ground-truth class indices.
        y_pred (list or array): Predicted class indices.
        class_names (list[str]): Class names in index order.
        save_path (str): Output file path for the PNG plot.

    The function computes a confusion matrix, normalizes it per class,
    and visualizes it as a heatmap. The plot is saved to the specified
    location, and the results directory is created if necessary.
    """
    Path("results").mkdir(exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
