"""
Utility functions for visualizing training progress.

This module provides plotting helpers for:
- training and validation loss curves
- training and validation accuracy curves

All plots are saved as PNG files under results/plots/. The directory is created
automatically if it does not exist. These visualizations are used to assess
model convergence, detect overfitting, and support the final project report.
"""

import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_curves(epochs, list_of_training_accuracies, list_of_training_losses, list_of_validation_accuracies, list_of_validation_losses, save_dir="results/plots"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    x_axis_scaling = range(1, len(list_of_training_accuracies) + 1)
    
    # Loss curve
    plt.figure(figsize=(8, 5))
    
    plt.plot(x_axis_scaling, list_of_training_losses, label="Training Loss")
    plt.plot(x_axis_scaling, list_of_validation_losses, label="Validation Loss")
    plt.xlim(1, len(x_axis_scaling))
    plt.xticks(x_axis_scaling)
    
    plt.title("SimpleCNN Training and Validation Loss Over 20 Epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    
    plt.plot(x_axis_scaling, list_of_training_accuracies, label="Training Accuracy")
    plt.plot(x_axis_scaling, list_of_validation_accuracies, label="Validation Accuracy")
    plt.xlim(1, len(x_axis_scaling))
    plt.xticks(x_axis_scaling)
    
    plt.title("SimpleCNN Training and Validation Accuracy Over 20 Epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/accuracy_curve.png")
    plt.close()
