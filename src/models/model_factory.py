"""
Factory utilities for constructing model architectures used in training and inference.

This module centralizes the creation of supported neural network architectures.
It provides a single entry point (`create_model`) that instantiates either a
lightweight custom CNN or a pretrained backbone with a task-specific classifier
head.

Supported architectures:
- "simplecnn": a small convolutional network defined in convolutional_neural_network.py
- "resnet18": an ImageNet-pretrained ResNet-18 with its final fully connected
  layer replaced to match the number of target classes

All returned models are untrained with respect to the tomato leaf dataset.
Pretrained backbones (e.g., ResNet-18) are initialized with ImageNet weights
and then fine-tuned during training.

Functions:
    create_model(model_architecture: str, num_classes: int) -> nn.Module
        Build and return the requested architecture with the correct output dimension.
"""

import torch.nn as nn
from torchvision import models
from .convolutional_neural_network import SimpleCNN


def create_model(model_architecture: str, num_classes: int) -> nn.Module:
    """
    Create a model architecture based on the selected model type.

    Args:
        model_architecture (str): Name of the architecture ("simplecnn", "resnet18").
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Instantiated model.
    """
    if model_architecture == "simplecnn":
        return SimpleCNN(num_classes)

    if model_architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model type: {model_architecture}")
