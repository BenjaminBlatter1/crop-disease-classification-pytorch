"""
Model creation utilities for the tomato leaf disease classification project.

This module provides a helper function for constructing a ResNet-18 model
pretrained on ImageNet and adapting its final classification layer to match
the number of target classes.
"""

import torch.nn as nn
import torchvision.models as models

def create_model(num_classes):
    """
    Create a ResNet-18 model with a custom classification head.

    The function loads a pretrained ResNet-18 backbone, replaces the final
    fully connected layer with a new layer sized for the target number of
    classes, and returns the modified model.

    Args:
        num_classes (int): Number of output classes for classification.

    Returns:
        nn.Module: A ResNet-18 model with a customized final layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
