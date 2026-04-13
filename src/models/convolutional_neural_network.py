"""
Convolutional neural network architecture used for tomato leaf disease classification.

This module defines a compact CNN consisting of three convolution-ReLU-maxpool
blocks followed by a fully connected classifier head. The architecture is designed
for RGB images resized to 224x224 pixels and produces class logits suitable for
multi-class classification using cross-entropy loss.

The model follows a common design pattern:
- 3x3 convolutions with padding to preserve spatial dimensions
- doubling the number of channels in deeper layers
- max-pooling to progressively reduce spatial resolution
- a fully connected head operating on flattened feature maps
"""

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A small convolutional neural network for image classification.

    The network consists of:
    - Three convolutional blocks, each containing:
        - a 3x3 convolution with padding=1
        - a ReLU activation
        - a 2x2 max-pooling layer that halves spatial resolution
    - A classifier head that flattens the feature maps and applies:
        - a fully connected layer with 256 hidden units
        - ReLU activation
        - dropout regularization
        - a final linear layer mapping to class logits

    Expected input shape:
        (batch_size, 3, 224, 224)

    After the convolutional stack, the feature tensor has shape:
        (batch_size, 128, 28, 28)

    Args:
        num_classes (int):
            Number of output classes. Determines the size of the final
            linear layer in the classifier head.

    Attributes:
        features (nn.Sequential):
            Convolutional feature extractor producing a tensor of shape
            (batch_size, 128, 28, 28) for 224x224 inputs.
        classifier (nn.Sequential):
            Fully connected classification head mapping the flattened
            feature tensor to class logits.
    """
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels, 224×224 → 112×112 spatial size
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64 channels, 112×112 → 56×56 spatial size
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128 channels, 56×56 → 28×28 spatial size
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # After three pooling operations:
        #   spatial size: 224 → 112 → 56 → 28
        #   channels: 3 → 32 → 64 → 128
        # Flattened feature size: 128 * 28 * 28 = 100352 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # Hidden layer with 256 units:
            # - expressive enough for non-linear combinations
            # - small enough to avoid overfitting
            # - common size in compact CNNs
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer: one logit per class
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Compute the forward pass of the network.

        Args:
            x (torch.Tensor):
                Input batch of images with shape (batch_size, 3, H, W).
                For correct operation, H and W should be 224.

        Returns:
            torch.Tensor:
                Logits of shape (batch_size, num_classes), suitable for
                use with torch.nn.CrossEntropyLoss.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
