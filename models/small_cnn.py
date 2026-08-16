"""A small convolutional neural network for Month 1 / Week 3.

The same architecture accepts one-channel MNIST images and three-channel
CIFAR-10 images.  This makes the dataset comparison easier to understand:
the input changes, but the basic convolution -> pooling -> classification
pipeline stays the same.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Extract local image features and classify them into ten classes."""

    def __init__(
        self,
        input_channels: int,
        convolution_channels: tuple[int, int] = (32, 64),
        hidden_units: int = 128,
        dropout: float = 0.25,
        number_of_classes: int = 10,
    ) -> None:
        super().__init__()
        first_channels, second_channels = convolution_channels

        self.features = nn.Sequential(
            # A 3x3 filter looks for a small local pattern such as an edge.
            nn.Conv2d(input_channels, first_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # Pooling halves width/height while keeping the strongest response.
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # This fixed output size lets one model support both 28x28 and 32x32.
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second_channels * 4 * 4, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, number_of_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized class score (logit) per class and image."""
        feature_maps = self.features(images)
        return self.classifier(feature_maps)
