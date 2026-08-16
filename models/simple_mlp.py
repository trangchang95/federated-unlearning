"""A first neural network for MNIST (Month 1, Week 2).

This is intentionally a fully connected network, not a CNN.  A CNN is the
next week's topic.  Keeping the first network small makes the five essential
parts visible: inputs, weights, activations, logits, and a loss used to update
the weights.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """Map a 28x28 MNIST image to ten unnormalized class scores (logits)."""

    def __init__(self, hidden_units: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run forward propagation and return one score for each digit."""
        return self.network(images)
