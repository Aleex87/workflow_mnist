"""
This file defines the PyTorch neural network architecture.

Responsibilities:
- Define the model class
- Define forward pass
- Keep architecture separated from training logic
"""

import torch
import torch.nn as nn


class DigitsMLP(nn.Module):
    """A simple MLP for 8x8 digit classification (10 classes)."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)