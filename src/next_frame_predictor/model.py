"""Predict next visual latent from fused visual+action latent."""

import torch
import torch.nn as nn

from src.utils.blocks import ConvBlock, ResidualBlock


class NextFrameLatentPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        hidden_channels: int = 64,
        out_channels: int = 16,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
