"""Decoder from Fusion latents to standard latents"""

import torch
import torch.nn as nn

from src.utils.blocks import ConvBlock, ResidualBlock


class LatentDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(24, 24, kernel_size=3, padding=1),
            ResidualBlock(24),
            nn.Conv2d(24, 8, kernel_size=1),
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        latent, _ = inputs
        return self.net(latent)
