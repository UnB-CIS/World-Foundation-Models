"""Model common Blocks"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Block of Convolution layer with activation and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.activation(self.batch_norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """ResNet inspired block"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.batch_norm_1 = nn.BatchNorm2d(channels)
        self.batch_norm_2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor):
        residual = x
        out = F.relu(self.batch_norm_1(self.conv1(x)))
        out = self.batch_norm_2(self.conv2(out))
        return F.relu(out + residual)
