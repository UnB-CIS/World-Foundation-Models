"""VAE class"""

import torch
import torch.nn as nn
from src.utils.blocks import ConvBlock, ResidualBlock


class VAEEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 32, 3, stride=1, padding=1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.utils.spectral_norm(nn.Conv2d(256, 16, kernel_size=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class VAEDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.decoder_main = nn.Sequential(
            ConvBlock(8, 64, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            ResidualBlock(32),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(32, 16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
        )

        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.final_conv.weight, gain=0.02)

        assert self.final_conv.bias is not None

        nn.init.constant_(self.final_conv.bias, 0.0)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_main(z)
        x = self.final_conv(x)

        return torch.clamp(x, 0, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstructed = self._decode(z)

        return reconstructed


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = VAEEncoder()

        self.decoder = VAEDecoder()

    def reparametrization(
        self, mean: torch.Tensor, log_variance: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mean)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)

        mean, log_variance = torch.chunk(encoded, 2, dim=1)
        log_variance = torch.clamp(log_variance, -10, 10)
        z = self.reparametrization(mean, log_variance)

        reconstructed = self.decoder(z)

        return reconstructed, encoded
