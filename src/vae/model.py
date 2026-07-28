"""VAE class"""

import torch
import torch.nn as nn

from src.utils.blocks import ConvBlock, ResidualBlock


class VAEEncoder(nn.Module):
    def __init__(self, latent_channels: int = 16) -> None:
        super().__init__()

        self.latent_channels = latent_channels

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
            nn.utils.spectral_norm(nn.Conv2d(256, 2 * latent_channels, kernel_size=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        action_latent_channels: int = 16,
    ) -> None:
        super().__init__()

        self.latent_channels = latent_channels
        self.action_latent_channels = action_latent_channels

        self.decoder_main = nn.Sequential(
            ConvBlock(
                latent_channels + action_latent_channels, 128, kernel_size=1, padding=0
            ),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(128, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            ResidualBlock(32),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.final_conv.weight, gain=0.02)
        assert self.final_conv.bias is not None
        nn.init.constant_(self.final_conv.bias, 0.0)

    def _prepare_input(
        self,
        z: torch.Tensor,
        z_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.action_latent_channels <= 0:
            return z

        batch_size, _, height, width = z.shape

        if z_action is not None:
            if z_action.dim() == 2:
                z_action_tiled = (
                    z_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
                )
            elif z_action.dim() == 4:
                z_action_tiled = z_action
            else:
                raise ValueError(
                    f"z_action deve ser 2D ou 4D, recebido {tuple(z_action.shape)}"
                )
        else:
            z_action_tiled = torch.zeros(
                batch_size,
                self.action_latent_channels,
                height,
                width,
                device=z.device,
                dtype=z.dtype,
            )

        return torch.cat([z, z_action_tiled], dim=1)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_main(z)
        x = self.final_conv(x)
        return torch.clamp(x, 0, 1)

    def forward(
        self,
        z: torch.Tensor,
        z_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_input = self._prepare_input(z, z_action)
        reconstructed = self._decode(z_input)
        return reconstructed


class VAE(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        action_latent_channels: int = 16,
    ) -> None:
        super().__init__()

        self.latent_channels = latent_channels
        self.action_latent_channels = action_latent_channels

        self.encoder = VAEEncoder(latent_channels=latent_channels)
        self.decoder = VAEDecoder(
            latent_channels=latent_channels,
            action_latent_channels=action_latent_channels,
        )

    def reparametrization(
        self,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(
        self,
        x: torch.Tensor,
        z_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)

        mean, log_variance = torch.chunk(encoded, 2, dim=1)
        log_variance = torch.clamp(log_variance, -10, 10)
        z = self.reparametrization(mean, log_variance)

        reconstructed = self.decoder(z, z_action)

        return reconstructed, encoded
