"""Model Definition for Action Encoder"""

import torch
import torch.nn as nn


class ActionTextEncoder(nn.Module):
    """
    Deterministic encoder that maps input vector to latent space.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class ActionTextDecoder(nn.Module):
    """
    Deterministic decoder that reconstructs action vector from latent space.
    """

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, output_dim)
        )

    def forward(self, z):
        x = self.net(z)
        return torch.sigmoid(x)


class ActionTextModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim) -> None:
        super().__init__()
        self.encoder = ActionTextEncoder(input_dim=input_dim, latent_dim=latent_dim)

        self.decoder = ActionTextDecoder(latent_dim=latent_dim, output_dim=output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)):
            x = x[0]
        z = self.encoder(x)
        reconstructed = self.decoder(z)

        return reconstructed, z
