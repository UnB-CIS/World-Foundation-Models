#!/usr/bin/env python
# coding: utf-8

"""
Latent Fusion Module
--------------------

Implementa Spatial Broadcast Fusion entre:

• z_visual → tensor visual (B, C, H, W)
• z_action → vetor latente da ação (B, A)

Saída:

• z_fused → tensor combinado (B, C+A, H, W)

Seguro para importação.
Testes executam apenas se rodar diretamente.
"""

import torch
import torch.nn as nn


# =========================================================
# FUSER
# =========================================================

class SpatialBroadcastFuser(nn.Module):
    """
    Spatial Broadcast Fusion

    Expande o vetor latente da ação espacialmente e concatena
    com o tensor visual no eixo dos canais.
    """

    def __init__(self, height=None, width=None):
        super().__init__()

        self.height = height
        self.width = width


    def forward(self, z_visual: torch.Tensor, z_action: torch.Tensor):

        """
        Args:

        z_visual : (B, C, H, W)

        z_action : (B, A)

        Returns:

        z_fused : (B, C+A, H, W)
        """

        if z_visual.dim() != 4:
            raise ValueError(
                f"z_visual deve ser 4D (B,C,H,W), recebido {z_visual.shape}"
            )

        if z_action.dim() != 2:
            raise ValueError(
                f"z_action deve ser 2D (B,A), recebido {z_action.shape}"
            )

        B, C, H, W = z_visual.shape

        height = self.height if self.height is not None else H
        width = self.width if self.width is not None else W

        # Confere consistência
        if height != H or width != W:
            raise ValueError(
                f"Dimensões incompatíveis: visual=({H},{W}) vs fuser=({height},{width})"
            )

        # (B, A) → (B, A, 1, 1)
        z_action_expanded = z_action.unsqueeze(-1).unsqueeze(-1)

        # (B, A, H, W)
        z_action_tiled = z_action_expanded.expand(-1, -1, height, width)

        # concatena canais
        z_fused = torch.cat(
            [z_visual, z_action_tiled],
            dim=1
        )

        return z_fused


# =========================================================
# PROBE (opcional — para debug e validação)
# =========================================================

class ActionProbe(nn.Module):
    """
    Rede auxiliar que tenta recuperar z_action do tensor fundido.

    Usado apenas para testes de integridade da fusão.
    """

    def __init__(self, in_channels, action_dim):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(
            in_channels,
            action_dim
        )


    def forward(self, x):

        x = self.gap(x)

        x = x.flatten(1)

        return self.linear(x)


# =========================================================
# TESTE
# =========================================================

def test_fusion():

    print("Testing SpatialBroadcastFuser...")

    batch_size = 32
    visual_channels = 8
    action_dim = 16
    H = 8
    W = 8

    z_visual = torch.randn(
        batch_size,
        visual_channels,
        H,
        W
    )

    z_action = torch.randn(
        batch_size,
        action_dim
    )

    fuser = SpatialBroadcastFuser()

    z_fused = fuser(
        z_visual,
        z_action
    )

    print("Visual shape :", z_visual.shape)
    print("Action shape :", z_action.shape)
    print("Fused shape  :", z_fused.shape)

    expected_channels = visual_channels + action_dim

    assert z_fused.shape == (
        batch_size,
        expected_channels,
        H,
        W
    )

    print("Fusion OK")


def test_probe():

    print("\nTesting probe recovery...")

    batch_size = 32
    visual_channels = 8
    action_dim = 16
    H = 8
    W = 8

    z_visual = torch.randn(
        batch_size,
        visual_channels,
        H,
        W
    )

    z_action = torch.randn(
        batch_size,
        action_dim
    )

    fuser = SpatialBroadcastFuser()

    z_fused = fuser(
        z_visual,
        z_action
    ).detach()

    probe = ActionProbe(
        visual_channels + action_dim,
        action_dim
    )

    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=0.01
    )

    loss_fn = nn.MSELoss()

    for i in range(300):

        optimizer.zero_grad()

        recovered = probe(z_fused)

        loss = loss_fn(
            recovered,
            z_action
        )

        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print(f"Iter {i} loss {loss.item():.6f}")

    if loss.item() < 0.01:
        print("Probe recovery OK")
    else:
        print("Warning: probe recovery high loss")




