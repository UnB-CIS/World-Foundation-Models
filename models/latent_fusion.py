#!/usr/bin/env python
# coding: utf-8

"""
Latent Fusion Module
--------------------

Implementa Spatial Broadcast Fusion com suporte a Frame Stacking entre:

• z_visual_stack → lista de tensores visuais ou tensor empilhado (B, K*C, H, W)
• z_action       → vetor latente da ação (B, A)

Saída:

• z_fused → tensor combinado (B, (K*C)+A, H, W)

A principal alteração é que agora a fusão ocorre com os últimos K frames para contexto, em que K = 4
"""

import torch
import torch.nn as nn


# =========================================================
# FUSER
# =========================================================

class SpatialBroadcastFuser(nn.Module):
    """
    Spatial Broadcast Fusion com Frame Stacking

    Recebe o histórico de frames, concatena-os no eixo dos canais (caso
    seja uma lista), expande o vetor latente da ação espacialmente 
    e concatena tudo em um único tensor.
    """
    def __init__(self, height=None, width=None):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, z_visual_stack, z_action: torch.Tensor):
        """
        Args:
        z_visual_stack : Tensor (B, K*C, H, W) ou Lista/Tupla de Tensors [(B, C, H, W), ...]
                         representando os K últimos frames.
        z_action : (B, A) representando a ação no tempo t.
        Returns:
        z_fused : (B, (K*C)+A, H, W)
        """

        # Se receber uma lista de frames (ex: t-3, t-2, ...), empilha nos canais
        if isinstance(z_visual_stack, (list, tuple)):
            z_visual_stack = torch.cat(z_visual_stack, dim=1)
        if z_visual_stack.dim() != 4:
            raise ValueError(
                f"z_visual_stack deve ser 4D (B,C,H,W), recebido {z_visual_stack.shape}"
            )
        if z_action.dim() != 2:
            raise ValueError(
                f"z_action deve ser 2D (B,A), recebido {z_action.shape}"
            )

        B, C_total, H, W = z_visual_stack.shape

        height = self.height if self.height is not None else H
        width = self.width if self.width is not None else W

        # Confere consistência
        if height != H or width != W:
            raise ValueError(
                f"Dimensões incompatíveis: visual=({H},{W}) vs fuser=({height},{width})"
            )

        # (B, A) → (B, A, 1, 1)
        z_action_expanded = z_action.unsqueeze(-1).unsqueeze(-1)

        # Copia a ação na grade HxW vezes na grade de altura e largura
        # (B, A, H, W)
        z_action_tiled = z_action_expanded.expand(-1, -1, height, width)

        # concatena canais (Histórico Visual + Ação)
        z_fused = torch.cat([z_visual_stack, z_action_tiled],dim=1)

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
        self.gap = nn.AdaptiveAvgPool2d(1) # Reduz cada fatia [H,W] a um valor médio, gerando [B, C, 1, 1]
        self.linear = nn.Linear(
            in_channels,
            action_dim
        )

    def forward(self, x):
        x = self.gap(x)          # [B, C, H, W] -> [B, C, 1, 1]
        x = x.flatten(1)         # [B, C, 1, 1] -> [B, C]
        return self.linear(x)    # [B, C] -> [B, action_dim]    


# =========================================================
# TESTE
# =========================================================

def test_fusion():

    print("Testing SpatialBroadcastFuser with 4-Frame Stacking...")

    batch_size = 32
    visual_channels = 8
    num_frames = 4  # alterado para 4 frames
    action_dim = 16
    H = 8
    W = 8

    # Simulando o envio de uma lista de 4 frames isolados
    z_visual_list = [
        torch.randn(batch_size, visual_channels, H, W) for _ in range(num_frames)
    ]

    z_action = torch.randn(
        batch_size,
        action_dim
    )

    fuser = SpatialBroadcastFuser()

    z_fused = fuser(
        z_visual_list,
        z_action
    )

    print(f"Visual shape (per frame) : {z_visual_list[0].shape}")
    print(f"Action shape             : {z_action.shape}")
    print(f"Fused shape              : {z_fused.shape}")

    expected_channels = (visual_channels * num_frames) + action_dim

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
    num_frames = 4  # alterado para 4 frames
    action_dim = 16
    H = 8
    W = 8

    z_visual_stack = torch.randn(
        batch_size,
        visual_channels * num_frames,
        H,
        W
    )

    z_action = torch.randn(
        batch_size,
        action_dim
    )

    fuser = SpatialBroadcastFuser()

    z_fused = fuser(
        z_visual_stack,
        z_action
    ).detach()

    probe = ActionProbe(
        (visual_channels * num_frames) + action_dim,
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


if __name__ == "__main__":
    test_fusion()
    test_probe()

'''
Resultado ao rodar:

    Fusion OK

    Testing probe recovery...
    Iter 0 loss 1.079126
    Iter 100 loss 0.019034
    Iter 200 loss 0.001652
    Probe recovery OK
'''