import torch
import torch.nn as nn


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
            raise ValueError(f"z_action deve ser 2D (B,A), recebido {z_action.shape}")

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
        z_fused = torch.cat([z_visual_stack, z_action_tiled], dim=1)

        return z_fused


class ActionProbe(nn.Module):
    """
    Rede auxiliar que tenta recuperar z_action do tensor fundido.
    Usado apenas para testes de integridade da fusão.
    """

    def __init__(self, in_channels, action_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(
            1
        )  # Reduz cada fatia [H,W] a um valor médio, gerando [B, C, 1, 1]
        self.linear = nn.Linear(in_channels, action_dim)

    def forward(self, x):
        x = self.gap(x)  # [B, C, H, W] -> [B, C, 1, 1]
        x = x.flatten(1)  # [B, C, 1, 1] -> [B, C]
        return self.linear(x)  # [B, C] -> [B, action_dim]
