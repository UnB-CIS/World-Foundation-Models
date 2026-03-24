"""Loss functions for VAE."""

import torch


def aggressive_beta(
    epoch: int,
    batch_idx: int,
    num_batches: int,
    warmup_epochs: int = 8,
) -> float:
    """
    MUCH faster beta ramp for difficult data
    Reach β=1.0 in 8 epochs (was 15)
    """
    total_steps = warmup_epochs * num_batches
    current_step = epoch * num_batches + batch_idx

    if current_step >= total_steps:
        return 1.0

    progress = current_step / total_steps
    return progress**0.3  # Faster than square root


def gradient_weighted_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EXTREME weighting for 99.7% background data
    Focus entirely on edges (balls/ground)
    """
    batch_size = x.size(0)

    # Sobel edge detection
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)

    x_padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
    grad_x = torch.nn.functional.conv2d(x_padded, sobel_x)
    grad_y = torch.nn.functional.conv2d(x_padded, sobel_y)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    # Normalize
    grad_max = gradient_magnitude.max()
    if grad_max > 1e-8:
        gradient_magnitude = gradient_magnitude / grad_max

    # EXTREME weighting: 1-100x (was 1-21x)
    mse = (x_hat - x) ** 2
    weights = 0.1 + 100.0 * gradient_magnitude  # Background=0.1, edges=100

    recon_loss = (mse * weights).sum() / batch_size

    # Free bits KL
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim_clamped = torch.clamp(kl_per_dim - 1.5, min=0.0)  # Lower free bits
    kl_loss = kl_per_dim_clamped.sum() / batch_size

    actual_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, actual_kl
