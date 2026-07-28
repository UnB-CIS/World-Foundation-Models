import torch
from torch.utils.data import DataLoader

from .model import VAE
from .training_state import VAETrainingState
from .loss import gradient_weighted_loss, aggressive_beta


def _make_zero_action(images: torch.Tensor, model: VAE) -> torch.Tensor:
    batch_size = images.size(0)
    spatial_size = images.size(-1) // 8
    return torch.zeros(
        batch_size,
        model.action_latent_channels,
        spatial_size,
        spatial_size,
        device=images.device,
        dtype=images.dtype,
    )


def validate_vae(
    model: VAE,
    test_loader: DataLoader,
    model_training_state: VAETrainingState,
    device: torch.device | str,
) -> dict[str, float]:
    model.eval()
    val_loss_total = val_recon = val_kl = 0
    beta = 1.0

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Pass explicit zero action — consistent with inference when no
            # click is present, and with how vae_loss_fn trains the model.
            z_action = _make_zero_action(images, model)
            reconstructed, encoded = model(images, z_action=z_action)
            mu, logvar = torch.chunk(encoded, 2, dim=1)

            loss_val, recon_loss, kl_loss = gradient_weighted_loss(
                images, reconstructed, mu, logvar, beta=beta
            )

            val_loss_total += loss_val.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()

    model.train()
    return {
        "val_loss": val_loss_total / len(test_loader),
        "val_recon": val_recon / len(test_loader),
        "val_kl": val_kl / len(test_loader),
    }


def vae_loss_fn(
    model_input: torch.Tensor,
    model_output: tuple[torch.Tensor, torch.Tensor],
    epoch: int,
    batch_idx: int,
    num_batches: int,
    model_training_state: VAETrainingState,
    **kwargs,
) -> torch.Tensor:
    recon, encoded = model_output

    mu, logvar = torch.chunk(encoded, 2, dim=1)
    beta = aggressive_beta(epoch, batch_idx, num_batches, warmup_epochs=8)

    loss, recon_loss, kl_loss = gradient_weighted_loss(
        model_input, recon, mu, logvar, beta=beta
    )

    loss_dict = {
        "loss": loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "beta": beta,
    }

    model_training_state.update_loss_metrics(loss_dict)

    return loss
