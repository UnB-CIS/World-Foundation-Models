import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


def plot_training_history(train_losses: list[float], val_losses: list[float]) -> None:
    """
    Plot training and validation losses.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training", marker='o')
    plt.plot(val_losses, label="Validation", marker='s')
    plt.title("Reconstruction Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def plot_with_best_epoch(train_losses: list[float], val_losses: list[float]) -> None:
    """
    Plot training and validation losses with best epoch marked.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    best_epoch = int(np.argmin(val_losses))

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training", marker='o')
    plt.plot(val_losses, label="Validation", marker='s')
    plt.axvline(best_epoch, linestyle='--', alpha=0.6, label=f"Best @ {best_epoch}")
    plt.title("Reconstruction Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


def save_reconstruction_vae_samples(
    model: nn.Module, images: torch.Tensor, epoch: int, batch_idx: int, save_dir: str
) -> None:
    """Save reconstruction samples during training."""
    model.eval()
    with torch.no_grad():
        sample_images = images[:4]
        reconstructed, _ = model(sample_images)

        comparison = torch.cat([sample_images, reconstructed], dim=3)

        filepath = os.path.join(
            save_dir, f"epoch_{epoch + 1}_batch_{batch_idx + 1}.png"
        )
        torchvision.utils.save_image(comparison, filepath, nrow=1)
    model.train()


def save_validation_vae_samples(
    model: nn.Module,
    test_loader: DataLoader,
    global_epoch: int,
    device: str | torch.device,
    save_dir: str,
    phase_name: str = "",
) -> None:
    """Save validation set reconstructions."""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    val_images = next(iter(test_loader))[:8].to(device)

    with torch.no_grad():
        val_recon, _ = model(val_images)

    comparison = torch.cat([val_images, val_recon], dim=0)

    filename = f"{phase_name}_val_epoch_{global_epoch:03d}.png"
    filepath = os.path.join(save_dir, filename)
    torchvision.utils.save_image(comparison, filepath, nrow=8, padding=2)

    model.train()
