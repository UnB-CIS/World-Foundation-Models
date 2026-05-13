"""
VAE for Video
"""

import os
import math
import shutil
from typing import Tuple
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import train_test_split


# ===================== Video Preprocessing =====================

def extract_frames(video_path, frames_dir, frame_step=5):
    """Extract frames from video at specified interval."""
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_step == 0:
            frame_path = os.path.join(frames_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(
        f"✅ Extraídos {saved_count} frames (de {frame_count} totais) para {frames_dir}"
    )


def process_videos(videos_folder, frames_root="./frames", frame_step=5):
    """Process all videos in folder and extract frames."""
    for video_file in os.listdir(videos_folder):
        if video_file.lower().endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(videos_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            frames_dir = os.path.join(frames_root, video_name)
            if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
                extract_frames(video_path, frames_dir, frame_step=frame_step)


# ===================== Loss Functions =====================

def aggressive_beta(epoch, batch_idx, num_batches, warmup_epochs=8):
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


def gradient_weighted_loss(x, x_hat, mu, logvar, beta=1.0):
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


# ===================== Training Utilities =====================

def save_validation_samples(
    model, test_loader, global_epoch, device, save_dir, phase_name=""
):
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


def save_reconstruction_samples(model, images, epoch, batch_idx, save_dir):
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


def validate(model, test_loader, device, beta=1.0):
    """Validation with weighting."""
    model.eval()
    val_loss = val_recon = val_kl = 0

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            reconstructed, encoded = model(images)
            mu, logvar = torch.chunk(encoded, 2, dim=1)

            loss, recon_loss, kl_loss = gradient_weighted_loss(
                images, reconstructed, mu, logvar, beta=beta
            )

            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()

    model.train()
    return (
        val_loss / len(test_loader),
        val_recon / len(test_loader),
        val_kl / len(test_loader),
    )


def save_checkpoint(model, optimizer, epoch, loss, filepath, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, filepath)

    if not is_best:
        print(f"Checkpoint saved: {filepath}")


def plot_training_curves(history, save_dir):
    """
    Plot and save training curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Total Loss
    axes[0, 0].plot(
        epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2
    )
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction Loss
    axes[0, 1].plot(
        epochs, history['train_recon'], 'b-', label='Train Recon', linewidth=2
    )
    axes[0, 1].plot(epochs, history['val_recon'], 'r-', label='Val Recon', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL Divergence
    axes[1, 0].plot(epochs, history['train_kl'], 'b-', label='Train KL', linewidth=2)
    axes[1, 0].plot(epochs, history['val_kl'], 'r-', label='Val KL', linewidth=2)
    axes[1, 0].set_title('KL Divergence', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight'
    )
    plt.close()


def train(
    model, train_loader, test_loader, device, num_epochs=50, save_dir="./saved_models"
):
    """
    Training for 99.7% background data.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./reconstructed_samples", exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Higher LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
        "learning_rates": [],
        "betas": [],
    }

    avg_train_loss = 0
    best_val_loss = float("inf")
    num_batches = len(train_loader)

    print("\n" + "=" * 70)
    print(f"Training Starting - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print("=" * 70 + "\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        epoch_beta_sum = 0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            beta = aggressive_beta(epoch, batch_idx, num_batches, warmup_epochs=8)
            epoch_beta_sum += beta

            # Forward
            recon, encoded = model(images)
            mu, logvar = torch.chunk(encoded, 2, dim=1)

            # Gradient weighting
            loss, recon_loss, kl_loss = gradient_weighted_loss(
                images, recon, mu, logvar, beta=beta
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=2.0
            )  # Higher clip
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"E[{epoch+1:02d}/{num_epochs}] "
                    f"B[{batch_idx+1:03d}/{num_batches}] | "
                    f"L:{loss.item():.3f} R:{recon_loss.item():.3f} "
                    f"KL:{kl_loss.item():.3f} β:{beta:.3f}"
                )

                if beta > 0.2 and kl_loss.item() < 1.0:
                    print(f"     KL={kl_loss.item():.2f} still low at β={beta:.2f}")

            if (batch_idx + 1) % 100 == 0:
                save_reconstruction_samples(
                    model, images, epoch, batch_idx, "./reconstructed_samples"
                )

        # Epoch metrics
        avg_train_loss = train_loss / num_batches
        avg_train_recon = train_recon / num_batches
        avg_train_kl = train_kl / num_batches
        avg_beta = epoch_beta_sum / num_batches

        # Validation
        val_loss, val_recon, val_kl = validate(
            model, test_loader, device, beta=avg_beta
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_recon'].append(avg_train_recon)
        history['train_kl'].append(avg_train_kl)
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)
        history['learning_rates'].append(current_lr)
        history['betas'].append(avg_beta)

        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"{'-'*70}")
        print(
            f"  Train: L={avg_train_loss:.4f} R={avg_train_recon:.4f} KL={avg_train_kl:.4f}"
        )
        print(f"  Val:   L={val_loss:.4f} R={val_recon:.4f} KL={val_kl:.4f}")
        print(f"  Beta={avg_beta:.3f} LR={current_lr:.2e}")

        if avg_beta >= 0.8:
            if avg_train_kl < 1.0:
                print(f"    CRITICAL: KL={avg_train_kl:.2f} < 1.0")
            elif avg_train_kl < 2.0:
                print(f"     WARNING: KL={avg_train_kl:.2f} < 2.0")
            else:
                print(f"    KL healthy: {avg_train_kl:.2f}")
        else:
            print(f"  ℹ  Warmup: β={avg_beta:.2f}")

        print(f"{'='*70}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(save_dir, 'best_model.pth'),
                is_best=True,
            )
            print(f"  Best model saved! (Val Loss: {val_loss:.4f})\n")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_train_loss,
                os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )
            save_validation_samples(
                model, test_loader, epoch, device, "./reconstructed_samples"
            )

        torch.cuda.empty_cache()

    save_checkpoint(
        model,
        optimizer,
        num_epochs - 1,
        avg_train_loss,
        os.path.join(save_dir, "final_model.pth"),
    )

    print("\n  Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 70 + "\n")

    return history


# ===================== Dataset =====================

class VideoFramesDataset(Dataset):
    """Dataset for video frames."""
    
    def __init__(self, frames_dir, transform=None):
        # Recursively find all .jpg frames
        self.frame_files = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(frames_dir)
                for f in files
                if f.lower().endswith(".jpg")
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        img_path = self.frame_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# ===================== Model Architecture =====================

class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""
    
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.activation(self.batch_norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, channels) -> None:
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

    def forward(self, x):
        residual = x
        out = F.relu(self.batch_norm_1(self.conv1(x)))
        out = self.batch_norm_2(self.conv2(out))
        return F.relu(out + residual)


class VAE(nn.Module):
    """Variational Autoencoder for video frames."""
    
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

        # Simple final layer
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        # Initialize
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.final_conv.weight, gain=0.02)
        nn.init.constant_(self.final_conv.bias, 0.0)

    def reparametrization(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decoder(self, z):
        x = self.decoder_main(z)
        x = self.final_conv(x)
        return torch.clamp(x, 0, 1)

    def forward(self, x):
        encoded = self.encoder(x)
        mean, log_variance = torch.chunk(encoded, 2, dim=1)
        log_variance = torch.clamp(log_variance, -10, 10)
        z = self.reparametrization(mean, log_variance)
        reconstructed = self.decoder(z)
        return reconstructed, encoded


# ===================== Main Training Script =====================

def main():
    """Main training function."""
    
    # Configuration
    videos_folder = "dataset/scenario1/videos"  # Folder with videos
    frames_root = "frames"
    frame_step = 5
    batch_size = 256
    num_epochs = 30
    save_dir = "./saved_models"

    # Extract frames from all videos
    print("Extracting frames from videos...")
    process_videos(videos_folder, frames_root, frame_step)

    # Transform for greyscale images
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    # Create dataset from frames
    print("Creating dataset...")
    full_dataset = VideoFramesDataset(frames_root, transform=transform)

    # Split data into training and testing
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"train_dataset: {len(train_dataset)} frames")
    print(f"test_dataset: {len(test_dataset)} frames")
    print(f"device: {device}")

    # Create and train model
    print("\n" + "=" * 70)
    print("EXTREME Training Config -", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    print("Device:", device)
    print("Learning Rate: 1e-3 (higher)")
    print("Epochs:", num_epochs)
    print("Beta warmup: 8 epochs (ultra-fast)")
    print("Free bits: 1.5 nats (lower)")
    print("Edge weighting: 100x (extreme)")
    print("Data: 99.7% background → needs extreme measures")
    print("=" * 70 + "\n")

    model = VAE().to(device)

    # Train model
    history = train(model, train_loader, test_loader, device, num_epochs, save_dir)

    # Save trained weights
    torch.save(model.state_dict(), "vae_weights.pth")
    print("Pesos do VAE salvos")

    # Plot training curves
    plot_training_curves(history, save_dir)


if __name__ == "__main__":
    main()