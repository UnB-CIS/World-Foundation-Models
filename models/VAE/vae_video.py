# vae_video.py

from typing import Tuple
import os
from datetime import datetime
from PIL import Image

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Video preprocessing
# ============================================================

def extract_frames(video_path, frames_dir, frame_step=5):
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

    print(f"Extracted {saved_count} frames from {video_path}")


def process_videos(videos_folder, frames_root="./frames", frame_step=5):

    for video_file in os.listdir(videos_folder):

        if video_file.lower().endswith((".mp4", ".avi", ".mov")):

            video_path = os.path.join(videos_folder, video_file)

            video_name = os.path.splitext(video_file)[0]

            frames_dir = os.path.join(frames_root, video_name)

            if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:

                extract_frames(video_path, frames_dir, frame_step)


# ============================================================
# Loss functions
# ============================================================

def aggressive_beta(epoch, batch_idx, num_batches, warmup_epochs=8):

    total_steps = warmup_epochs * num_batches
    current_step = epoch * num_batches + batch_idx

    if current_step >= total_steps:
        return 1.0

    progress = current_step / total_steps

    return progress ** 0.3


def gradient_weighted_loss(x, x_hat, mu, logvar, beta=1.0):

    batch_size = x.size(0)

    sobel_x = torch.tensor(
        [[-1,0,1],[-2,0,2],[-1,0,1]],
        dtype=torch.float32,
        device=x.device
    ).view(1,1,3,3)

    sobel_y = torch.tensor(
        [[-1,-2,-1],[0,0,0],[1,2,1]],
        dtype=torch.float32,
        device=x.device
    ).view(1,1,3,3)

    x_padded = F.pad(x, (1,1,1,1), mode="replicate")

    grad_x = F.conv2d(x_padded, sobel_x)
    grad_y = F.conv2d(x_padded, sobel_y)

    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    grad_max = gradient_magnitude.max()

    if grad_max > 1e-8:
        gradient_magnitude = gradient_magnitude / grad_max

    mse = (x_hat - x) ** 2

    weights = 0.1 + 100.0 * gradient_magnitude

    recon_loss = (mse * weights).sum() / batch_size

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    kl_per_dim_clamped = torch.clamp(kl_per_dim - 1.5, min=0.0)

    kl_loss = kl_per_dim_clamped.sum() / batch_size

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


# ============================================================
# Dataset
# ============================================================

class VideoFramesDataset(Dataset):

    def __init__(self, frames_dir, transform=None):

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


# ============================================================
# Model blocks
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU(inplace=True)


    def forward(self, x):

        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)


    def forward(self, x):

        residual = x

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.bn2(self.conv2(x))

        return F.relu(x + residual)


# ============================================================
# VAE Model
# ============================================================

class VAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualBlock(64),

            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ResidualBlock(128),

            nn.Conv2d(128,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            ResidualBlock(256),

            nn.Conv2d(256,16,1)
        )

        self.decoder_main = nn.Sequential(

            ConvBlock(8,64,1,padding=0),

            nn.Upsample(scale_factor=2),

            ConvBlock(64,32),

            ResidualBlock(32),

            nn.Upsample(scale_factor=2),

            ConvBlock(32,16),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(16,32,3,padding=1)
        )

        self.final_conv = nn.Conv2d(32,1,3,padding=1)


    def reparametrization(self, mean, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return eps * std + mean


    def decoder(self, z):

        x = self.decoder_main(z)

        x = self.final_conv(x)

        return torch.clamp(x,0,1)


    def forward(self, x):

        encoded = self.encoder(x)

        mean, logvar = torch.chunk(encoded, 2, dim=1)

        logvar = torch.clamp(logvar, -10, 10)

        z = self.reparametrization(mean, logvar)

        reconstructed = self.decoder(z)

        return reconstructed, encoded


# ============================================================
# Training function
# ============================================================

def train(model, train_loader, test_loader, device, num_epochs=50, save_dir="./saved_models"):

    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0

        for images in train_loader:

            images = images.to(device)

            recon, encoded = model(images)

            mu, logvar = torch.chunk(encoded,2,dim=1)

            loss, _, _ = gradient_weighted_loss(images, recon, mu, logvar)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")

        if avg_loss < best_val_loss:

            best_val_loss = avg_loss

            torch.save(
                model.state_dict(),
                os.path.join(save_dir,"vae_best.pth")
            )
