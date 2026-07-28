"""Standalone trainer for next-frame latent prediction using frozen VAE."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.vae.model import VAE
from src.next_frame_predictor.dataset import NextFrameDataset
from src.next_frame_predictor.model import NextFrameLatentPredictor

VISUAL_LATENT_CHANNELS = 16
ACTION_LATENT_CHANNELS = 16
FUSED_LATENT_CHANNELS = VISUAL_LATENT_CHANNELS + ACTION_LATENT_CHANNELS
MODEL_FRAME_SIZE = 64


@dataclass
class NextFrameTrainerConfig:
    data_dir: str
    vae_weights_path: str
    save_dir: str
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    train_ratio: float = 0.9
    max_norm: float = 1.0
    latent_loss_weight: float = 1.0
    frame_loss_weight: float = 0.25
    foreground_pixel_weight: float = 50.0
    white_pixel_threshold: float = 0.98
    sample_dir: Optional[str] = None
    save_samples_every: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "cuda"
    seed: int = 42


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    is_best: bool = False,
) -> None:
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


def frame_from_tensor(frame_tensor: torch.Tensor) -> np.ndarray:
    frame = frame_tensor.detach().cpu().squeeze().numpy()
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0).astype(np.uint8)


def weighted_frame_l1_loss(
    predicted_frame: torch.Tensor,
    target_frame: torch.Tensor,
    foreground_pixel_weight: float,
    white_pixel_threshold: float,
) -> torch.Tensor:
    non_white_mask = target_frame < white_pixel_threshold
    weights = torch.ones_like(target_frame)
    weights = torch.where(
        non_white_mask,
        torch.full_like(target_frame, foreground_pixel_weight),
        weights,
    )

    absolute_error = torch.abs(predicted_frame - target_frame)
    return (absolute_error * weights).sum() / weights.sum().clamp_min(1.0)


def extract_action_latent_from_fused(fused_latent: torch.Tensor) -> torch.Tensor:
    action_broadcast = fused_latent[:, VISUAL_LATENT_CHANNELS:, :, :]
    return action_broadcast.mean(dim=(-2, -1))


def load_latent_dataset(data_dir: str) -> NextFrameDataset:
    all_files = [
        torch.load(os.path.join(data_dir, f), weights_only=False)
        for f in sorted(os.listdir(data_dir))
        if f.endswith(".pt")
    ]

    if not all_files:
        raise FileNotFoundError(f"No processed .pt files found in {data_dir}")

    random.shuffle(all_files)
    return NextFrameDataset(all_files)


def load_frozen_vae(
    vae_weights_path: str,
    device: torch.device,
) -> VAE:
    vae = VAE(action_latent_channels=ACTION_LATENT_CHANNELS).to(device)

    if not os.path.exists(vae_weights_path):
        raise FileNotFoundError(f"VAE weights not found at {vae_weights_path}")

    ckpt = torch.load(vae_weights_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    vae.load_state_dict(state_dict)

    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    return vae


def validate(
    predictor: NextFrameLatentPredictor,
    vae: VAE,
    val_loader: DataLoader,
    device: torch.device,
    latent_loss_weight: float,
    frame_loss_weight: float,
    foreground_pixel_weight: float,
    white_pixel_threshold: float,
) -> dict[str, float]:
    predictor.eval()

    latent_loss_fn = nn.MSELoss()

    val_total = 0.0
    val_latent = 0.0
    val_frame = 0.0

    with torch.no_grad():
        for fused_latent, target_next_latent in val_loader:
            fused_latent = fused_latent.to(device)
            target_next_latent = target_next_latent.to(device)

            pred_next_latent = predictor(fused_latent)
            action_latent = extract_action_latent_from_fused(fused_latent)

            pred_frame = vae.decoder(pred_next_latent, action_latent)
            target_frame = vae.decoder(target_next_latent, action_latent)

            latent_loss = latent_loss_fn(pred_next_latent, target_next_latent)
            frame_loss = weighted_frame_l1_loss(
                pred_frame,
                target_frame,
                foreground_pixel_weight,
                white_pixel_threshold,
            )

            total_loss = (
                latent_loss_weight * latent_loss + frame_loss_weight * frame_loss
            )

            val_total += total_loss.item()
            val_latent += latent_loss.item()
            val_frame += frame_loss.item()

    predictor.train()

    return {
        "val_loss": val_total / len(val_loader),
        "val_latent_loss": val_latent / len(val_loader),
        "val_frame_loss": val_frame / len(val_loader),
    }


def save_visual_samples(
    predictor: NextFrameLatentPredictor,
    vae: VAE,
    val_loader: DataLoader,
    device: torch.device,
    sample_dir: str,
    epoch: int,
) -> None:
    os.makedirs(sample_dir, exist_ok=True)

    try:
        fused_latent, target_next_latent = next(iter(val_loader))
    except StopIteration:
        return

    fused_latent = fused_latent[:8].to(device)
    target_next_latent = target_next_latent[:8].to(device)

    predictor.eval()
    with torch.no_grad():
        pred_next_latent = predictor(fused_latent)
        action_latent = extract_action_latent_from_fused(fused_latent)

        pred_frame = vae.decoder(pred_next_latent, action_latent)
        target_frame = vae.decoder(target_next_latent, action_latent)

    rows = []
    separator = np.full((MODEL_FRAME_SIZE, 4), 255, dtype=np.uint8)

    for idx in range(pred_frame.shape[0]):
        target = frame_from_tensor(target_frame[idx])
        predicted = frame_from_tensor(pred_frame[idx])
        difference = np.abs(
            predicted.astype(np.int16) - target.astype(np.int16)
        ).astype(np.uint8)

        row = np.concatenate(
            [target, separator, predicted, separator, difference], axis=1
        )
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    output_path = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
    cv2.imwrite(output_path, grid)

    predictor.train()


def train_next_frame_predictor(cfg: NextFrameTrainerConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    os.makedirs(cfg.save_dir, exist_ok=True)
    sample_dir = cfg.sample_dir or os.path.join(cfg.save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    dataset = load_latent_dataset(cfg.data_dir)

    train_size = int(len(dataset) * cfg.train_ratio)
    val_size = len(dataset) - train_size
    if val_size == 0:
        raise ValueError("Dataset too small to create validation split.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    predictor = NextFrameLatentPredictor(
        in_channels=FUSED_LATENT_CHANNELS,
        hidden_channels=64,
        out_channels=VISUAL_LATENT_CHANNELS,
    ).to(device)
    vae = load_frozen_vae(cfg.vae_weights_path, device)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    latent_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    print("\n" + "=" * 70)
    print(
        f"Next-frame predictor training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {cfg.num_epochs}")
    print(
        f"Samples: total={len(dataset)} train={len(train_dataset)} val={len(val_dataset)}"
    )
    print("=" * 70 + "\n")

    for epoch in range(cfg.num_epochs):
        flag = True
        predictor.train()

        train_total = 0.0
        train_latent = 0.0
        train_frame = 0.0

        for fused_latent, target_next_latent in train_loader:
            fused_latent = fused_latent.to(device)
            target_next_latent = target_next_latent.to(device)

            if flag:
                print("AQUI", fused_latent.shape, target_next_latent.shape)
                flag = False

            pred_next_latent = predictor(fused_latent)
            action_latent = extract_action_latent_from_fused(fused_latent)

            pred_frame = vae.decoder(pred_next_latent, action_latent)
            target_frame = vae.decoder(target_next_latent, action_latent)

            latent_loss = latent_loss_fn(pred_next_latent, target_next_latent)
            frame_loss = weighted_frame_l1_loss(
                pred_frame,
                target_frame,
                cfg.foreground_pixel_weight,
                cfg.white_pixel_threshold,
            )

            total_loss = (
                cfg.latent_loss_weight * latent_loss
                + cfg.frame_loss_weight * frame_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                predictor.parameters(), max_norm=cfg.max_norm
            )
            optimizer.step()

            train_total += total_loss.item()
            train_latent += latent_loss.item()
            train_frame += frame_loss.item()

        val_metrics = validate(
            predictor=predictor,
            vae=vae,
            val_loader=val_loader,
            device=device,
            latent_loss_weight=cfg.latent_loss_weight,
            frame_loss_weight=cfg.frame_loss_weight,
            foreground_pixel_weight=cfg.foreground_pixel_weight,
            white_pixel_threshold=cfg.white_pixel_threshold,
        )

        avg_train_total = train_total / len(train_loader)
        avg_train_latent = train_latent / len(train_loader)
        avg_train_frame = train_frame / len(train_loader)

        scheduler.step(val_metrics["val_loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"train={avg_train_total:.6f} latent={avg_train_latent:.6f} frame={avg_train_frame:.6f} | "
            f"val={val_metrics['val_loss']:.6f} latent={val_metrics['val_latent_loss']:.6f} "
            f"frame={val_metrics['val_frame_loss']:.6f} | lr={current_lr:.2e}"
        )

        if (epoch + 1) % cfg.save_samples_every == 0:
            save_visual_samples(
                predictor=predictor,
                vae=vae,
                val_loader=val_loader,
                device=device,
                sample_dir=sample_dir,
                epoch=epoch + 1,
            )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model=predictor,
                optimizer=optimizer,
                epoch=epoch,
                loss=best_val_loss,
                filepath=os.path.join(cfg.save_dir, "best_model.pth"),
                is_best=True,
            )
            print(f"  Best model saved! (Val Loss: {best_val_loss:.6f})")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model=predictor,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_train_total,
                filepath=os.path.join(
                    cfg.save_dir, f"checkpoint_epoch_{epoch + 1}.pth"
                ),
            )

        torch.cuda.empty_cache()

    save_checkpoint(
        model=predictor,
        optimizer=optimizer,
        epoch=cfg.num_epochs - 1,
        loss=best_val_loss,
        filepath=os.path.join(cfg.save_dir, "final_model.pth"),
    )

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
