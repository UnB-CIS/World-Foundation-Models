import os
import torch
import pytest
from PIL import Image
from unittest.mock import patch

from training.vae_trainer import VAETrainer


def _make_fake_frames(frames_root: str, n: int = 20, size: int = 16):
    video_dir = os.path.join(frames_root, "fake_video")
    os.makedirs(video_dir, exist_ok=True)
    for i in range(n):
        img = Image.fromarray(
            torch.randint(0, 255, (size, size)).byte().numpy(), mode="L"
        )
        img.save(os.path.join(video_dir, f"frame_{i:04d}.jpg"))


def test_vae_trainer_builds(vae_cfg, tmp_path):
    """Trainer instantiates all components without error."""
    _make_fake_frames(vae_cfg.data.frames_root)

    # Skip actual video extraction — frames already exist
    with patch("src.vae.dataset.process_videos"):
        trainer = VAETrainer(vae_cfg)

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert len(trainer.train_loader.dataset) > 0
    assert len(trainer.test_loader.dataset) > 0


def test_vae_trainer_fit(vae_cfg, tmp_path):
    """Full training loop runs for 2 epochs and returns valid history."""
    _make_fake_frames(vae_cfg.data.frames_root)

    with patch("src.vae.dataset.process_videos"):
        trainer = VAETrainer(vae_cfg)

    history = trainer.fit()

    # History keys
    assert "train_loss" in history
    assert "val_loss" in history
    assert "train_kl" in history
    assert "betas" in history

    # Correct number of epochs recorded
    assert len(history["train_loss"]) == vae_cfg.training.num_epochs
    assert len(history["val_loss"]) == vae_cfg.training.num_epochs

    # Losses are finite
    assert all(torch.isfinite(torch.tensor(v)) for v in history["train_loss"])
    assert all(torch.isfinite(torch.tensor(v)) for v in history["val_loss"])

    # Checkpoint saved
    assert os.path.exists(os.path.join(vae_cfg.paths.save_dir, "best_model.pth"))

    # Config saved
    assert os.path.exists(os.path.join(vae_cfg.paths.save_dir, "config.yaml"))
