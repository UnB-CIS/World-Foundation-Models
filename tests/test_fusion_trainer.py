import os
import torch
import pytest

from training.fusion_trainer import FusionTrainer


def _make_fake_pt_files(source_dir: str, n_files: int = 10, seq_len: int = 5):
    os.makedirs(source_dir, exist_ok=True)
    for i in range(n_files):
        data = {
            "x": torch.randn(seq_len, 32, 8, 8),
            "y": torch.randn(seq_len, 16, 8, 8),
        }
        torch.save(data, os.path.join(source_dir, f"sample_{i:03d}.pt"))


def test_fusion_trainer_builds(fusion_cfg):
    _make_fake_pt_files(fusion_cfg.data.source_dir)

    trainer = FusionTrainer(fusion_cfg)

    assert trainer.model is not None
    assert len(trainer.train_loader.dataset) > 0
    assert len(trainer.test_loader.dataset) > 0


def test_fusion_trainer_fit(fusion_cfg):
    _make_fake_pt_files(fusion_cfg.data.source_dir)

    trainer = FusionTrainer(fusion_cfg)
    history = trainer.fit()

    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == fusion_cfg.training.num_epochs

    assert all(torch.isfinite(torch.tensor(v)) for v in history["train_loss"])
    assert all(torch.isfinite(torch.tensor(v)) for v in history["val_loss"])

    assert os.path.exists(os.path.join(fusion_cfg.paths.save_dir, "best_model.pth"))
    assert os.path.exists(os.path.join(fusion_cfg.paths.save_dir, "config.yaml"))
