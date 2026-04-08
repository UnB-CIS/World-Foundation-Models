import os
import torch

from training.action_encoder_trainer import ActionEncoderTrainer


def test_action_encoder_trainer_builds(action_encoder_cfg):
    trainer = ActionEncoderTrainer(action_encoder_cfg)

    assert trainer.model is not None
    assert len(trainer.train_loader.dataset) > 0
    assert len(trainer.test_loader.dataset) > 0

    total = action_encoder_cfg.data.num_samples
    expected_train = total - int(total * action_encoder_cfg.data.test_split)
    assert len(trainer.train_loader.dataset) == expected_train


def test_action_encoder_trainer_fit(action_encoder_cfg):
    trainer = ActionEncoderTrainer(action_encoder_cfg)
    history = trainer.fit()

    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == action_encoder_cfg.training.num_epochs

    assert all(torch.isfinite(torch.tensor(v)) for v in history["train_loss"])
    assert all(torch.isfinite(torch.tensor(v)) for v in history["val_loss"])

    assert os.path.exists(
        os.path.join(action_encoder_cfg.paths.save_dir, "best_model.pth")
    )
    assert os.path.exists(
        os.path.join(action_encoder_cfg.paths.save_dir, "config.yaml")
    )
