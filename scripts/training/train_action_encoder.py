import os
import json
from datetime import datetime

import torch
import hydra
from omegaconf import DictConfig

from training.action_encoder_trainer import ActionEncoderTrainer


def train(cfg: DictConfig) -> None:
    print(f"Device: {cfg.device} | CUDA available: {torch.cuda.is_available()}")

    trainer = ActionEncoderTrainer(cfg)

    print(
        f"Train: {len(trainer.train_loader.dataset)} samples  "
        f"Val: {len(trainer.test_loader.dataset)} samples  "
        f"Batches/epoch: {len(trainer.train_loader)}"
    )

    history = trainer.fit()

    filename = os.path.join(
        cfg.paths.save_dir, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    from hydra import compose, initialize
    import sys

    overrides = sys.argv[1:]

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="train_action_encoder", overrides=overrides)
        train(cfg)
