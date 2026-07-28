import os
import sys
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from training.next_frame_trainer import (
    NextFrameTrainerConfig,
    train_next_frame_predictor,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_next_frame_predictor",
)
def main(cfg: DictConfig) -> None:
    print("\n" + "=" * 70)
    print("NEXT FRAME PREDICTOR TRAINING")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70 + "\n")

    set_seed(cfg.seed)

    trainer_cfg = NextFrameTrainerConfig(
        data_dir=cfg.data.processed_dir,
        vae_weights_path=cfg.paths.vae_weights,
        save_dir=cfg.paths.save_dir,
        batch_size=cfg.data.batch_size,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.lr,
        train_ratio=cfg.data.train_split,
        max_norm=cfg.training.max_norm,
        latent_loss_weight=cfg.training.latent_loss_weight,
        frame_loss_weight=cfg.training.frame_loss_weight,
        foreground_pixel_weight=cfg.training.foreground_pixel_weight,
        white_pixel_threshold=cfg.training.white_pixel_threshold,
        sample_dir=cfg.paths.samples_dir,
        save_samples_every=cfg.training.save_samples_every,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        device=cfg.device,
        seed=cfg.seed,
    )

    train_next_frame_predictor(trainer_cfg)


if __name__ == "__main__":
    main()
