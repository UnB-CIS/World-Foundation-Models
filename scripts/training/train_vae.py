import os
import json
from datetime import datetime

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from training.vae_trainer import VAETrainer


def train(cfg: DictConfig) -> None:
    print(f"Device: {cfg.device} | CUDA available: {torch.cuda.is_available()}")

    trainer = VAETrainer(cfg)

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

    print("VAE was trained successfully.")


if __name__ == "__main__":
    from hydra import compose, initialize

    # Pega os argumentos da linha de comando, ignorando o nome do script
    import sys

    overrides = sys.argv[1:]

    # Inicializa o Hydra manualmente sem usar argparse
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="train_vae", overrides=overrides)
        train(cfg)
