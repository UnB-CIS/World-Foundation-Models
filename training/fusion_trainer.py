from typing import Callable

import os
import random

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.fusion.decoder import LatentDecoder
from src.fusion.dataset import LatentDataset
from src.fusion.training_state import LatentTrainingState
from src.fusion.utils import validate_latent_decoder, latent_decoder_loss_fn

from training.base_trainer import BaseTrainer


class FusionTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    @property
    def loss_fn(self) -> Callable:
        return latent_decoder_loss_fn

    @property
    def validate_fn(self) -> Callable:
        return validate_latent_decoder

    def _build_model(self) -> LatentDecoder:
        return LatentDecoder()

    def _build_optimizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.training.lr)

    def _build_scheduler(self) -> LRScheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.training.scheduler_factor,
            patience=self.cfg.training.scheduler_patience,
        )

    def _build_dataloaders(self) -> tuple:
        all_files = [
            torch.load(os.path.join(self.cfg.data.source_dir, f), weights_only=False)
            for f in sorted(os.listdir(self.cfg.data.source_dir))
            if f.endswith(".pt")
        ]
        random.shuffle(all_files)

        split_idx = int(self.cfg.data.test_split * len(all_files))

        train_files = all_files[split_idx:]
        test_files = all_files[:split_idx]

        train_dataset = LatentDataset(train_files)
        test_dataset = LatentDataset(test_files)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.compute.num_workers,
            pin_memory=self.cfg.compute.pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.compute.num_workers,
            pin_memory=self.cfg.compute.pin_memory,
        )

        return train_loader, test_loader

    def _build_training_state(self) -> LatentTrainingState:
        return LatentTrainingState()
