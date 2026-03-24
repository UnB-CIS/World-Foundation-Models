from typing import Callable

from omegaconf import DictConfig

import torch
from torch.optim.lr_scheduler import LRScheduler
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torch.optim.optimizer import Optimizer as Optimizer

from src.vae.model import VAE
from training.base_trainer import BaseTrainer
from src.vae.utils import validate_vae, vae_loss_fn
from src.vae.training_state import VAETrainingState
from src.vae.dataset import VideoFramesDataset, process_videos


class VAETrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    @property
    def loss_fn(self) -> Callable:
        return vae_loss_fn

    @property
    def validate_fn(self) -> Callable:
        return validate_vae

    def _build_model(self) -> VAE:
        return VAE()

    def _build_optimizer(self) -> Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.lr,
        )

    def _build_scheduler(self) -> LRScheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.training.scheduler_factor,
            patience=self.cfg.training.scheduler_patience,
        )

    def _build_dataloaders(self) -> tuple:
        process_videos(
            videos_folder=self.cfg.data.videos_folder,
            frames_root=self.cfg.data.frames_root,
            frame_step=self.cfg.data.frame_step,
        )

        transform = T.Compose(
            [
                T.Resize((self.cfg.data.img_size, self.cfg.data.img_size)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )

        full_dataset = VideoFramesDataset(
            frames_dir=self.cfg.data.frames_root,
            transform=transform,
        )

        test_size = int(len(full_dataset) * self.cfg.data.test_split)
        train_size = len(full_dataset) - test_size

        train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.compute.num_workers,
            pin_memory=self.cfg.compute.pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.compute.num_workers,
            pin_memory=self.cfg.compute.pin_memory,
        )

        return train_loader, test_loader

    def _build_training_state(self) -> VAETrainingState:
        return VAETrainingState()
