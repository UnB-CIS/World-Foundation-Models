from typing import Callable

import os
import torch
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf

from src.utils.training import train
from src.utils.training_state import ModelTrainingState


class BaseTrainer(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.train_loader, self.test_loader = self._build_dataloaders()
        self.training_state = self._build_training_state()

    @property
    @abstractmethod
    def loss_fn(self) -> Callable: ...

    @property
    @abstractmethod
    def validate_fn(self) -> Callable: ...

    @abstractmethod
    def _build_model(self) -> torch.nn.Module: ...

    @abstractmethod
    def _build_optimizer(self) -> torch.optim.Optimizer: ...

    @abstractmethod
    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler: ...

    @abstractmethod
    def _build_dataloaders(self) -> tuple: ...

    @abstractmethod
    def _build_training_state(self) -> ModelTrainingState: ...

    def fit(self) -> dict:
        os.makedirs(self.cfg.paths.save_dir, exist_ok=True)
        self._save_config()

        return train(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            loss_fn=self.loss_fn,
            validate_fn=self.validate_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            max_norm=self.cfg.training.max_norm,
            model_training_state=self.training_state,
            num_epochs=self.cfg.training.num_epochs,
            device=self.device,
            save_dir=self.cfg.paths.save_dir,
        )

    def _save_config(self) -> None:
        path = os.path.join(self.cfg.paths.save_dir, "config.yaml")
        OmegaConf.save(self.cfg, path)
