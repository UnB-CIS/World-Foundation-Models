from typing import Any

import torch
import torch.nn as nn

from torch.optim import Optimizer

from abc import ABC, abstractmethod


class ModelTrainingState(ABC):
    history: dict[str, Any]

    @abstractmethod
    def set_training_metrics(self) -> None: ...

    @abstractmethod
    def update_loss_metrics(self, loss_dict: dict[str, torch.Tensor]) -> None: ...

    @abstractmethod
    def update_history_dict(
        self, val_metrics: dict[str, torch.Tensor], current_lr: float, num_batches: int
    ) -> None: ...

    @abstractmethod
    def training_step_output(
        self, batch_idx: int, epoch: int, num_epochs: int, num_batches: int
    ) -> None: ...

    @abstractmethod
    def training_epoch_output(self, epoch: int, num_epochs: int) -> None: ...

    @abstractmethod
    def training_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int | None,
        num_epochs: int,
        save_dir: str,
        **kwargs,
    ) -> None: ...
