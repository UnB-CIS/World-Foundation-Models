from typing import Callable

import random
from omegaconf import DictConfig

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import TensorDataset, DataLoader, random_split


from src.action_encoder.model import ActionTextModel
from src.action_encoder.encoding import encoding_function
from src.action_encoder.training_state import ActionEncoderTrainingState
from src.action_encoder.utils import validate_action_encoder, action_encoder_loss_fn

from training.base_trainer import BaseTrainer


class ActionEncoderTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    @property
    def loss_fn(self) -> Callable:
        return action_encoder_loss_fn

    @property
    def validate_fn(self) -> Callable:
        return validate_action_encoder

    def _build_model(self) -> ActionTextModel:
        return ActionTextModel(
            input_dim=self.cfg.model.input_dim,
            latent_dim=self.cfg.model.latent_dim,
            output_dim=self.cfg.model.output_dim,
        )

    def _build_optimizer(self) -> Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.lr,
        )

    def _build_scheduler(self) -> LRScheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

    def _build_dataloaders(self) -> tuple:
        samples = []
        n_positive = int(self.cfg.data.num_samples * self.cfg.data.positive_ratio)
        n_negative = self.cfg.data.num_samples - n_positive

        type_encoding = dict(self.cfg.scenario.type_encoding)
        object_encoding = dict(self.cfg.scenario.object_encoding)
        screen_w = self.cfg.scenario.screen_width
        screen_h = self.cfg.scenario.screen_height

        for _ in range(n_positive):
            action = {
                "type": "mouse_down",
                "object": "ball",
                "pos": [
                    random.randint(50, screen_w - 50),
                    random.randint(50, screen_h - 50),
                ],
            }
            samples.append(
                encoding_function(
                    type_encoding=type_encoding,
                    object_encoding=object_encoding,
                    screen_width=screen_w,
                    screen_height=screen_h,
                    input_vector_dim=self.cfg.model.input_dim,
                    action_data=action,
                )
            )

        for _ in range(n_negative):
            samples.append(
                encoding_function(
                    type_encoding=type_encoding,
                    object_encoding=object_encoding,
                    screen_width=screen_w,
                    screen_height=screen_h,
                    input_vector_dim=self.cfg.model.input_dim,
                )
            )

        full_tensor = torch.stack(samples)

        dataset = TensorDataset(full_tensor, full_tensor)

        test_size = int(len(dataset) * self.cfg.data.test_split)
        train_size = len(dataset) - test_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size])

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

    def _build_training_state(self) -> ActionEncoderTrainingState:
        return ActionEncoderTrainingState()
