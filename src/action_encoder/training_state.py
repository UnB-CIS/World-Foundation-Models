import os
import torch
import torch.nn as nn

from torch.optim import Optimizer

from src.utils.training import save_checkpoint
from src.utils.training_state import ModelTrainingState


class ActionEncoderTrainingState(ModelTrainingState):
    def __init__(self) -> None:
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
        }
        self.set_training_metrics()

    def set_training_metrics(self) -> None:
        self.loss = 0.0
        self.sum_loss = 0.0

    def update_loss_metrics(self, loss_dict: dict[str, torch.Tensor]) -> None:
        self.loss = loss_dict["loss"].item()
        self.sum_loss += loss_dict["loss"].item()

    def update_history_dict(
        self, val_metrics: dict[str, torch.Tensor], current_lr: float, num_batches: int
    ) -> None:
        val_total_loss = val_metrics["val_loss"]

        self.history['train_loss'].append(self.sum_loss / num_batches)

        self.history['val_loss'].append(val_total_loss)

        self.history['learning_rates'].append(current_lr)

        self.set_training_metrics()

    def training_step_output(
        self, batch_idx: int, epoch: int, num_epochs: int, num_batches: int
    ) -> None:
        if (batch_idx + 1) % 20 == 0:
            print(
                f"Epoch[{epoch+1:02d}/{num_epochs}] "
                f"Batch[{batch_idx+1:03d}/{num_batches}] | "
                f"Loss: {self.loss:.3f} "
            )

    def training_epoch_output(self, epoch: int, num_epochs: int) -> None:
        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"{'-'*70}")
        print(f"  Train: L={self.history['train_loss'][-1]:.4f}")
        print(f"  Val:   L={self.history['val_loss'][-1]:.4f}")
        print(f"{'='*70}\n")

    def training_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int | None,
        num_epochs: int,
        save_dir: str,
        **kwargs,
    ) -> None:
        checkpoint_name = "final_model.pth"
        epoch_name = num_epochs

        if epoch is not None:
            checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(
            model,
            optimizer,
            epoch_name,
            self.history["train_loss"][-1],
            os.path.join(save_dir, checkpoint_name),
        )
