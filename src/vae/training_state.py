import os
import torch
import torch.nn as nn

from torch.optim import Optimizer

from src.utils.training import save_checkpoint
from src.utils.visualization import save_validation_vae_samples
from src.utils.training_state import ModelTrainingState


class VAETrainingState(ModelTrainingState):
    def __init__(self) -> None:

        self.history = {
            "train_loss": [],
            "train_recon": [],
            "train_kl": [],
            "val_loss": [],
            "val_recon": [],
            "val_kl": [],
            "learning_rates": [],
            "betas": [],
        }
        self.set_training_metrics()

    def set_training_metrics(self) -> None:
        self.loss = 0.0
        self.recon = 0.0
        self.kl = 0.0
        self.beta = 0.0

        self.sum_loss = 0.0
        self.sum_recon = 0.0
        self.sum_kl = 0.0
        self.sum_beta = 0.0

    def update_loss_metrics(self, loss_dict: dict[str, torch.Tensor]) -> None:
        self.loss = loss_dict["loss"].item()
        self.recon = loss_dict["recon_loss"].item()
        self.kl = loss_dict["kl_loss"].item()
        self.beta = loss_dict["beta"]

        self.sum_loss += loss_dict["loss"].item()
        self.sum_recon += loss_dict["recon_loss"].item()
        self.sum_kl += loss_dict["kl_loss"].item()
        self.sum_beta += loss_dict["beta"]

    def update_history_dict(
        self, val_metrics: dict[str, torch.Tensor], current_lr: float, num_batches: int
    ) -> None:
        val_total_loss, val_recon, val_kl = val_metrics.values()

        self.history['train_loss'].append(self.sum_loss / num_batches)
        self.history['train_recon'].append(self.sum_recon / num_batches)
        self.history['train_kl'].append(self.sum_kl / num_batches)
        self.history['betas'].append(self.sum_beta / num_batches)

        self.history['val_loss'].append(val_total_loss)
        self.history['val_recon'].append(val_recon)
        self.history['val_kl'].append(val_kl)
        self.history['learning_rates'].append(current_lr)

        self.set_training_metrics()

    def training_step_output(
        self, batch_idx: int, epoch: int, num_epochs: int, num_batches: int
    ) -> None:
        if (batch_idx + 1) % 20 == 0:
            print(
                f"Epoch[{epoch+1:02d}/{num_epochs}] "
                f"Batch[{batch_idx+1:03d}/{num_batches}] | "
                f"Loss: {self.loss:.3f} R:{self.recon:.3f} "
                f"KL Div:{self.kl:.3f} β:{self.beta:.3f}"
            )

            if self.beta > 0.2 and self.kl < 1.0:
                print(f"     KL={self.kl:.2f} still low at β={self.beta:.2f}")

    def training_epoch_output(self, epoch: int, num_epochs: int) -> None:
        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"{'-'*70}")
        print(
            f"  Train: L={self.history['train_loss'][-1]:.4f} R={self.history['train_recon'][-1]:.4f} KL={self.history['train_kl'][-1]:.4f}"
        )
        print(
            f"  Val:   L={self.history['val_loss'][-1]:.4f} R={self.history['val_recon'][-1]:.4f} KL={self.history['val_kl'][-1]:.4f}"
        )
        print(
            f"  Beta={self.history['betas'][-1]:.3f} LR={self.history['learning_rates'][-1]:.2e}"
        )

        if self.history['betas'][-1] >= 0.8:
            if self.history['train_kl'][-1] < 1.0:
                print(f"    CRITICAL: KL={self.history['train_kl'][-1]:.2f} < 1.0")
            elif self.history['train_kl'][-1] < 2.0:
                print(f"     WARNING: KL={self.history['train_kl'][-1]:.2f} < 2.0")
            else:
                print(f"    KL healthy: {self.history['train_kl'][-1]:.2f}")
        else:
            print(f"  ℹ  Warmup: β={self.history['betas'][-1]:.2f}")

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
        test_loader = kwargs["test_loader"]
        device = kwargs["device"]

        checkpoint_name = "final_model.pth"
        epoch_name = num_epochs - 1
        if epoch is not None:
            checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth"
            save_validation_vae_samples(
                model, test_loader, epoch, device, kwargs["samples_dir"]
            )
            epoch_name = epoch
        save_checkpoint(
            model,
            optimizer,
            epoch_name,
            self.history["train_loss"][-1],
            os.path.join(save_dir, checkpoint_name),
        )
