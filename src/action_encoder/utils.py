import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import ActionTextModel
from .training_state import ActionEncoderTrainingState


def validate_action_encoder(
    model: ActionTextModel,
    test_loader: DataLoader,
    device: torch.device | str,
    **kwargs,
) -> dict[str, float]:
    val_loss_total = 0

    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            reconstruction, _ = model(data)
            val_loss = nn.MSELoss()(reconstruction, data)
            val_loss_total += val_loss.item()

    model.train()
    return {
        "val_loss": val_loss_total / len(test_loader),
    }


def action_encoder_loss_fn(
    model_input: torch.Tensor,
    model_output: tuple[torch.Tensor, torch.Tensor],
    model_training_state: ActionEncoderTrainingState,
    **kwargs,
) -> torch.Tensor:

    data, _ = model_input

    reconstructed, _ = model_output

    loss = nn.MSELoss()(reconstructed, data)

    loss_dict = {"loss": loss}

    model_training_state.update_loss_metrics(loss_dict)

    return loss
