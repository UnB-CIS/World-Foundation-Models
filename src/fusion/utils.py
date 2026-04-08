import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import ActionProbe
from .decoder import LatentDecoder
from .training_state import LatentTrainingState


def validate_ActionProbe(
    model: ActionProbe,
    test_loader: DataLoader,
    device: torch.device | str,
    **kwargs,
) -> dict[str, float]:
    val_loss_total = 0

    fuser = kwargs["spatial_broadcast_fuser"]

    model.eval()
    with torch.no_grad():
        for z_value, z_action in test_loader:
            z_value = z_value.to(device)
            z_action = z_action.to(device)

            z_fused = fuser(z_value, z_action).detach()

            reconstruction = model(z_fused)
            val_loss = nn.MSELoss()(reconstruction, z_action)
            val_loss_total += val_loss.item()

    model.train()
    return {
        "val_loss": val_loss_total / len(test_loader),
    }


def validate_latent_decoder(
    model: LatentDecoder, test_loader: DataLoader, device: str, **kwargs
):
    model.eval()

    val_loss_total = 0

    with torch.no_grad():
        for latent, encoding in test_loader:
            latent = latent.to(device)
            encoding = encoding.to(device)

            pred_encoding = model((latent, encoding))

            loss = nn.MSELoss()(pred_encoding, encoding)

            val_loss_total += loss.item()

    model.train()

    return {
        "val_loss": val_loss_total / len(test_loader),
    }


def ActionProbe_loss_fn(
    model_input: list[torch.Tensor],
    model_output: tuple[torch.Tensor, torch.Tensor],
    model_training_state: LatentTrainingState,
    **kwargs,
) -> torch.Tensor:
    _, z_action = model_input

    loss = nn.MSELoss()(model_output, z_action)

    loss_dict = {"loss": loss}

    model_training_state.update_loss_metrics(loss_dict)

    return loss


def latent_decoder_loss_fn(
    model_input: list[torch.Tensor],
    model_output: tuple[torch.Tensor, torch.Tensor],
    model_training_state: LatentTrainingState,
    **kwargs,
) -> torch.Tensor:
    _, encoding = model_input

    loss = nn.MSELoss()(model_output, encoding)

    loss_dict = {"loss": loss}

    model_training_state.update_loss_metrics(loss_dict)

    return loss
