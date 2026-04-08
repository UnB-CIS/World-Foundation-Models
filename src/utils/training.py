from typing import Callable, Any
from src.utils.training_state import ModelTrainingState

import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from torch.utils.data import DataLoader

from src.action_encoder.encoding import encoding_function


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    is_best=False,
):
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, filepath)

    if not is_best:
        print(f"Checkpoint saved: {filepath}")


def test_single_action(
    encoder: nn.Module,
    decoder: nn.Module,
    type_encoding: dict,
    object_encoding: dict,
    screen_width: int = 800,
    screen_height: int = 600,
    input_vector_dim: int = 4,
):
    """
    Test reconstruction of indi ual actions.

    Args:
        encoder: TextEncoder model
        decoder: TextDecoder model
    """
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Test 1 - No Action
        v_none_original = encoding_function(
            type_encoding=type_encoding,
            object_encoding=object_encoding,
            screen_width=screen_width,
            screen_height=screen_height,
            input_vector_dim=input_vector_dim,
        )
        z_none = encoder(v_none_original.unsqueeze(0))
        reconstruction_none = decoder(z_none).squeeze(0)

        print("Test 1: No Action Reconstruction")
        print(f"Original:     {v_none_original.tolist()}")
        print(f"Reconstructed: {[round(x, 4) for x in reconstruction_none.tolist()]}")
        print()

        # Test 2 - Mouse Down Action
        pos_x, pos_y = 640, 120
        test_action = {"type": "mouse_down", "object": "ball", "pos": [pos_x, pos_y]}

        v_down_original = encoding_function(
            type_encoding=type_encoding,
            object_encoding=object_encoding,
            action_data=test_action,
            screen_width=screen_width,
            screen_height=screen_height,
            input_vector_dim=input_vector_dim,
        )
        z_down = encoder(v_down_original.unsqueeze(0))
        reconstruction_down = decoder(z_down).squeeze(0)

        x_expected = pos_x / screen_width
        y_expected = pos_y / screen_height

        print("Test 2: Mouse Down Action Reconstruction")
        print(f"Original:     {v_down_original.tolist()}")
        print(f"Reconstructed: {[round(x, 4) for x in reconstruction_down.tolist()]}")
        print()

        print("Position components (normalized):")
        print(f"Original:     x={x_expected:.4f}, y={y_expected:.4f}")
        print(
            f"Reconstructed: x={reconstruction_down[2]:.4f}, y={reconstruction_down[3]:.4f}"
        )

        max_error = torch.abs(reconstruction_down - v_down_original).max().item()
        print(f"\nMax absolute error: {max_error:.6f}")


def train_step(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
    max_norm: float,
    epoch: int,
    num_epochs: int,
    num_batches: int,
    model_training_state: ModelTrainingState,
    device: torch.device | str,
):
    loss = {}

    for batch_idx, inputs in enumerate(train_loader):
        if isinstance(inputs, (list, tuple)):
            inputs = [i.to(device) for i in inputs]
        else:
            inputs = inputs.to(device)

        # Forward
        output = model(inputs)

        loss = loss_fn(
            model_input=inputs,
            model_output=output,
            model_training_state=model_training_state,
            epoch=epoch,
            batch_idx=batch_idx,
            num_batches=num_batches,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        model_training_state.training_step_output(
            batch_idx=batch_idx,
            epoch=epoch,
            num_epochs=num_epochs,
            num_batches=num_batches,
        )
    return loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: Callable,
    validate_fn: Callable,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    max_norm: float,
    model_training_state: ModelTrainingState,
    num_epochs: int = 50,
    device: torch.device | str = "cpu",
    save_dir: str = "./checkpoints/model_checkpoints",
    samples_dir: str = "./checkpoints/samples",
) -> dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    num_batches = len(train_loader)

    print("\n" + "=" * 70)
    print(f"Training Starting - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print("=" * 70 + "\n")

    for epoch in range(num_epochs):
        model.train()
        _ = train_step(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            max_norm=max_norm,
            epoch=epoch,
            num_epochs=num_epochs,
            num_batches=num_batches,
            model_training_state=model_training_state,
            device=device,
        )

        # Validation
        val_metrics = validate_fn(
            model=model,
            test_loader=test_loader,
            device=device,
            model_training_state=model_training_state,
        )
        val_total_loss = val_metrics["val_loss"]

        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        model_training_state.update_history_dict(
            val_metrics=val_metrics,
            current_lr=current_lr,
            num_batches=num_batches,
        )

        model_training_state.training_epoch_output(
            epoch=epoch,
            num_epochs=num_epochs,
        )

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_total_loss,
                filepath=os.path.join(save_dir, 'best_model.pth'),
                is_best=True,
            )
            print(f"  Best model saved! (Val Loss: {val_total_loss:.4f})\n")

        if (epoch + 1) % 5 == 0:
            model_training_state.training_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                save_dir=save_dir,
                test_loader=test_loader,
                device=device,
                num_epochs=num_epochs,
                samples_dir=samples_dir,
            )

        torch.cuda.empty_cache()

    model_training_state.training_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=None,
        save_dir=save_dir,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
    )

    print("\n  Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 70 + "\n")

    return model_training_state.history
