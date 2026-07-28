import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "scenario_1", "processed")
VAE_WEIGHTS = os.path.join(PROJECT_ROOT, "checkpoints", "vae", "best_model.pth")
TEXT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "checkpoints", "action_encoder", "best_model.pth"
)
WORLD_MODEL_WEIGHTS = os.path.join(CURRENT_DIR, "world_model_weights.pth")
TRAINING_SAMPLES_DIR = os.path.join(CURRENT_DIR, "world_model_samples")

sys.path.insert(0, PROJECT_ROOT)

from src.vae.model import VAE
from src.fusion.model import SpatialBroadcastFuser
from src.action_encoder.model import ActionTextModel
from src.action_encoder.encoding import encoding_function

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MODEL_FRAME_SIZE = 64

VISUAL_LATENT_CHANNELS = 16
ACTION_LATENT_CHANNELS = 16
FUSED_LATENT_CHANNELS = VISUAL_LATENT_CHANNELS + ACTION_LATENT_CHANNELS  # 32

DEFAULT_MEMORY_FRAMES = 10
DEFAULT_MEMORY_STRIDE = 5
DEFAULT_FOREGROUND_PIXEL_WEIGHT = 50.0
DEFAULT_WHITE_PIXEL_THRESHOLD = 0.98
DEFAULT_ROLLOUT_STEPS = 4

# Click loss multiplier — frames with a non-zero action in the fused latent
# are weighted this many times higher in the latent loss, forcing the model
# to learn click response rather than ignoring rare click events.
CLICK_LOSS_WEIGHT = 5.0

TYPE_ENCODING = {"mouse_down": [1.0], "none": [0.0]}
OBJECT_ENCODING = {"ball": [1.0], "none": [0.0]}


def preprocess_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[..., 0]
    if frame.max() > 1.0:
        frame = frame / 255.0
    return torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)


def frame_from_tensor(frame_tensor: torch.Tensor) -> np.ndarray:
    frame = frame_tensor.detach().cpu().squeeze().numpy()
    frame = np.clip(frame, 0.0, 1.0)
    # f_min = frame.min()
    # f_max = frame.max()
    # if f_max - f_min > 1e-6:
    #     frame = (frame - f_min) / (f_max - f_min)
    return (frame * 255.0).astype(np.uint8)


def select_spaced_history(
    history: List[torch.Tensor], memory_frames: int, memory_stride: int
) -> torch.Tensor:
    if not history:
        raise ValueError("Historico vazio para memoria temporal.")
    if memory_frames <= 0:
        raise ValueError("memory_frames deve ser maior que zero.")
    if memory_stride <= 0:
        raise ValueError("memory_stride deve ser maior que zero.")

    last_index = len(history) - 1
    selected = []
    for offset in range(memory_frames - 1, -1, -1):
        source_index = max(0, last_index - offset * memory_stride)
        selected.append(history[source_index])
    return torch.stack(selected, dim=1)


def extract_action_latent(fused: torch.Tensor) -> torch.Tensor:
    return fused[:, VISUAL_LATENT_CHANNELS:, ...].mean(dim=(-2, -1))


def is_click_frame(fused: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Returns a boolean mask (B,) indicating which samples had a click action.

    A click frame has non-zero action channels in the fused latent. Because
    the action encoder bias has been zeroed, null actions produce exactly zero
    action channels, so any value above threshold means a real click happened.
    """
    action_magnitude = fused[:, VISUAL_LATENT_CHANNELS:, ...].abs().mean(dim=(1, 2, 3))
    return action_magnitude > threshold


@dataclass
class PredictionResult:
    frame: np.ndarray
    model_status: str


# -----------------------------------------------------------------------------
# ConvLSTM Cell
# -----------------------------------------------------------------------------


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, _, h, w = x.shape
        if state is None:
            hx = torch.zeros(
                b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype
            )
            cx = torch.zeros(
                b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype
            )
        else:
            hx, cx = state

        combined = torch.cat([x, hx], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, (hx, cx)


class LatentTransitionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


# -----------------------------------------------------------------------------
# ConvLSTM Transition Model — predicts DELTA latent, not absolute latent.
#
# Why delta prediction fixes action conditioning:
# For null frames the delta is small (just physics drift).
# For click frames the delta contains a strong localised spike where the new
# ball appeared. Training on deltas concentrates the loss on what changed,
# making click events the dominant signal rather than a minor perturbation
# in an otherwise stable background.
# -----------------------------------------------------------------------------


class LatentTransitionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = FUSED_LATENT_CHANNELS,
        hidden_channels: int = 64,
        out_channels: int = VISUAL_LATENT_CHANNELS,
        memory_frames: int = DEFAULT_MEMORY_FRAMES,
    ) -> None:
        super().__init__()
        self.memory_frames = memory_frames

        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.convlstm1 = ConvLSTMCell(hidden_channels, hidden_channels)
        self.convlstm2 = ConvLSTMCell(hidden_channels, hidden_channels)

        # Output projects to DELTA — not absolute next latent
        self.output_proj = nn.Sequential(
            LatentTransitionBlock(hidden_channels),
            LatentTransitionBlock(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, fused_history: torch.Tensor) -> torch.Tensor:
        """Returns predicted delta: (mu_next - mu_current)."""
        if fused_history.dim() == 4:
            fused_history = fused_history.unsqueeze(1)
        if fused_history.dim() != 5:
            raise ValueError(
                f"fused_history deve ser 5D (B,T,C,H,W), recebido {tuple(fused_history.shape)}"
            )

        batch_size, time_steps, channels, height, width = fused_history.shape

        if channels != FUSED_LATENT_CHANNELS:
            raise ValueError(
                f"Esperado {FUSED_LATENT_CHANNELS} canais fundidos, recebido {channels}."
            )

        if time_steps < self.memory_frames:
            pad_count = self.memory_frames - time_steps
            pad = fused_history[:, :1].expand(
                batch_size, pad_count, channels, height, width
            )
            fused_history = torch.cat([pad, fused_history], dim=1)
        elif time_steps > self.memory_frames:
            fused_history = fused_history[:, -self.memory_frames :]

        h2 = None
        state1 = state2 = None
        for t in range(fused_history.shape[1]):
            x = self.input_proj(fused_history[:, t])
            h1, state1 = self.convlstm1(x, state1)
            h2, state2 = self.convlstm2(h1, state2)

        return self.output_proj(h2)


# -----------------------------------------------------------------------------
# Model bundle
# -----------------------------------------------------------------------------


class ModelBundle:
    def __init__(
        self,
        device: Optional[str] = None,
        world_model_weights: Optional[str] = WORLD_MODEL_WEIGHTS,
        memory_frames: int = DEFAULT_MEMORY_FRAMES,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA foi solicitada, mas nao esta disponivel. Usando CPU.")
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)

        self.vae = VAE(
            latent_channels=VISUAL_LATENT_CHANNELS,
            action_latent_channels=ACTION_LATENT_CHANNELS,
        ).to(self.device)

        self.full_action_model = ActionTextModel(
            input_dim=4,
            latent_dim=ACTION_LATENT_CHANNELS,
            output_dim=4,
        ).to(self.device)
        self.text_encoder = self.full_action_model.encoder

        self.fuser = SpatialBroadcastFuser(height=8, width=8).to(self.device)
        self.transition_model = LatentTransitionModel(memory_frames=memory_frames).to(
            self.device
        )
        self.memory_frames = memory_frames
        self.world_model_weights = world_model_weights
        self.transition_weights_loaded = False
        self._load_weights()
        self.vae.eval()
        self.full_action_model.eval()
        self.fuser.eval()
        self.transition_model.eval()

    def _load_weights(self) -> None:
        if not os.path.exists(VAE_WEIGHTS):
            raise FileNotFoundError(f"Pesos do VAE nao encontrados em {VAE_WEIGHTS}")
        try:
            vae_raw = torch.load(
                VAE_WEIGHTS, map_location=self.device, weights_only=True
            )
            self.vae.load_state_dict(vae_raw.get("model_state_dict", vae_raw))
            print(f"VAE carregado OK ({VAE_WEIGHTS})")
        except RuntimeError as exc:
            raise RuntimeError(
                f"Falha ao carregar pesos do VAE de {VAE_WEIGHTS}."
            ) from exc

        if not os.path.exists(TEXT_WEIGHTS):
            raise FileNotFoundError(
                f"Pesos do Action Encoder nao encontrados em {TEXT_WEIGHTS}"
            )
        try:
            text_raw = torch.load(
                TEXT_WEIGHTS, map_location=self.device, weights_only=True
            )
            self.full_action_model.load_state_dict(
                text_raw.get("model_state_dict", text_raw)
            )
            print(f"Action Encoder carregado OK ({TEXT_WEIGHTS})")
        except RuntimeError as exc:
            raise RuntimeError(
                f"Falha ao carregar pesos do Action Encoder de {TEXT_WEIGHTS}."
            ) from exc

        if self.world_model_weights and os.path.exists(self.world_model_weights):
            try:
                wm_raw = torch.load(
                    self.world_model_weights,
                    map_location=self.device,
                    weights_only=True,
                )
                self.transition_model.load_state_dict(
                    wm_raw.get("model_state_dict", wm_raw)
                )
                self.transition_weights_loaded = True
                print(f"World model carregado OK ({self.world_model_weights})")
            except RuntimeError as exc:
                raise RuntimeError(
                    "Checkpoint do world model incompatível. Retreine com este arquivo."
                ) from exc
        else:
            print(
                "AVISO: Nenhum checkpoint do world model encontrado. "
                "Usando pesos aleatórios (predições serão ruins)."
            )

    @torch.no_grad()
    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Returns mu (VISUAL_LATENT_CHANNELS channels)."""
        frame_tensor = preprocess_frame(frame, self.device)
        encoded = self.vae.encoder(frame_tensor)
        mu, _ = torch.chunk(encoded, 2, dim=1)
        return mu

    @torch.no_grad()
    def fuse(self, frame: np.ndarray, action: Optional[dict]) -> torch.Tensor:
        mu = self.encode_frame(frame)

        action_vector = (
            encoding_function(
                type_encoding=TYPE_ENCODING,
                object_encoding=OBJECT_ENCODING,
                screen_width=SCREEN_WIDTH,
                screen_height=SCREEN_HEIGHT,
                input_vector_dim=4,
                action_data=action,
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # Null baseline subtraction — zero input → zero embedding
        null_vector = torch.zeros_like(action_vector)
        null_embedding = self.text_encoder(null_vector)
        action_latent = self.text_encoder(action_vector) - null_embedding

        return self.fuser(mu, action_latent)

    @torch.no_grad()
    def decode_latent(
        self, latent: torch.Tensor, action_latent: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        latent = latent.to(self.device)
        if action_latent is not None:
            action_latent = action_latent.to(self.device)
        decoded = self.vae.decoder(latent, action_latent)
        return frame_from_tensor(decoded)


# -----------------------------------------------------------------------------
# World model — inference
# -----------------------------------------------------------------------------


class WorldModel:
    def __init__(
        self,
        world_model_weights: Optional[str] = WORLD_MODEL_WEIGHTS,
        device: Optional[str] = None,
        memory_frames: int = DEFAULT_MEMORY_FRAMES,
        memory_stride: int = DEFAULT_MEMORY_STRIDE,
    ) -> None:
        if memory_stride <= 0:
            raise ValueError("memory_stride deve ser maior que zero.")
        self.models = ModelBundle(
            device=device,
            world_model_weights=world_model_weights,
            memory_frames=memory_frames,
        )
        self.memory_frames = memory_frames
        self.memory_stride = memory_stride

    def predict(
        self,
        frame: np.ndarray,
        action: Optional[dict],
        fused_history: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[PredictionResult, torch.Tensor]:
        # 1. Encode current frame + action
        fused = self.models.fuse(frame, action)

        # 2. Build temporal history
        history = [item.to(self.models.device) for item in (fused_history or [])]
        history.append(fused.detach())
        temporal_input = select_spaced_history(
            history, self.memory_frames, self.memory_stride
        )

        # 3. Get current visual latent (needed for delta prediction)
        current_mu = self.models.encode_frame(frame)

        # 4. Predict DELTA, then add to current latent to get next latent
        # This is the key inference change: next = current + delta
        # For null frames delta ≈ small physics drift
        # For click frames delta contains a strong spike where the ball appeared
        with torch.no_grad():
            delta = self.models.transition_model(temporal_input)
        next_latent = current_mu + delta

        # 5. Decode — pass None so decoder receives zeros, matching training
        next_frame = self.models.decode_latent(next_latent, action_latent=None)

        model_status = (
            f"pesos carregados de {os.path.basename(self.models.world_model_weights)}"
            if self.models.transition_weights_loaded and self.models.world_model_weights
            else "sem checkpoint do world model, usando pesos aleatorios"
        )
        return (
            PredictionResult(frame=next_frame, model_status=model_status),
            fused.detach(),
        )


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class FinalModelDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = DATASET_DIR,
        max_samples: Optional[int] = None,
        memory_frames: int = DEFAULT_MEMORY_FRAMES,
        memory_stride: int = DEFAULT_MEMORY_STRIDE,
        rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
    ) -> None:
        if memory_stride <= 0:
            raise ValueError("memory_stride deve ser maior que zero.")
        if rollout_steps <= 0:
            raise ValueError("rollout_steps deve ser maior que zero.")
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        files = sorted(
            os.path.join(dataset_dir, name)
            for name in os.listdir(dataset_dir)
            if name.endswith(".pt")
        )
        if not files:
            raise FileNotFoundError(
                f"Nenhum dataset processado encontrado em {dataset_dir}"
            )

        for path in files:
            sample = torch.load(path, map_location="cpu", weights_only=True)
            x = sample["x"].to(torch.float32)  # (N, FUSED_LATENT_CHANNELS, 8, 8)
            y = sample["y"].to(torch.float32)  # (N, VISUAL_LATENT_CHANNELS, 8, 8)

            if x.shape[1] != FUSED_LATENT_CHANNELS:
                raise ValueError(
                    f"Dataset em {path} tem x com {x.shape[1]} canais, "
                    f"esperado {FUSED_LATENT_CHANNELS}. Reconstrua o dataset."
                )
            if y.shape[1] != VISUAL_LATENT_CHANNELS:
                raise ValueError(
                    f"Dataset em {path} tem y com {y.shape[1]} canais, "
                    f"esperado {VISUAL_LATENT_CHANNELS}. Reconstrua o dataset."
                )

            for frame_idx in range(x.shape[0] - rollout_steps + 1):
                history_indices = [
                    max(0, frame_idx - offset * memory_stride)
                    for offset in range(memory_frames - 1, -1, -1)
                ]
                history = x[history_indices]
                target_latents = y[frame_idx : frame_idx + rollout_steps]
                action_start = frame_idx + 1
                action_end = frame_idx + rollout_steps
                future_action_latents = x[
                    action_start:action_end, VISUAL_LATENT_CHANNELS:
                ]
                self.samples.append((history, target_latents, future_action_latents))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
            if max_samples is not None and len(self.samples) >= max_samples:
                self.samples = self.samples[:max_samples]
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history, target_latents, future_action_latents = self.samples[index]
        return history.clone(), target_latents.clone(), future_action_latents.clone()


# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------


def weighted_frame_l1_loss(
    predicted_frame: torch.Tensor,
    target_frame: torch.Tensor,
    foreground_pixel_weight: float,
    white_pixel_threshold: float,
) -> torch.Tensor:
    non_white_mask = target_frame < white_pixel_threshold
    weights = torch.ones_like(target_frame)
    weights = torch.where(
        non_white_mask,
        torch.full_like(target_frame, foreground_pixel_weight),
        weights,
    )
    absolute_error = torch.abs(predicted_frame - target_frame)
    return (absolute_error * weights).sum() / weights.sum().clamp_min(1.0)


def rollout_batch(
    models: ModelBundle,
    fused_history: torch.Tensor,
    target_latents: torch.Tensor,
    future_action_latents: torch.Tensor,
    memory_frames: int,
    memory_stride: int,
    latent_loss_fn: nn.Module,
    frame_loss_weight: float,
    foreground_pixel_weight: float,
    white_pixel_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history_items = [fused_history[:, step] for step in range(fused_history.shape[1])]
    rollout_steps = target_latents.shape[1]
    total_latent_loss = torch.zeros((), device=fused_history.device)
    total_frame_loss = torch.zeros((), device=fused_history.device)
    predicted_latents = []

    for step in range(rollout_steps):
        temporal_input = select_spaced_history(
            history_items, memory_frames, memory_stride
        )

        # Current visual latent is the visual channels of the most recent fused frame
        current_visual = history_items[-1][:, :VISUAL_LATENT_CHANNELS]

        # Predict DELTA, then reconstruct absolute next latent
        delta = models.transition_model(temporal_input)
        predicted_latent = current_visual + delta

        # Target DELTA — what the model should have predicted
        target_latent = target_latents[:, step]
        target_delta = target_latent - current_visual

        # Detect click frames in the current history step
        current_fused = history_items[-1]
        click_mask = is_click_frame(current_fused).float()  # (B,)
        # Expand to (B, C, H, W) for broadcasting
        click_weight = (1.0 + (CLICK_LOSS_WEIGHT - 1.0) * click_mask).view(-1, 1, 1, 1)

        # Weighted latent delta loss — click frames penalised CLICK_LOSS_WEIGHT× more
        raw_latent_loss = latent_loss_fn(delta, target_delta)
        # Per-sample loss for weighting
        per_sample_delta_loss = (delta - target_delta).abs().mean(dim=(1, 2, 3))
        weighted_latent_loss = (per_sample_delta_loss * click_weight.squeeze()).mean()

        # Frame loss for visual sharpness
        current_action_latent = extract_action_latent(current_fused)
        predicted_frame = models.vae.decoder(predicted_latent, current_action_latent)
        target_frame = models.vae.decoder(target_latent, current_action_latent)
        frame_loss = weighted_frame_l1_loss(
            predicted_frame,
            target_frame,
            foreground_pixel_weight,
            white_pixel_threshold,
        )

        total_latent_loss = total_latent_loss + weighted_latent_loss
        total_frame_loss = total_frame_loss + frame_loss
        predicted_latents.append(predicted_latent)

        if step < rollout_steps - 1:
            predicted_encoded = models.vae.encoder(predicted_frame)
            predicted_mu, _ = torch.chunk(predicted_encoded, 2, dim=1)
            next_action_latent = future_action_latents[:, step]
            _, _, h, w = predicted_mu.shape
            next_action_broadcast = (
                torch.nn.functional.interpolate(
                    next_action_latent, size=(h, w), mode="nearest"
                )
                if next_action_latent.shape[-2:] != (h, w)
                else next_action_latent
            )
            next_fused = torch.cat([predicted_mu, next_action_broadcast], dim=1)
            history_items.append(next_fused)

    avg_latent_loss = total_latent_loss / rollout_steps
    avg_frame_loss = total_frame_loss / rollout_steps
    total_loss = avg_latent_loss + frame_loss_weight * avg_frame_loss
    return (
        total_loss,
        avg_latent_loss,
        avg_frame_loss,
        torch.stack(predicted_latents, dim=1),
    )


# -----------------------------------------------------------------------------
# Visual samples
# -----------------------------------------------------------------------------


def save_visual_samples(
    models: ModelBundle,
    val_loader: DataLoader,
    sample_dir: str,
    epoch: int,
    memory_frames: int,
    memory_stride: int,
    latent_loss_fn: nn.Module,
    frame_loss_weight: float,
    foreground_pixel_weight: float,
    white_pixel_threshold: float,
    device: torch.device,
) -> None:
    os.makedirs(sample_dir, exist_ok=True)
    try:
        fused_history, target_latents, future_action_latents = next(iter(val_loader))
    except StopIteration:
        return

    fused_history = fused_history[:1].to(device)
    target_latents = target_latents[:1].to(device)
    future_action_latents = future_action_latents[:1].to(device)

    models.transition_model.eval()
    with torch.no_grad():
        _, _, _, predicted_latents = rollout_batch(
            models,
            fused_history,
            target_latents,
            future_action_latents,
            memory_frames,
            memory_stride,
            latent_loss_fn,
            frame_loss_weight,
            foreground_pixel_weight,
            white_pixel_threshold,
        )
        last_fused = fused_history[:, -1]
        action_latent = extract_action_latent(last_fused)
        predicted_frames = torch.stack(
            [
                models.vae.decoder(
                    predicted_latents[0, step].unsqueeze(0), action_latent
                ).squeeze(0)
                for step in range(predicted_latents.shape[1])
            ]
        )
        target_frames = torch.stack(
            [
                models.vae.decoder(
                    target_latents[0, step].unsqueeze(0), action_latent
                ).squeeze(0)
                for step in range(target_latents.shape[1])
            ]
        )

    rows = []
    separator = np.full((MODEL_FRAME_SIZE, 4), 255, dtype=np.uint8)
    for step in range(target_frames.shape[0]):
        target = frame_from_tensor(target_frames[step])
        predicted = frame_from_tensor(predicted_frames[step])
        difference = np.abs(
            predicted.astype(np.int16) - target.astype(np.int16)
        ).astype(np.uint8)
        rows.append(
            np.concatenate(
                [target, separator, predicted, separator, difference], axis=1
            )
        )

    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), grid)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_world_model(
    dataset_dir: str = DATASET_DIR,
    output_path: str = WORLD_MODEL_WEIGHTS,
    batch_size: int = 128,
    num_epochs: int = 80,
    learning_rate: float = 3e-4,
    train_ratio: float = 0.9,
    frame_loss_weight: float = 1.0,
    foreground_pixel_weight: float = DEFAULT_FOREGROUND_PIXEL_WEIGHT,
    white_pixel_threshold: float = DEFAULT_WHITE_PIXEL_THRESHOLD,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    memory_frames: int = DEFAULT_MEMORY_FRAMES,
    memory_stride: int = DEFAULT_MEMORY_STRIDE,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
    sample_dir: str = TRAINING_SAMPLES_DIR,
    save_samples_every: int = 1,
    history_dir: Optional[str] = None,
) -> dict:
    import json
    from datetime import datetime

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada mas nao disponivel.")

    models = ModelBundle(
        device=str(resolved_device),
        world_model_weights=None,
        memory_frames=memory_frames,
    )
    models.transition_model.train()
    models.vae.eval()
    models.full_action_model.eval()
    models.fuser.eval()

    for module in (models.vae, models.full_action_model, models.fuser):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    dataset = FinalModelDataset(
        dataset_dir=dataset_dir,
        max_samples=max_samples,
        memory_frames=memory_frames,
        memory_stride=memory_stride,
        rollout_steps=rollout_steps,
    )
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    if val_size == 0:
        raise ValueError("Dataset pequeno demais para separar validacao.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(models.transition_model.parameters(), lr=learning_rate)

    # OneCycleLR: ramps up to learning_rate in first ~30% of training then
    # anneals down to near zero. This reliably converges faster than a flat lr
    # with ReduceLROnPlateau for models that were previously improving slowly.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,  # start lr = learning_rate / 10
        final_div_factor=100.0,  # end lr = start_lr / 100
    )

    latent_loss_fn = nn.L1Loss()
    best_val_loss = float("inf")

    # History dict — mirrors structure used by VAE trainer
    history: dict = {
        "train_loss": [],
        "train_latent_loss": [],
        "train_frame_loss": [],
        "val_loss": [],
        "val_latent_loss": [],
        "val_frame_loss": [],
        "learning_rates": [],
    }

    print(f"Treinando world model ConvLSTM + delta prediction em {resolved_device}")
    print(
        f"Amostras: total={len(dataset)} treino={len(train_dataset)} validacao={len(val_dataset)}"
    )
    print(f"Click loss weight: {CLICK_LOSS_WEIGHT}x")
    print(
        f"Loss: L1 delta latent (click-weighted) + weighted frame L1 (weight={frame_loss_weight})"
    )
    print(f"Scheduler: OneCycleLR max_lr={learning_rate:.2e} epochs={num_epochs}")

    for epoch in range(num_epochs):
        models.transition_model.train()
        train_loss_total = train_latent_total = train_frame_total = 0.0

        for fused_history, target_latents, future_action_latents in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            fused_history = fused_history.to(resolved_device)
            target_latents = target_latents.to(resolved_device)
            future_action_latents = future_action_latents.to(resolved_device)

            loss, latent_loss, frame_loss, _ = rollout_batch(
                models,
                fused_history,
                target_latents,
                future_action_latents,
                memory_frames,
                memory_stride,
                latent_loss_fn,
                frame_loss_weight,
                foreground_pixel_weight,
                white_pixel_threshold,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                models.transition_model.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()  # OneCycleLR steps per batch, not per epoch

            train_loss_total += loss.item()
            train_latent_total += latent_loss.item()
            train_frame_total += frame_loss.item()

        models.transition_model.eval()
        val_loss_total = val_latent_total = val_frame_total = 0.0

        with torch.no_grad():
            for fused_history, target_latents, future_action_latents in val_loader:
                fused_history = fused_history.to(resolved_device)
                target_latents = target_latents.to(resolved_device)
                future_action_latents = future_action_latents.to(resolved_device)

                loss, latent_loss, frame_loss, _ = rollout_batch(
                    models,
                    fused_history,
                    target_latents,
                    future_action_latents,
                    memory_frames,
                    memory_stride,
                    latent_loss_fn,
                    frame_loss_weight,
                    foreground_pixel_weight,
                    white_pixel_threshold,
                )
                val_loss_total += loss.item()
                val_latent_total += latent_loss.item()
                val_frame_total += frame_loss.item()

        avg_train = train_loss_total / len(train_loader)
        avg_train_latent = train_latent_total / len(train_loader)
        avg_train_frame = train_frame_total / len(train_loader)
        avg_val = val_loss_total / len(val_loader)
        avg_val_latent = val_latent_total / len(val_loader)
        avg_val_frame = val_frame_total / len(val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(avg_train)
        history["train_latent_loss"].append(avg_train_latent)
        history["train_frame_loss"].append(avg_train_frame)
        history["val_loss"].append(avg_val)
        history["val_latent_loss"].append(avg_val_latent)
        history["val_frame_loss"].append(avg_val_frame)
        history["learning_rates"].append(current_lr)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train={avg_train:.6f} | "
            f"val={avg_val:.6f} latent={avg_val_latent:.6f} frame={avg_val_frame:.6f} | "
            f"lr={current_lr:.2e}"
        )

        if (epoch + 1) % save_samples_every == 0:
            save_visual_samples(
                models,
                val_loader,
                sample_dir,
                epoch + 1,
                memory_frames,
                memory_stride,
                latent_loss_fn,
                frame_loss_weight,
                foreground_pixel_weight,
                white_pixel_threshold,
                resolved_device,
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {"model_state_dict": models.transition_model.state_dict()}, output_path
            )
            print(f"Melhor checkpoint salvo em {output_path}")

    print(f"Treinamento concluido. Melhor val loss: {best_val_loss:.6f}")

    # Save history JSON — same pattern as VAE trainer
    save_dir = history_dir or os.path.dirname(output_path)
    os.makedirs(save_dir, exist_ok=True)
    history_path = os.path.join(
        save_dir, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Histórico salvo em {history_path}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model-weights", default=WORLD_MODEL_WEIGHTS)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--frame-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--foreground-pixel-weight", type=float, default=DEFAULT_FOREGROUND_PIXEL_WEIGHT
    )
    parser.add_argument(
        "--white-pixel-threshold", type=float, default=DEFAULT_WHITE_PIXEL_THRESHOLD
    )
    parser.add_argument("--memory-frames", type=int, default=DEFAULT_MEMORY_FRAMES)
    parser.add_argument("--memory-stride", type=int, default=DEFAULT_MEMORY_STRIDE)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--sample-dir", default=TRAINING_SAMPLES_DIR)
    parser.add_argument("--save-samples-every", type=int, default=1)
    parser.add_argument(
        "--history-dir",
        default=None,
        help="Directory to save training history JSON. Defaults to same dir as world-model-weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_world_model(
        dataset_dir=args.dataset_dir,
        output_path=args.world_model_weights,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        frame_loss_weight=args.frame_loss_weight,
        foreground_pixel_weight=args.foreground_pixel_weight,
        white_pixel_threshold=args.white_pixel_threshold,
        max_samples=args.max_samples,
        device=args.device,
        memory_frames=args.memory_frames,
        memory_stride=args.memory_stride,
        rollout_steps=args.rollout_steps,
        sample_dir=args.sample_dir,
        save_samples_every=args.save_samples_every,
        history_dir=args.history_dir,
    )


if __name__ == "__main__":
    main()
