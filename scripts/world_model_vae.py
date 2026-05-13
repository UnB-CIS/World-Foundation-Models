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
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "scripts", "dataset", "dataset_processado")
VAE_WEIGHTS = os.path.join(MODELS_ROOT, "VAE", "vae_weights.pth")
TEXT_WEIGHTS = os.path.join(MODELS_ROOT, "Text_Encoder", "text_encoder_weights.pth")
WORLD_MODEL_WEIGHTS = os.path.join(CURRENT_DIR, "world_model_weights.pth")
TRAINING_SAMPLES_DIR = os.path.join(CURRENT_DIR, "world_model_samples")

sys.path.insert(0, os.path.join(MODELS_ROOT, "VAE"))
sys.path.insert(0, os.path.join(MODELS_ROOT, "Text_Encoder"))
sys.path.insert(0, MODELS_ROOT)

from vae_video import VAE  # noqa: E402
from Text_Encoder import TextEncoder, encoding_function  # noqa: E402
from latent_fusion import SpatialBroadcastFuser  # noqa: E402


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MODEL_FRAME_SIZE = 64
ENCODED_FRAME_CHANNELS = 16
VISUAL_LATENT_CHANNELS = 8
ACTION_LATENT_CHANNELS = 16
FUSED_LATENT_CHANNELS = ENCODED_FRAME_CHANNELS + ACTION_LATENT_CHANNELS
DEFAULT_MEMORY_FRAMES = 10
DEFAULT_MEMORY_STRIDE = 5
DEFAULT_FOREGROUND_PIXEL_WEIGHT = 50.0
DEFAULT_WHITE_PIXEL_THRESHOLD = 0.98
DEFAULT_ROLLOUT_STEPS = 4


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
    return (frame * 255.0).astype(np.uint8)


def select_spaced_history(history: List[torch.Tensor], memory_frames: int, memory_stride: int) -> torch.Tensor:
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


@dataclass
class PredictionResult:
    frame: np.ndarray
    model_status: str


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
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels,
                hidden_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.ReLU(inplace=True),
        )
        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LatentTransitionBlock(hidden_channels),
            LatentTransitionBlock(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, fused_history: torch.Tensor) -> torch.Tensor:
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
            pad = fused_history[:, :1].expand(batch_size, pad_count, channels, height, width)
            fused_history = torch.cat([pad, fused_history], dim=1)
        elif time_steps > self.memory_frames:
            fused_history = fused_history[:, -self.memory_frames :]

        temporal_input = fused_history.permute(0, 2, 1, 3, 4)
        temporal_features = self.temporal_encoder(temporal_input)
        summarized_features = temporal_features[:, :, -1]
        return self.spatial_decoder(summarized_features)


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
        self.vae = VAE().to(self.device)
        self.text_encoder = TextEncoder().to(self.device)
        self.fuser = SpatialBroadcastFuser(height=8, width=8).to(self.device)
        self.transition_model = LatentTransitionModel(memory_frames=memory_frames).to(self.device)
        self.memory_frames = memory_frames
        self.world_model_weights = world_model_weights
        self.transition_weights_loaded = False
        self._load_weights()
        self.vae.eval()
        self.text_encoder.eval()
        self.fuser.eval()
        self.transition_model.eval()

    def _load_weights(self) -> None:
        if not os.path.exists(VAE_WEIGHTS):
            raise FileNotFoundError(f"Pesos do VAE nao encontrados em {VAE_WEIGHTS}")
        if not os.path.exists(TEXT_WEIGHTS):
            raise FileNotFoundError(f"Pesos do Text Encoder nao encontrados em {TEXT_WEIGHTS}")

        self.vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=self.device, weights_only=True))
        self.text_encoder.load_state_dict(torch.load(TEXT_WEIGHTS, map_location=self.device, weights_only=True))
        if self.world_model_weights and os.path.exists(self.world_model_weights):
            try:
                self.transition_model.load_state_dict(
                    torch.load(self.world_model_weights, map_location=self.device, weights_only=True)
                )
                self.transition_weights_loaded = True
            except RuntimeError as exc:
                raise RuntimeError(
                    "Checkpoint do world model incompatível com a arquitetura temporal atual. "
                    "Treine novamente o preditor com memoria."
                ) from exc

    @torch.no_grad()
    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame_tensor = preprocess_frame(frame, self.device)
        return self.vae.encoder(frame_tensor)

    @torch.no_grad()
    def fuse(self, frame: np.ndarray, action: Optional[dict]) -> torch.Tensor:
        mu = self.encode_frame(frame)
        action_vector = encoding_function(action).unsqueeze(0).to(self.device)
        action_latent = self.text_encoder(action_vector)
        return self.fuser(mu, action_latent)

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> np.ndarray:
        latent = latent.to(self.device)
        decoded = self.vae.decoder(latent)
        return frame_from_tensor(decoded)


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
        fused = self.models.fuse(frame, action)
        history = [item.to(self.models.device) for item in (fused_history or [])]
        history.append(fused.detach())
        temporal_input = select_spaced_history(
            history,
            self.memory_frames,
            self.memory_stride,
        )
        next_latent = self.models.transition_model(temporal_input)
        next_frame = self.models.decode_latent(next_latent)
        model_status = (
            f"pesos carregados de {os.path.basename(self.models.world_model_weights)}"
            if self.models.transition_weights_loaded and self.models.world_model_weights
            else "sem checkpoint do world model, usando pesos aleatorios"
        )
        return PredictionResult(
            frame=next_frame,
            model_status=model_status,
        ), fused.detach()


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
        self.memory_frames = memory_frames
        self.memory_stride = memory_stride
        self.rollout_steps = rollout_steps
        files = sorted(
            os.path.join(dataset_dir, name)
            for name in os.listdir(dataset_dir)
            if name.endswith(".pt")
        )
        if not files:
            raise FileNotFoundError(f"Nenhum dataset processado encontrado em {dataset_dir}")

        for path in files:
            sample = torch.load(path, map_location="cpu", weights_only=True)
            x = sample["x"].to(torch.float32)
            y = sample["y"].to(torch.float32)
            decoder_target = y[:, :VISUAL_LATENT_CHANNELS]
            for frame_idx in range(x.shape[0] - rollout_steps + 1):
                history_indices = [
                    max(0, frame_idx - offset * memory_stride)
                    for offset in range(memory_frames - 1, -1, -1)
                ]
                history = x[history_indices]
                target_latents = decoder_target[frame_idx : frame_idx + rollout_steps]
                action_start = frame_idx + 1
                action_end = frame_idx + rollout_steps
                future_action_latents = x[action_start:action_end, ENCODED_FRAME_CHANNELS:]
                self.samples.append((history, target_latents, future_action_latents))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
            if max_samples is not None and len(self.samples) >= max_samples:
                self.samples = self.samples[:max_samples]
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history, target_latents, future_action_latents = self.samples[index]
        return history.clone(), target_latents.clone(), future_action_latents.clone()


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
        temporal_input = select_spaced_history(history_items, memory_frames, memory_stride)
        predicted_latent = models.transition_model(temporal_input)
        target_latent = target_latents[:, step]

        latent_loss = latent_loss_fn(predicted_latent, target_latent)
        predicted_frame = models.vae.decoder(predicted_latent)
        target_frame = models.vae.decoder(target_latent)
        frame_loss = weighted_frame_l1_loss(
            predicted_frame,
            target_frame,
            foreground_pixel_weight,
            white_pixel_threshold,
        )

        total_latent_loss = total_latent_loss + latent_loss
        total_frame_loss = total_frame_loss + frame_loss
        predicted_latents.append(predicted_latent)

        if step < rollout_steps - 1:
            predicted_encoded = models.vae.encoder(predicted_frame)
            next_action_latent = future_action_latents[:, step]
            next_fused = torch.cat([predicted_encoded, next_action_latent], dim=1)
            history_items.append(next_fused)

    avg_latent_loss = total_latent_loss / rollout_steps
    avg_frame_loss = total_frame_loss / rollout_steps
    total_loss = avg_latent_loss + frame_loss_weight * avg_frame_loss
    return total_loss, avg_latent_loss, avg_frame_loss, torch.stack(predicted_latents, dim=1)


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
        predicted_frames = models.vae.decoder(predicted_latents.squeeze(0))
        target_frames = models.vae.decoder(target_latents.squeeze(0))

    rows = []
    separator = np.full((MODEL_FRAME_SIZE, 4), 255, dtype=np.uint8)
    for step in range(target_frames.shape[0]):
        target = frame_from_tensor(target_frames[step])
        predicted = frame_from_tensor(predicted_frames[step])
        difference = np.abs(predicted.astype(np.int16) - target.astype(np.int16)).astype(np.uint8)
        rows.append(np.concatenate([target, separator, predicted, separator, difference], axis=1))

    grid = np.concatenate(rows, axis=0)
    output_path = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
    cv2.imwrite(output_path, grid)


def train_world_model(
    dataset_dir: str = DATASET_DIR,
    output_path: str = WORLD_MODEL_WEIGHTS,
    batch_size: int = 128,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    train_ratio: float = 0.9,
    frame_loss_weight: float = 0.25,
    foreground_pixel_weight: float = DEFAULT_FOREGROUND_PIXEL_WEIGHT,
    white_pixel_threshold: float = DEFAULT_WHITE_PIXEL_THRESHOLD,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    memory_frames: int = DEFAULT_MEMORY_FRAMES,
    memory_stride: int = DEFAULT_MEMORY_STRIDE,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
    sample_dir: str = TRAINING_SAMPLES_DIR,
    save_samples_every: int = 1,
) -> None:
    if memory_stride <= 0:
        raise ValueError("memory_stride deve ser maior que zero.")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps deve ser maior que zero.")
    if save_samples_every <= 0:
        raise ValueError("save_samples_every deve ser maior que zero.")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Treinamento em GPU foi solicitado, mas CUDA nao esta disponivel neste ambiente."
        )
    models = ModelBundle(
        device=str(resolved_device),
        world_model_weights=None,
        memory_frames=memory_frames,
    )
    models.transition_model.train()
    models.vae.eval()
    models.text_encoder.eval()
    models.fuser.eval()

    for module in (models.vae, models.text_encoder, models.fuser):
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
    latent_loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    print(f"Treinando world model em {resolved_device}")
    print(f"Amostras: total={len(dataset)} treino={len(train_dataset)} validacao={len(val_dataset)}")
    print(f"Memoria temporal: {memory_frames} frames fundidos com stride {memory_stride}")
    print(f"Rollout multi-step: {rollout_steps} passos")
    print(f"Amostras visuais: {sample_dir} a cada {save_samples_every} epoca(s)")
    print(
        "Frame loss ponderada: "
        f"foreground_weight={foreground_pixel_weight} "
        f"white_threshold={white_pixel_threshold}"
    )

    for epoch in range(num_epochs):
        models.transition_model.train()
        train_loss_total = 0.0
        train_latent_total = 0.0
        train_frame_total = 0.0

        for fused_history, target_latents, future_action_latents in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} treino",
            leave=False,
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
            optimizer.step()

            train_loss_total += loss.item()
            train_latent_total += latent_loss.item()
            train_frame_total += frame_loss.item()

        models.transition_model.eval()
        val_loss_total = 0.0
        val_latent_total = 0.0
        val_frame_total = 0.0

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

        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_latent = train_latent_total / len(train_loader)
        avg_train_frame = train_frame_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_latent = val_latent_total / len(val_loader)
        avg_val_frame = val_frame_total / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train={avg_train_loss:.6f} latent={avg_train_latent:.6f} frame={avg_train_frame:.6f} | "
            f"val={avg_val_loss:.6f} latent={avg_val_latent:.6f} frame={avg_val_frame:.6f}"
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(models.transition_model.state_dict(), output_path)
            print(f"Melhor checkpoint salvo em {output_path}")

    print(f"Treinamento concluido. Melhor val loss: {best_val_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o preditor latente do world model 2D.")
    parser.add_argument(
        "--world-model-weights",
        default=WORLD_MODEL_WEIGHTS,
        help="Checkpoint do preditor de latente do world model.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DATASET_DIR,
        help="Diretorio com os arquivos .pt do dataset processado.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas do treino.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size do treino.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate do treino.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Dispositivo do treino. O padrao exige GPU via CUDA.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita a quantidade de amostras do dataset durante o treino.",
    )
    parser.add_argument(
        "--frame-loss-weight",
        type=float,
        default=0.25,
        help="Peso da loss no frame reconstruido via decoder.",
    )
    parser.add_argument(
        "--foreground-pixel-weight",
        type=float,
        default=DEFAULT_FOREGROUND_PIXEL_WEIGHT,
        help="Peso dos pixels nao brancos na loss visual.",
    )
    parser.add_argument(
        "--white-pixel-threshold",
        type=float,
        default=DEFAULT_WHITE_PIXEL_THRESHOLD,
        help="Pixels abaixo desse valor sao tratados como conteudo relevante.",
    )
    parser.add_argument(
        "--memory-frames",
        type=int,
        default=DEFAULT_MEMORY_FRAMES,
        help="Quantidade de frames fundidos anteriores usados pela memoria temporal.",
    )
    parser.add_argument(
        "--memory-stride",
        type=int,
        default=DEFAULT_MEMORY_STRIDE,
        help="Intervalo entre frames selecionados para a memoria temporal.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
        help="Quantidade de passos futuros previstos durante o treino multi-step.",
    )
    parser.add_argument(
        "--sample-dir",
        default=TRAINING_SAMPLES_DIR,
        help="Diretorio para salvar amostras visuais de validacao.",
    )
    parser.add_argument(
        "--save-samples-every",
        type=int,
        default=1,
        help="Frequencia, em epocas, para salvar amostras visuais.",
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
    )


if __name__ == "__main__":
    main()
