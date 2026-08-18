"""Carregamento de episodios e inferencia em lote do world model.

O simulador interativo (`scripts/world_model_simulator.py`) roda um passo por
clique do usuario. Para avaliacao precisamos dos mesmos passos, mas em milhares
de frames, entao aqui a mesma matematica e executada em lote:

    frames -> mu (VAE) -> fusao com a acao -> historico temporal ->
    ConvLSTM (delta) -> mu_proximo = mu + delta -> decoder -> frame predito

Nada e reimplementado: os pesos, o encoder de acao e a regra `next = current +
delta` sao exatamente os de `scripts/world_model_vae.py`.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from world_model_vae import (  # noqa: E402
    ACTION_LATENT_CHANNELS,
    MODEL_FRAME_SIZE,
    OBJECT_ENCODING,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TYPE_ENCODING,
    VISUAL_LATENT_CHANNELS,
    ModelBundle,
)

from src.action_encoder.encoding import encoding_function  # noqa: E402

# Fonte padrao: os episodios usados para construir o dataset de treino.
TRAIN_POOL_ROOT = os.path.join(PROJECT_ROOT, "scripts", "dataset")
# Conjunto inedito gerado por `scripts/eval/generate_holdout.py`.
HOLDOUT_ROOT = os.path.join(PROJECT_ROOT, "data", "holdout")
DATASET_ROOTS = {"train_pool": TRAIN_POOL_ROOT, "holdout": HOLDOUT_ROOT}
TARGET_FPS = 60.0
LATENT_GRID = 8


@dataclass
class Episode:
    """Um episodio: frames reais em [0,1] e o mapa {indice_frame: acao}."""

    name: str
    family: str
    frames: np.ndarray  # (T, 64, 64) float32 em [0, 1]
    actions: dict[int, dict]
    video_path: str

    @property
    def length(self) -> int:
        return int(self.frames.shape[0])

    def action_list(self) -> list[dict | None]:
        return [self.actions.get(t) for t in range(self.length)]


def episode_family(name: str) -> str:
    """Agrupa episodios pelo lote de geracao (prefixo de timestamp)."""
    match = re.match(r"(?:random_input|holdout)_(\d+)", name)
    return match.group(1) if match else "desconhecido"


def _action_frame_index(action_time: float) -> int:
    """Indice do ultimo frame ANTES da bola aparecer.

    O simulador injeta a bola na primeira iteracao com `tempo_sim >= action_time`
    e desenha o frame ja com ela. Logo a bola surge no frame `ceil(t*fps)` e o
    frame condicionado pela acao (o que ainda nao a contem) e o anterior.
    `dataset_builder.py` usa `int(t*fps)`, que coincide com este valor exceto
    quando `t*fps` e inteiro; aqui usamos a versao causalmente correta.
    """
    return max(0, math.ceil(round(action_time * TARGET_FPS, 6)) - 1)


def resolve_root(dataset: str) -> str:
    root = DATASET_ROOTS.get(dataset, dataset)
    if not os.path.isdir(os.path.join(root, "videos")):
        raise FileNotFoundError(f"Fonte de episodios invalida: {root}")
    return root


def list_episodes(pattern: str = "*", dataset: str = "train_pool") -> list[tuple[str, str]]:
    """Retorna [(nome_base, caminho_video)] com JSON de acoes correspondente."""
    root = resolve_root(dataset)
    found: dict[str, str] = {}
    for video in sorted(glob.glob(os.path.join(root, "videos", f"{pattern}.mp4"))):
        stem = os.path.splitext(os.path.basename(video))[0]
        base = re.sub(r"_auto.*$", "", stem)
        found.setdefault(base, video)
    return [
        (base, video)
        for base, video in sorted(found.items())
        if os.path.exists(os.path.join(root, "inputs", f"{base}.json"))
    ]


def load_episode(
    name: str,
    video_path: str,
    max_frames: int | None = None,
    dataset: str = "train_pool",
) -> Episode:
    capture = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (MODEL_FRAME_SIZE, MODEL_FRAME_SIZE)))
        if max_frames is not None and len(frames) >= max_frames:
            break
    capture.release()
    if not frames:
        raise RuntimeError(f"Video sem frames legiveis: {video_path}")

    with open(os.path.join(resolve_root(dataset), "inputs", f"{name}.json")) as handle:
        raw_actions = json.load(handle)

    actions = {_action_frame_index(item["time"]): item for item in raw_actions}
    return Episode(
        name=name,
        family=episode_family(name),
        frames=np.stack(frames).astype(np.float32) / 255.0,
        actions=actions,
        video_path=video_path,
    )


class WorldModelRunner:
    """Executa o world model em lote sobre episodios inteiros."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        memory_frames: int = 10,
        memory_stride: int = 1,
        batch_size: int = 96,
    ) -> None:
        self.bundle = ModelBundle(
            device=device,
            world_model_weights=checkpoint,
            memory_frames=memory_frames,
        )
        if not self.bundle.transition_weights_loaded:
            raise RuntimeError(f"Checkpoint do world model nao carregado: {checkpoint}")
        self.checkpoint = checkpoint
        self.memory_frames = memory_frames
        self.memory_stride = memory_stride
        self.batch_size = batch_size
        self.device = self.bundle.device

    # ------------------------------------------------------------------
    # Blocos elementares
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """(T,64,64) em [0,1] -> mu (T,16,8,8)."""
        tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(1).to(self.device)
        outputs = []
        for start in range(0, tensor.shape[0], self.batch_size):
            encoded = self.bundle.vae.encoder(tensor[start : start + self.batch_size])
            mu, _ = torch.chunk(encoded, 2, dim=1)
            outputs.append(mu)
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def decode(self, mu: torch.Tensor) -> np.ndarray:
        """mu (T,16,8,8) -> frames (T,64,64) em [0,1].

        `action_latent=None` reproduz a inferencia do simulador, em que o
        decoder recebe zeros nos canais de acao.
        """
        outputs = []
        for start in range(0, mu.shape[0], self.batch_size):
            decoded = self.bundle.vae.decoder(mu[start : start + self.batch_size], None)
            outputs.append(decoded.squeeze(1).clamp(0.0, 1.0).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    @torch.no_grad()
    def action_latents(self, actions: list[dict | None]) -> torch.Tensor:
        """Lista de acoes -> (T,16), com subtracao da linha de base nula."""
        vectors = torch.stack(
            [
                encoding_function(
                    type_encoding=TYPE_ENCODING,
                    object_encoding=OBJECT_ENCODING,
                    screen_width=SCREEN_WIDTH,
                    screen_height=SCREEN_HEIGHT,
                    input_vector_dim=4,
                    action_data=action,
                )
                for action in actions
            ]
        ).to(self.device)
        null = self.bundle.text_encoder(torch.zeros_like(vectors[:1]))
        return self.bundle.text_encoder(vectors) - null

    def fuse(self, mu: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        """[mu | acao difundida espacialmente] -> (T,32,8,8)."""
        broadcast = action_latent[:, :, None, None].expand(
            -1, ACTION_LATENT_CHANNELS, LATENT_GRID, LATENT_GRID
        )
        return torch.cat([mu, broadcast], dim=1)

    def history_window(self) -> int:
        """Tamanho do buffer necessario para amostrar `memory_frames` espacados."""
        return (self.memory_frames - 1) * self.memory_stride + 1

    def build_history(self, fused: torch.Tensor) -> torch.Tensor:
        """(T,32,8,8) -> (T, memory_frames, 32,8,8) replicando o inicio.

        Reproduz `select_spaced_history` de `world_model_vae.py` para todos os
        instantes de uma vez.
        """
        length = fused.shape[0]
        offsets = torch.arange(self.memory_frames - 1, -1, -1, device=fused.device)
        times = torch.arange(length, device=fused.device)[:, None]
        indices = (times - offsets[None, :] * self.memory_stride).clamp_min(0)
        return fused[indices]

    @torch.no_grad()
    def predict_delta(self, history: torch.Tensor) -> torch.Tensor:
        """(N, memory_frames, 32,8,8) -> delta (N,16,8,8)."""
        outputs = []
        for start in range(0, history.shape[0], self.batch_size):
            outputs.append(
                self.bundle.transition_model(history[start : start + self.batch_size])
            )
        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # Predicao de um passo sobre um episodio inteiro
    # ------------------------------------------------------------------
    @torch.no_grad()
    def one_step(
        self, episode: Episode, use_actions: bool = True
    ) -> dict[str, torch.Tensor]:
        """Predicao teacher-forced: para todo t, prediz o estado em t+1.

        Retorna os latentes verdadeiros, o predito e o delta previsto.
        """
        mu = self.encode(episode.frames)
        actions = episode.action_list() if use_actions else [None] * episode.length
        fused = self.fuse(mu, self.action_latents(actions))
        delta = self.predict_delta(self.build_history(fused))
        return {
            "mu_true": mu,
            "mu_predicted": mu + delta,
            "delta": delta,
            "fused": fused,
        }

    # ------------------------------------------------------------------
    # Rollout em malha aberta
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        episode: Episode,
        starts: list[int],
        horizon: int,
        feedback: str = "pixel",
        use_actions: bool = True,
    ) -> np.ndarray:
        """Rollout autoregressivo a partir de varios instantes em paralelo.

        `feedback="pixel"` decodifica e recodifica a predicao a cada passo,
        exatamente como o treino (`rollout_batch`) e como o simulador
        interativo. `feedback="latent"` realimenta o latente direto, sem passar
        pelo decoder.

        Retorna frames preditos (len(starts), horizon, 64, 64).
        """
        if feedback not in {"pixel", "latent"}:
            raise ValueError("feedback deve ser 'pixel' ou 'latent'")

        mu_all = self.encode(episode.frames)
        actions = episode.action_list() if use_actions else [None] * episode.length
        action_latent_all = self.action_latents(actions)
        fused_all = self.fuse(mu_all, action_latent_all)

        window = self.history_window()
        start_tensor = torch.tensor(starts, device=self.device)
        offsets = torch.arange(window - 1, -1, -1, device=self.device)
        buffer_indices = (start_tensor[:, None] - offsets[None, :]).clamp_min(0)
        buffer = fused_all[buffer_indices]  # (S, window, 32, 8, 8)
        mu = mu_all[start_tensor]

        predictions = np.empty(
            (len(starts), horizon, MODEL_FRAME_SIZE, MODEL_FRAME_SIZE),
            dtype=np.float32,
        )
        last_index = episode.length - 1
        for step in range(horizon):
            history = buffer[:, :: self.memory_stride]
            mu = mu + self.predict_delta(history)
            frame = self.decode(mu)
            predictions[:, step] = frame
            if feedback == "pixel":
                mu = self.encode(frame)
            if step == horizon - 1:
                break
            next_times = (start_tensor + step + 1).clamp_max(last_index)
            next_fused = self.fuse(mu, action_latent_all[next_times])
            buffer = torch.cat([buffer[:, 1:], next_fused[:, None]], dim=1)
        return predictions


def split_episodes(
    episodes: list[tuple[str, str]], test_ratio: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split deterministico. O manifesto de treino original nao foi versionado,
    entao o split e reproduzivel por semente e documentado no relatorio."""
    ordered = sorted(episodes)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    cut = max(1, int(len(ordered) * test_ratio))
    test_indices = set(permutation[:cut].tolist())
    test = [ordered[i] for i in sorted(test_indices)]
    train = [ordered[i] for i in range(len(ordered)) if i not in test_indices]
    return train, test


__all__ = [
    "Episode",
    "WorldModelRunner",
    "list_episodes",
    "load_episode",
    "split_episodes",
    "VISUAL_LATENT_CHANNELS",
]
