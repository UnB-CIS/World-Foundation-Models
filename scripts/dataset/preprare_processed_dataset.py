import json
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm

from collections import OrderedDict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from src.vae.model import VAE
from src.action_encoder.model import ActionTextEncoder
from src.action_encoder.encoding import encoding_function
from src.fusion.model import SpatialBroadcastFuser

PROJECT_ROOT = Path(project_root)
print(PROJECT_ROOT)
INPUT_FOLDER_JSON = PROJECT_ROOT / "data" / "scenario_1" / "inputs"
INPUT_FOLDER_VIDEO = PROJECT_ROOT / "data" / "scenario_1" / "videos"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "scenario_1" / "processed_next_frame"

VAE_WEIGHTS = PROJECT_ROOT / "checkpoints" / "vae" / "best_model.pth"
TEXT_WEIGHTS = PROJECT_ROOT / "checkpoints" / "action_encoder" / "best_model.pth"

TARGET_FPS = 60.0
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

VISUAL_LATENT_CHANNELS = 16
ACTION_LATENT_CHANNELS = 16
FUSED_LATENT_CHANNELS = VISUAL_LATENT_CHANNELS + ACTION_LATENT_CHANNELS

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

TYPE_ENCODING = {
    "mouse_down": [1.0],
    "none": [0.0],
}

OBJECT_ENCODING = {
    "ball": [1.0],
    "none": [0.0],
}


def load_models():
    vae = VAE(
        latent_channels=VISUAL_LATENT_CHANNELS,
        action_latent_channels=ACTION_LATENT_CHANNELS,
    ).to(DEVICE)
    vae_ckpt = torch.load(VAE_WEIGHTS, map_location=DEVICE, weights_only=True)
    vae_state = vae_ckpt.get("model_state_dict", vae_ckpt)
    vae.load_state_dict(vae_state, strict=True)
    vae.eval()

    text_encoder = ActionTextEncoder().to(DEVICE)
    text_ckpt = torch.load(TEXT_WEIGHTS, map_location=DEVICE, weights_only=True)
    text_state = text_ckpt.get("model_state_dict", text_ckpt)

    remapped_state = OrderedDict()
    for key, value in text_state.items():
        if key.startswith("encoder."):
            remapped_state[key[len("encoder.") :]] = value

    text_encoder.load_state_dict(remapped_state, strict=True)
    text_encoder.eval()

    fuser = SpatialBroadcastFuser(height=8, width=8).to(DEVICE)
    fuser.eval()

    return vae, text_encoder, fuser


def map_actions_to_frames(json_path, fps):
    with open(json_path, "r") as f:
        actions_list = json.load(f)

    action_map = {}
    for action in actions_list:
        frame_idx = int(action["time"] * fps)
        action_map[frame_idx] = action

    return action_map


def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype(np.float32) / 255.0
    tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


@torch.no_grad()
def process_step(frame_curr, frame_next, action_data, models):
    vae, text_encoder, fuser = models

    t_curr = preprocess_frame(frame_curr)
    t_next = preprocess_frame(frame_next)

    action_vector = (
        encoding_function(
            type_encoding=TYPE_ENCODING,
            object_encoding=OBJECT_ENCODING,
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
            input_vector_dim=4,
            action_data=action_data,
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    encoded_curr = vae.encoder(t_curr)
    mu_curr, _ = torch.chunk(encoded_curr, 2, dim=1)  # (1, 16, 8, 8)

    encoded_next = vae.encoder(t_next)
    mu_next, _ = torch.chunk(encoded_next, 2, dim=1)  # (1, 16, 8, 8)

    action_latent = text_encoder(action_vector)  # (1, 16)
    fused_latent = fuser(mu_curr, action_latent)  # (1, 32, 8, 8)

    return fused_latent.squeeze(0).cpu(), mu_next.squeeze(0).cpu()


def process_video_sequence(video_path, json_path, models):
    action_map = map_actions_to_frames(json_path, TARGET_FPS)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    inputs_list = []
    targets_list = []
    frame_idx = 0

    ret, frame_curr = cap.read()
    if not ret:
        cap.release()
        return None

    while True:
        ret, frame_next = cap.read()
        if not ret:
            break

        current_action = action_map.get(frame_idx, None)
        x, y = process_step(frame_curr, frame_next, current_action, models)

        inputs_list.append(x)  # (32, 8, 8)
        targets_list.append(y)  # (16, 8, 8)

        frame_curr = frame_next
        frame_idx += 1

    cap.release()

    if not inputs_list:
        return None

    return {
        "x": torch.stack(inputs_list),  # (T, 32, 8, 8)
        "y": torch.stack(targets_list),  # (T, 16, 8, 8)
    }


def main():
    models = load_models()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not INPUT_FOLDER_JSON.exists():
        print(f"Pasta não encontrada: {INPUT_FOLDER_JSON}")
        return

    json_files = sorted(
        [f for f in os.listdir(INPUT_FOLDER_JSON) if f.endswith(".json")]
    )
    print(f"Iniciando processamento de {len(json_files)} simulações...")
    print(f"Usando dispositivo: {DEVICE}")

    count = 0
    for json_file in tqdm(json_files):
        prefix = json_file.replace(".json", "")
        video_file = next(
            (
                v
                for v in os.listdir(INPUT_FOLDER_VIDEO)
                if v.startswith(prefix) and v.endswith(".mp4")
            ),
            None,
        )

        if not video_file:
            continue

        try:
            dataset_tensors = process_video_sequence(
                INPUT_FOLDER_VIDEO / video_file,
                INPUT_FOLDER_JSON / json_file,
                models,
            )
        except Exception as e:
            print(f"Erro processando {json_file}: {e}")
            continue

        if dataset_tensors is not None:
            save_name = json_file.replace(".json", ".pt")
            torch.save(dataset_tensors, OUTPUT_FOLDER / save_name)
            count += 1

    print(f"Sucesso! {count} arquivos .pt gerados em {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
