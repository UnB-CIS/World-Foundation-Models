import torch
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
import sys

# ==============================================================================
# CONFIGURAÇÃO DE IMPORTS
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

try:
    from src.vae.model import VAE
    from src.action_encoder.model import ActionTextEncoder
    from src.action_encoder.encoding import encoding_function
    from src.fusion.model import SpatialBroadcastFuser

    print("OK: Sem erros de importação")

except ImportError as e:
    print(f"\nERRO DE IMPORTAÇÃO: {e}")
    sys.exit()

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

PROJECT_ROOT = project_root
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "scenario_1")
INPUT_FOLDER_JSON = os.path.join(DATA_DIR, "inputs")
INPUT_FOLDER_VIDEO = os.path.join(DATA_DIR, "videos")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "scenario_1", "processed")

VAE_WEIGHTS = os.path.join(
    PROJECT_ROOT, "checkpoints", "model_checkpoints", "vae_weights.pth"
)
TEXT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "checkpoints", "model_checkpoints", "text_encoder_weights.pth"
)

TARGET_FPS = 60.0
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Usando dispositivo: {DEVICE}")

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================


def load_models():
    """Carrega VAE, ActionTextEncoder e Fuser com pesos pré-treinados."""

    # 1. VAE
    vae = VAE().to(DEVICE)
    if os.path.exists(VAE_WEIGHTS):
        vae.load_state_dict(
            torch.load(VAE_WEIGHTS, map_location=DEVICE, weights_only=True)
        )
        print("Pesos VAE carregados")
    else:
        print(f"AVISO: Pesos VAE não encontrados em {VAE_WEIGHTS}")
        sys.exit()
    vae.eval()

    # 2. Action Encoder
    text_enc = ActionTextEncoder(input_dim=4, latent_dim=16).to(DEVICE)
    if os.path.exists(TEXT_WEIGHTS):
        text_enc.load_state_dict(
            torch.load(TEXT_WEIGHTS, map_location=DEVICE, weights_only=True)
        )
        print("Pesos Action Encoder carregados.")
    else:
        print(f"AVISO: Pesos Action Encoder não encontrados em {TEXT_WEIGHTS}")
        sys.exit()
    text_enc.eval()

    # 3. Fuser
    fuser = SpatialBroadcastFuser(height=8, width=8).to(DEVICE)

    return vae, text_enc, fuser


def map_actions_to_frames(json_path, fps):
    """Lê o JSON e cria um dicionário {frame_index: action_data}."""
    with open(json_path, 'r') as f:
        actions_list = json.load(f)

    action_map = {}
    for action in actions_list:
        frame_idx = int(action['time'] * fps)
        action_map[frame_idx] = action

    return action_map


def preprocess_frame(frame):
    """Prepara frame para passar como entrada do VAE (64x64, Grayscale, Tensor)."""
    frame = cv2.resize(frame, (64, 64))
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = frame.astype(np.float32) / 255.0
    tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def process_step(frame_curr, frame_next, action_data, models):
    """
    Recebe frames brutos e dados da ação.
    Retorna:
       - Input  X: Latente Fundido  (24, 8, 8)
       - Target Y: Latente Visual Futuro (8, 8, 8)
    """
    vae, text_enc, fuser = models

    t_curr = preprocess_frame(frame_curr)
    t_next = preprocess_frame(frame_next)

    type_encoding = {"mouse_down": [1.0], "none": [0.0]}
    object_encoding = {"ball": [1.0], "none": [0.0]}

    vec_action = (
        encoding_function(
            type_encoding=type_encoding,
            object_encoding=object_encoding,
            screen_width=800,
            screen_height=600,
            input_vector_dim=4,
            action_data=action_data,
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        mu_curr, _ = vae.encode(t_curr)
        mu_next, _ = vae.encode(t_next)
        emb_action = text_enc(vec_action)
        z_fused = fuser(mu_curr, emb_action)

    return z_fused.squeeze(0).cpu(), mu_next.squeeze(0).cpu()


def process_video_sequence(video_path, json_path, models):
    """Itera sobre todos os frames do vídeo e empilha os resultados."""
    action_map = map_actions_to_frames(json_path, TARGET_FPS)

    cap = cv2.VideoCapture(video_path)
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

        inputs_list.append(x)
        targets_list.append(y)

        frame_curr = frame_next
        frame_idx += 1

    cap.release()

    if not inputs_list:
        return None

    return {'x': torch.stack(inputs_list), 'y': torch.stack(targets_list)}


# ==============================================================================
# LOOP PRINCIPAL (MAIN)
# ==============================================================================


def main():
    models = load_models()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(INPUT_FOLDER_JSON):
        print(f"Pasta não encontrada: {INPUT_FOLDER_JSON}")
        return

    json_files = [f for f in os.listdir(INPUT_FOLDER_JSON) if f.endswith('.json')]
    print(f"Iniciando processamento de {len(json_files)} simulações...")

    count = 0
    for json_file in tqdm(json_files):
        prefix = json_file.replace('.json', '')
        video_file = next(
            (
                v
                for v in os.listdir(INPUT_FOLDER_VIDEO)
                if v.startswith(prefix) and v.endswith('.mp4')
            ),
            None,
        )

        if not video_file:
            continue

        try:
            dataset_tensors = process_video_sequence(
                os.path.join(INPUT_FOLDER_VIDEO, video_file),
                os.path.join(INPUT_FOLDER_JSON, json_file),
                models,
            )
        except Exception as e:
            print(f"Erro processando {json_file}: {e}")
            continue

        if dataset_tensors:
            save_name = json_file.replace('.json', '.pt')
            torch.save(dataset_tensors, os.path.join(OUTPUT_FOLDER, save_name))
            count += 1

    print(f"Sucesso! {count} arquivos .pt gerados em {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
