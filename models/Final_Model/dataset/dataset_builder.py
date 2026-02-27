import torch
import torch.nn as nn
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
import sys
import os


# ==============================================================================
# CONFIGURAÇÃO DE IMPORTS 
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
models_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
vae_path = os.path.join(models_root, "VAE")
text_path = os.path.join(models_root, "Text_Encoder")

# Adiciona ao sistema
sys.path.append(vae_path)
sys.path.append(text_path)
sys.path.append(models_root)

try:
    from vae_video import VAE 
    from Text_Encoder import TextEncoder, encoding_function
    from latent_fusion import SpatialBroadcastFuser

    print("OK: Sem erros de importação")

except ImportError as e:
    print(f"\nERRO DE IMPORTAÇÃO: {e}")
    sys.exit()


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Caminhos
BASE_DIR = current_dir
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "scenario1")
INPUT_FOLDER_JSON = os.path.join(DATASET_ROOT, "inputs")
INPUT_FOLDER_VIDEO = os.path.join(DATASET_ROOT, "videos")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "dataset_processado")

# Caminho dos pesos
VAE_WEIGHTS = os.path.join(vae_path, "vae_weights.pth")
TEXT_WEIGHTS = os.path.join(text_path, "text_encoder_weights.pth")

TARGET_FPS = 60.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo {DEVICE}")


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def load_models():
    """Carrega VAE, TextEncoder e Fuser com pesos pré-treinados."""
    
    # 1. VAE
    vae = VAE().to(DEVICE)
    if os.path.exists(VAE_WEIGHTS):
        vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=DEVICE))
        print("Pesos VAE carregados")
    else:
        print(f"AVISO: Pesos VAE não encontrados em {VAE_WEIGHTS}")
        sys.exit()
    vae.eval()
    
    # 2. Text Encoder
    text_enc = TextEncoder().to(DEVICE)
    if os.path.exists(TEXT_WEIGHTS):
        text_enc.load_state_dict(torch.load(TEXT_WEIGHTS, map_location=DEVICE))
        print("Pesos Text Encoder carregados.")
    else:
        print(f"AVISO: Pesos Text Encoder não encontrados em {TEXT_WEIGHTS}")
        sys.exit()
    text_enc.eval()
    
    # 3. Fuser
    fuser = SpatialBroadcastFuser(height=8, width=8).to(DEVICE)
    
    return vae, text_enc, fuser


def map_actions_to_frames(json_path, fps):
    """
    Lê o JSON e cria um dicionário {frame_index: action_data}.
    """
    # CORRIGIDO: agora abre o arquivo json_path corretamente
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
    
    # Normalizar [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Tensor (Batch=1, Channel=1, H, W)
    tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def process_step(frame_curr, frame_next, action_data, models):
    """
    Recebe frames brutos e dados da ação.
    Retorna:
       - Input X: Latente Fundido (24, 8, 8)
       - Target Y: Latente Visual Futuro (8, 8, 8)
    """
    vae, text_enc, fuser = models
    
    # Preparar Tensores de Imagem
    t_curr = preprocess_frame(frame_curr)
    t_next = preprocess_frame(frame_next)
    
    # Preparar Tensor de Ação
    # Chama encoding_function, do encoder de texto
    vec_action = encoding_function(action_data).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Codificar Visual Atual (t) no VAE
        mu_curr, _ = vae.encode(t_curr)
        
        # Codificar Visual Futuro (t+1), o alvo
        mu_next, _ = vae.encode(t_next)
        
        # Codificar Texto/Ação
        emb_action = text_enc(vec_action)
        
        # Fusão (Spatial Broadcast)
        z_fused = fuser(mu_curr, emb_action)
        
    return z_fused.squeeze(0).cpu(), mu_next.squeeze(0).cpu()


def process_video_sequence(video_path, json_path, models):
    """
    Itera sobre todos os frames do vídeo, chama process_step e empilha os resultados.
    """
    # Abrir Vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return None

    # Usa FPS real do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Carregar Mapa de Ações
    action_map = map_actions_to_frames(json_path, fps)
    
    inputs_list = []
    targets_list = []
    
    frame_idx = 0
    ret, frame_curr = cap.read()
    
    if not ret:
        cap.release()
        return None

    # Loop Frame a Frame
    while True:
        ret, frame_next = cap.read()
        # Se chegou ao fim do vídeo, descarta ultimo frame como entrada
        if not ret:
            break
        
        # Pega a ação correspondente a este frame (ou None)
        current_action = action_map.get(frame_idx, None)
        
        # Chama função auxiliar que processa o passo individual
        x, y = process_step(frame_curr, frame_next, current_action, models)
        
        inputs_list.append(x)
        targets_list.append(y)
        
        # Avança
        frame_curr = frame_next
        frame_idx += 1
        
    cap.release()
    
    if len(inputs_list) == 0:
        return None
        
    # Empilha tudo em um tensor grande (Time, Channels, H, W)
    return {
        'x': torch.stack(inputs_list),
        'y': torch.stack(targets_list)
    }


# ==============================================================================
# LOOP PRINCIPAL (MAIN)
# ==============================================================================

def main():
    # 1. Carregar Modelos
    models = load_models()
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 2. Listar Arquivos
    json_files = [f for f in os.listdir(INPUT_FOLDER_JSON) if f.endswith('.json')]
    print(f"Iniciando processamento de {len(json_files)} simulações...")
    
    count = 0
    
    for json_file in tqdm(json_files):
        # Encontra o vídeo correspondente
        prefix = json_file.replace('.json', '')
        video_file = None
        for v in os.listdir(INPUT_FOLDER_VIDEO):
            if v.startswith(prefix) and v.endswith('.mp4'):
                video_file = v
                break
        
        if not video_file:
            continue
            
        full_video_path = os.path.join(INPUT_FOLDER_VIDEO, video_file)
        full_json_path = os.path.join(INPUT_FOLDER_JSON, json_file)
        
        # 3. Processar Vídeo
        try:
            dataset_tensors = process_video_sequence(full_video_path, full_json_path, models)
        except Exception as e:
            print(f"Erro processando {json_file}: {e}")
            continue
        
        if dataset_tensors:
            # 4. Salvar Resultado
            save_name = json_file.replace('.json', '.pt')
            torch.save(dataset_tensors, os.path.join(OUTPUT_FOLDER, save_name))
            count += 1
            
    print(f"Sucesso! {count} arquivos .pt gerados em {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()