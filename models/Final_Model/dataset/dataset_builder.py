import torch
import torch.nn as nn
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
import sys
import os


###########################################################
# IMPORTAÇÕES E DEFINIÇÕES DE CLASSES
###########################################################

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
