"""
World Model VAE
===============

Terceira e ultima etapa do pipeline.

Pipeline completo:
  Simulacao -> video + JSON de acoes
  video + JSON -> VAE.encode + TextEncoder + Fuser -> dataset_processado (.pt)
  dataset_processado -> WorldModelVAE -> predicao de latente -> VAE.decode -> frame gerado

Este modulo:
  1. Define o WorldModelVAE: aprende a mapear z_fused (24,8,8) -> z_next (8,8,8)
  2. Treina o modelo no dataset_processado
  3. Gera novos frames decodificando as predicoes pelo VAE original
"""

import os
import sys

# Evita crash do Qt em ambientes sem display (headless)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


# ==============================================================================
# CONFIGURACAO DE CAMINHOS
# ==============================================================================

current_dir  = os.path.dirname(os.path.abspath(__file__))
models_root  = os.path.abspath(os.path.join(current_dir, ".."))
vae_path     = os.path.join(models_root, "VAE")
text_path    = os.path.join(models_root, "Text_Encoder")

sys.path.append(vae_path)
sys.path.append(text_path)
sys.path.append(models_root)

try:
    from vae_video import VAE
    from Text_Encoder import TextEncoder, encoding_function
    from latent_fusion import SpatialBroadcastFuser
    print("OK: VAE, TextEncoder e Fuser importados com sucesso")
except ImportError as e:
    print(f"ERRO DE IMPORTACAO: {e}")
    sys.exit()


# ==============================================================================
# CONFIGURACOES
# ==============================================================================

DATASET_FOLDER = os.path.join(current_dir, "dataset", "dataset_processado")
VAE_WEIGHTS    = os.path.join(vae_path, "vae_weights.pth")
TEXT_WEIGHTS   = os.path.join(text_path, "text_encoder_weights.pth")
OUTPUT_DIR     = os.path.join(current_dir, "world_model_outputs")
WEIGHTS_PATH   = os.path.join(current_dir, "world_model_weights.pth")

# Dimensoes (devem coincidir com o que o dataset_builder gerou)
# x = cat([vae_enc_output(16), action_embedding(16)]) => 32 canais
# y = vae_enc_output completo (16 canais = mu + logvar concatenados)
FUSED_CHANNELS  = 32    # canais de z_fused  (entrada do WorldModelVAE)
VISUAL_CHANNELS = 16    # canais de z_visual (saida prevista)
SPATIAL_H       = 8
SPATIAL_W       = 8
LATENT_DIM      = 64    # dimensao do espaco latente interno

# Treinamento
BATCH_SIZE    = 64
NUM_EPOCHS    = 150
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")


# ==============================================================================
# DATASET
# ==============================================================================

SEQ_LEN = 32  # comprimento da janela de sequencia para TBPTT


class SequenceDataset(Dataset):
    """
    Dataset de SEQUENCIAS consecutivas para treinar o GRU com TBPTT.

    Cada arquivo .pt contem uma simulacao completa.
    O dataset expoe janelas de SEQ_LEN frames consecutivos (sem frames estaticos).

    Cada item retornado:
      x_seq: (SEQ_LEN, 32, 8, 8) — latentes fundidos consecutivos
      y_seq: (SEQ_LEN, 16, 8, 8) — latentes alvo consecutivos
    """

    def __init__(self, folder: str, seq_len: int = SEQ_LEN):
        self.sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.seq_len = seq_len

        pt_files = sorted(f for f in os.listdir(folder) if f.endswith(".pt"))
        if not pt_files:
            raise FileNotFoundError(f"Nenhum arquivo .pt encontrado em {folder}")

        STATIC_THRESHOLD = 0.01

        total_pairs = 0
        skipped     = 0
        print(f"Carregando {len(pt_files)} arquivos .pt (modo sequencia, L={seq_len})...")
        for fname in tqdm(pt_files):
            data  = torch.load(os.path.join(folder, fname), map_location="cpu", weights_only=True)
            x_full = data["x"]   # (T, 32, 8, 8)
            y_full = data["y"]   # (T, 16, 8, 8)
            T = x_full.shape[0]

            # Filtra frames estaticos e guarda indices validos
            valid_idx = [0]
            for t in range(1, T):
                diff = (y_full[t] - y_full[t - 1]).abs().mean().item()
                if diff < STATIC_THRESHOLD:
                    skipped += 1
                else:
                    valid_idx.append(t)

            # Constroi sub-sequencias consecutivas (stride = seq_len // 2)
            stride = max(1, seq_len // 2)
            for start in range(0, len(valid_idx) - seq_len + 1, stride):
                idxs = valid_idx[start : start + seq_len]
                # Sequencias sao validas somente se forem de frames consecutivos
                # (sem gaps causados pelo filtro de frames estaticos)
                if idxs[-1] - idxs[0] <= seq_len * 3:  # tolerancia de gaps
                    x_win = x_full[idxs]  # (SEQ_LEN, 32, 8, 8)
                    y_win = y_full[idxs]  # (SEQ_LEN, 16, 8, 8)
                    self.sequences.append((x_win, y_win))
                    total_pairs += seq_len

        print(
            f"Sequencias: {len(self.sequences)} | "
            f"Pares efetivos: {total_pairs} | "
            f"Frames estaticos descartados: {skipped}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# ==============================================================================
# ARQUITETURA DO WORLD MODEL VAE
# ==============================================================================

class ResBlock(nn.Module):
    """Bloco residual convolucional 1x1 -> 3x3 -> 1x1."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)


class WorldModelVAE(nn.Module):
    """
    VAE que prediz o latente visual futuro z_{t+1}
    a partir do latente fundido z_fused_t (visual_t + acao_t).

    Encoder: (B, 24, 8, 8) -> mu, logvar em R^LATENT_DIM
    Decoder: z in R^LATENT_DIM -> z_next_pred (B, 8, 8, 8)
    """

    def __init__(
        self,
        in_channels:    int = FUSED_CHANNELS,
        out_channels:   int = VISUAL_CHANNELS,
        latent_dim:     int = LATENT_DIM,
        spatial_h:      int = SPATIAL_H,
        spatial_w:      int = SPATIAL_W,
    ):
        super().__init__()

        self.latent_dim  = latent_dim
        self.out_channels = out_channels
        self.spatial_h   = spatial_h
        self.spatial_w   = spatial_w

        # --- Encoder ---
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )
        enc_flat = 128 * spatial_h * spatial_w
        self.fc_mu     = nn.Linear(enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat, latent_dim)

        # --- Decoder ---
        dec_hidden = 64 * spatial_h * spatial_w
        self.fc_dec = nn.Linear(latent_dim, dec_hidden)
        self.dec_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        """x: (B, 24, 8, 8) -> mu, logvar: (B, latent_dim)"""
        h = self.enc_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor):
        """z: (B, latent_dim) -> z_next: (B, 8, 8, 8)"""
        h = self.fc_dec(z).view(-1, 64, self.spatial_h, self.spatial_w)
        return self.dec_conv(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, -10, 10)
        z      = self.reparametrize(mu, logvar)
        z_next = self.decode(z)
        return z_next, mu, logvar


# ==============================================================================
# ARQUITETURA RECORRENTE COM GRU (TEMPORAL STATE)
# ==============================================================================

class WorldModelVAERecurrent(nn.Module):
    """
    Variante do WorldModelVAE com celula GRU entre encoder e decoder.

    A celula GRU mantém um estado oculto h_t que captura contexto temporal
    ao longo do rollout autoregressivo.

    Encoder: x_fused (B, 32, 8, 8) -> mu, logvar em R^LATENT_DIM
    GRU:     z + h_{t-1} -> h_t (R^LATENT_DIM)
    Decoder: h_t -> z_next (B, VISUAL_CHANNELS, 8, 8)
    """

    def __init__(
        self,
        in_channels:    int = FUSED_CHANNELS,
        out_channels:   int = VISUAL_CHANNELS,
        latent_dim:     int = LATENT_DIM,
        spatial_h:      int = SPATIAL_H,
        spatial_w:      int = SPATIAL_W,
    ):
        super().__init__()

        self.latent_dim   = latent_dim
        self.out_channels = out_channels
        self.spatial_h    = spatial_h
        self.spatial_w    = spatial_w

        # --- Encoder (igual ao WorldModelVAE) ---
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )
        enc_flat = 128 * spatial_h * spatial_w
        self.fc_mu     = nn.Linear(enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat, latent_dim)

        # --- GRU temporal ---
        self.gru = nn.GRUCell(latent_dim, latent_dim)

        # --- Decoder (igual ao WorldModelVAE) ---
        dec_hidden = 64 * spatial_h * spatial_w
        self.fc_dec = nn.Linear(latent_dim, dec_hidden)
        self.dec_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        h = self.enc_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor):
        h = self.fc_dec(z).view(-1, 64, self.spatial_h, self.spatial_w)
        return self.dec_conv(h)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        """
        Args:
            x      : (B, FUSED_CHANNELS, H, W)
            hidden : (B, LATENT_DIM) estado oculto anterior; None = zeros

        Returns:
            z_next  : (B, VISUAL_CHANNELS, H, W)
            mu      : (B, LATENT_DIM)
            logvar  : (B, LATENT_DIM)
            hidden  : (B, LATENT_DIM) novo estado oculto
        """
        B = x.size(0)
        if hidden is None:
            hidden = torch.zeros(B, self.latent_dim, device=x.device)

        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, -10, 10)
        z      = self.reparametrize(mu, logvar)

        hidden = self.gru(z, hidden)      # integra contexto temporal
        z_next = self.decode(hidden)
        return z_next, mu, logvar, hidden


# ==============================================================================
# LOSS
# ==============================================================================

# Peso do foreground (bolas escuras) em relacao ao background branco.
# background = pixels > FG_THRESHOLD; bolas = pixels <= FG_THRESHOLD
FG_THRESHOLD      = 0.8   # limiar de brilho: abaixo = foreground
FG_WEIGHT         = 15.0  # bolas valem 15x mais que o fundo
PIXEL_WEIGHT      = 5.0   # peso relativo do pixel loss vs latent MSE
DYNAMIC_THRESHOLD = 0.05  # diff de pixel minima para considerar movimento real


def world_model_loss(z_pred, z_target, mu, logvar, beta=1.0, vae=None, prev_frame_gt=None):
    """
    Loss = latent_MSE + PIXEL_WEIGHT * fg_weighted_pixel_MSE + beta * KL

    - latent_MSE         : MSE entre z_pred e z_target no espaco latente
    - fg_weighted_pixel  : decodifica pelo VAE e penaliza mais pixels escuros
                           que sao DINAMICOS (bolas em movimento). O chao e
                           escuro mas estatico, entao nao recebe peso extra.
    - beta pequeno (0.01-0.10): modelo preditor, nao generativo

    vae          : VAE congelado para computar pixel loss (opcional; se None, ignora)
    prev_frame_gt: frame anterior decodificado (B,1,64,64) para mascara dinamica
    Retorna: (total, recon, actual_kl, frame_gt)
             frame_gt e None se vae=None, senao (B,1,64,64) para proximo passo
    """
    B = z_pred.size(0)

    # Latent MSE
    recon = F.mse_loss(z_pred, z_target, reduction="sum") / B

    # Pixel loss ponderado por foreground dinamico
    frame_gt   = None
    pixel_loss = torch.tensor(0.0, device=z_pred.device)
    if vae is not None:
        mu_pred, _  = torch.chunk(z_pred,   2, dim=1)  # (B, 8, 8, 8)
        mu_gt,   _  = torch.chunk(z_target, 2, dim=1)  # (B, 8, 8, 8)
        with torch.no_grad():
            frame_gt  = vae.decoder(mu_gt.detach())           # (B, 1, 64, 64)
            dark_mask = (frame_gt < FG_THRESHOLD).float()     # escuro = bola ou chao
            if prev_frame_gt is not None:
                # Peso extra apenas em pixels escuros QUE MUDARAM — exclui chao estatico
                dynamic_mask = (frame_gt - prev_frame_gt).abs() > DYNAMIC_THRESHOLD
                fg_mask = dark_mask * dynamic_mask.float()
            else:
                fg_mask = dark_mask  # primeiro frame da sequencia: sem frame anterior
            weight = 1.0 + (FG_WEIGHT - 1.0) * fg_mask       # (B, 1, 64, 64)
        frame_pred = vae.decoder(mu_pred)                     # (B, 1, 64, 64) — diferenciavel
        pixel_diff = (frame_pred - frame_gt) ** 2
        pixel_loss = (pixel_diff * weight).sum() / (B * frame_gt.numel() / B)

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    actual_kl  = kl_per_dim.sum() / B
    kl         = torch.clamp(kl_per_dim - 0.1, min=0.0).sum() / B

    total = recon + PIXEL_WEIGHT * pixel_loss + beta * kl
    return total, recon, actual_kl, frame_gt


# ==============================================================================
# TREINAMENTO
# ==============================================================================

def _tbptt_step(model, x_seq, y_seq, beta, optimizer, is_recurrent, vae=None):
    """
    Executa um passo de TBPTT (Truncated Backprop Through Time) em uma sequencia.

    x_seq : (B, SEQ_LEN, 32, 8, 8)
    y_seq : (B, SEQ_LEN, 16, 8, 8)
    vae   : VAE congelado para pixel loss ponderado (opcional)
    """
    B, S = x_seq.shape[:2]
    hidden        = None
    prev_frame_gt = None
    total_loss = total_recon = total_kl = 0.0

    optimizer.zero_grad()

    for t in range(S):
        x_t = x_seq[:, t]
        y_t = y_seq[:, t]

        if is_recurrent:
            z_pred, mu, logvar, hidden = model(x_t, hidden)
            hidden = hidden.detach()
        else:
            z_pred, mu, logvar = model(x_t)

        loss, recon, kl, frame_gt_curr = world_model_loss(
            z_pred, y_t, mu, logvar, beta, vae=vae, prev_frame_gt=prev_frame_gt
        )
        if frame_gt_curr is not None:
            prev_frame_gt = frame_gt_curr.detach()
        (loss / S).backward()

        total_loss  += loss.item()
        total_recon += recon.item()
        total_kl    += kl.item()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()

    return total_loss / S, total_recon / S, total_kl / S


def train_world_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, vae=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    history = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss":   [], "val_recon":   [], "val_kl":   [],
        "lr":         [],
    }
    best_val_recon = float("inf")
    is_recurrent   = isinstance(model, WorldModelVAERecurrent)

    print(f"\n{'='*60}")
    print("Treinamento World Model VAE" + (" (TBPTT)" if is_recurrent else ""))
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dispositivo: {DEVICE} | Epocas: {num_epochs}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        # beta pequeno e fixo: modelo preditor, nao generativo
        # sobe de 0.01 ate 0.10 nos primeiros 10 epocas e fica la
        beta = min(0.10, 0.01 * (epoch + 1))

        # --- Treino ---
        model.train()
        t_loss = t_recon = t_kl = 0.0

        for x_seq, y_seq in train_loader:
            x_seq = x_seq.to(DEVICE)
            y_seq = y_seq.to(DEVICE)

            if is_recurrent:
                # x_seq: (B, SEQ_LEN, 32, 8, 8) — treina com TBPTT
                loss, recon, kl = _tbptt_step(
                    model, x_seq, y_seq, beta, optimizer, is_recurrent=True, vae=vae
                )
            else:
                # Para o modelo nao-recorrente, processa cada passo do batch
                B, S = x_seq.shape[:2]
                x_flat = x_seq.view(B * S, *x_seq.shape[2:])
                y_flat = y_seq.view(B * S, *y_seq.shape[2:])
                z_pred, mu, logvar = model(x_flat)
                loss_t, recon_t, kl_t, _ = world_model_loss(z_pred, y_flat, mu, logvar, beta, vae=vae)
                optimizer.zero_grad()
                loss_t.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                loss, recon, kl = loss_t.item(), recon_t.item(), kl_t.item()

            t_loss  += loss
            t_recon += recon
            t_kl    += kl

        n = len(train_loader)
        t_loss /= n; t_recon /= n; t_kl /= n

        # --- Validacao ---
        model.eval()
        v_loss = v_recon = v_kl = 0.0

        with torch.no_grad():
            for x_seq, y_seq in val_loader:
                x_seq = x_seq.to(DEVICE)
                y_seq = y_seq.to(DEVICE)

                B, S = x_seq.shape[:2]
                hidden_val = None

                for t in range(S):
                    x_t = x_seq[:, t]
                    y_t = y_seq[:, t]
                    if is_recurrent:
                        z_pred, mu, logvar, hidden_val = model(x_t, hidden_val)
                    else:
                        z_pred, mu, logvar = model(x_t)
                    loss_t, recon_t, kl_t, _ = world_model_loss(z_pred, y_t, mu, logvar, beta, vae=None)
                    v_loss  += loss_t.item() / S
                    v_recon += recon_t.item() / S
                    v_kl    += kl_t.item() / S

        nv = len(val_loader)
        v_loss /= nv; v_recon /= nv; v_kl /= nv

        scheduler.step(v_recon)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(t_loss)
        history["train_recon"].append(t_recon)
        history["train_kl"].append(t_kl)
        history["val_loss"].append(v_loss)
        history["val_recon"].append(v_recon)
        history["val_kl"].append(v_kl)
        history["lr"].append(lr)

        print(
            f"Epoca [{epoch+1:03d}/{num_epochs}] | "
            f"Train L={t_loss:.4f} R={t_recon:.4f} KL={t_kl:.4f} | "
            f"Val L={v_loss:.4f} R={v_recon:.4f} | "
            f"b={beta:.3f} LR={lr:.2e}"
        )

        if v_recon < best_val_recon:
            best_val_recon = v_recon
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> Melhor modelo salvo (Val Recon: {v_recon:.4f})")

    print(f"\nTreinamento concluido. Melhor Val Recon: {best_val_recon:.4f}")
    return history


# ==============================================================================
# GERACAO DE FRAMES
# ==============================================================================

def load_original_vae() -> VAE:
    """Carrega o VAE original (necessario para decodificar latentes em pixels)."""
    vae = VAE().to(DEVICE)
    if not os.path.exists(VAE_WEIGHTS):
        raise FileNotFoundError(f"Pesos do VAE nao encontrados: {VAE_WEIGHTS}")
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=DEVICE, weights_only=True))
    vae.eval()
    print("VAE original carregado.")
    return vae


def generate_frames(
    world_model: WorldModelVAE,
    vae: VAE,
    x_fused_seq: torch.Tensor,
    num_steps: int = None,
) -> list[np.ndarray]:
    """
    Gera uma sequencia de frames a partir de latentes fundidos.

    Args:
        world_model  : WorldModelVAE treinado
        vae          : VAE original (para decodificacao pixel)
        x_fused_seq  : (T, 24, 8, 8) - sequencia de latentes fundidos
        num_steps    : quantos frames gerar (padrao: todos)

    Returns:
        lista de arrays numpy (64, 64) com valores em [0, 1]
    """
    world_model.eval()
    vae.eval()

    T = x_fused_seq.shape[0]
    if num_steps is None:
        num_steps = T

    is_recurrent = isinstance(world_model, WorldModelVAERecurrent)
    hidden = None

    frames = []
    with torch.no_grad():
        for t in range(min(num_steps, T)):
            x_t = x_fused_seq[t].unsqueeze(0).to(DEVICE)  # (1, FUSED, 8, 8)

            # Prediz o latente do proximo frame (modo deterministico: usa mu)
            if is_recurrent:
                enc_next, _, _, hidden = world_model(x_t, hidden)
            else:
                mu, _ = world_model.encode(x_t)
                enc_next = world_model.decode(mu)          # (1, 16, 8, 8)

            # O VAE decoder espera z de 8 canais (mu apos reparametrizacao)
            # enc_next tem 16 canais (mu + logvar concatenados)
            mu_next, _ = torch.chunk(enc_next, 2, dim=1)  # (1, 8, 8, 8)

            # Decodifica o latente em pixels pelo VAE original
            frame_tensor = vae.decoder(mu_next)           # (1, 1, 64, 64)
            frame_np     = frame_tensor.squeeze().cpu().numpy()  # (64, 64)
            frames.append(frame_np)

    return frames


def _preprocess_frame(frame_np: np.ndarray) -> torch.Tensor:
    """Converte frame numpy (H, W) [0,1] para tensor (1, 1, 64, 64) no DEVICE."""
    img = cv2.resize(frame_np, (64, 64))
    tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def load_rollout_models(world_model: WorldModelVAE):
    """
    Carrega todos os modelos necessarios para o rollout autoregressivo.
    Retorna (world_model, vae, text_enc, fuser).
    """
    vae = load_original_vae()

    text_enc = TextEncoder().to(DEVICE)
    if not os.path.exists(TEXT_WEIGHTS):
        raise FileNotFoundError(f"Pesos do TextEncoder nao encontrados: {TEXT_WEIGHTS}")
    text_enc.load_state_dict(torch.load(TEXT_WEIGHTS, map_location=DEVICE, weights_only=True))
    text_enc.eval()
    print("TextEncoder carregado.")

    fuser = SpatialBroadcastFuser(height=SPATIAL_H, width=SPATIAL_W).to(DEVICE)

    return vae, text_enc, fuser


def autoregressive_rollout(
    world_model: WorldModelVAE,
    vae: VAE,
    text_enc,
    fuser,
    initial_frame: np.ndarray,
    actions: list,
) -> list[np.ndarray]:
    """
    Gera uma sequencia de frames de forma autoregressiva (ciclo completo).

    A cada passo t:
      frame_t -> VAE.encoder -> z_visual_t (16, 8, 8)
      action_t -> TextEncoder -> z_action_t (16,)
      fuser(z_visual_t, z_action_t) -> x_fused_t (32, 8, 8)
      WorldModelVAE(x_fused_t) -> z_next_pred (16, 8, 8)
      VAE.decoder(z_next_pred[:8]) -> frame_{t+1}  <- vira input do proximo passo

    Args:
        world_model  : WorldModelVAE treinado
        vae          : VAE original
        text_enc     : TextEncoder treinado
        fuser        : SpatialBroadcastFuser
        initial_frame: array numpy (64, 64) em [0, 1] — frame inicial real
        actions      : lista de dicts de acao (ou None para sem acao) por passo

    Returns:
        lista de arrays numpy (64, 64) com os frames gerados (sem o frame inicial)
    """
    world_model.eval()
    vae.eval()
    text_enc.eval()

    is_recurrent = isinstance(world_model, WorldModelVAERecurrent)
    hidden = None

    frames = []
    current_frame = initial_frame.copy()

    with torch.no_grad():
        for action_data in actions:
            # 1. Encodar frame atual com o VAE
            frame_t = _preprocess_frame(current_frame)       # (1, 1, 64, 64)
            z_visual = vae.encoder(frame_t)                  # (1, 16, 8, 8)

            # 2. Encodar acao
            vec_action = encoding_function(action_data).unsqueeze(0).to(DEVICE)
            z_action   = text_enc(vec_action)                # (1, 16)

            # 3. Fundir visual + acao
            x_fused = fuser(z_visual, z_action)              # (1, 32, 8, 8)

            # 4. Predizer latente do proximo frame
            if is_recurrent:
                enc_next, _, _, hidden = world_model(x_fused, hidden)
            else:
                mu_wm, _ = world_model.encode(x_fused)
                enc_next  = world_model.decode(mu_wm)        # (1, 16, 8, 8)

            # 5. Decodificar em pixels (VAE decoder espera 8 canais = mu)
            mu_next, _ = torch.chunk(enc_next, 2, dim=1)     # (1, 8, 8, 8)
            frame_next  = vae.decoder(mu_next)               # (1, 1, 64, 64)

            # 6. Converter para numpy e usar como proximo frame
            current_frame = frame_next.squeeze().cpu().numpy()
            frames.append(current_frame.copy())

    return frames


def save_frames_as_video(frames: list[np.ndarray], output_path: str, fps: int = 30):
    """Salva lista de frames numpy (grayscale) como video MP4."""
    h, w = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)
    for frame in frames:
        writer.write((frame * 255).clip(0, 255).astype(np.uint8))
    writer.release()
    print(f"Video salvo em: {output_path}")


def save_frames_as_grid(
    frames: list[np.ndarray], output_path: str, cols: int = 10
):
    """Salva frames como grade de imagens PNG."""
    n    = len(frames)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(frames[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Grade de frames salva em: {output_path}")


# ==============================================================================
# CURVAS DE TREINAMENTO
# ==============================================================================

def plot_training_curves(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss Total")
    axes[0].set_xlabel("Epoca")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_recon"], label="Train")
    axes[1].plot(epochs, history["val_recon"],   label="Val")
    axes[1].set_title("Reconstruction Loss (MSE)")
    axes[1].set_xlabel("Epoca")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_kl"], label="Train KL")
    axes[2].set_title("KL Divergencia")
    axes[2].set_xlabel("Epoca")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "world_model_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Curvas de treinamento salvas em: {path}")


# ==============================================================================
# AVALIACAO VISUAL: REAL vs PREDITO
# ==============================================================================

def evaluate_visual(
    world_model: WorldModelVAE,
    vae: VAE,
    val_loader: DataLoader,
    save_dir: str,
    num_samples: int = 20,
):
    """
    Gera um grid comparando frames reais vs frames preditos pelo modelo.

    Para cada amostra do conjunto de validacao:
      - Passa x_fused pelo WorldModelVAE para obter z_next_pred
      - Decoda z_next_pred pelo VAE original -> frame predito
      - Decoda y (ground truth) pelo VAE original -> frame real
      - Exibe ambos lado a lado com o MSE pixel

    Salva a grade como PNG em save_dir/visual_comparison.png.
    """
    world_model.eval()
    vae.eval()

    samples_x, samples_y = [], []
    for x, y in val_loader:
        # Se for dataset de sequencias (B, SEQ_LEN, C, H, W), usa apenas t=0
        if x.dim() == 5:
            x = x[:, 0]  # (B, C, H, W)
            y = y[:, 0]
        samples_x.append(x)
        samples_y.append(y)
        if sum(s.shape[0] for s in samples_x) >= num_samples:
            break

    x_batch = torch.cat(samples_x, dim=0)[:num_samples].to(DEVICE)
    y_batch = torch.cat(samples_y, dim=0)[:num_samples].to(DEVICE)

    is_recurrent = isinstance(world_model, WorldModelVAERecurrent)

    with torch.no_grad():
        # Predicao do world model (modo deterministico: usa mu)
        if is_recurrent:
            enc_pred, _, _, _ = world_model(x_batch)
        else:
            mu_wm, _ = world_model.encode(x_batch)
            enc_pred  = world_model.decode(mu_wm)     # (N, 16, 8, 8)
        mu_pred, _ = torch.chunk(enc_pred, 2, dim=1)  # (N, 8, 8, 8) = mu

        # Ground truth
        mu_gt, _ = torch.chunk(y_batch, 2, dim=1)     # (N, 8, 8, 8)

        # Decodifica em pixels
        frames_pred = vae.decoder(mu_pred)             # (N, 1, 64, 64)
        frames_real = vae.decoder(mu_gt)               # (N, 1, 64, 64)

    frames_pred_np = frames_pred.squeeze(1).cpu().numpy()  # (N, 64, 64)
    frames_real_np = frames_real.squeeze(1).cpu().numpy()  # (N, 64, 64)

    mse_per = ((frames_pred_np - frames_real_np) ** 2).mean(axis=(1, 2))
    mean_mse = mse_per.mean()

    # Grid: cada coluna = (real, predito) para um sample
    cols = min(10, num_samples)
    rows = 2 * ((num_samples + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(rows, cols)

    sample_idx = 0
    for row_pair in range(rows // 2):
        for col in range(cols):
            if sample_idx >= num_samples:
                axes[row_pair * 2,     col].axis("off")
                axes[row_pair * 2 + 1, col].axis("off")
                continue
            ax_real = axes[row_pair * 2,     col]
            ax_pred = axes[row_pair * 2 + 1, col]
            ax_real.imshow(frames_real_np[sample_idx], cmap="gray", vmin=0, vmax=1)
            ax_real.set_title(f"Real {sample_idx}", fontsize=6)
            ax_real.axis("off")
            ax_pred.imshow(frames_pred_np[sample_idx], cmap="gray", vmin=0, vmax=1)
            ax_pred.set_title(f"MSE={mse_per[sample_idx]:.4f}", fontsize=6)
            ax_pred.axis("off")
            sample_idx += 1

    plt.suptitle(f"Real (topo) vs Predito (base) | MSE medio: {mean_mse:.4f}", fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, "visual_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Comparacao visual salva em: {path} | MSE medio: {mean_mse:.4f}")
    return mean_mse


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dataset
    if not os.path.exists(DATASET_FOLDER):
        print(f"ERRO: dataset_processado nao encontrado: {DATASET_FOLDER}")
        print("Execute primeiro models/Final_Model/dataset/dataset_builder.py")
        return

    dataset    = SequenceDataset(DATASET_FOLDER, seq_len=SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {train_size} sequencias | Val: {val_size} sequencias")

    # 2. Carrega VAE congelado para pixel loss ponderado durante treino
    vae_train = load_original_vae()
    print(f"Pixel loss: FG_WEIGHT={FG_WEIGHT}x | threshold={FG_THRESHOLD} | peso={PIXEL_WEIGHT}")

    # 3. Criar e treinar World Model VAE (versao recorrente com GRU)
    model = WorldModelVAERecurrent().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"WorldModelVAERecurrent: {total_params:,} parametros")

    history = train_world_model(
        model, train_loader, val_loader, num_epochs=NUM_EPOCHS, vae=vae_train
    )

    # 4. Curvas
    plot_training_curves(history, OUTPUT_DIR)

    # 5. Carregar melhor modelo e modelos auxiliares
    print("\nCarregando modelos para geracao...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    vae, text_enc, fuser = load_rollout_models(model)

    # 5.5 Avaliacao visual: grade real vs predito
    evaluate_visual(model, vae, val_loader, OUTPUT_DIR, num_samples=20)

    # 6. Buscar frame inicial e acoes do dataset de simulacao
    project_root   = os.path.abspath(os.path.join(current_dir, "..", ".."))
    video_folder   = os.path.join(project_root, "dataset", "scenario1", "videos")
    json_folder    = os.path.join(project_root, "dataset", "scenario1", "inputs")

    video_files = sorted(f for f in os.listdir(video_folder) if f.endswith(".mp4"))
    json_files  = sorted(f for f in os.listdir(json_folder)  if f.endswith(".json"))

    if not video_files or not json_files:
        print("AVISO: Videos ou JSONs nao encontrados. Pulando rollout autoregressivo.")
        print(f"\nPronto! Resultados salvos em: {OUTPUT_DIR}")
        return

    # Pega o primeiro video/json como exemplo
    video_path = os.path.join(video_folder, video_files[0])
    json_path  = os.path.join(json_folder,  json_files[0])

    # Extrai primeiro frame real do video
    cap = cv2.VideoCapture(video_path)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        print("AVISO: Nao foi possivel ler o video. Pulando rollout.")
        print(f"\nPronto! Resultados salvos em: {OUTPUT_DIR}")
        return

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    initial_frame = cv2.resize(frame_gray, (64, 64)).astype(np.float32) / 255.0

    # Carrega acoes do JSON
    import json
    with open(json_path) as f:
        actions_list = json.load(f)

    TARGET_FPS = 60.0
    action_map = {int(a["time"] * TARGET_FPS): a for a in actions_list}
    num_steps  = 120
    actions    = [action_map.get(t, None) for t in range(num_steps)]

    # 6. Rollout autoregressivo
    print(f"Rollout autoregressivo: {num_steps} passos a partir do frame inicial real...")
    frames_ar = autoregressive_rollout(model, vae, text_enc, fuser, initial_frame, actions)

    save_frames_as_grid(
        frames_ar,
        os.path.join(OUTPUT_DIR, "rollout_frames_grid.png"),
        cols=10,
    )
    save_frames_as_video(
        frames_ar,
        os.path.join(OUTPUT_DIR, "rollout_video.mp4"),
        fps=30,
    )

    print(f"\nPronto! Resultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
