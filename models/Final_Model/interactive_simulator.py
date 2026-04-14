"""
Simulador Interativo - World Model
===================================

Usa o WorldModelVAERecurrent treinado para prever frames em tempo real.
A cada passo, o modelo prediz o proximo frame dado o frame atual + acao do usuario.

Controles:
  SPACE / ENTER  : avanca um frame (aplica acao pendente se houver)
  Click esquerdo : define acao mouse_down na posicao clicada
  Click direito  : cancela acao pendente
  R              : reinicia do frame inicial
  A              : avanca automaticamente (sem acao) ate pressionar A novamente
  S              : salva o frame atual como PNG
  ESC / Q        : encerra
"""

import os
import sys

# Pygame usa SDL (nao Qt), mas cv2 precisa da flag para nao abrir janela Qt
os.environ.setdefault("SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR", "0")

try:
    import pygame
except ImportError:
    print("pygame nao encontrado. Instale com: pip install pygame")
    sys.exit(1)

import cv2
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Configuracao de caminhos
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
models_root = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, os.path.join(models_root, "VAE"))
sys.path.insert(0, os.path.join(models_root, "Text_Encoder"))
sys.path.insert(0, models_root)
sys.path.insert(0, current_dir)

from vae_video import VAE
from Text_Encoder import TextEncoder, encoding_function
from latent_fusion import SpatialBroadcastFuser

# Importa constantes e arquitetura do world model (sem executar main)
from world_model_vae import (
    WorldModelVAERecurrent,
    DEVICE,
    SPATIAL_H,
    SPATIAL_W,
)

# -----------------------------------------------------------------------------
# Constantes de display
# -----------------------------------------------------------------------------
FRAME_W       = 64          # tamanho interno do frame do modelo
FRAME_H       = 64
SCALE         = 8           # fator de escala: 64 * 8 = 512px
RENDER_W      = FRAME_W * SCALE   # 512
RENDER_H      = FRAME_H * SCALE   # 512
SIDEBAR_W     = 260
WINDOW_W      = RENDER_W + SIDEBAR_W
WINDOW_H      = RENDER_H + 40     # 40px de header

SIM_W         = 800         # espaco de simulacao original (para mapear clicks)
SIM_H         = 600

# Offset da area de render dentro da janela
RENDER_X      = 0
RENDER_Y      = 40          # abaixo do header

# Caminhos dos pesos
WEIGHTS_PATH  = os.path.join(current_dir, "world_model_weights.pth")
VAE_WEIGHTS   = os.path.join(models_root, "VAE", "vae_weights.pth")
TEXT_WEIGHTS  = os.path.join(models_root, "Text_Encoder", "text_encoder_weights.pth")

# Cores
C_BG          = (25,  25,  35)
C_HEADER      = (40,  40,  55)
C_SIDEBAR     = (35,  35,  48)
C_TEXT        = (210, 210, 220)
C_DIM         = (110, 110, 130)
C_ACTION      = (255,  80,  80)
C_GREEN       = ( 80, 220, 100)
C_BORDER      = ( 70,  70,  90)
C_KEY         = (180, 180, 200)


# =============================================================================
# Carregamento de modelos
# =============================================================================

def load_models():
    print("Carregando modelos...")

    wm = WorldModelVAERecurrent().to(DEVICE)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Pesos do World Model nao encontrados: {WEIGHTS_PATH}\n"
            "Execute o treinamento primeiro: python3 world_model_vae.py"
        )
    wm.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    wm.eval()
    print(f"  WorldModelVAERecurrent  [{DEVICE}]")

    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=DEVICE, weights_only=True))
    vae.eval()
    print("  VAE original")

    text_enc = TextEncoder().to(DEVICE)
    text_enc.load_state_dict(torch.load(TEXT_WEIGHTS, map_location=DEVICE, weights_only=True))
    text_enc.eval()
    print("  TextEncoder")

    fuser = SpatialBroadcastFuser(height=SPATIAL_H, width=SPATIAL_W).to(DEVICE)
    print("Modelos prontos.\n")

    return wm, vae, text_enc, fuser


# =============================================================================
# World Model step
# =============================================================================

def model_step(
    wm, vae, text_enc, fuser,
    frame_np: np.ndarray,
    action_data: dict | None,
    hidden: torch.Tensor | None,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Executa um passo do world model.

    Args:
        frame_np    : (64, 64) float32 [0, 1] — frame atual
        action_data : dict de acao (com pos em coords SIM) ou None
        hidden      : estado oculto do GRU (ou None para inicializar)

    Returns:
        (next_frame_np, new_hidden)
    """
    with torch.no_grad():
        # Encode frame com VAE
        img = cv2.resize(frame_np, (64, 64))
        frame_t = (
            torch.from_numpy(img.astype(np.float32))
            .unsqueeze(0).unsqueeze(0)
            .to(DEVICE)
        )
        z_visual = vae.encoder(frame_t)              # (1, 16, 8, 8)

        # Encode acao
        vec_action = encoding_function(action_data).unsqueeze(0).to(DEVICE)
        z_action   = text_enc(vec_action)            # (1, 16)

        # Funde visual + acao
        x_fused = fuser(z_visual, z_action)          # (1, 32, 8, 8)

        # Prediz proximo latente com GRU
        enc_next, _, _, new_hidden = wm(x_fused, hidden)
        mu_next, _ = torch.chunk(enc_next, 2, dim=1) # (1, 8, 8, 8)

        # Decodifica em pixels
        frame_next = vae.decoder(mu_next)            # (1, 1, 64, 64)

    next_np = frame_next.squeeze().cpu().numpy().clip(0.0, 1.0)
    return next_np, new_hidden


# =============================================================================
# Utilitarios de display
# =============================================================================

def frame_to_surface(frame_np: np.ndarray) -> pygame.Surface:
    """Converte frame numpy (H, W) [0,1] em Surface pygame escalada."""
    scaled = cv2.resize(
        frame_np, (RENDER_W, RENDER_H), interpolation=cv2.INTER_NEAREST
    )
    rgb = (scaled * 255).clip(0, 255).astype(np.uint8)
    rgb3 = np.stack([rgb, rgb, rgb], axis=-1)          # (H, W, 3)
    # pygame.surfarray espera (W, H, 3)
    return pygame.surfarray.make_surface(rgb3.transpose(1, 0, 2))


def draw_text(surface, font, text, pos, color=C_TEXT):
    surf = font.render(text, True, color)
    surface.blit(surf, pos)


def display_click_to_sim(display_x, display_y):
    """Converte coordenadas de display (dentro da area de render) para coords SIM."""
    rel_x = (display_x - RENDER_X) / RENDER_W
    rel_y = (display_y - RENDER_Y) / RENDER_H
    return rel_x * SIM_W, rel_y * SIM_H


def sim_to_display(sim_x, sim_y):
    """Converte coords SIM para coordenadas de display."""
    dx = int(RENDER_X + sim_x / SIM_W * RENDER_W)
    dy = int(RENDER_Y + sim_y / SIM_H * RENDER_H)
    return dx, dy


def get_initial_frame() -> np.ndarray:
    """Tenta carregar primeiro frame de um video real; senao retorna frame em branco."""
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    video_folder = os.path.join(project_root, "dataset", "scenario1", "videos")

    if os.path.exists(video_folder):
        videos = sorted(f for f in os.listdir(video_folder) if f.endswith(".mp4"))
        if videos:
            cap = cv2.VideoCapture(os.path.join(video_folder, videos[0]))
            ret, frame_bgr = cap.read()
            cap.release()
            if ret:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
                print("Frame inicial: primeiro frame do video real.")
                return frame

    print("Frame inicial: frame em branco (video nao encontrado).")
    return np.zeros((64, 64), dtype=np.float32)


# =============================================================================
# Main
# =============================================================================

def main():
    wm, vae, text_enc, fuser = load_models()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("World Model - Simulador Interativo")

    font_mono  = pygame.font.SysFont("monospace", 13)
    font_title = pygame.font.SysFont("monospace", 15, bold=True)
    font_small = pygame.font.SysFont("monospace", 11)
    clock = pygame.time.Clock()

    # -------------------------------------------------------------------------
    # Estado do simulador
    # -------------------------------------------------------------------------
    current_frame   : np.ndarray          = get_initial_frame()
    hidden          : torch.Tensor | None = None
    pending_action  : dict | None         = None   # acao aguardando SPACE
    step_count      : int                 = 0
    saved_count     : int                 = 0
    auto_play       : bool                = False   # avanco automatico
    history         : list[np.ndarray]    = [current_frame.copy()]
    history_hidden  : list                = [None]  # hidden states

    output_dir = os.path.join(current_dir, "world_model_outputs")
    os.makedirs(output_dir, exist_ok=True)

    def advance(action=None):
        """Avanca um frame, atualizando estado."""
        nonlocal current_frame, hidden, step_count
        current_frame, hidden = model_step(
            wm, vae, text_enc, fuser, current_frame, action, hidden
        )
        step_count += 1
        history.append(current_frame.copy())
        history_hidden.append(hidden)

    def reset():
        nonlocal current_frame, hidden, pending_action, step_count, auto_play
        current_frame  = get_initial_frame()
        hidden         = None
        pending_action = None
        step_count     = 0
        auto_play      = False
        history.clear()
        history_hidden.clear()
        history.append(current_frame.copy())
        history_hidden.append(None)

    # -------------------------------------------------------------------------
    # Loop principal
    # -------------------------------------------------------------------------
    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    # Avanca um frame com (ou sem) acao pendente
                    advance(pending_action)
                    pending_action = None

                elif event.key == pygame.K_r:
                    reset()
                    print("Reset!")

                elif event.key == pygame.K_a:
                    auto_play = not auto_play
                    print(f"Auto-play: {'ON' if auto_play else 'OFF'}")

                elif event.key == pygame.K_s:
                    path = os.path.join(output_dir, f"sim_frame_{saved_count:04d}.png")
                    img_u8 = (current_frame * 255).clip(0, 255).astype(np.uint8)
                    # Salva em resolucao maior para melhor visualizacao
                    img_big = cv2.resize(img_u8, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(path, img_big)
                    saved_count += 1
                    print(f"Frame salvo: {path}")

                elif event.key == pygame.K_BACKSPACE:
                    # Volta um frame no historico
                    if len(history) > 1:
                        history.pop()
                        history_hidden.pop()
                        current_frame = history[-1].copy()
                        hidden        = history_hidden[-1]
                        step_count    = max(0, step_count - 1)
                        pending_action = None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # Verifica se o clique foi dentro da area de render
                if (RENDER_X <= mx < RENDER_X + RENDER_W and
                        RENDER_Y <= my < RENDER_Y + RENDER_H):

                    if event.button == 1:  # clique esquerdo: define acao
                        sim_x, sim_y = display_click_to_sim(mx, my)
                        pending_action = {
                            "type":   "mouse_down",
                            "object": "ball",
                            "pos":    [sim_x, sim_y],
                        }

                    elif event.button == 3:  # clique direito: cancela acao
                        pending_action = None

        # Auto-play: avanca automaticamente sem acao
        if auto_play:
            advance(action=None)
            pending_action = None

        # -------------------------------------------------------------------------
        # Renderizacao
        # -------------------------------------------------------------------------
        screen.fill(C_BG)

        # --- Header ---
        pygame.draw.rect(screen, C_HEADER, (0, 0, WINDOW_W, RENDER_Y))
        draw_text(screen, font_title,
                  "World Model Simulator",
                  (10, 10), C_TEXT)
        mode_str = "AUTO" if auto_play else "MANUAL"
        draw_text(screen, font_mono,
                  f"Frame: {step_count:04d}  |  Modo: {mode_str}",
                  (280, 10), C_GREEN if auto_play else C_DIM)

        # --- Frame atual ---
        frame_surf = frame_to_surface(current_frame)
        screen.blit(frame_surf, (RENDER_X, RENDER_Y))

        # Borda da area de render
        pygame.draw.rect(screen, C_BORDER,
                         (RENDER_X, RENDER_Y, RENDER_W, RENDER_H), 1)

        # Marcador de acao pendente
        if pending_action is not None:
            pos = pending_action["pos"]
            px, py = sim_to_display(pos[0], pos[1])

            # Circulo + crosshair
            pygame.draw.circle(screen, C_ACTION, (px, py), 10, 2)
            pygame.draw.line(screen, C_ACTION, (px - 16, py), (px + 16, py), 1)
            pygame.draw.line(screen, C_ACTION, (px, py - 16), (px, py + 16), 1)

        # --- Sidebar ---
        sx = RENDER_W + 10
        sy = RENDER_Y + 10

        draw_text(screen, font_title, "CONTROLES", (sx, sy), C_TEXT)
        sy += 22

        controls = [
            ("SPACE/ENTER", "Avancar frame"),
            ("CLICK ESQ",   "Definir acao"),
            ("CLICK DIR",   "Cancelar acao"),
            ("BACKSPACE",   "Voltar frame"),
            ("A",           "Auto-play on/off"),
            ("R",           "Reiniciar"),
            ("S",           "Salvar frame"),
            ("Q / ESC",     "Sair"),
        ]
        for key, desc in controls:
            draw_text(screen, font_small, f"  {key:<12}", (sx, sy), C_KEY)
            draw_text(screen, font_small, desc, (sx + 110, sy), C_DIM)
            sy += 17

        sy += 12
        pygame.draw.line(screen, C_BORDER, (sx, sy), (sx + SIDEBAR_W - 20, sy), 1)
        sy += 10

        draw_text(screen, font_title, "ACAO PENDENTE", (sx, sy), C_TEXT)
        sy += 20

        if pending_action is not None:
            pos = pending_action["pos"]
            draw_text(screen, font_small, "mouse_down", (sx, sy), C_ACTION)
            sy += 16
            draw_text(screen, font_small,
                      f"  x: {pos[0]:.0f}  y: {pos[1]:.0f}",
                      (sx, sy), C_ACTION)
            sy += 16
            draw_text(screen, font_small,
                      "  [SPACE para aplicar]",
                      (sx, sy), C_DIM)
        else:
            draw_text(screen, font_small, "Nenhuma", (sx, sy), C_DIM)
            sy += 16
            draw_text(screen, font_small,
                      "  Clique na simulacao",
                      (sx, sy), C_DIM)
            sy += 16
            draw_text(screen, font_small,
                      "  para definir um click",
                      (sx, sy), C_DIM)

        sy += 30
        pygame.draw.line(screen, C_BORDER, (sx, sy), (sx + SIDEBAR_W - 20, sy), 1)
        sy += 10

        draw_text(screen, font_title, "HISTORICO", (sx, sy), C_TEXT)
        sy += 20
        draw_text(screen, font_small,
                  f"  {len(history)} frames armazenados",
                  (sx, sy), C_DIM)
        sy += 16
        draw_text(screen, font_small,
                  f"  {saved_count} frames salvos",
                  (sx, sy), C_DIM)

        pygame.display.flip()

    pygame.quit()
    print("Simulador encerrado.")


if __name__ == "__main__":
    main()
