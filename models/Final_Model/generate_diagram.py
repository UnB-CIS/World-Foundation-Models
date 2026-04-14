"""
Gera diagrama da arquitetura do World Model.
Salva em world_model_outputs/architecture_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "world_model_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────────────
C_INPUT   = "#4A90D9"   # blue  – inputs
C_VAE     = "#E8A838"   # amber – VAE encoder/decoder
C_TEXT    = "#7B68EE"   # violet – TextEncoder
C_FUSER   = "#52B06A"   # green – SpatialBroadcastFuser
C_WM      = "#E05A5A"   # red   – WorldModelVAERecurrent
C_HIDDEN  = "#BFA0CC"   # lilac – hidden state
C_ARROW   = "#444444"
C_BG      = "#F7F7F7"
C_LOOP    = "#888888"

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# ── helper functions ──────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel="", color="#AAAAAA", fontsize=9, subsize=7.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=1.4,
                          edgecolor="white",
                          facecolor=color,
                          zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.15, label,   ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)
        ax.text(x + w/2, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=subsize,  color="white", alpha=0.88, zorder=4)
    else:
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)

def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW, lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.05, my + 0.12, label, ha="left", va="center",
                fontsize=7, color=color, zorder=5)

def varrow(ax, x, y0, y1, label="", color=C_ARROW, lw=1.5):
    arrow(ax, x, y0, x, y1, label, color, lw)

def harrow(ax, x0, x1, y, label="", color=C_ARROW, lw=1.5):
    arrow(ax, x0, y, x1, y, label, color, lw)

# ─────────────────────────────────────────────────────────────────────────────
# ROW LAYOUT  (y positions, bottom-up)
# Row 0 (y~1.0):  frame_t  →  VAE enc  →  (space)  →  WorldModel  →  (space)  →  VAE dec  →  frame_t+1
# Row 1 (y~4.5):  action   →  TextEnc  →  SpatialFuser  (feeds into WorldModel from below)
# ─────────────────────────────────────────────────────────────────────────────

# ── ROW 0 — main pipeline ────────────────────────────────────────────────────
BH = 1.1   # box height
BW = 1.8   # box width

# frame_t  (input)
box(ax,  0.3, 4.2, BW, BH, "frame_t", "(1,1,64,64)", C_INPUT)

# VAE Encoder
box(ax,  2.5, 4.2, BW, BH, "VAE\nEncoder", "Conv2d × 4\n→ z_visual", C_VAE)

# z_visual label
ax.text(4.65, 4.75, "z_visual\n(1,16,8,8)", ha="center", va="center",
        fontsize=7.5, color=C_VAE, zorder=5)

# SpatialBroadcastFuser
box(ax,  5.5, 4.2, BW, BH, "Spatial\nBroadcast\nFuser", "cat(z_vis, tile(z_act))\n→ x_fused (1,32,8,8)", C_FUSER)

# WorldModelVAERecurrent — larger box
WMX, WMY, WMW, WMH = 8.0, 3.0, 3.8, 3.5
box(ax, WMX, WMY, WMW, WMH, "", "", C_WM, fontsize=9)

# inner components of WorldModel
ax.text(WMX + WMW/2, WMY + WMH - 0.35, "WorldModelVAERecurrent",
        ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=5)

inner_items = [
    (WMX+0.2, WMY+2.4, "conv_enc  →  Flatten"),
    (WMX+0.2, WMY+1.85,"fc_mu / fc_logvar"),
    (WMX+0.2, WMY+1.30,"reparametrize  →  z"),
    (WMX+0.2, WMY+0.75,"GRUCell(z, hidden)  →  h'"),
    (WMX+0.2, WMY+0.20,"fc_dec  →  conv_dec  →  z_next"),
]
for ix, iy, itxt in inner_items:
    ax.text(ix, iy, itxt, ha="left", va="center",
            fontsize=7.5, color="white", alpha=0.9, zorder=5,
            fontfamily="monospace")
    if iy > WMY + 0.3:
        ax.plot([WMX+0.15, WMX+WMW-0.15], [iy-0.27, iy-0.27],
                color="white", lw=0.4, alpha=0.35, zorder=4)

# hidden state box
box(ax, WMX+0.3, WMY-1.4, WMW-0.6, 0.85, "hidden state  (1, 64)", "(GRU memory — persists across steps)", C_HIDDEN, fontsize=8, subsize=7)

# chunk → mu_next
box(ax, 12.3, 4.2, BW, BH, "chunk(z_next)", "mu_next\n(1,8,8,8)", C_WM)

# VAE Decoder
box(ax, 14.5, 4.2, BW, BH, "VAE\nDecoder", "ConvTranspose2d × 4\n→ frame_t+1", C_VAE)

# frame_t+1
box(ax, 16.7, 4.2, BW, BH, "frame_t+1", "(1,1,64,64)", C_INPUT)

# ── ROW 1 — action branch ────────────────────────────────────────────────────
box(ax, 0.3, 1.3, BW, BH, "action", '{"type","x","y","z"}', C_INPUT)
box(ax, 2.5, 1.3, BW, BH, "TextEncoder", "MLP 4→64→16\n→ z_action (1,16)", C_TEXT)

# ── ARROWS ───────────────────────────────────────────────────────────────────
# frame → VAE enc
harrow(ax, 0.3+BW, 2.5, 4.75)
# VAE enc → fuser
harrow(ax, 2.5+BW, 5.5, 4.75, "z_visual")

# action → text enc
harrow(ax, 0.3+BW, 2.5, 1.85)
# text enc → fuser (vertical)
ax.annotate("", xy=(6.4, 4.2), xytext=(6.4, 1.3+BH),
            arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.5,
                            connectionstyle="arc3,rad=0.0"), zorder=2)
ax.text(6.55, 3.2, "z_action\n(1,16)", ha="left", va="center",
        fontsize=7.5, color=C_TEXT, zorder=5)

# fuser → WorldModel
harrow(ax, 5.5+BW, WMX, 4.75, "x_fused\n(1,32,8,8)")

# WorldModel → chunk
harrow(ax, WMX+WMW, 12.3, 4.75, "z_next\n(1,16,8,8)")

# chunk → VAE dec
harrow(ax, 12.3+BW, 14.5, 4.75, "mu_next\n(1,8,8,8)")

# VAE dec → frame_t+1
harrow(ax, 14.5+BW, 16.7, 4.75)

# WorldModel ↔ hidden state
ax.annotate("", xy=(WMX+WMW/2, WMY), xytext=(WMX+WMW/2, WMY-0.55),
            arrowprops=dict(arrowstyle="<->", color=C_HIDDEN, lw=1.8,
                            connectionstyle="arc3,rad=0.0"), zorder=2)

# ── autoregressive loop arrow ─────────────────────────────────────────────────
# frame_t+1 loops back to become frame_t
loop_x0 = 16.7 + BW/2
loop_x1 = 0.3  + BW/2
loop_y  = 5.6
ax.annotate("", xy=(loop_x1, loop_y), xytext=(loop_x0, loop_y),
            arrowprops=dict(arrowstyle="->", color=C_LOOP, lw=1.5,
                            connectionstyle="arc3,rad=-0.25"), zorder=2)
ax.text((loop_x0+loop_x1)/2, loop_y + 0.7,
        "autoregressive loop", ha="center", va="bottom",
        fontsize=8, color=C_LOOP, style="italic", zorder=5)

# ── frozen VAE note ───────────────────────────────────────────────────────────
ax.text(14.5 + BW/2, 3.85,
        "frozen during\ntraining",
        ha="center", va="center", fontsize=6.5, color=C_VAE,
        alpha=0.85, zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_VAE, lw=0.8, alpha=0.7))

# ── foreground-weighted loss note ─────────────────────────────────────────────
ax.text(11.2, 7.0,
        "Foreground-weighted pixel loss\n"
        "(FG_WEIGHT=15 × for dark pixels)\n"
        "via frozen VAE decoder",
        ha="center", va="center", fontsize=7.5, color="#555",
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#AAAAAA", lw=0.9, alpha=0.85))
# dashed arrow from loss note to VAE dec
ax.annotate("", xy=(14.5 + BW/2, 5.3), xytext=(11.2, 6.65),
            arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.0,
                            linestyle="dashed", connectionstyle="arc3,rad=0.0"),
            zorder=2)

# ── TBPTT note ────────────────────────────────────────────────────────────────
ax.text(9.9, 7.4,
        "TBPTT  |  SEQ_LEN=16  |  hidden.detach() after each window",
        ha="center", va="center", fontsize=7.5, color="#333", zorder=5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_WM, lw=0.9, alpha=0.85))

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_INPUT,  "Input / Output"),
    (C_VAE,    "VAE (frozen)"),
    (C_TEXT,   "TextEncoder"),
    (C_FUSER,  "SpatialBroadcastFuser"),
    (C_WM,     "WorldModelVAERecurrent"),
    (C_HIDDEN, "GRU hidden state"),
]
lx, ly = 0.3, 8.5
for i, (col, lbl) in enumerate(legend_items):
    rect = FancyBboxPatch((lx + i*2.9, ly), 0.4, 0.35,
                          boxstyle="round,pad=0.05",
                          linewidth=0,
                          facecolor=col, zorder=4)
    ax.add_patch(rect)
    ax.text(lx + i*2.9 + 0.5, ly + 0.17, lbl,
            ha="left", va="center", fontsize=7.5, zorder=5)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(9, 9.6, "World Foundation Model — Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#222")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "architecture_diagram.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Diagrama salvo em: {out_path}")
