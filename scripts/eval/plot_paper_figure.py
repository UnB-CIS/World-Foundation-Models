"""Gera a figura de rollout do artigo IEEE, com rotulos em ingles.

As figuras de `run_world_model_eval.py --stage figures` sao em portugues, para o
relatorio interno. O artigo precisa da mesma curva em ingles e no formato de uma
coluna do IEEEtran (~3.5 in de largura).

Uso:
    python scripts/eval/plot_paper_figure.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(PROJECT_ROOT, "scripts", "eval", "outputs", "resultados.json")
OUTPUT = os.path.join(PROJECT_ROOT, "IEEE_Article_2026", "figures", "rollout_cpe.pdf")

LABELS = {
    "world_model": ("Proposed world model", "#c96a3d", "-", "o"),
    "static": ("Persistence", "#4a4a4a", "--", None),
    "constant_velocity": ("Constant velocity", "#3d7fc9", "-.", None),
    "vae_ceiling": ("VAE ceiling (oracle)", "#5f7f4f", ":", None),
}


def main() -> None:
    with open(RESULTS) as handle:
        summary = json.load(handle)
    rollout = summary["rollout"]

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7})
    figure, axis = plt.subplots(figsize=(3.45, 2.35))
    for name, (label, color, style, marker) in LABELS.items():
        values = rollout[name]
        axis.plot(
            np.arange(1, len(values) + 1),
            values,
            label=label,
            color=color,
            linestyle=style,
            marker=marker,
            markersize=3,
            linewidth=1.6 if name == "world_model" else 1.1,
        )
    axis.set_xlabel("Prediction horizon (frames ahead)")
    axis.set_ylabel("Ball position error (px)")
    axis.set_xlim(1, len(rollout["world_model"]))
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout(pad=0.2)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(f"Figura salva em {OUTPUT}")


if __name__ == "__main__":
    main()
