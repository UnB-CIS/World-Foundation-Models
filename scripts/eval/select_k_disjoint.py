"""Varredura de k em episodios de validacao DISJUNTOS do conjunto de teste.

Por que existe: `run_world_model_eval.py --stage selection` usa os primeiros
SELECTION_EPISODES episodios, que sao um subconjunto dos primeiros
TEST_EPISODES usados na avaliacao. Escolher k ali significa ajustar um
hiperparametro dentro do conjunto de teste. Este script varre k nos episodios
que sobram (indices >= TEST_EPISODES), garantindo separacao total.

Os valores que o artigo cita na Secao V-A saem daqui.

Uso:
    python scripts/eval/select_k_disjoint.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, "..", ".."))
for path in (PROJECT_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from wm_runner import WorldModelRunner, list_episodes, load_episode  # noqa: E402

CHECKPOINT = os.path.join(PROJECT_ROOT, "scripts", "world_model_weights.pth")
OUTPUT = os.path.join(EVAL_DIR, "outputs", "selection_disjoint.json")
TEST_EPISODES = 24  # espelha run_world_model_eval.TEST_EPISODES
WARMUP_FRAMES = 20  # espelha run_world_model_eval.WARMUP_FRAMES
K_VALUES = (1, 2, 4, 6, 8, 10, 12)


def skill_score(runner: WorldModelRunner, episodes) -> dict[str, float]:
    """1 - sum(MSE_modelo) / sum(MSE_persistencia), agregado sobre o conjunto."""
    numerator = denominator = 0.0
    for episode in episodes:
        outputs = runner.one_step(episode)
        mu, predicted = outputs["mu_true"], outputs["mu_predicted"]
        window = slice(WARMUP_FRAMES, episode.length - 1)
        target = mu[1:][window].numpy()
        current = mu[:-1][window].numpy()
        estimate = predicted[:-1][window].numpy()
        numerator += ((estimate - target) ** 2).mean(axis=(1, 2, 3)).sum()
        denominator += ((current - target) ** 2).mean(axis=(1, 2, 3)).sum()
    return {
        "skill_score": float(1.0 - numerator / denominator),
        "error_ratio": float(numerator / denominator),
    }


def main() -> None:
    catalogue = list_episodes("*", dataset="holdout")
    held_back = catalogue[TEST_EPISODES:]
    if not held_back:
        raise SystemExit(
            f"Nenhum episodio sobrando: gere mais de {TEST_EPISODES} com "
            "scripts/eval/generate_holdout.py"
        )

    episodes = [load_episode(name, path, dataset="holdout") for name, path in held_back]
    print(f"Validacao disjunta: {[e.name for e in episodes]}")
    print(f"(os primeiros {TEST_EPISODES} episodios formam o conjunto de teste)\n")

    results = {}
    for k in K_VALUES:
        runner = WorldModelRunner(
            checkpoint=CHECKPOINT, device="cpu", memory_frames=k, memory_stride=1
        )
        results[k] = skill_score(runner, episodes)
        print(
            f"  k={k:2d}: skill = {results[k]['skill_score']:+.4f}  "
            f"(MSE ratio {results[k]['error_ratio']:.4f})"
        )

    best = max(results, key=lambda k: results[k]["skill_score"])
    print(f"\nargmax = k={best} ({results[best]['skill_score']:+.4f})")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as handle:
        json.dump(
            {
                "checkpoint": os.path.basename(CHECKPOINT),
                "memory_stride": 1,
                "warmup_frames": WARMUP_FRAMES,
                "validation_episodes": [e.name for e in episodes],
                "test_episodes_excluded": TEST_EPISODES,
                "sweep": {str(k): v for k, v in results.items()},
                "argmax_k": best,
            },
            handle,
            indent=2,
        )
    print(f"Salvo em {OUTPUT}")


if __name__ == "__main__":
    main()
