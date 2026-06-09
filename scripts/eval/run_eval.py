"""Avaliação da Seção 6: baselines + modelo proposto, logado no MLflow."""

from __future__ import annotations

import glob
import os
import random
import re
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from eval_adapters import (
    NULL_ACTION, build_configs, build_samples,
    copy_last_frame, constant_velocity,
    load_models, video_to_json_path,
)
from eval_simples import run_simple_eval

CKPT_DIR          = "checkpoints"
VIDEOS_GLOB       = "scripts/dataset/videos/*.mp4"
INPUTS_DIR        = "scripts/dataset/inputs"
TEST_SPLIT        = 0.2
SEED              = 42
K                 = 2
MLFLOW_EXPERIMENT = "wfm_eval"


def _collect_test_videos():
    all_videos = sorted(glob.glob(VIDEOS_GLOB))

    seen: dict[str, str] = {}
    for v in all_videos:
        stem = os.path.splitext(os.path.basename(v))[0]
        base = re.sub(r"_auto_.*$", "", stem)
        if base not in seen:
            seen[base] = v
    unique = sorted(seen.values())

    valid = [v for v in unique if os.path.exists(video_to_json_path(v, INPUTS_DIR))]
    if len(valid) < len(unique):
        print(f"[aviso] {len(unique) - len(valid)} episódio(s) sem JSON ignorados.")

    # split determinístico — nenhum trainer salvou manifesto com seed fixo
    random.seed(SEED)
    shuffled = valid[:]
    random.shuffle(shuffled)
    test_videos = shuffled[:max(1, int(len(shuffled) * TEST_SPLIT))]

    print(f"Episódios válidos: {len(valid)}  |  teste (seed={SEED}): {len(test_videos)}")
    return test_videos


def _build_samples(test_videos):
    samples = []
    for vid in test_videos:
        ep = build_samples(vid, video_to_json_path(vid, INPUTS_DIR), k=K)
        samples.extend(ep)
        print(f"  {os.path.basename(vid):60s} -> {len(ep)} samples")
    n_action = sum(1 for s in samples if s["is_action_frame"])
    print(f"Total: {len(samples)} samples  ({n_action} de ação)")
    return samples


def _run_phase(configs, samples):
    run_simple_eval(
        configs, samples, NULL_ACTION,
        experiment=MLFLOW_EXPERIMENT,
        params={"k": K, "test_split": TEST_SPLIT, "seed": SEED},
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    test_videos = _collect_test_videos()
    if not test_videos:
        sys.exit("Nenhum vídeo de teste encontrado. Verifique VIDEOS_GLOB.")

    samples = _build_samples(test_videos)
    if not samples:
        sys.exit("Nenhum sample extraído.")

    print("\n[1/2] Baselines...")
    _run_phase({"copy_last_frame": copy_last_frame, "constant_velocity": constant_velocity}, samples)

    print("\n[2/2] Modelo...")
    models     = load_models(CKPT_DIR, device)
    all_cfgs   = build_configs(models, device)
    model_cfgs = {k: all_cfgs[k] for k in ("proposto", "no_action")}
    _run_phase(model_cfgs, samples)

    print(f"\nConcluído. Experimento MLflow: '{MLFLOW_EXPERIMENT}'")
    print("  no_action  -> action_sensitivity esperada ≈ 0")
    print("  proposto   -> action_sensitivity esperada > 0")


if __name__ == "__main__":
    main()
