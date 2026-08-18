"""Avaliacao quantitativa do world model 2D, com registro no MLflow.

Responde as duas hipoteses do artigo com evidencia numerica:

  H1 (dinamica)  - o modelo prediz o proximo estado da bola melhor do que
                   baselines triviais (persistencia e velocidade constante).
  H2 (acao)      - a predicao e condicionada pela acao: um clique altera o
                   frame predito de forma especifica e localizada.

Estagios:
  selection  varre checkpoints x (memory_frames, memory_stride) num subconjunto
             de validacao e escolhe a melhor configuracao pelo skill score.
  evaluation roda a melhor configuracao no conjunto de teste: predicao de um
             passo, rollout em malha aberta e resposta a acao.
  best       registra a configuracao vencedora e o checkpoint no MLflow.

Uso:
    python scripts/eval/run_world_model_eval.py --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, "..", ".."))
for path in (PROJECT_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from wm_metrics import (  # noqa: E402
    MIN_BALL_AREA,
    RELATIVE_THRESHOLD,
    SCALE_X,
    SCALE_Y,
    Ball,
    bootstrap_ci,
    centroid_position_error,
    detect_balls,
    extrapolate_balls,
    latent_metrics,
    match_balls,
    moving_indices,
    nanmean,
    subset_position_error,
    pixel_metrics,
)
from wm_runner import (  # noqa: E402
    Episode,
    WorldModelRunner,
    list_episodes,
    load_episode,
    split_episodes,
)

CHECKPOINTS = {
    "delta_convlstm_v2": os.path.join(PROJECT_ROOT, "scripts", "world_model_weights.pth"),
    "delta_convlstm_v1": os.path.join(
        PROJECT_ROOT, "scripts", "world_model_weights_useful.pth"
    ),
}
# Eq. (1) e (5) do artigo definem a janela como os k frames CONTIGUOS mais
# recentes, o que corresponde a stride 1. A grade "spaced" existe apenas para
# medir o quanto se perde ao desviar dessa definicao.
MEMORY_GRID_CONTIGUOUS = [(k, 1) for k in (1, 2, 4, 6, 8, 10, 12)]
MEMORY_GRID_SPACED = [(10, 5), (10, 2), (6, 2), (4, 3)]
PREDICTORS = (
    "world_model",
    "world_model_no_action",
    "latent_persistence",
    "constant_velocity",
    "vae_ceiling",
    # Baselines na forma literal da Secao V do artigo (espaco de pixels).
    "pixel_persistence",
    "pixel_constant_velocity",
)

TRACKING_URI = pathlib.Path(PROJECT_ROOT, "mlruns").as_uri()
EXPERIMENT_SELECTION = "wfm_world_model_selection"
EXPERIMENT_EVALUATION = "wfm_world_model_evaluation"
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "scripts", "eval", "outputs")

SELECTION_EPISODES = 6
TEST_EPISODES = 24
ROLLOUT_HORIZON = 16
ROLLOUT_STARTS_PER_EPISODE = 6
SEED = 42
TEST_RATIO = 0.2
WARMUP_FRAMES = 20  # descarta o inicio do episodio (cena vazia, sem dinamica)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
@dataclass
class Config:
    checkpoint_name: str
    checkpoint_path: str
    memory_frames: int
    memory_stride: int
    feedback: str = "pixel"


def _mlflow():
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow


def _log_metrics(mlflow, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            mlflow.log_metric(key, value)


def _detect_sequence(frames: np.ndarray) -> list[list[Ball]]:
    return [detect_balls(frame) for frame in frames]


def _select_test_episodes(
    limit: int, family: str | None, dataset: str
) -> list[tuple[str, str]]:
    """Episodios de teste.

    Em `holdout` todos os episodios sao ineditos por construcao (foram gerados
    depois do treino), entao nao ha split. Em `train_pool` aplicamos um split
    deterministico, mas o resultado continua sendo potencialmente in-sample —
    o manifesto de treino original nao foi versionado.
    """
    pattern = f"*{family}*" if family else "*"
    episodes = list_episodes(pattern, dataset=dataset)
    if not episodes:
        raise RuntimeError(f"Nenhum episodio encontrado para o padrao {pattern}")
    if dataset == "holdout":
        return episodes[:limit]
    _, test = split_episodes(episodes, TEST_RATIO, SEED)
    return test[:limit]


# ---------------------------------------------------------------------------
# Estagio 1 - selecao de configuracao
# ---------------------------------------------------------------------------
def run_selection(args) -> Config:
    mlflow = _mlflow()
    mlflow.set_experiment(EXPERIMENT_SELECTION)

    episodes = [
        load_episode(name, path, max_frames=args.max_frames, dataset=args.dataset)
        for name, path in _select_test_episodes(
            SELECTION_EPISODES, args.family, args.dataset
        )
    ]
    print(f"Selecao em {len(episodes)} episodios de validacao")

    results: list[tuple[float, Config, dict[str, float]]] = []
    grid = {
        "contiguous": MEMORY_GRID_CONTIGUOUS,
        "spaced": MEMORY_GRID_SPACED,
        "all": MEMORY_GRID_CONTIGUOUS + MEMORY_GRID_SPACED,
    }[args.history]
    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            print(f"  [pular] checkpoint ausente: {checkpoint_path}")
            continue
        for memory_frames, memory_stride in grid:
            config = Config(
                checkpoint_name=checkpoint_name,
                checkpoint_path=checkpoint_path,
                memory_frames=memory_frames,
                memory_stride=memory_stride,
            )
            runner = WorldModelRunner(
                checkpoint=checkpoint_path,
                device=args.device,
                memory_frames=memory_frames,
                memory_stride=memory_stride,
            )
            metrics = _selection_metrics(runner, episodes)
            results.append((metrics["skill_score"], config, metrics))

            with mlflow.start_run(
                run_name=f"{checkpoint_name}_mf{memory_frames}_ms{memory_stride}"
            ):
                mlflow.log_params(
                    {
                        "checkpoint": checkpoint_name,
                        "checkpoint_file": os.path.basename(checkpoint_path),
                        "memory_frames": memory_frames,
                        "memory_stride": memory_stride,
                        "n_episodes": len(episodes),
                        "seed": SEED,
                        # Sem estes dois a varredura fica ambigua: o mesmo
                        # checkpoint tem skill oposto em lotes diferentes.
                        "dataset": args.dataset,
                        "episode_family": args.family or "todas",
                    }
                )
                _log_metrics(mlflow, metrics)
            print(
                f"  {checkpoint_name:20s} mf={memory_frames:2d} ms={memory_stride} "
                f"skill={metrics['skill_score']:+.4f} "
                f"latent_mse={metrics['latent_mse']:.5f} "
                f"cosine={metrics['delta_cosine']:+.3f}"
            )

    if not results:
        raise RuntimeError("Nenhuma configuracao avaliada.")
    results.sort(key=lambda item: item[0], reverse=True)
    best_score, best_config, best_metrics = results[0]
    print(
        f"\nMelhor configuracao: {best_config.checkpoint_name} "
        f"mf={best_config.memory_frames} ms={best_config.memory_stride} "
        f"(skill={best_score:+.4f})"
    )
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "best_config.json"), "w") as handle:
        json.dump(
            {"config": asdict(best_config), "selection_metrics": best_metrics},
            handle,
            indent=2,
        )
    return best_config


def _selection_metrics(
    runner: WorldModelRunner, episodes: list[Episode]
) -> dict[str, float]:
    """Metricas latentes agregadas — baratas, sem decodificar frames."""
    numerator = denominator = 0.0
    cosines: list[float] = []
    ratios: list[float] = []
    for episode in episodes:
        outputs = runner.one_step(episode)
        mu_true = outputs["mu_true"]
        mu_predicted = outputs["mu_predicted"]
        window = slice(WARMUP_FRAMES, episode.length - 1)
        target = mu_true[1:][window]
        current = mu_true[:-1][window]
        predicted = mu_predicted[:-1][window]

        numerator += float(((predicted - target) ** 2).mean(dim=(1, 2, 3)).sum())
        denominator += float(((current - target) ** 2).mean(dim=(1, 2, 3)).sum())

        true_delta = (target - current).flatten(1)
        predicted_delta = (predicted - current).flatten(1)
        true_norm = true_delta.norm(dim=1)
        predicted_norm = predicted_delta.norm(dim=1)
        valid = (true_norm > 1e-6) & (predicted_norm > 1e-6)
        if valid.any():
            cosine = (true_delta[valid] * predicted_delta[valid]).sum(dim=1) / (
                true_norm[valid] * predicted_norm[valid]
            )
            cosines.extend(cosine.tolist())
            ratios.extend((predicted_norm[valid] / true_norm[valid]).tolist())

    count = sum(episode.length - 1 - WARMUP_FRAMES for episode in episodes)
    return {
        "latent_mse": numerator / count,
        "latent_mse_persistence": denominator / count,
        "skill_score": 1.0 - numerator / denominator,
        "delta_cosine": nanmean(cosines),
        "delta_magnitude_ratio": nanmean(ratios),
        "n_frames": float(count),
    }


# ---------------------------------------------------------------------------
# Estagio 2a - predicao de um passo
# ---------------------------------------------------------------------------
def run_one_step(config: Config, episodes: list[Episode], args) -> dict:
    """Compara modelo, ablacao sem acao e baselines no horizonte de 1 passo."""
    runner = WorldModelRunner(
        checkpoint=config.checkpoint_path,
        device=args.device,
        memory_frames=config.memory_frames,
        memory_stride=config.memory_stride,
    )
    per_predictor: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    latent_rows: list[dict[str, float]] = []
    qualitative: list[np.ndarray] = []

    for episode_index, episode in enumerate(episodes):
        outputs = runner.one_step(episode)
        outputs_no_action = runner.one_step(episode, use_actions=False)
        mu_true = outputs["mu_true"]
        decoded = {
            "world_model": runner.decode(outputs["mu_predicted"][:-1]),
            "world_model_no_action": runner.decode(
                outputs_no_action["mu_predicted"][:-1]
            ),
            "latent_persistence": runner.decode(mu_true[:-1]),
            "vae_ceiling": runner.decode(mu_true[1:]),
        }
        # Baselines triviais NA FORMA LITERAL DA SECAO V DO ARTIGO, em espaco de
        # pixels: x_{t+1} = x_t e x_{t+1} = clip(2 x_t - x_{t-1}, 0, 1). Elas nao
        # passam pelo VAE, entao nao carregam erro de reconstrucao — sao mantidas
        # ao lado das versoes no dominio decodificado justamente para tornar essa
        # diferenca mensuravel.
        current_raw = episode.frames[:-1]
        previous_raw = episode.frames[
            np.maximum(np.arange(len(current_raw)) - 1, 0)
        ]
        decoded["pixel_persistence"] = current_raw
        decoded["pixel_constant_velocity"] = np.clip(
            2.0 * current_raw - previous_raw, 0.0, 1.0
        )
        ground_truth = episode.frames[1:]
        detections = {name: _detect_sequence(frames) for name, frames in decoded.items()}
        ground_truth_all = _detect_sequence(episode.frames)
        detections["ground_truth"] = ground_truth_all[1:]
        ground_truth_current = ground_truth_all[:-1]

        # Baseline de velocidade constante em espaco de trajetoria: usa as
        # deteccoes do dominio decodificado, para nao levar vantagem sobre o
        # modelo por nao passar pelo decoder.
        persistence_detections = detections["latent_persistence"]
        constant_velocity: list[list[Ball]] = []
        for index in range(len(persistence_detections)):
            previous = persistence_detections[max(0, index - 1)]
            constant_velocity.append(
                extrapolate_balls(previous, persistence_detections[index], horizon=1)
            )
        detections["constant_velocity"] = constant_velocity
        decoded["constant_velocity"] = decoded["latent_persistence"]

        start = WARMUP_FRAMES
        stop = len(ground_truth)
        # O laco e por frame (e nao por preditor) para garantir que todas as
        # listas fiquem alinhadas amostra a amostra — os testes pareados
        # dependem disso.
        for index in range(start, stop):
            truth_balls = detections["ground_truth"][index]
            if not truth_balls:
                continue

            moving = moving_indices(ground_truth_current[index], truth_balls)
            moving_errors = {
                name: subset_position_error(truth_balls, detections[name][index], moving)
                for name in PREDICTORS
            }
            if not all(np.isfinite(value) for value in moving_errors.values()):
                moving_errors = {}

            for name in PREDICTORS:
                store = per_predictor[name]
                errors = centroid_position_error(truth_balls, detections[name][index])
                store["cpe"].append(errors["cpe"])
                store["count_error"].append(errors["count_error"])
                store["detection_ratio"].append(errors["count_ratio"])
                store["_episode"].append(float(episode_index))
                store["_frame"].append(float(index))
                if moving_errors:
                    store["cpe_moving"].append(moving_errors[name])
                    store["_episode_moving"].append(float(episode_index))
                if name == "constant_velocity":
                    # Baseline definido apenas em espaco de trajetoria: nao gera
                    # um frame proprio, entao metricas de pixel nao se aplicam.
                    continue
                pixels = pixel_metrics(ground_truth[index], decoded[name][index])
                store["px_mse"].append(pixels["mse"])
                store["px_mae"].append(pixels["mae"])
                store["px_psnr"].append(pixels["psnr"])
                store["px_ssim"].append(pixels["ssim"])

        for index in range(start, stop):
            latent_rows.append(
                latent_metrics(
                    mu_true[index].cpu().numpy(),
                    mu_true[index + 1].cpu().numpy(),
                    outputs["mu_predicted"][index].cpu().numpy(),
                )
            )

        if len(qualitative) < 6:
            index = min(stop - 1, start + 120)
            qualitative.append(
                np.stack(
                    [
                        episode.frames[index],
                        ground_truth[index],
                        decoded["world_model"][index],
                        decoded["vae_ceiling"][index],
                        decoded["latent_persistence"][index],
                    ]
                )
            )

    summary = {
        name: {
            key: nanmean(values)
            for key, values in store.items()
            if not key.startswith("_")
        }
        for name, store in per_predictor.items()
    }
    for name, store in per_predictor.items():
        low, high = bootstrap_ci(store["cpe"])
        summary[name]["cpe_ci_low"] = low
        summary[name]["cpe_ci_high"] = high
        moving_low, moving_high = bootstrap_ci(store["cpe_moving"])
        summary[name]["cpe_moving_ci_low"] = moving_low
        summary[name]["cpe_moving_ci_high"] = moving_high
        summary[name]["n_samples"] = float(len(store["cpe"]))

    if not latent_rows:
        raise RuntimeError(
            "Nenhum frame avaliado. Episodios curtos demais para WARMUP_FRAMES."
        )
    latent_summary = {
        key: nanmean([row[key] for row in latent_rows]) for key in latent_rows[0]
    }
    # O skill score precisa ser calculado sobre os erros AGREGADOS. A media dos
    # skills por frame diverge: em frames estaticos o erro da persistencia e
    # ~0 e a razao explode, o que dominaria a media.
    total_error = float(np.nansum([row["latent_mse"] for row in latent_rows]))
    total_persistence = float(
        np.nansum([row["latent_mse_persistence"] for row in latent_rows])
    )
    latent_summary["skill_score"] = (
        1.0 - total_error / total_persistence if total_persistence > 0 else float("nan")
    )
    latent_summary["latent_win_rate"] = float(
        np.mean(
            [
                row["latent_mse"] < row["latent_mse_persistence"]
                for row in latent_rows
            ]
        )
    )
    paired = _paired_test(
        per_predictor["world_model"]["cpe"], per_predictor["latent_persistence"]["cpe"]
    )
    paired_cv = _paired_test(
        per_predictor["world_model"]["cpe"], per_predictor["constant_velocity"]["cpe"]
    )
    # Frames do mesmo episodio sao correlacionados: o teste por frame superestima
    # a significancia. O teste por episodio usa a unidade experimental correta.
    episode_ids = per_predictor["world_model"]["_episode"]
    paired_episode = _paired_test_by_episode(
        per_predictor["world_model"]["cpe"],
        per_predictor["latent_persistence"]["cpe"],
        episode_ids,
    )
    paired_episode_cv = _paired_test_by_episode(
        per_predictor["world_model"]["cpe"],
        per_predictor["constant_velocity"]["cpe"],
        episode_ids,
    )
    moving_ids = per_predictor["world_model"]["_episode_moving"]
    paired_moving = _paired_test_by_episode(
        per_predictor["world_model"]["cpe_moving"],
        per_predictor["latent_persistence"]["cpe_moving"],
        moving_ids,
    )
    paired_moving_cv = _paired_test_by_episode(
        per_predictor["world_model"]["cpe_moving"],
        per_predictor["constant_velocity"]["cpe_moving"],
        moving_ids,
    )
    _dump_per_frame(per_predictor, [episode.name for episode in episodes])
    return {
        "per_predictor": summary,
        "latent": latent_summary,
        "paired_vs_persistence": paired,
        "paired_vs_constant_velocity": paired_cv,
        "paired_by_episode_vs_persistence": paired_episode,
        "paired_by_episode_vs_constant_velocity": paired_episode_cv,
        "paired_moving_vs_persistence": paired_moving,
        "paired_moving_vs_constant_velocity": paired_moving_cv,
        "qualitative": qualitative,
    }


def _episode_means(values: list[float], episode_ids: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    groups = np.asarray(episode_ids, dtype=np.int64)[: array.size]
    return np.array(
        [nanmean(array[groups == group]) for group in np.unique(groups)],
        dtype=np.float64,
    )


def _paired_test_by_episode(
    model: list[float], baseline: list[float], episode_ids: list[float]
) -> dict[str, float]:
    from scipy import stats

    a = _episode_means(model, episode_ids)
    b = _episode_means(baseline, episode_ids)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 3:
        return _empty_paired_result("n_episodes")
    difference = b - a
    low, high = bootstrap_ci(difference)
    try:
        _, p_value = stats.wilcoxon(a, b)
    except ValueError:
        p_value = float("nan")
    return {
        "mean_difference": float(difference.mean()),
        "difference_ci_low": low,
        "difference_ci_high": high,
        "win_rate": float((difference > 0).mean()),
        "p_value": float(p_value),
        "n_episodes": float(a.size),
    }


def _dump_per_frame(per_predictor: dict, episode_names: list[str]) -> None:
    """Salva o CPE por frame — insumo bruto para reanalise e para o artigo."""
    import csv

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, "cpe_por_frame.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["preditor", "episodio", "frame", "cpe"])
        for name, store in per_predictor.items():
            for value, episode, frame in zip(
                store["cpe"], store["_episode"], store["_frame"], strict=False
            ):
                writer.writerow(
                    [name, episode_names[int(episode)], int(frame), f"{value:.4f}"]
                )


def _empty_paired_result(count_key: str) -> dict[str, float]:
    return {
        "mean_difference": float("nan"),
        "difference_ci_low": float("nan"),
        "difference_ci_high": float("nan"),
        "win_rate": float("nan"),
        "p_value": float("nan"),
        count_key: 0.0,
    }


def _paired_test(model: list[float], baseline: list[float]) -> dict[str, float]:
    """Teste pareado (Wilcoxon) entre modelo e baseline no mesmo frame."""
    from scipy import stats

    a = np.asarray(model, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    size = min(a.size, b.size)
    a, b = a[:size], b[:size]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 10:
        return _empty_paired_result("n_pairs")
    difference = b - a  # positivo => modelo melhor que baseline
    low, high = bootstrap_ci(difference)
    try:
        _, p_value = stats.wilcoxon(a, b)
    except ValueError:
        p_value = float("nan")
    return {
        "mean_difference": float(difference.mean()),
        "difference_ci_low": low,
        "difference_ci_high": high,
        "win_rate": float((difference > 0).mean()),
        "p_value": float(p_value),
        "n_pairs": float(a.size),
    }


# ---------------------------------------------------------------------------
# Estagio 2b - rollout em malha aberta
# ---------------------------------------------------------------------------
def run_rollout(config: Config, episodes: list[Episode], args) -> dict:
    """Erro de posicao da bola em funcao do horizonte de predicao."""
    runner = WorldModelRunner(
        checkpoint=config.checkpoint_path,
        device=args.device,
        memory_frames=config.memory_frames,
        memory_stride=config.memory_stride,
    )
    horizon = args.horizon
    curves: dict[str, list[list[float]]] = {
        name: [[] for _ in range(horizon)]
        for name in ("world_model", "static", "constant_velocity", "vae_ceiling")
    }
    strip: dict[str, np.ndarray] | None = None
    # Deslocamento vertical acumulado das bolas: mede diretamente se o modelo
    # reproduz a queda por gravidade, e nao apenas se erra pouco.
    displacement: dict[str, list[list[float]]] = {
        "real": [[] for _ in range(horizon)],
        "predito": [[] for _ in range(horizon)],
    }

    for episode in episodes:
        usable = episode.length - horizon - 1
        if usable <= WARMUP_FRAMES + 2:
            continue
        starts = np.linspace(
            WARMUP_FRAMES + 2, usable, ROLLOUT_STARTS_PER_EPISODE, dtype=int
        ).tolist()
        predictions = runner.rollout(
            episode, starts, horizon, feedback=config.feedback
        )
        # Deteccoes no dominio decodificado, calculadas uma unica vez: as
        # baselines partem do mesmo estado reconstruido que o modelo recebe.
        reconstructed_frames = runner.decode(runner.encode(episode.frames))
        reconstructed = _detect_sequence(reconstructed_frames)
        ground_truth = _detect_sequence(episode.frames)

        if strip is None:
            middle = len(starts) // 2
            begin = starts[middle]
            window = slice(begin + 1, begin + 1 + horizon)
            strip = {
                "real": episode.frames[window].copy(),
                "reconstruido": reconstructed_frames[window].copy(),
                "predito": predictions[middle].copy(),
                "episodio": episode.name,
                "inicio": begin,
            }

        for position, start in enumerate(starts):
            previous = reconstructed[start - 1]
            current = reconstructed[start]
            for step in range(horizon):
                truth = ground_truth[start + step + 1]
                if not truth:
                    continue
                model_balls = detect_balls(predictions[position, step])
                curves["world_model"][step].append(
                    centroid_position_error(truth, model_balls)["cpe"]
                )
                curves["static"][step].append(
                    centroid_position_error(truth, current)["cpe"]
                )
                curves["constant_velocity"][step].append(
                    centroid_position_error(
                        truth, extrapolate_balls(previous, current, step + 1)
                    )["cpe"]
                )
                curves["vae_ceiling"][step].append(
                    centroid_position_error(truth, reconstructed[start + step + 1])[
                        "cpe"
                    ]
                )
                displacement["real"][step].append(_mean_vertical_shift(current, truth))
                displacement["predito"][step].append(
                    _mean_vertical_shift(current, model_balls)
                )

    result: dict = {
        name: [nanmean(values) for values in steps] for name, steps in curves.items()
    }
    result["_strip"] = strip
    result["_displacement"] = {
        name: [nanmean(values) for values in steps]
        for name, steps in displacement.items()
    }
    return result


def _mean_vertical_shift(start_balls: list[Ball], target_balls: list[Ball]) -> float:
    """Deslocamento vertical medio (px na tela) entre dois conjuntos de bolas."""
    pairs, _ = match_balls(start_balls, target_balls)
    if not pairs:
        return float("nan")
    return float(
        np.mean(
            [
                (target_balls[target].y - start_balls[source].y) * SCALE_Y
                for source, target in pairs
            ]
        )
    )


# ---------------------------------------------------------------------------
# Estagio 2c - condicionamento pela acao (H2)
# ---------------------------------------------------------------------------
def run_action_analysis(config: Config, episodes: list[Episode], args) -> dict:
    runner = WorldModelRunner(
        checkpoint=config.checkpoint_path,
        device=args.device,
        memory_frames=config.memory_frames,
        memory_stride=config.memory_stride,
    )
    click_sensitivity: list[float] = []
    null_sensitivity: list[float] = []
    localization: list[float] = []
    ground_truth_localization: list[float] = []
    heatmap_sample: tuple[np.ndarray, np.ndarray, tuple[float, float]] | None = None

    for episode in episodes:
        with_action = runner.one_step(episode, use_actions=True)
        without_action = runner.one_step(episode, use_actions=False)
        frames_action = runner.decode(with_action["mu_predicted"])
        frames_null = runner.decode(without_action["mu_predicted"])
        difference = np.abs(frames_action - frames_null)

        click_indices = sorted(
            index
            for index in episode.actions
            if WARMUP_FRAMES <= index < episode.length - 1
        )
        click_set = set(click_indices)
        control_indices = [
            index
            for index in range(WARMUP_FRAMES, episode.length - 1)
            if index not in click_set
        ]

        click_sensitivity.extend(difference[index].mean() for index in click_indices)
        null_sensitivity.extend(difference[index].mean() for index in control_indices)

        for index in click_indices:
            action = episode.actions[index]
            click_x = action["pos"][0] / SCALE_X
            click_y = action["pos"][1] / SCALE_Y
            peak_y, peak_x = np.unravel_index(
                difference[index].argmax(), difference[index].shape
            )
            localization.append(
                float(
                    np.hypot((peak_x - click_x) * SCALE_X, (peak_y - click_y) * SCALE_Y)
                )
            )
            ground_truth_localization.append(
                _ground_truth_spawn_error(episode, index, (click_x, click_y))
            )
            if heatmap_sample is None and difference[index].max() > 0.02:
                heatmap_sample = (
                    episode.frames[index],
                    difference[index],
                    (click_x, click_y),
                )

    separation = _paired_summary(click_sensitivity, null_sensitivity)
    return {
        "sensitivity_click": nanmean(click_sensitivity),
        "sensitivity_control": nanmean(null_sensitivity),
        "sensitivity_ratio": nanmean(click_sensitivity) / max(
            nanmean(null_sensitivity), 1e-9
        ),
        "sensitivity_p_value": separation["p_value"],
        "sensitivity_effect_size": separation["effect_size"],
        "spawn_localization_error_px": nanmean(localization),
        "spawn_hit_rate_100px": float(
            np.mean([value <= 100 for value in localization]) if localization else np.nan
        ),
        "spawn_hit_rate_200px": float(
            np.mean([value <= 200 for value in localization]) if localization else np.nan
        ),
        "ground_truth_spawn_error_px": nanmean(ground_truth_localization),
        "n_click_frames": float(len(click_sensitivity)),
        "n_control_frames": float(len(null_sensitivity)),
        "_heatmap": heatmap_sample,
    }


def _ground_truth_spawn_error(
    episode: Episode, index: int, click: tuple[float, float]
) -> float:
    """Distancia entre a bola que realmente surgiu e a posicao do clique.

    Serve de controle: se este valor nao for proximo de zero, o alinhamento
    acao/frame esta errado e qualquer conclusao sobre H2 seria invalida.
    """
    before = detect_balls(episode.frames[index])
    after = detect_balls(episode.frames[index + 1])
    if not after:
        return float("nan")
    pairs, _ = match_balls(after, before)
    matched = {a for a, _ in pairs}
    new_balls = [ball for position, ball in enumerate(after) if position not in matched]
    candidates = new_balls or after
    distances = [
        float(np.hypot((ball.x - click[0]) * SCALE_X, (ball.y - click[1]) * SCALE_Y))
        for ball in candidates
    ]
    return min(distances)


def _paired_summary(treatment: list[float], control: list[float]) -> dict[str, float]:
    from scipy import stats

    a = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return {"p_value": float("nan"), "effect_size": float("nan")}
    _, p_value = stats.mannwhitneyu(a, b, alternative="greater")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    effect = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
    return {"p_value": float(p_value), "effect_size": float(effect)}


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def save_figures(one_step: dict, rollout: dict, action: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    paths: list[str] = []

    # 1. Curva de erro por horizonte
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    labels = {
        "world_model": "World model (proposto)",
        "static": "Persistencia (bola parada)",
        "constant_velocity": "Velocidade constante",
        "vae_ceiling": "Teto do VAE (oraculo)",
    }
    styles = {
        "world_model": dict(color="#c96a3d", linewidth=2.4, marker="o", markersize=4),
        "static": dict(color="#4a4a4a", linewidth=1.6, linestyle="--"),
        "constant_velocity": dict(color="#3d7fc9", linewidth=1.6, linestyle="-."),
        "vae_ceiling": dict(color="#7aa06a", linewidth=1.4, linestyle=":"),
    }
    for name, values in rollout.items():
        if name.startswith("_"):
            continue
        horizons = np.arange(1, len(values) + 1)
        axis.plot(horizons, values, label=labels.get(name, name), **styles.get(name, {}))
    axis.set_xlabel("Horizonte de predicao (frames a frente)")
    axis.set_ylabel("Erro de posicao da bola (px na tela 800x600)")
    axis.set_title("Rollout em malha aberta: erro por horizonte")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = os.path.join(ARTIFACT_DIR, "rollout_cpe_por_horizonte.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(path)

    # 2. Comparacao de um passo
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    names = [
        "vae_ceiling",
        "world_model",
        "constant_velocity",
        "latent_persistence",
        "world_model_no_action",
    ]
    def series(metric: str, low: str, high: str):
        values = [one_step["per_predictor"][name][metric] for name in names]
        errors = np.array(
            [
                [
                    one_step["per_predictor"][name][metric]
                    - one_step["per_predictor"][name][low]
                    for name in names
                ],
                [
                    one_step["per_predictor"][name][high]
                    - one_step["per_predictor"][name][metric]
                    for name in names
                ],
            ]
        )
        return values[::-1], np.clip(errors[:, ::-1], 0.0, None)

    positions = np.arange(len(names))
    all_values, all_errors = series("cpe", "cpe_ci_low", "cpe_ci_high")
    moving_values, moving_errors = series(
        "cpe_moving", "cpe_moving_ci_low", "cpe_moving_ci_high"
    )
    axis.barh(
        positions + 0.2,
        all_values,
        height=0.38,
        xerr=all_errors,
        color="#9a8f7a",
        label="todas as bolas",
    )
    axis.barh(
        positions - 0.2,
        moving_values,
        height=0.38,
        xerr=moving_errors,
        color="#c96a3d",
        label="apenas bolas em movimento",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(names[::-1])
    axis.set_xlabel("Erro de posicao da bola em 1 passo (px)")
    axis.set_title("Predicao de 1 passo (IC 95% por bootstrap)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    path = os.path.join(ARTIFACT_DIR, "erro_um_passo.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(path)

    # 3. Grade qualitativa
    samples = one_step["qualitative"]
    if samples:
        columns = [
            "frame t (real)",
            "frame t+1 (real)",
            "predito",
            "teto do VAE",
            "persistencia",
        ]
        figure, axes = plt.subplots(
            len(samples), len(columns), figsize=(2.0 * len(columns), 2.0 * len(samples))
        )
        axes = np.atleast_2d(axes)
        for row, sample in enumerate(samples):
            for column in range(len(columns)):
                axes[row, column].imshow(sample[column], cmap="gray", vmin=0, vmax=1)
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(columns[column], fontsize=9)
        figure.tight_layout()
        path = os.path.join(ARTIFACT_DIR, "amostras_qualitativas.png")
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(path)

    # 4. Tira temporal do rollout: real (topo) vs predito (base)
    strip = rollout.get("_strip")
    if strip is not None:
        rows = ["real", "reconstruido", "predito"]
        labels = {
            "real": "real",
            "reconstruido": "reconstruido (teto do VAE)",
            "predito": "predito (rollout)",
        }
        steps = min(8, strip["real"].shape[0])
        indices = np.linspace(0, strip["real"].shape[0] - 1, steps, dtype=int)
        figure, axes = plt.subplots(len(rows), steps, figsize=(1.6 * steps, 5.4))
        for column, index in enumerate(indices):
            for row, key in enumerate(rows):
                axes[row, column].imshow(strip[key][index], cmap="gray", vmin=0, vmax=1)
                axes[row, column].axis("off")
            axes[0, column].set_title(f"t+{index + 1}", fontsize=9)
        for row, key in enumerate(rows):
            axes[row, 0].text(
                -0.3,
                0.5,
                labels[key],
                transform=axes[row, 0].transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=8,
            )
        figure.suptitle(
            f"Rollout em malha aberta — {strip['episodio']} (inicio no frame "
            f"{strip['inicio']}). A linha do meio mostra quanto contraste o "
            f"decoder do VAE perde mesmo com o latente correto.",
            fontsize=9,
        )
        figure.tight_layout()
        path = os.path.join(ARTIFACT_DIR, "rollout_tira_temporal.png")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(path)

    # 5. Queda por gravidade: deslocamento vertical real vs predito
    displacement = rollout.get("_displacement")
    if displacement:
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        horizons = np.arange(1, len(displacement["real"]) + 1)
        axis.plot(
            horizons,
            displacement["real"],
            color="#4a4a4a",
            marker="s",
            markersize=4,
            label="real",
        )
        axis.plot(
            horizons,
            displacement["predito"],
            color="#c96a3d",
            marker="o",
            markersize=4,
            label="predito pelo world model",
        )
        axis.axhline(0.0, color="#3d7fc9", linestyle="--", label="persistencia (0)")
        axis.set_xlabel("Horizonte de predicao (frames)")
        axis.set_ylabel("Deslocamento vertical acumulado (px)")
        axis.set_title("O modelo reproduz a queda por gravidade?")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = os.path.join(ARTIFACT_DIR, "deslocamento_vertical.png")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(path)

    # 6. Heatmap do efeito da acao
    sample = action.get("_heatmap")
    if sample is not None:
        base, difference, (click_x, click_y) = sample
        figure, axis = plt.subplots(figsize=(4.4, 4.0))
        axis.imshow(base, cmap="gray", vmin=0, vmax=1)
        image = axis.imshow(difference, cmap="hot", alpha=0.65)
        axis.scatter([click_x], [click_y], s=90, facecolors="none", edgecolors="cyan")
        axis.set_title("|predicao com acao - sem acao|\ncirculo = posicao do clique")
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046)
        figure.tight_layout()
        path = os.path.join(ARTIFACT_DIR, "efeito_da_acao.png")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Estagio 2 - orquestracao da avaliacao
# ---------------------------------------------------------------------------
def run_evaluation(config: Config, args) -> dict:
    mlflow = _mlflow()
    mlflow.set_experiment(EXPERIMENT_EVALUATION)

    selected = _select_test_episodes(args.episodes, args.family, args.dataset)
    episodes = [
        load_episode(name, path, max_frames=args.max_frames, dataset=args.dataset)
        for name, path in selected
    ]
    print(f"Avaliacao em {len(episodes)} episodios de teste")
    for episode in episodes[:5]:
        print(f"  {episode.name}  T={episode.length}  cliques={len(episode.actions)}")

    started = time.time()
    one_step = run_one_step(config, episodes, args)
    print(f"  [1/3] predicao de 1 passo concluida ({time.time() - started:.0f}s)")
    rollout = run_rollout(config, episodes, args)
    print(f"  [2/3] rollout concluido ({time.time() - started:.0f}s)")
    action = run_action_analysis(config, episodes, args)
    print(f"  [3/3] analise de acao concluida ({time.time() - started:.0f}s)")

    figures = save_figures(one_step, rollout, action)

    common_params = {
        "checkpoint": config.checkpoint_name,
        "checkpoint_file": os.path.basename(config.checkpoint_path),
        "memory_frames": config.memory_frames,
        "memory_stride": config.memory_stride,
        "rollout_feedback": config.feedback,
        "n_test_episodes": len(episodes),
        "episode_family": args.family or "todas",
        "dataset": args.dataset,
        "split_seed": SEED,
        "test_ratio": TEST_RATIO,
        "warmup_frames": WARMUP_FRAMES,
        "detector_relative_threshold": RELATIVE_THRESHOLD,
        "detector_min_area": MIN_BALL_AREA,
    }

    for name, metrics in one_step["per_predictor"].items():
        with mlflow.start_run(run_name=f"one_step__{name}"):
            mlflow.log_params({**common_params, "predictor": name})
            _log_metrics(mlflow, metrics)
            if name == "world_model":
                _log_metrics(mlflow, one_step["latent"])
                _log_metrics(
                    mlflow,
                    {
                        f"vs_persistence_{key}": value
                        for key, value in one_step["paired_vs_persistence"].items()
                    },
                )
                _log_metrics(
                    mlflow,
                    {
                        f"vs_constant_velocity_{key}": value
                        for key, value in one_step["paired_vs_constant_velocity"].items()
                    },
                )
                for label in ("persistence", "constant_velocity"):
                    _log_metrics(
                        mlflow,
                        {
                            f"por_episodio_vs_{label}_{key}": value
                            for key, value in one_step[
                                f"paired_by_episode_vs_{label}"
                            ].items()
                        },
                    )
                    _log_metrics(
                        mlflow,
                        {
                            f"bolas_moveis_vs_{label}_{key}": value
                            for key, value in one_step[
                                f"paired_moving_vs_{label}"
                            ].items()
                        },
                    )

    for name, values in rollout.items():
        if name.startswith("_"):
            continue
        with mlflow.start_run(run_name=f"rollout__{name}"):
            mlflow.log_params({**common_params, "predictor": name})
            for step, value in enumerate(values, start=1):
                if np.isfinite(value):
                    mlflow.log_metric("rollout_cpe", value, step=step)
            finite = [value for value in values if np.isfinite(value)]
            _log_metrics(
                mlflow,
                {
                    "rollout_cpe_mean": nanmean(values),
                    "rollout_cpe_final": finite[-1] if finite else float("nan"),
                    "rollout_horizon": float(len(values)),
                },
            )
            if name == "world_model":
                displacement = rollout["_displacement"]
                for step, value in enumerate(displacement["predito"], start=1):
                    if np.isfinite(value):
                        mlflow.log_metric("rollout_queda_predita_px", value, step=step)
                for step, value in enumerate(displacement["real"], start=1):
                    if np.isfinite(value):
                        mlflow.log_metric("rollout_queda_real_px", value, step=step)
                _log_metrics(
                    mlflow,
                    {
                        "queda_capturada_fracao": (
                            displacement["predito"][-1] / displacement["real"][-1]
                            if displacement["real"][-1]
                            else float("nan")
                        )
                    },
                )

    with mlflow.start_run(run_name="action_conditioning"):
        mlflow.log_params({**common_params, "predictor": "world_model"})
        _log_metrics(
            mlflow, {k: v for k, v in action.items() if not k.startswith("_")}
        )

    summary = {
        "config": asdict(config),
        "params": common_params,
        "one_step": {
            "per_predictor": one_step["per_predictor"],
            "latent": one_step["latent"],
            "paired_vs_persistence": one_step["paired_vs_persistence"],
            "paired_vs_constant_velocity": one_step["paired_vs_constant_velocity"],
            "paired_by_episode_vs_persistence": one_step[
                "paired_by_episode_vs_persistence"
            ],
            "paired_by_episode_vs_constant_velocity": one_step[
                "paired_by_episode_vs_constant_velocity"
            ],
            "paired_moving_vs_persistence": one_step["paired_moving_vs_persistence"],
            "paired_moving_vs_constant_velocity": one_step[
                "paired_moving_vs_constant_velocity"
            ],
        },
        "rollout": {k: v for k, v in rollout.items() if not k.startswith("_")},
        "deslocamento_vertical": rollout["_displacement"],
        "action": {k: v for k, v in action.items() if not k.startswith("_")},
        "episodes": [episode.name for episode in episodes],
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    summary_path = os.path.join(ARTIFACT_DIR, "resultados.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    with mlflow.start_run(run_name="resumo_avaliacao"):
        mlflow.log_params(common_params)
        _log_metrics(
            mlflow,
            {
                "headline_cpe_1step": one_step["per_predictor"]["world_model"]["cpe"],
                "headline_cpe_1step_persistence": one_step["per_predictor"][
                    "latent_persistence"
                ]["cpe"],
                "headline_skill_score": one_step["latent"]["skill_score"],
                "headline_rollout_cpe_final": rollout["world_model"][-1],
                "headline_rollout_cpe_final_persistence": rollout["static"][-1],
                "headline_action_ratio": action["sensitivity_ratio"],
            },
        )
        mlflow.log_artifact(summary_path)
        for path in figures:
            mlflow.log_artifact(path)

    _print_report(summary)
    return summary


def _print_report(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("RESUMO — predicao de 1 passo (erro de posicao da bola, px em 800x600)")
    print("=" * 78)
    print(
        f"{'preditor':26s} {'CPE':>8s} {'IC95%':>18s} {'CPE_mov':>8s} "
        f"{'PSNR':>7s} {'SSIM':>6s}"
    )
    for name, metrics in summary["one_step"]["per_predictor"].items():
        print(
            f"{name:26s} {metrics['cpe']:8.2f} "
            f"[{metrics['cpe_ci_low']:7.2f},{metrics['cpe_ci_high']:7.2f}] "
            f"{metrics.get('cpe_moving', float('nan')):8.2f} "
            f"{metrics.get('px_psnr', float('nan')):7.2f} "
            f"{metrics.get('px_ssim', float('nan')):6.3f}"
        )
    latent = summary["one_step"]["latent"]
    print(
        f"\nLatente: MSE={latent['latent_mse']:.5f} "
        f"persistencia={latent['latent_mse_persistence']:.5f} "
        f"skill={latent['skill_score']:+.4f} cos={latent['delta_cosine']:+.3f}"
    )
    for label, title in (
        ("persistence", "persistencia"),
        ("constant_velocity", "velocidade constante"),
    ):
        by_frame = summary["one_step"][f"paired_vs_{label}"]
        by_episode = summary["one_step"][f"paired_by_episode_vs_{label}"]
        print(
            f"vs {title:20s} por frame: ganho={by_frame['mean_difference']:+.2f} px "
            f"win={by_frame['win_rate']:.1%} p={by_frame['p_value']:.1e} | "
            f"por episodio: ganho={by_episode['mean_difference']:+.2f} px "
            f"[{by_episode['difference_ci_low']:+.2f},"
            f"{by_episode['difference_ci_high']:+.2f}] "
            f"win={by_episode['win_rate']:.1%} p={by_episode['p_value']:.1e}"
        )
    print("\nRollout (CPE por horizonte):")
    for label, title in (
        ("persistence", "persistencia"),
        ("constant_velocity", "velocidade constante"),
    ):
        moving = summary["one_step"][f"paired_moving_vs_{label}"]
        print(
            f"So bolas em movimento vs {title:20s}: "
            f"ganho={moving['mean_difference']:+.2f} px "
            f"win={moving['win_rate']:.1%} p={moving['p_value']:.1e}"
        )
    horizons = [1, 2, 4, 8, 12, 16]
    header = "".join(f"{h:>9d}" for h in horizons)
    print(f"{'preditor':26s}{header}")
    for name, values in summary["rollout"].items():
        row = "".join(
            f"{values[h - 1]:9.1f}" if h <= len(values) else " " * 9 for h in horizons
        )
        print(f"{name:26s}{row}")
    action = summary["action"]
    print(
        f"\nAcao: sensibilidade clique={action['sensitivity_click']:.5f} "
        f"controle={action['sensitivity_control']:.5f} "
        f"razao={action['sensitivity_ratio']:.2f}x p={action['sensitivity_p_value']:.2e}"
    )
    print(
        f"      erro de localizacao do spawn={action['spawn_localization_error_px']:.1f} px "
        f"(controle real={action['ground_truth_spawn_error_px']:.1f} px)"
    )
    print("=" * 78)


# ---------------------------------------------------------------------------
# Estagio 3 - registro da melhor configuracao
# ---------------------------------------------------------------------------
def register_best(config: Config, summary: dict) -> None:
    mlflow = _mlflow()
    mlflow.set_experiment(EXPERIMENT_EVALUATION)
    import mlflow.pytorch

    runner = WorldModelRunner(
        checkpoint=config.checkpoint_path,
        device="cpu",
        memory_frames=config.memory_frames,
        memory_stride=config.memory_stride,
    )
    with mlflow.start_run(run_name="melhores_parametros"):
        mlflow.log_params(
            {
                **summary["params"],
                "arquitetura": "ConvLSTM x2 (hidden=64) + 2 blocos residuais, saida delta",
                "latente_visual": 16,
                "latente_acao": 16,
                "grade_latente": "8x8",
                "frame": "64x64 escala de cinza",
                "regra_de_predicao": "mu_{t+1} = mu_t + f(historico fundido)",
            }
        )
        predictors = summary["one_step"]["per_predictor"]
        by_episode = summary["one_step"]["paired_by_episode_vs_persistence"]
        moving = summary["one_step"]["paired_moving_vs_persistence"]
        rollout = summary["rollout"]
        _log_metrics(
            mlflow,
            {
                # Predicao de 1 passo — modelo e baselines lado a lado, para que
                # este run sozinho conte a historia completa.
                "cpe_1step": predictors["world_model"]["cpe"],
                "cpe_1step_bolas_moveis": predictors["world_model"]["cpe_moving"],
                "cpe_1step_persistencia": predictors["latent_persistence"]["cpe"],
                "cpe_1step_persistencia_bolas_moveis": predictors[
                    "latent_persistence"
                ]["cpe_moving"],
                "cpe_1step_velocidade_constante": predictors["constant_velocity"][
                    "cpe"
                ],
                "cpe_1step_teto_vae": predictors["vae_ceiling"]["cpe"],
                "psnr_1step": predictors["world_model"]["px_psnr"],
                "ssim_1step": predictors["world_model"]["px_ssim"],
                # Espaco latente
                "skill_score": summary["one_step"]["latent"]["skill_score"],
                "delta_cosine": summary["one_step"]["latent"]["delta_cosine"],
                "latent_win_rate": summary["one_step"]["latent"]["latent_win_rate"],
                # Significancia (unidade experimental = episodio)
                "ganho_vs_persistencia_px": by_episode["mean_difference"],
                "ganho_vs_persistencia_p": by_episode["p_value"],
                "ganho_bolas_moveis_px": moving["mean_difference"],
                "ganho_bolas_moveis_p": moving["p_value"],
                # Rollout
                "rollout_cpe_h1": rollout["world_model"][0],
                "rollout_cpe_h16": rollout["world_model"][-1],
                "rollout_cpe_h16_persistencia": rollout["static"][-1],
                "rollout_cpe_h16_velocidade_constante": rollout["constant_velocity"][
                    -1
                ],
                "queda_real_h16_px": summary["deslocamento_vertical"]["real"][-1],
                "queda_predita_h16_px": summary["deslocamento_vertical"]["predito"][-1],
                # Acao (H2)
                "action_sensitivity_ratio": summary["action"]["sensitivity_ratio"],
                "action_sensitivity_p": summary["action"]["sensitivity_p_value"],
                "spawn_erro_px": summary["action"]["spawn_localization_error_px"],
                "spawn_erro_controle_px": summary["action"][
                    "ground_truth_spawn_error_px"
                ],
            },
        )
        mlflow.set_tag(
            "descricao",
            "Configuracao vencedora do world model (checkpoint + memoria temporal).",
        )
        mlflow.set_tag("relatorio", "docs/relatorio_avaliacao_world_model.md")
        mlflow.set_tag("harness", "scripts/eval/run_world_model_eval.py")
        example = torch.zeros(
            1, config.memory_frames, 32, 8, 8, dtype=torch.float32
        )
        mlflow.pytorch.log_model(
            pytorch_model=runner.bundle.transition_model,
            name="latent_transition_model",
            input_example=example.numpy(),
        )
        mlflow.log_artifact(os.path.join(ARTIFACT_DIR, "resultados.json"))
        mlflow.log_artifact(config.checkpoint_path)
    print("Melhores parametros e modelo registrados no MLflow.")


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["selection", "evaluation", "figures", "best", "all"],
        default="all",
        help=(
            "figures: redesenha as figuras a partir de resultados.json; "
            "best: registra a melhor configuracao no MLflow sem recomputar."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dataset",
        default="holdout",
        help="holdout (episodios ineditos) ou train_pool (episodios do dataset original).",
    )
    parser.add_argument(
        "--history",
        choices=["contiguous", "spaced", "all"],
        default="contiguous",
        help="contiguous: janela dos k frames mais recentes, como nas eq. (1) e (5).",
    )
    parser.add_argument("--episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=ROLLOUT_HORIZON)
    parser.add_argument(
        "--family",
        default=None,
        help="Prefixo de timestamp do lote de episodios (ex.: 20260609).",
    )
    parser.add_argument("--checkpoint", default=None, help="Forca um checkpoint.")
    parser.add_argument("--memory-frames", type=int, default=None)
    parser.add_argument("--memory-stride", type=int, default=None)
    return parser.parse_args()


def rebuild_figures() -> None:
    """Redesenha as figuras derivadas de `resultados.json`, sem recomputar nada."""
    with open(os.path.join(ARTIFACT_DIR, "resultados.json")) as handle:
        summary = json.load(handle)
    rollout = dict(summary["rollout"])
    rollout["_displacement"] = summary.get("deslocamento_vertical")
    paths = save_figures(
        {"per_predictor": summary["one_step"]["per_predictor"], "qualitative": []},
        rollout,
        {},
    )
    print("Figuras regeneradas:")
    for path in paths:
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.stage == "figures":
        rebuild_figures()
        return

    if args.stage == "best":
        with open(os.path.join(ARTIFACT_DIR, "resultados.json")) as handle:
            summary = json.load(handle)
        stored = summary["config"]
        register_best(
            Config(
                checkpoint_name=stored["checkpoint_name"],
                checkpoint_path=stored["checkpoint_path"],
                memory_frames=stored["memory_frames"],
                memory_stride=stored["memory_stride"],
                feedback=stored["feedback"],
            ),
            summary,
        )
        return

    config: Config | None = None
    if args.stage in {"selection", "all"}:
        config = run_selection(args)
    if args.checkpoint or config is None:
        path = args.checkpoint or CHECKPOINTS["delta_convlstm_v1"]
        name = next(
            (key for key, value in CHECKPOINTS.items() if value == path),
            os.path.basename(path),
        )
        config = Config(
            checkpoint_name=name,
            checkpoint_path=path,
            memory_frames=args.memory_frames or 10,
            memory_stride=args.memory_stride or 1,
        )
    if args.memory_frames:
        config.memory_frames = args.memory_frames
    if args.memory_stride:
        config.memory_stride = args.memory_stride

    if args.stage in {"evaluation", "all"}:
        summary = run_evaluation(config, args)
        register_best(config, summary)


if __name__ == "__main__":
    main()
