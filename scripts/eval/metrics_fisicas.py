"""
Métricas de pixel e detecção de bolinhas para o World Model 2D.

Convenção: frame (H, W) em [0, 1], bolinhas são objetos ESCUROS sobre fundo claro,
centróides reportados como (x=coluna, y=linha).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass
class Ball:
    x: float
    y: float
    r: float
    area: int


def _to_2d(frame) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Esperado frame 2D após squeeze, recebi shape {arr.shape}")
    return arr


def detect_balls(
    frame,
    threshold: float = 0.4,
    min_area: int = 4,
    max_area: Optional[int] = None,
    max_aspect: float = 3.0,
) -> list[Ball]:
    """
    Binariza (pixel < threshold), rotula componentes conexas e filtra por área e aspecto.
    max_aspect descarta faixas horizontais (e.g. chão) que entrem no threshold.
    """
    arr = _to_2d(frame)
    binary = arr < threshold
    labels, n = ndimage.label(binary)

    balls: list[Ball] = []
    for lbl in range(1, n + 1):
        ys, xs = np.where(labels == lbl)
        area = int(xs.size)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        aspect = max(h, w) / max(min(h, w), 1)
        if aspect > max_aspect:
            continue
        balls.append(
            Ball(
                x=float(xs.mean()),
                y=float(ys.mean()),
                r=float(np.sqrt(area / np.pi)),
                area=area,
            )
        )
    return balls


def match_balls(
    gt: Sequence[Ball], pred: Sequence[Ball]
) -> tuple[list[tuple[int, int]], list[float]]:
    """Atribuição ótima gt↔pred via algoritmo Húngaro (distância euclidiana)."""
    if not gt or not pred:
        return [], []
    cost = np.empty((len(gt), len(pred)), dtype=np.float32)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            cost[i, j] = np.hypot(g.x - p.x, g.y - p.y)
    rows, cols = linear_sum_assignment(cost)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    dists = [float(cost[r, c]) for r, c in pairs]
    return pairs, dists


def centroid_position_error(gt_frame, pred_frame, **detect_kw) -> dict:
    """CPE médio sobre bolinhas casadas. Reportar sempre junto de count_error."""
    gt = detect_balls(gt_frame, **detect_kw)
    pred = detect_balls(pred_frame, **detect_kw)
    _, dists = match_balls(gt, pred)
    return {
        "cpe": float(np.mean(dists)) if dists else float("nan"),
        "count_error": abs(len(gt) - len(pred)),
        "n_gt": len(gt),
        "n_pred": len(pred),
    }


def action_response_hit(
    prev_frame,
    pred_frame,
    click_xy_px: tuple[float, float],
    epsilon: float = 4.0,
    match_radius: float = 3.0,
    **detect_kw,
) -> bool:
    """
    True se o frame predito contém uma bolinha nova a menos de epsilon pixels do clique.
    click_xy_px deve estar em pixels (cx = x_norm * W, cy = y_norm * H).
    """
    prev = detect_balls(prev_frame, **detect_kw)
    pred = detect_balls(pred_frame, **detect_kw)
    cx, cy = click_xy_px
    for p in pred:
        is_new = all(np.hypot(p.x - q.x, p.y - q.y) > match_radius for q in prev)
        near_click = np.hypot(p.x - cx, p.y - cy) <= epsilon
        if is_new and near_click:
            return True
    return False


def pixel_metrics(gt_frame, pred_frame) -> dict:
    """MSE, MAE, PSNR e SSIM entre dois frames [0, 1]."""
    gt = _to_2d(gt_frame)
    pred = _to_2d(pred_frame)
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(np.abs(gt - pred)))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    try:
        from skimage.metrics import structural_similarity as ssim
        ss = float(ssim(gt, pred, data_range=1.0))
    except Exception:
        ss = float("nan")
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ss}


def _save_prediction_grid(triplets, path="pred_grid.png"):
    """Grade visual [anterior | real | predito]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(triplets)
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    if n == 1:
        axes = axes[None, :]
    cols = ["anterior", "real", "predito"]
    for i, (prev, gt, pred) in enumerate(triplets):
        for j, img in enumerate((prev, gt, pred)):
            axes[i, j].imshow(_to_2d(img), cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_model(
    predict_fn: Callable,
    samples: Iterable[dict],
    run_name: str,
    params: Optional[dict] = None,
    epsilon: float = 4.0,
    detect_kw: Optional[dict] = None,
    n_grid: int = 8,
) -> dict:
    import mlflow

    detect_kw = detect_kw or {}
    agg: dict[str, list[float]] = defaultdict(list)
    action_hits = action_total = 0
    grid: list = []

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)

        for s in samples:
            pred = predict_fn(s["history"], s["action"])
            gt = s["gt_next"]

            metrics = {**pixel_metrics(gt, pred),
                       **centroid_position_error(gt, pred, **detect_kw)}
            for k, v in metrics.items():
                if isinstance(v, float) and np.isnan(v):
                    continue
                agg[k].append(float(v))

            if s.get("click_px") is not None:
                prev = s["history"][-1]
                hit = action_response_hit(prev, pred, s["click_px"],
                                          epsilon=epsilon, **detect_kw)
                action_hits += int(hit)
                action_total += 1

            if len(grid) < n_grid:
                grid.append((s["history"][-1], gt, pred))

        for k, vals in agg.items():
            mlflow.log_metric(f"mean_{k}", float(np.mean(vals)))
        if action_total:
            mlflow.log_metric("action_response_acc", action_hits / action_total)
            mlflow.log_metric("action_frames_evaluated", action_total)

        if grid:
            path = _save_prediction_grid(grid)
            mlflow.log_artifact(path)

    return {k: float(np.mean(v)) for k, v in agg.items()}


def evaluate_rollout(
    predict_fn: Callable,
    initial_history: np.ndarray,
    future_actions: Sequence,
    gt_future: Sequence,
    run_name: Optional[str] = None,
    detect_kw: Optional[dict] = None,
) -> list[float]:
    """Rollout open-loop: realimenta o modelo com as próprias predições."""
    import mlflow

    detect_kw = detect_kw or {}
    history = list(initial_history)
    cpe_curve: list[float] = []

    ctx = mlflow.start_run(run_name=run_name) if run_name else _NullCtx()
    with ctx:
        for h, (action, gt) in enumerate(zip(future_actions, gt_future)):
            pred = predict_fn(np.stack(history), action)
            cpe = centroid_position_error(gt, pred, **detect_kw)["cpe"]
            cpe_curve.append(cpe)
            if run_name and not np.isnan(cpe):
                mlflow.log_metric("rollout_cpe_by_horizon", cpe, step=h)
            history = history[1:] + [pred]
    return cpe_curve


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _draw_ball(frame, cx, cy, r, value=0.05):
    h, w = frame.shape
    yy, xx = np.ogrid[:h, :w]
    frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = value
    return frame


def _self_test():
    H = W = 64
    bg = np.ones((H, W), dtype=np.float32)

    f = bg.copy()
    for cx, cy in [(16, 40), (32, 20), (48, 50)]:
        _draw_ball(f, cx, cy, r=3)
    assert len(detect_balls(f)) == 3

    g = bg.copy()
    for cx, cy in [(17, 41), (32, 22), (47, 50)]:
        _draw_ball(g, cx, cy, r=3)
    res = centroid_position_error(f, g)
    assert res["count_error"] == 0 and 0 < res["cpe"] < 4

    prev = bg.copy()
    for cx, cy in [(16, 40), (32, 20)]:
        _draw_ball(prev, cx, cy, r=3)
    pred_hit  = _draw_ball(prev.copy(), 45, 30, r=3)
    pred_miss = prev.copy()
    assert action_response_hit(prev, pred_hit,  (45, 30), epsilon=4) is True
    assert action_response_hit(prev, pred_miss, (45, 30), epsilon=4) is False

    pm = pixel_metrics(f, g)
    print(f"mse={pm['mse']:.5f}  psnr={pm['psnr']:.2f}  ssim={pm['ssim']:.4f}")
    print("OK")


if __name__ == "__main__":
    _self_test()
