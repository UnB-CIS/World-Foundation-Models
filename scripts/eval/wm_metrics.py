"""Metricas de avaliacao do world model 2D.

Convencao: frame (H, W) em [0, 1], bolas sao objetos ESCUROS sobre fundo claro.
Centroides sao reportados como (x=coluna, y=linha) na grade 64x64 do modelo, e
os erros de posicao sao convertidos para pixels da tela original (800x600), que
e a unidade fisica do cenario.

O detector usa limiar ADAPTATIVO (relativo ao contraste de cada frame) porque o
decoder do VAE nao reproduz o preto absoluto das bolas: no frame real a bola vale
~0.0 sobre fundo 1.0, enquanto no frame decodificado ela vale ~0.5. Um limiar fixo
detectaria bolas apenas nos frames reais, inviabilizando qualquer comparacao.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MODEL_FRAME_SIZE = 64

# Fatores de conversao da grade do modelo (64x64) para a tela do cenario.
SCALE_X = SCREEN_WIDTH / MODEL_FRAME_SIZE  # 12.5 px por celula
SCALE_Y = SCREEN_HEIGHT / MODEL_FRAME_SIZE  # 9.375 px por celula

# Parametros do detector, calibrados na secao "calibracao" do relatorio:
# o valor de RELATIVE_THRESHOLD e o que iguala a contagem de bolas entre o frame
# real e o teto do VAE (decode(encode(frame_real))) no conjunto de validacao.
RELATIVE_THRESHOLD = 0.25
MIN_BALL_AREA = 2
MAX_BALL_ASPECT = 3.0
MIN_CONTRAST = 0.06


@dataclass(frozen=True)
class Ball:
    """Bola detectada. x/y em celulas da grade 64x64 (subpixel)."""

    x: float
    y: float
    area: int
    mass: float


def to_2d(frame) -> np.ndarray:
    array = np.squeeze(np.asarray(frame, dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"Esperado frame 2D apos squeeze, recebido {array.shape}")
    return array


def detect_balls(
    frame,
    relative_threshold: float = RELATIVE_THRESHOLD,
    min_area: int = MIN_BALL_AREA,
    max_aspect: float = MAX_BALL_ASPECT,
    min_contrast: float = MIN_CONTRAST,
) -> list[Ball]:
    """Segmenta bolas por componentes conexas com limiar adaptativo.

    O fundo e estimado pela mediana do frame; o limiar fica a
    `relative_threshold` do caminho entre o fundo e o pixel mais escuro.
    O centroide e ponderado pela "massa" (fundo - intensidade), o que da
    resolucao subpixel — essencial porque o deslocamento tipico da bola entre
    dois frames consecutivos e menor que uma celula da grade 64x64.
    """
    array = to_2d(frame)
    background = float(np.median(array))
    contrast = background - float(array.min())
    if contrast < min_contrast:
        return []

    binary = array < background - relative_threshold * contrast
    labels, count = ndimage.label(binary)

    balls: list[Ball] = []
    for label in range(1, count + 1):
        mask = labels == label
        ys, xs = np.where(mask)
        area = int(xs.size)
        if area < min_area:
            continue
        height = ys.max() - ys.min() + 1
        width = xs.max() - xs.min() + 1
        aspect = max(height, width) / max(min(height, width), 1)
        if aspect > max_aspect:
            # Descarta faixas alongadas (o chao entra no limiar em frames escuros).
            continue
        weights = np.clip(background - array[mask], 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            continue
        balls.append(
            Ball(
                x=float((xs * weights).sum() / total),
                y=float((ys * weights).sum() / total),
                area=area,
                mass=total,
            )
        )
    return balls


def match_balls(
    reference: list[Ball], candidate: list[Ball]
) -> tuple[list[tuple[int, int]], list[float]]:
    """Casamento otimo (algoritmo Hungaro) em distancia euclidiana na tela 800x600."""
    if not reference or not candidate:
        return [], []
    cost = np.empty((len(reference), len(candidate)), dtype=np.float64)
    for i, a in enumerate(reference):
        for j, b in enumerate(candidate):
            cost[i, j] = np.hypot((a.x - b.x) * SCALE_X, (a.y - b.y) * SCALE_Y)
    rows, cols = linear_sum_assignment(cost)
    pairs = list(zip(rows.tolist(), cols.tolist(), strict=True))
    return pairs, [float(cost[r, c]) for r, c in pairs]


def centroid_position_error(
    reference_balls: list[Ball], predicted_balls: list[Ball]
) -> dict[str, float]:
    """CPE medio (px da tela 800x600) sobre bolas casadas + erro de contagem."""
    _, distances = match_balls(reference_balls, predicted_balls)
    return {
        "cpe": float(np.mean(distances)) if distances else float("nan"),
        "cpe_max": float(np.max(distances)) if distances else float("nan"),
        "count_error": float(abs(len(reference_balls) - len(predicted_balls))),
        "count_ratio": (
            float(len(predicted_balls) / len(reference_balls))
            if reference_balls
            else float("nan")
        ),
        "n_reference": float(len(reference_balls)),
        "n_predicted": float(len(predicted_balls)),
    }


def pixel_metrics(reference, predicted) -> dict[str, float]:
    """MSE, MAE, PSNR e SSIM entre dois frames em [0, 1]."""
    a = to_2d(reference)
    b = to_2d(predicted)
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    try:
        from skimage.metrics import structural_similarity

        ssim = float(structural_similarity(a, b, data_range=1.0))
    except Exception:  # pragma: no cover - skimage e opcional
        ssim = float("nan")
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def latent_metrics(
    mu_current: np.ndarray,
    mu_next_true: np.ndarray,
    mu_next_predicted: np.ndarray,
) -> dict[str, float]:
    """Metricas no espaco latente, que isolam o modelo de transicao do VAE.

    - `latent_mse`: erro do modelo.
    - `latent_mse_persistence`: erro da persistencia (mu_next = mu_current).
    - `skill_score`: 1 - MSE_modelo / MSE_persistencia. Positivo => o modelo
      prediz o proximo estado melhor do que assumir que nada muda.
    - `delta_cosine`: cosseno entre a variacao predita e a real. Mede se o
      modelo acerta a DIRECAO da mudanca, independente da magnitude.
    - `delta_magnitude_ratio`: |delta_pred| / |delta_true|. ~1 e o ideal;
      >1 indica que o modelo injeta movimento inexistente.
    """
    true_delta = (mu_next_true - mu_current).ravel()
    predicted_delta = (mu_next_predicted - mu_current).ravel()
    error = float(np.mean((mu_next_predicted - mu_next_true) ** 2))
    persistence = float(np.mean((mu_current - mu_next_true) ** 2))

    true_norm = float(np.linalg.norm(true_delta))
    predicted_norm = float(np.linalg.norm(predicted_delta))
    cosine = (
        float(np.dot(true_delta, predicted_delta) / (true_norm * predicted_norm))
        if true_norm > 1e-8 and predicted_norm > 1e-8
        else float("nan")
    )
    return {
        "latent_mse": error,
        "latent_mse_persistence": persistence,
        "skill_score": 1.0 - error / persistence if persistence > 1e-12 else float("nan"),
        "delta_cosine": cosine,
        "delta_magnitude_ratio": (
            predicted_norm / true_norm if true_norm > 1e-8 else float("nan")
        ),
    }


def extrapolate_balls(
    previous: list[Ball], current: list[Ball], horizon: int
) -> list[Ball]:
    """Baseline de velocidade constante em espaco de trajetoria.

    Casa as bolas entre t-1 e t, estima a velocidade por diferenca finita e
    projeta linearmente `horizon` passos. Bolas sem par mantem a posicao.
    """
    pairs, _ = match_balls(current, previous)
    previous_by_current = {c: p for c, p in pairs}
    extrapolated: list[Ball] = []
    for index, ball in enumerate(current):
        source = previous_by_current.get(index)
        if source is None:
            extrapolated.append(ball)
            continue
        vx = ball.x - previous[source].x
        vy = ball.y - previous[source].y
        extrapolated.append(
            Ball(
                x=float(np.clip(ball.x + vx * horizon, 0, MODEL_FRAME_SIZE - 1)),
                y=float(np.clip(ball.y + vy * horizon, 0, MODEL_FRAME_SIZE - 1)),
                area=ball.area,
                mass=ball.mass,
            )
        )
    return extrapolated


def moving_indices(
    before: list[Ball], after: list[Ball], threshold_px: float = 3.0
) -> set[int]:
    """Indices de `after` que se deslocaram mais que `threshold_px` desde `before`.

    Bolas paradas no chao sao triviais para a persistencia e diluem qualquer
    comparacao de dinamica; restringir as bolas em movimento isola exatamente a
    pergunta "o modelo sabe para onde a bola vai?".

    Bolas NOVAS (sem par no frame anterior) ficam de fora de proposito: elas
    testam a resposta a acao (H2), nao a dinamica (H1).
    """
    if not before or not after:
        return set()
    pairs, distances = match_balls(after, before)
    return {
        index
        for (index, _), distance in zip(pairs, distances, strict=True)
        if distance > threshold_px
    }


def subset_position_error(
    reference_balls: list[Ball], predicted_balls: list[Ball], subset: set[int]
) -> float:
    """CPE restrito a um subconjunto de bolas de referencia.

    O casamento e feito sobre os conjuntos COMPLETOS e so depois filtrado. Casar
    apenas o subconjunto permitiria que uma bola em movimento fosse atribuida a
    uma bola parada da predicao, o que enviesaria a comparacao entre preditores.
    """
    if not subset:
        return float("nan")
    pairs, distances = match_balls(reference_balls, predicted_balls)
    selected = [
        distance
        for (source, _), distance in zip(pairs, distances, strict=True)
        if source in subset
    ]
    return float(np.mean(selected)) if selected else float("nan")


def nanmean(values) -> float:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def bootstrap_ci(
    values, confidence: float = 0.95, iterations: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """IC por bootstrap da media. Usado para reportar incerteza das metricas."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, array.size, size=(iterations, array.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
