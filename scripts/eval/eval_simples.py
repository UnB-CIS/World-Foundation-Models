"""Harness de avaliação enxuta: métricas de pixel, sensibilidade à ação e heatmap."""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from metrics_fisicas import pixel_metrics, _to_2d


def action_sensitivity(predict_fn, history, action, null_action):
    """(1/HW)·Σ|pred(a) − pred(0)| a partir do mesmo estado. ~0 para modelo cego à ação."""
    pred_a = _to_2d(predict_fn(history, action))
    pred_0 = _to_2d(predict_fn(history, null_action))
    diff = np.abs(pred_a - pred_0)
    return float(diff.mean()), diff


def save_diff_heatmap(diff_map, base_frame, path="action_heatmap.png"):
    """Mapa |com_ação − sem_ação| sobreposto ao frame anterior, como artefato."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(_to_2d(base_frame), cmap="gray", vmin=0, vmax=1)
    hm = ax.imshow(diff_map, cmap="hot", alpha=0.6)
    ax.axis("off")
    ax.set_title("Efeito da ação  |com − sem|")
    fig.colorbar(hm, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def run_simple_eval(configs, samples, null_action,
                    experiment="wfm_simple", params=None):
    import mlflow

    samples = list(samples)
    mlflow.set_experiment(experiment)

    for name, predict_fn in configs.items():
        with mlflow.start_run(run_name=name):
            if params:
                mlflow.log_params(params)

            agg = defaultdict(list)
            sens_vals = []
            heatmap_done = False

            for s in samples:
                pred = predict_fn(s["history"], s["action"])
                for k, v in pixel_metrics(s["gt_next"], pred).items():
                    if not (isinstance(v, float) and np.isnan(v)):
                        agg[k].append(float(v))

                if s.get("is_action_frame"):
                    val, diff = action_sensitivity(
                        predict_fn, s["history"], s["action"], null_action)
                    sens_vals.append(val)
                    if not heatmap_done:            # 1 heatmap por configuração
                        path = save_diff_heatmap(diff, s["history"][-1])
                        mlflow.log_artifact(path)
                        heatmap_done = True

            for k, vals in agg.items():
                mlflow.log_metric(f"mean_{k}", float(np.mean(vals)))
            if sens_vals:
                mlflow.log_metric("action_sensitivity", float(np.mean(sens_vals)))

    print("Avaliação concluída. Rode `mlflow ui` para comparar os runs.")


# ---------------------------------------------------------------------------
# Demo / auto-teste (roda sem MLflow nem torch)
# ---------------------------------------------------------------------------
def _demo():
    from metrics_fisicas import _draw_ball

    H = W = 64
    bg = np.ones((H, W), dtype=np.float32)
    f = bg.copy()
    _draw_ball(f, 20, 50, r=3)                 # uma bolinha já em repouso
    history = np.stack([f, f])                  # janela k=2
    null = np.zeros(4, dtype=np.float32)
    action = np.array([45 / W, 30 / H, 0, 0], dtype=np.float32)  # clique em (45,30)

    # predict_fns sintéticos para validar a lógica
    def proposto(h, a):
        out = h[-1].copy()
        if a is not None and np.any(a[:2]):     # tem clique -> insere bolinha
            _draw_ball(out, a[0] * W, a[1] * H, r=3)
        return out

    def no_action(h, a):                         # ignora a ação
        return h[-1]

    copy_last = lambda h, a: h[-1]

    print("Sensibilidade à ação (esperado: proposto alto, resto ~0):")
    for name, fn in [("proposto", proposto), ("no_action", no_action), ("copy", copy_last)]:
        val, _ = action_sensitivity(fn, history, action, null)
        print(f"  {name:16s} = {val:.4f}")

    _, diff = action_sensitivity(proposto, history, action, null)
    out = save_diff_heatmap(diff, history[-1], "/home/claude/demo_heatmap.png")
    print(f"\nHeatmap salvo em {out}")
    print("Demo OK")


if __name__ == "__main__":
    _demo()
