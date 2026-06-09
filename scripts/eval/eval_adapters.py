from __future__ import annotations

import json
import numpy as np
import torch

ACT_DIM = 4
NULL_ACTION = np.zeros(ACT_DIM, dtype=np.float32)

_TYPE_ENCODING   = {"mouse_down": [1.0], "none": [0.0]}
_OBJECT_ENCODING = {"ball": [1.0], "none": [0.0]}


def _last_frame_tensor(history, device):
    frame = np.asarray(history[-1], dtype=np.float32)
    return torch.from_numpy(frame)[None, None].to(device)


def make_encode_mu(vae_encoder):
    """Extrai apenas mu do encoder, compatível com saída (mu, logvar) ou (1,16,H,W)."""
    @torch.no_grad()
    def encode_mu(frame_t):
        out = vae_encoder(frame_t)
        if isinstance(out, (tuple, list)):
            return out[0]
        mu, _ = torch.chunk(out, 2, dim=1)
        return mu
    return encode_mu


def make_model_predict_fn(
    encode_mu, action_encoder, fuser, latent_decoder, vae_decoder,
    device="cpu", force_null=False,
):
    """force_null=True implementa a ablação No-Action: arquitetura completa, ação sempre nula."""
    @torch.no_grad()
    def predict_fn(history, action):
        frame_t = _last_frame_tensor(history, device)
        mu_t    = encode_mu(frame_t)

        a = NULL_ACTION if (force_null or action is None) else np.asarray(action, dtype=np.float32)
        a_t = torch.from_numpy(a)[None].to(device)

        a_emb   = action_encoder(a_t)
        fused   = fuser(mu_t, a_emb)
        # LatentDecoder.forward espera tupla (latent, _); o segundo elemento é ignorado internamente
        mu_pred = latent_decoder((fused, None))
        frame_pred = vae_decoder(mu_pred)
        return frame_pred.squeeze().clamp(0.0, 1.0).cpu().numpy()

    return predict_fn


def copy_last_frame(history, action):
    return np.asarray(history[-1], dtype=np.float32)


def constant_velocity(history, action):
    """Extrapolação linear: clip(2·x_t − x_{t-1}, 0, 1). Requer k >= 2."""
    h = np.asarray(history, dtype=np.float32)
    return np.clip(2.0 * h[-1] - h[-2], 0.0, 1.0)


def load_models(ckpt_dir="checkpoints", device="cpu"):
    from src.vae.model import VAE
    from src.action_encoder.model import ActionTextModel
    from src.fusion.model import SpatialBroadcastFuser
    from src.fusion.decoder import LatentDecoder

    def _load(module, path):
        sd = torch.load(path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        module.load_state_dict(sd)
        return module.to(device).eval()

    # O checkpoint do VAE salva VAE() inteiro; encoder e decoder são submodules
    vae     = _load(VAE(), f"{ckpt_dir}/vae/best_model.pth")
    vae_enc = vae.encoder
    vae_dec = vae.decoder

    # O checkpoint salva ActionTextModel (encoder + decoder); usamos só o encoder
    act_model = _load(
        ActionTextModel(input_dim=4, latent_dim=16, output_dim=4),
        f"{ckpt_dir}/action_encoder/best_model.pth",
    )
    act_enc = act_model.encoder

    # SpatialBroadcastFuser não tem parâmetros treináveis
    fuser   = SpatialBroadcastFuser().to(device).eval()
    lat_dec = _load(LatentDecoder(), f"{ckpt_dir}/fusion/best_model.pth")

    return dict(vae_enc=vae_enc, vae_dec=vae_dec, act_enc=act_enc,
                fuser=fuser, lat_dec=lat_dec)


def build_configs(models, device="cpu"):
    encode_mu = make_encode_mu(models["vae_enc"])
    common = dict(
        action_encoder=models["act_enc"], fuser=models["fuser"],
        latent_decoder=models["lat_dec"], vae_decoder=models["vae_dec"],
        device=device,
    )
    return {
        "proposto":          make_model_predict_fn(encode_mu, **common, force_null=False),
        "no_action":         make_model_predict_fn(encode_mu, **common, force_null=True),
        "copy_last_frame":   copy_last_frame,
        "constant_velocity": constant_velocity,
    }


def _make_action_vec(event, x_norm, y_norm):
    """Layout [type, object, x_norm, y_norm] — idêntico ao usado no treino."""
    from src.action_encoder.encoding import encoding_function
    action_data = {
        "type":   event.get("type", "mouse_down"),
        "object": event.get("object", "ball"),
        "pos":    [int(x_norm * 800), int(y_norm * 600)],
    }
    return encoding_function(
        type_encoding=_TYPE_ENCODING,
        object_encoding=_OBJECT_ENCODING,
        action_data=action_data,
        screen_width=800,
        screen_height=600,
    ).numpy().astype(np.float32)


def video_to_json_path(video_path, inputs_dir="scripts/dataset/inputs"):
    """Vídeos têm sufixo _auto_DATA_HORA que os JSONs não têm; strip via regex."""
    import os, re
    stem = os.path.splitext(os.path.basename(video_path))[0]
    base = re.sub(r"_auto_.*$", "", stem)
    return os.path.join(inputs_dir, base + ".json")


def build_samples(video_path, json_path, k=2, fps=60, screen=(800, 600)):
    import cv2

    cap, frames = cv2.VideoCapture(video_path), []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(g, (64, 64)).astype(np.float32) / 255.0)
    cap.release()
    if len(frames) < k + 1:
        return []
    frames = np.stack(frames)

    W, H   = screen
    events = json.load(open(json_path))
    if isinstance(events, dict):
        events = events.get("events", [events])

    action_at: dict[int, np.ndarray] = {}
    for ev in events:
        idx    = int(round(ev["time"] * fps))
        x_norm = ev["pos"][0] / W
        y_norm = ev["pos"][1] / H  # coordenadas de tela: Y já aponta para baixo, sem flip
        action_at[idx] = _make_action_vec(ev, x_norm, y_norm)

    samples = []
    for t in range(k - 1, len(frames) - 1):
        a = action_at.get(t)
        samples.append({
            "history":        frames[t - k + 1: t + 1],
            "action":         a if a is not None else NULL_ACTION,
            "gt_next":        frames[t + 1],
            "is_action_frame": a is not None,
        })
    return samples


def _self_test():
    import torch.nn as nn
    torch.manual_seed(0)

    class Enc(nn.Module):
        def __init__(self): super().__init__(); self.c = nn.Conv2d(1, 16, 8, 8)
        def forward(self, x): return self.c(x)

    class Act(nn.Module):
        def __init__(self): super().__init__(); self.f = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        def forward(self, a): return self.f(a)

    class Fuser(nn.Module):
        def forward(self, mu, a_emb):
            b, _, h, w = mu.shape
            return torch.cat([mu, a_emb[:, :, None, None].expand(b, 16, h, w)], dim=1)

    class LatDec(nn.Module):
        def __init__(self): super().__init__(); self.c = nn.Conv2d(24, 8, 1)
        def forward(self, inputs):
            latent, _ = inputs
            return self.c(latent)

    class Dec(nn.Module):
        def __init__(self): super().__init__(); self.c = nn.ConvTranspose2d(8, 1, 8, 8)
        def forward(self, x): return torch.sigmoid(self.c(x))

    models  = dict(vae_enc=Enc().eval(), vae_dec=Dec().eval(), act_enc=Act().eval(),
                   fuser=Fuser().eval(), lat_dec=LatDec().eval())
    configs = build_configs(models, device="cpu")

    hist   = np.random.rand(2, 64, 64).astype(np.float32)
    action = np.array([1.0, 0.7, 0.45, 1.0], dtype=np.float32)

    for name, fn in configs.items():
        out = fn(hist, action)
        assert out.shape == (64, 64), f"{name}: shape {out.shape}"
        assert 0.0 <= out.min() and out.max() <= 1.0001, f"{name}: fora de [0,1]"
        print(f"  {name:18s} -> {out.shape}  [{out.min():.2f}, {out.max():.2f}]")

    from eval_simples import action_sensitivity
    s_prop,  _ = action_sensitivity(configs["proposto"],  hist, action, NULL_ACTION)
    s_noact, _ = action_sensitivity(configs["no_action"], hist, action, NULL_ACTION)
    print(f"\n  proposto={s_prop:.4f}  no_action={s_noact:.4f}")
    assert s_prop > 0 and abs(s_noact) < 1e-9


if __name__ == "__main__":
    _self_test()
