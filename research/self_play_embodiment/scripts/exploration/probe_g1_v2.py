"""G1 probe v2 — trustworthy eval: episode-level holdout (no adjacent-frame
leakage), single-frame (T=2) vs clip (T=16) readout, and per-joint R²."""

import os
import time

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
OUT, REPO = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256"
dev, dt = "cuda", torch.bfloat16

d = np.load(OUT + "/g1_buffer.npz")
imgs, states = d["images"], d["states"]
ep = [tuple(map(int, b)) for b in d["ep_bounds"]]

from transformers import AutoVideoProcessor, VJEPA2Model

model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, dt)


def encode(clips):
    x = torch.from_numpy(clips).to(dev).permute(0, 1, 4, 2, 3).to(dt) / 255.0
    x = (x - mean) / std
    with torch.no_grad():
        out = model(pixel_values_videos=x)
    h = out.last_hidden_state if getattr(out, "last_hidden_state", None) is not None else out[0]
    return h.float().mean(1).cpu().numpy()


samples = [(f, eid) for eid, (s, e) in enumerate(ep) for f in range(s + 15, e)]
rng = np.random.default_rng(0)
rng.shuffle(samples)
samples = samples[:1500]
frames = np.array([f for f, _ in samples])
eids = np.array([e for _, e in samples])
test_eps = {8, 9}
tr, te = ~np.isin(eids, list(test_eps)), np.isin(eids, list(test_eps))
Y = states[frames].astype(np.float64)
print(f"samples={len(frames)} train={tr.sum()} test={te.sum()} (held-out episodes {test_eps})", flush=True)


def ridge_pred(Xtr, Ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym))
    return Xte @ W + ym


def r2(Yte, P):
    return 1 - ((Yte - P) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9)


for T in [2, 16]:
    try:
        lat, t0 = [], time.time()
        for k in range(0, len(frames), 8):
            sel = frames[k : k + 8]
            lat.append(encode(np.stack([imgs[f - T + 1 : f + 1] for f in sel])))
        lat = np.concatenate(lat).astype(np.float64)
        rr = r2(Y[te], ridge_pred(lat[tr], Y[tr], lat[te]))
        pj = ", ".join(f"{v:.2f}" for v in rr)
        print(
            f"[T={T:2d}] vjepa meanR2={rr.mean():.3f}  per-joint=[{pj}]  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    except Exception as ex:
        print(f"[T={T}] failed: {type(ex).__name__} {str(ex)[:90]}", flush=True)

flat = (imgs[frames][:, ::4, ::4, :].reshape(len(frames), -1).astype(np.float64)) / 255.0
mu = flat[tr].mean(0)
_, _, Vt = np.linalg.svd(flat[tr] - mu, full_matrices=False)
comp = Vt[:256]
rp = r2(Y[te], ridge_pred((flat[tr] - mu) @ comp.T, Y[tr], (flat[te] - mu) @ comp.T))
print(f"[pixelPCA] meanR2={rp.mean():.3f}", flush=True)
