"""Reframed G1 (world-centric): does the FROZEN V-JEPA latent encode the *world*
(peg / socket / end-effector positions) on our sim renders?

- held-out-episode split (no adjacent-frame leakage)
- per-target R² grouped by peg / socket / L-grip / R-grip
- raw-pixel-PCA baseline
- non-degeneracy check: participation ratio (effective rank) of the latent
Run: python probe_world.py
"""

import json
import os
import time

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
OUT, REPO = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256"
dev, dt = "cuda", torch.bfloat16

d = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs, world = d["images"], d["world"].astype(np.float64)
ep = [tuple(map(int, b)) for b in d["ep_bounds"]]
keys = [str(k) for k in d["world_keys"]]
print(f"buffer: images={imgs.shape} world={world.shape} episodes={len(ep)} targets={keys}", flush=True)

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


T = 2  # near-single-frame: static world legibility, cheap
samples = [(f, eid) for eid, (s, e) in enumerate(ep) for f in range(s + T - 1, e)]
rng = np.random.default_rng(0)
rng.shuffle(samples)
samples = samples[:1800]
frames = np.array([f for f, _ in samples])
eids = np.array([e for _, e in samples])
n_test_ep = max(2, len(ep) // 5)
test_eps = set(range(len(ep) - n_test_ep, len(ep)))
tr, te = ~np.isin(eids, list(test_eps)), np.isin(eids, list(test_eps))
Y = world[frames]
print(f"samples={len(frames)} train={tr.sum()} test={te.sum()} (held-out {n_test_ep} episodes)", flush=True)

lat, t0 = [], time.time()
for k in range(0, len(frames), 16):
    sel = frames[k : k + 16]
    lat.append(encode(np.stack([imgs[f - T + 1 : f + 1] for f in sel])))
lat = np.concatenate(lat).astype(np.float64)
print(f"latents {lat.shape} in {time.time() - t0:.0f}s", flush=True)

# non-degeneracy: participation ratio of latent covariance (effective dimensionality)
C = np.cov((lat - lat.mean(0)).T)
ev = np.clip(np.linalg.eigvalsh(C), 0, None)
pr = (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)
print(f"latent effective rank (participation ratio) = {pr:.1f} / {lat.shape[1]}", flush=True)


def ridge_pred(Xtr, Ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym))
    return Xte @ W + ym


def r2(Yte, P):
    return 1 - ((Yte - P) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9)


def pca_reduce(Xtr, Xte, k):
    m = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
    comp = Vt[:k]
    return (Xtr - m) @ comp.T, (Xte - m) @ comp.T


Ltr, Lte = pca_reduce(
    lat[tr], lat[te], 64
)  # latent eff-rank ~15 -> reduce before linear probe (kills extrapolation blowup)
rv = r2(Y[te], ridge_pred(Ltr, Y[tr], Lte))
flat = (imgs[frames][:, ::4, ::4, :].reshape(len(frames), -1).astype(np.float64)) / 255.0
Ptr, Pte = pca_reduce(flat[tr], flat[te], 64)
rp = r2(Y[te], ridge_pred(Ptr, Y[tr], Pte))

keep = Y[tr].std(0) > 1e-3  # drop near-constant dims (objects on the table -> z constant -> R2 explodes)
groups, axn = ["peg", "socket", "L-grip", "R-grip"], ["x", "y", "z"]
print("\n=== reframed G1: read WORLD positions from frozen latent (held-out episodes, varying dims only) ===")
print(f"{'target':8s} {'dims':>6s} {'V-JEPA R2':>10s} {'pixel R2':>10s}")
for i, g in enumerate(groups):
    idx = [i * 3 + j for j in range(3) if keep[i * 3 + j]]
    if not idx:
        print(f"{g:8s} {'const':>6s}")
        continue
    dims = "".join(axn[j] for j in range(3) if keep[i * 3 + j])
    print(f"{g:8s} {dims:>6s} {rv[idx].mean():10.3f} {rp[idx].mean():10.3f}")
ai = [k for k in range(len(keep)) if keep[k]]
ee = [k for k in range(6, 12) if keep[k]]
print(f"{'ALL':8s} {'':>6s} {rv[ai].mean():10.3f} {rp[ai].mean():10.3f}")
print(f"{'EE-only':8s} {'':>6s} {rv[ee].mean():10.3f} {rp[ee].mean():10.3f}")
verdict = (
    "PASS" if rv[ee].mean() > 0.3 else "WEAK" if rv[ee].mean() > 0 else "FAIL"
)  # judge on the clean visible target (EE)
print(f"VERDICT (is the world legible at all?): {verdict}")
json.dump(
    {
        "vjepa_meanR2_varying": round(float(rv[ai].mean()), 3),
        "pixel_meanR2_varying": round(float(rp[ai].mean()), 3),
        "vjepa_EE_R2": round(float(rv[ee].mean()), 3),
        "eff_rank": round(float(pr), 1),
        "verdict": verdict,
    },
    open(OUT + "/g1_world_result.json", "w"),
    indent=2,
)
