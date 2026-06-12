# ruff: noqa
"""G2: does action-conditioning improve world prediction on the frozen V-JEPA latent?

On held-out episodes, predict the latent CHANGE dz = z_{t+1} - z_t (PCA-64 space):
  - persistence  (dz = 0)                  baseline
  - action-free  f(z_t)
  - action-cond. f(z_t, a_t)
Embodiment signal = R2(action-cond) - R2(action-free). z_t is a STATIC single-frame
latent (frame repeated to fill the clip) so no velocity leaks in -> the action is the
only motion cue. Run: python probe_g2.py
"""

import json
import os
import time

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
OUT, REPO = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256"  # nosec B108
dev, dt = "cuda", torch.bfloat16

d = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs, actions = d["images"], d["actions"].astype(np.float64)
ep = [tuple(map(int, b)) for b in d["ep_bounds"]]
print(f"buffer images={imgs.shape} actions={actions.shape} episodes={len(ep)}", flush=True)

from transformers import AutoVideoProcessor, VJEPA2Model

model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, dt)
T = 2


def encode_frames(frame_ids, bs=16):
    out = np.zeros((len(frame_ids), 1408), np.float32)
    for k in range(0, len(frame_ids), bs):
        sel = frame_ids[k : k + bs]
        clips = np.stack([np.repeat(imgs[f : f + 1], T, 0) for f in sel])  # STATIC single-frame clip
        x = torch.from_numpy(clips).to(dev).permute(0, 1, 4, 2, 3).to(dt) / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            o = model(pixel_values_videos=x)
        h = o.last_hidden_state if getattr(o, "last_hidden_state", None) is not None else o[0]
        out[k : k + len(sel)] = h.float().mean(1).cpu().numpy()
    return out


pairs = [(t, t + 1, eid) for eid, (s, e) in enumerate(ep) for t in range(s, e - 1)]
rng = np.random.default_rng(0)
rng.shuffle(pairs)
pairs = pairs[:2500]
ts = np.array([p[0] for p in pairs])
t1s = np.array([p[1] for p in pairs])
eids = np.array([p[2] for p in pairs])
need = np.unique(np.concatenate([ts, t1s]))
print(f"{len(pairs)} transitions, encoding {len(need)} unique frames...", flush=True)
t0 = time.time()
lat = encode_frames(list(need))
print(f"encoded in {time.time() - t0:.0f}s", flush=True)
pos = {int(f): i for i, f in enumerate(need)}
Z0 = lat[[pos[int(t)] for t in ts]].astype(np.float64)
Z1 = lat[[pos[int(t)] for t in t1s]].astype(np.float64)
A = actions[ts]

n_test_ep = max(2, len(ep) // 5)
test_eps = set(range(len(ep) - n_test_ep, len(ep)))
tr = ~np.isin(eids, list(test_eps))
te = np.isin(eids, list(test_eps))
print(f"train transitions={tr.sum()} test={te.sum()} (held-out {n_test_ep} episodes)", flush=True)

m = Z0[tr].mean(0)
_, _, Vt = np.linalg.svd(Z0[tr] - m, full_matrices=False)
comp = Vt[:64]


def proj(Z):
    return (Z - m) @ comp.T


P0, P1 = proj(Z0), proj(Z1)
dZ = P1 - P0


def ridge_pred(Xtr, Ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym))
    return Xte @ W + ym


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


r_persist = r2(P1[te], P0[te])  # predict z_{t+1}=z_t
r_af = r2(dZ[te], ridge_pred(P0[tr], dZ[tr], P0[te]))  # dz from z_t
XAC = np.hstack([P0, A])
r_ac = r2(dZ[te], ridge_pred(XAC[tr], dZ[tr], XAC[te]))  # dz from z_t, a_t

print("\n=== G2: action-conditioned world prediction (held-out episodes, PCA-64) ===")
print(f"  persistence z_t+1=z_t :            R2(z)  = {r_persist:.3f}  (how autocorrelated frames are)")
print("  predict latent CHANGE dz:")
print(f"     action-free  f(z_t)      R2(dz) = {r_af:.3f}")
print(f"     action-cond. f(z_t,a_t)  R2(dz) = {r_ac:.3f}")
print(f"  >> EMBODIMENT SIGNAL (AC - action-free) = {r_ac - r_af:+.3f}")
verdict = "PASS" if (r_ac - r_af > 0.05 and r_ac > 0.1) else "WEAK" if r_ac > r_af else "FAIL"
print(f"  VERDICT: {verdict}")
json.dump(
    {
        "r2_persist": round(float(r_persist), 3),
        "r2_dz_actionfree": round(float(r_af), 3),
        "r2_dz_actioncond": round(float(r_ac), 3),
        "embodiment_gap": round(float(r_ac - r_af), 3),
        "verdict": verdict,
    },
    open(OUT + "/g2_result.json", "w"),
    indent=2,
)
