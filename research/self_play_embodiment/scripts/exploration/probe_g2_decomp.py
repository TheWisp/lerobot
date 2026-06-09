"""Decompose the G2 action-signal: is it the arm predicting its OWN visible motion
(trivial) or the action predicting WORLD changes it causes (the real embodiment)?

Uses ground-truth labels (peg/socket = world; grippers = body):
  Part 1: how much latent variance is attributable to body-state vs object-state.
  Part 2: action's marginal R2 gain at predicting dBODY vs dWORLD (all + contact-only).
held-out-episode split throughout."""

import os
import time

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
OUT, REPO = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256"
dev, dt = "cuda", torch.bfloat16
d = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs = d["images"]
actions = d["actions"].astype(np.float64)
W = d["world"].astype(np.float64)
ep = [tuple(map(int, b)) for b in d["ep_bounds"]]
N = len(imgs)
OBJ, BODY = slice(0, 6), slice(6, 12)  # world.npz cols: peg(0:3) socket(3:6) Lgrip(6:9) Rgrip(9:12)

from transformers import AutoVideoProcessor, VJEPA2Model

model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, dt)
Tc = 2


def encode_all(bs=32):
    out = np.zeros((N, 1408), np.float32)
    for k in range(0, N, bs):
        clips = np.stack([np.repeat(imgs[f : f + 1], Tc, 0) for f in range(k, min(k + bs, N))])
        x = torch.from_numpy(clips).to(dev).permute(0, 1, 4, 2, 3).to(dt) / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            o = model(pixel_values_videos=x)
        h = o.last_hidden_state if getattr(o, "last_hidden_state", None) is not None else o[0]
        out[k : k + len(clips)] = h.float().mean(1).cpu().numpy()
    return out


t0 = time.time()
lat = encode_all()
print(f"encoded {N} frames in {time.time() - t0:.0f}s", flush=True)

pairs = [(t, t + 1, eid) for eid, (s, e) in enumerate(ep) for t in range(s, e - 1)]
ts = np.array([p[0] for p in pairs])
t1s = np.array([p[1] for p in pairs])
eids = np.array([p[2] for p in pairs])
ne = len(ep)
fit = np.isin(eids, list(range(ne - 4)))
test = np.isin(eids, list(range(ne - 4, ne)))

m = lat[ts[fit]].mean(0)
_, _, Vt = np.linalg.svd(lat[ts[fit]] - m, full_matrices=False)
comp = Vt[:64]
Z = ((lat - m) @ comp.T).astype(np.float64)
Z0, A = Z[ts], actions[ts]
dW = W[t1s] - W[ts]


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    Wt = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym))
    return Xte @ Wt + ym


print("\n--- Part 1: latent content attribution (predict latent from ground-truth state, held-out) ---")
bt, ot = W[ts][:, BODY], W[ts][:, OBJ]
print(
    f"  latent var explained by BODY-state (grippers): R2 = {r2(Z0[test], ridge(bt[fit], Z0[fit], bt[test])):.3f}"
)
print(
    f"  latent var explained by WORLD-state (objects):  R2 = {r2(Z0[test], ridge(ot[fit], Z0[fit], ot[test])):.3f}"
)
print(
    f"  (also: can latent DECODE current object pos? R2 = {r2(ot[test], ridge(Z0[fit], ot[fit], Z0[test])):.3f})"
)


def gap(cols, mask):
    f, t = fit & mask, test & mask
    Y = dW[:, cols]
    af = r2(Y[t], ridge(Z0[f], Y[f], Z0[t]))
    ac = r2(Y[t], ridge(np.hstack([Z0, A])[f], Y[f], np.hstack([Z0, A])[t]))
    return af, ac, t.sum()


allm = np.ones(len(pairs), bool)
dobj = np.linalg.norm(dW[:, OBJ].reshape(-1, 2, 3), axis=2).max(1)  # per-step max object displacement
contact = dobj > 0.005
print("\n--- Part 2: does the action help predict CHANGE? body (self-motion) vs world (caused) ---")
af, ac, n = gap(BODY, allm)
print(f"  dBODY  (gripper motion)   n={n:5d}  free {af:.3f} | +action {ac:.3f} | gap {ac - af:+.3f}")
af, ac, n = gap(OBJ, allm)
print(f"  dWORLD (objects, all)     n={n:5d}  free {af:.3f} | +action {ac:.3f} | gap {ac - af:+.3f}")
af, ac, n = gap(OBJ, contact)
print(f"  dWORLD (objects, CONTACT) n={n:5d}  free {af:.3f} | +action {ac:.3f} | gap {ac - af:+.3f}")
print(
    f"  contact transitions: {(contact & test).sum()}/{test.sum()} test ({100 * contact.mean():.0f}% overall, >5mm/step)"
)
