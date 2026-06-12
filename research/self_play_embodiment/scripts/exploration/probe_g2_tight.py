# ruff: noqa
"""G2, tightened on current data (option a): does a *properly regularized* nonlinear
model beat linear at predicting dz = z_{t+1}-z_t from (z_t) vs (z_t, a_t)?

Fixes the overfit: 3-way EPISODE split (fit/val/test), early stopping on val,
dropout + weight decay, small val-selected MLP, and ALL within-episode transitions
(not subsampled). Reports fit/val/test R2 so overfitting is visible."""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MUJOCO_GL", "egl")
torch.manual_seed(0)
OUT, REPO = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256"  # nosec B108
dev, dt = "cuda", torch.bfloat16

d = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs, actions = d["images"], d["actions"].astype(np.float64)
ep = [tuple(map(int, b)) for b in d["ep_bounds"]]
N = len(imgs)
from transformers import AutoVideoProcessor, VJEPA2Model

model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, dt)
T = 2


def encode_all(bs=32):
    out = np.zeros((N, 1408), np.float32)
    for k in range(0, N, bs):
        clips = np.stack([np.repeat(imgs[f : f + 1], T, 0) for f in range(k, min(k + bs, N))])
        x = torch.from_numpy(clips).to(dev).permute(0, 1, 4, 2, 3).to(dt) / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            o = model(pixel_values_videos=x)
        h = o.last_hidden_state if getattr(o, "last_hidden_state", None) is not None else o[0]
        out[k : k + len(clips)] = h.float().mean(1).cpu().numpy()
    return out


t0 = time.time()
lat = encode_all()
print(f"encoded all {N} frames in {time.time() - t0:.0f}s", flush=True)

pairs = [(t, t + 1, eid) for eid, (s, e) in enumerate(ep) for t in range(s, e - 1)]
ts = np.array([p[0] for p in pairs])
t1s = np.array([p[1] for p in pairs])
eids = np.array([p[2] for p in pairs])
ne = len(ep)
test_eps, val_eps, fit_eps = set(range(ne - 4, ne)), set(range(ne - 8, ne - 4)), set(range(ne - 8))
fit, val, test = np.isin(eids, list(fit_eps)), np.isin(eids, list(val_eps)), np.isin(eids, list(test_eps))
print(
    f"transitions: fit={fit.sum()} val={val.sum()} test={test.sum()} (eps {len(fit_eps)}/{len(val_eps)}/{len(test_eps)})",
    flush=True,
)

m = lat[ts[fit]].mean(0)
_, _, Vt = np.linalg.svd(lat[ts[fit]] - m, full_matrices=False)
comp = Vt[:64]


def proj(Z):
    return ((Z - m) @ comp.T).astype(np.float64)


Z0, Z1, A = proj(lat[ts]), proj(lat[t1s]), actions[ts]
dZ = Z1 - Z0


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym))
    return Xte @ W + ym


def train_mlp(Xf, Xv, Xte, hid, wd, drop, max_ep=4000, patience=80):
    mu = Xf.mean(0)
    sd = Xf.std(0) + 1e-6
    TT = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    Xf_, Xv_, Xte_ = TT(Xf), TT(Xv), TT(Xte)
    Yf_ = torch.tensor(dZ[fit], dtype=torch.float32, device=dev)
    Yv_ = torch.tensor(dZ[val], dtype=torch.float32, device=dev)
    net = nn.Sequential(
        nn.Linear(Xf.shape[1], hid),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Linear(hid, hid),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Linear(hid, dZ.shape[1]),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    best, best_state, bad = 1e9, None, 0
    for e in range(max_ep):
        net.train()
        opt.zero_grad()
        ((net(Xf_) - Yf_) ** 2).mean().backward()
        opt.step()
        if e % 10 == 0:
            net.eval()
            with torch.no_grad():
                vl = ((net(Xv_) - Yv_) ** 2).mean().item()
            if vl < best - 1e-7:
                best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(TT(Xf)).cpu().numpy(), net(Xv_).cpu().numpy(), net(Xte_).cpu().numpy()


CFG = [(64, 3e-3, 0.1), (64, 1e-2, 0.1), (128, 3e-3, 0.1), (128, 1e-2, 0.2)]


def best_mlp(name, full):
    Xf, Xv, Xte = full[fit], full[val], full[test]
    bvr, pick = -1e9, None
    for h, wd, dr in CFG:
        pf, pv, pte = train_mlp(Xf, Xv, Xte, h, wd, dr)
        vr = r2(dZ[val], pv)
        if vr > bvr:
            bvr, pick = vr, (r2(dZ[fit], pf), vr, r2(dZ[test], pte), (h, wd, dr))
    print(
        f"  MLP {name:11s} fit={pick[0]:.3f} val={pick[1]:.3f} test={pick[2]:.3f}  cfg={pick[3]}", flush=True
    )
    return pick[2]


XAC = np.hstack([Z0, A])
print("\n=== G2 tightened (held-out TEST episodes; predict dz) ===")
print(f"  persistence R2(z) = {r2(Z1[test], Z0[test]):.3f}")
lin_af, lin_ac = (
    r2(dZ[test], ridge(Z0[fit], dZ[fit], Z0[test])),
    r2(dZ[test], ridge(XAC[fit], dZ[fit], XAC[test])),
)
print(f"  LINEAR  action-free {lin_af:.3f} | action-cond {lin_ac:.3f} | gap {lin_ac - lin_af:+.3f}")
mlp_af = best_mlp("action-free", Z0)
mlp_ac = best_mlp("action-cond", XAC)
print(f"  MLP(reg) action-free {mlp_af:.3f} | action-cond {mlp_ac:.3f} | gap {mlp_ac - mlp_af:+.3f}")
gap = mlp_ac - mlp_af
verdict = "PASS" if (gap > 0.08 and mlp_ac > 0.2) else "WEAK" if gap > 0.02 else "FAIL"
print(f"  >> EMBODIMENT SIGNAL (best regularized) = {gap:+.3f}   VERDICT: {verdict}")
json.dump(
    {
        "lin_gap": round(float(lin_ac - lin_af), 3),
        "mlp_af": round(float(mlp_af), 3),
        "mlp_ac": round(float(mlp_ac), 3),
        "mlp_gap": round(float(gap), 3),
        "verdict": verdict,
    },
    open(OUT + "/g2_tight_result.json", "w"),
    indent=2,
)
