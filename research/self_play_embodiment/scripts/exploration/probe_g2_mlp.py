# ruff: noqa
"""G2 with a nonlinear predictor. Linear can't capture the pose-dependent action
effect (Jacobian varies with configuration), so compare LINEAR vs small MLP for
both action-free f(z_t) and action-conditioned f(z_t, a_t). Target dz = z_{t+1}-z_t
in PCA-64 space; static single-frame latents; held-out-episode split."""

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
from transformers import AutoVideoProcessor, VJEPA2Model

model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, dt)
T = 2


def encode_frames(fids, bs=16):
    out = np.zeros((len(fids), 1408), np.float32)
    for k in range(0, len(fids), bs):
        sel = fids[k : k + bs]
        clips = np.stack([np.repeat(imgs[f : f + 1], T, 0) for f in sel])
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
t0 = time.time()
lat = encode_frames(list(need))
print(f"encoded {len(need)} frames in {time.time() - t0:.0f}s", flush=True)
pos = {int(f): i for i, f in enumerate(need)}
Z0 = lat[[pos[int(t)] for t in ts]].astype(np.float64)
Z1 = lat[[pos[int(t)] for t in t1s]].astype(np.float64)
A = actions[ts]
n_test = max(2, len(ep) // 5)
test_eps = set(range(len(ep) - n_test, len(ep)))
tr = ~np.isin(eids, list(test_eps))
te = np.isin(eids, list(test_eps))
m = Z0[tr].mean(0)
_, _, Vt = np.linalg.svd(Z0[tr] - m, full_matrices=False)
comp = Vt[:64]


def proj(Z):
    return (Z - m) @ comp.T


P0, P1 = proj(Z0), proj(Z1)
dZ = P1 - P0


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge_pred(Xtr, Ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym))
    return Xte @ W + ym


def mlp_pred(Xtr, Ytr, Xte, epochs=500, wd=1e-3):
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=dev)
    mu = Xt.mean(0)
    sd = Xt.std(0) + 1e-6
    Xt = (Xt - mu) / sd
    Xe = (Xe - mu) / sd
    net = nn.Sequential(
        nn.Linear(Xt.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, Yt.shape[1])
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    for e in range(epochs):
        opt.zero_grad()
        (((net(Xt) - Yt) ** 2).mean()).backward()
        opt.step()
    with torch.no_grad():
        return net(Xe).cpu().numpy()


XAC = np.hstack([P0, A])
res = {
    "persist": r2(P1[te], P0[te]),
    "lin_af": r2(dZ[te], ridge_pred(P0[tr], dZ[tr], P0[te])),
    "lin_ac": r2(dZ[te], ridge_pred(XAC[tr], dZ[tr], XAC[te])),
    "mlp_af": r2(dZ[te], mlp_pred(P0[tr], dZ[tr], P0[te])),
    "mlp_ac": r2(dZ[te], mlp_pred(XAC[tr], dZ[tr], XAC[te])),
}
print("\n=== G2: action-conditioned world prediction (held-out episodes, predict dz, PCA-64) ===")
print(f"  persistence R2(z) = {res['persist']:.3f}")
print(
    f"  LINEAR  action-free {res['lin_af']:.3f} | action-cond {res['lin_ac']:.3f} | gap {res['lin_ac'] - res['lin_af']:+.3f}"
)
print(
    f"  MLP     action-free {res['mlp_af']:.3f} | action-cond {res['mlp_ac']:.3f} | gap {res['mlp_ac'] - res['mlp_af']:+.3f}"
)
gap = res["mlp_ac"] - res["mlp_af"]
verdict = "PASS" if (gap > 0.08 and res["mlp_ac"] > 0.2) else "WEAK" if gap > 0.02 else "FAIL"
print(f"  >> EMBODIMENT SIGNAL (MLP gap) = {gap:+.3f}   VERDICT: {verdict}")
json.dump(
    {k: round(float(v), 3) for k, v in res.items()} | {"mlp_gap": round(float(gap), 3), "verdict": verdict},
    open(OUT + "/g2_mlp_result.json", "w"),
    indent=2,
)
