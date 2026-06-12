# ruff: noqa
"""G3-lite: fixed-teacher action-conditioned distillation (collapse-free reshaping).

Teacher = frozen V-JEPA (targets precomputed once). Student = V-JEPA with top-K blocks
unfrozen. Train: predictor([z_S_t, a_t]) -> teacher's pooled latent CHANGE, + anchor
keeping z_S_t near z_T_t. Then compare reshaped student vs frozen on:
  (1) action->Δz gap (should rise above frozen +0.04),
  (2) body/object decode (G1 legibility — must be preserved),
  (3) effective rank (no collapse).
Usage: python g3_lite.py smoke   |   python g3_lite.py [epochs]
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MUJOCO_GL", "egl")
torch.manual_seed(0)
OUT, REPO, dev = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256", "cuda"  # nosec B108
import time as _time

T0 = _time.time()
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
EPOCHS = 1 if SMOKE else int(os.environ.get("EPOCHS", 3))
MAXSTEPS = 20 if SMOKE else 10**9
K = int(os.environ.get("K", 6))
LAM = float(os.environ.get("LAM", 1.0))
LR_ENC = float(os.environ.get("LR_ENC", 1e-5))
LR_PRED, BATCH, Tc = 1e-3, 12, 2
print(f"config: K={K} LAM={LAM} LR_ENC={LR_ENC} EPOCHS={EPOCHS}", flush=True)

d = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs, actions, W = d["images"], d["actions"].astype(np.float32), d["world"].astype(np.float64)
ep = [tuple(map(int, b)) for b in d["ep_bounds"]]
N = len(imgs)
pairs = [(t, t + 1, eid) for eid, (s, e) in enumerate(ep) for t in range(s, e - 1)]
ts = np.array([p[0] for p in pairs])
t1s = np.array([p[1] for p in pairs])
pep = np.array([p[2] for p in pairs])
ne = len(ep)
test_ep = set(range(ne - 4, ne))
tr_idx = np.where(~np.isin(pep, list(test_ep)))[0]
print(f"{N} frames, {len(pairs)} pairs, train {len(tr_idx)}", flush=True)

from transformers import AutoVideoProcessor, VJEPA2Model

proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1)


def clip_of(idxs):
    c = np.stack([np.repeat(imgs[i : i + 1], Tc, 0) for i in idxs])
    return ((torch.from_numpy(c).permute(0, 1, 4, 2, 3).float() / 255.0) - mean) / std


def pooled(model, idxs, bs=32, grad=False, want_spatial=False):
    Ms, Ss = [], []
    for k in range(0, len(idxs), bs):
        x = clip_of(idxs[k : k + bs]).to(dev, next(model.parameters()).dtype)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            o = model(pixel_values_videos=x)
            h = o.last_hidden_state if getattr(o, "last_hidden_state", None) is not None else o[0]
            hf = h.float()
            b, nt, C = hf.shape
            g = int(round(nt**0.5))
            hh = hf.reshape(b, g, g, C)
            Ms.append(hh.mean((1, 2)) if grad else hh.mean((1, 2)).cpu().numpy())
            if want_spatial:
                Ss.append(hh.reshape(b, 4, g // 4, 4, g // 4, C).amax((2, 4)).reshape(b, -1).cpu().numpy())
    if grad:
        return Ms[0]
    return (np.concatenate(Ms), np.concatenate(Ss)) if want_spatial else np.concatenate(Ms)


# ---- teacher targets (precompute once, then free) ----
teacher = VJEPA2Model.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(dev).eval()
t0 = time.time()
zT = pooled(teacher, np.arange(N))
print(f"teacher precompute {time.time() - t0:.0f}s", flush=True)
del teacher
torch.cuda.empty_cache()
zT = torch.tensor(zT)

# ---- student (top-K unfrozen) ----
student = VJEPA2Model.from_pretrained(REPO, torch_dtype=torch.float32).to(dev)
for p in student.parameters():
    p.requires_grad = False
for blk in student.encoder.layer[-K:]:
    for p in blk.parameters():
        p.requires_grad = True
student.gradient_checkpointing_enable()
student.train()
print(
    f"student trainable: {sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6:.0f}M (top-{K})",
    flush=True,
)
pred = nn.Sequential(
    nn.Linear(1408 + 14, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1408)
).to(dev)
opt = torch.optim.Adam(
    [
        {"params": [p for p in student.parameters() if p.requires_grad], "lr": LR_ENC},
        {"params": pred.parameters(), "lr": LR_PRED},
    ]
)


def spool_grad(idxs):
    x = clip_of(idxs).to(dev, torch.float32)
    o = student(pixel_values_videos=x)
    h = o.last_hidden_state if getattr(o, "last_hidden_state", None) is not None else o[0]
    return h.float().mean(1)


step = 0
for epoch in range(EPOCHS):
    perm = np.random.permutation(tr_idx)
    lps, las = [], []
    for k in range(0, len(perm), BATCH):
        bi = perm[k : k + BATCH]
        a = torch.tensor(actions[ts[bi]], device=dev)
        target = (zT[t1s[bi]] - zT[ts[bi]]).to(dev)
        zt_teach = zT[ts[bi]].to(dev)
        zS = spool_grad(ts[bi])
        dhat = pred(torch.cat([zS, a], 1))
        lp = ((dhat - target) ** 2).mean()
        la = ((zS - zt_teach) ** 2).mean()
        loss = lp + LAM * la
        opt.zero_grad()
        loss.backward()
        opt.step()
        lps.append(lp.item())
        las.append(la.item())
        step += 1
        if step >= MAXSTEPS:
            break
    print(f"epoch {epoch}: L_pred={np.mean(lps):.4f} L_anchor={np.mean(las):.4f} (steps {step})", flush=True)
    if step >= MAXSTEPS:
        break

student.eval()


def rank(Z):
    C = np.cov((Z - Z.mean(0)).T)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    return (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)


zS_sample = pooled(student, np.arange(0, N, 8))
print(
    f"[check] student latent finite={np.isfinite(zS_sample).all()} eff_rank={rank(zS_sample.astype(np.float64)):.1f}",
    flush=True,
)
if SMOKE:
    print(f"SMOKE OK ({(_time.time() - T0) / 60:.1f} min)", flush=True)
    sys.exit(0)


# ---- full eval: reshaped student vs frozen teacher baseline (feat_cache) ----
def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, a=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xte @ np.linalg.solve(
        Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - Ytr.mean(0))
    ) + Ytr.mean(0)


def pca(Xtr, Xte, k):
    m = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
    c = Vt[:k]
    return (Xtr - m) @ c.T, (Xte - m) @ c.T


fe = np.zeros(N, int)
for eid, (s, e) in enumerate(ep):
    fe[s:e] = eid
ftr, fte = fe < ne - 4, fe >= ne - 4
ptr, pte = ~np.isin(pep, list(test_ep)), np.isin(pep, list(test_ep))


def action_gap(M):
    Z0, Z1 = M[ts], M[t1s]
    mm = Z0[ptr].mean(0)
    _, _, Vt = np.linalg.svd(Z0[ptr] - mm, full_matrices=False)
    c = Vt[:64]
    P0, P1 = (Z0 - mm) @ c.T, (Z1 - mm) @ c.T
    dZ = P1 - P0
    A = actions[ts].astype(np.float64)
    af = r2(dZ[pte], ridge(P0[ptr], dZ[ptr], P0[pte]))
    ac = r2(dZ[pte], ridge(np.hstack([P0, A])[ptr], dZ[ptr], np.hstack([P0, A])[pte]))
    return af, ac


def decode(S, tgt, mask):
    f, t = ftr & mask, fte & mask
    Ptr, Pte = pca(S[f], S[t], 200)
    return r2(tgt[t], ridge(Ptr, tgt[f], Pte))


peg, soc = W[:, 0:3], W[:, 3:6]
inb = (
    (peg[:, 2] < 0.15)
    & (soc[:, 2] < 0.15)
    & (np.abs(peg[:, 0]) < 0.4)
    & (np.abs(soc[:, 0]) < 0.4)
    & (np.abs(peg[:, 1] - 0.55) < 0.4)
    & (np.abs(soc[:, 1] - 0.55) < 0.4)
)
obj_xy, body_xy = W[:, [0, 1, 3, 4]], W[:, [6, 7, 9, 10]]
fc = np.load(OUT + "/feat_cache.npz")
Mt, St = fc["M"].astype(np.float64), fc["S"].astype(np.float64)
Ms, Ss = pooled(student, np.arange(N), want_spatial=True)
Ms, Ss = Ms.astype(np.float64), Ss.astype(np.float64)
print("\n=== FROZEN teacher vs RESHAPED student ===")
print(f"{'metric':22s}{'frozen':>10s}{'reshaped':>10s}")
af0, ac0 = action_gap(Mt)
af1, ac1 = action_gap(Ms)
print(f"{'action->Δz gap':22s}{ac0 - af0:+10.3f}{ac1 - af1:+10.3f}")
print(f"{'object-xy decode':22s}{decode(St, obj_xy, inb):10.3f}{decode(Ss, obj_xy, inb):10.3f}")
print(
    f"{'body-xy decode':22s}{decode(St, body_xy, np.ones(N, bool)):10.3f}{decode(Ss, body_xy, np.ones(N, bool)):10.3f}"
)
print(f"{'eff rank (pooled)':22s}{rank(Mt):10.1f}{rank(Ms):10.1f}")
print(f"total wall-clock {(_time.time() - T0) / 60:.1f} min", flush=True)
