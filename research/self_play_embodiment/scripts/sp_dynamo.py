# ruff: noqa
"""Fork-1 cheap gate (DynaMo-style): fine-tune DINOv2 top blocks with inverse + forward dynamics
(+ SimSiam stop-grad to prevent collapse, per DynaMo arXiv:2409.12192 & Levine-Stone-Zhang 2024).
Objective on self-play (random play): z_t = meanpool(DINOv2 patches);
  inverse: g_inv(z_t, z_{t+1}) -> d_t (achieved displacement)
  forward: g_fwd(z_t, d_t)    -> z_{t+1}.detach()  (SimSiam predictor; cosine; stop-grad target)
GATE: (a) no collapse (effective rank of z), (b) does the ADAPTED encoder decode gripper-xyz
BETTER than the FROZEN one (held-out eps)? Frozen baseline = dino_cache.M (cached frozen DINOv2).
If adapted > frozen without collapse -> Fork-1 adds info -> proceed to ACT demo-efficiency.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

OUT = "/tmp/selfplay_probe"  # nosec B108
dev = "cuda"
torch.manual_seed(0)
K = int(os.environ.get("K", "2"))  # unfrozen top blocks
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs = wb["images"]
states = wb["states"].astype(np.float32)
W = wb["world"].astype(np.float64)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
pairs = np.array([(t, t + 1, eid) for eid, (s, e) in enumerate(ep) for t in range(s, e - 1)])
ts, t1s, pep = pairs[:, 0], pairs[:, 1], pairs[:, 2]
tr = pep < ne - 4
d_act = (states[t1s] - states[ts]).astype(np.float32)
amu, asd = d_act[tr].mean(0), d_act[tr].std(0) + 1e-6
print(f"K={K} | {len(pairs)} transitions, train {tr.sum()}", flush=True)
m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(dev)
for p in m.parameters():
    p.requires_grad = False
for blk in m.encoder.layer[-K:]:
    for p in blk.parameters():
        p.requires_grad = True
m.gradient_checkpointing_enable()
m.train()
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(dev)
std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(dev)


def prep(idxs):  # uint8 256 -> normalized 224 tensor
    x = (
        torch.from_numpy(np.stack([np.array(Image.fromarray(imgs[i]).resize((224, 224))) for i in idxs]))
        .to(dev)
        .permute(0, 3, 1, 2)
        .float()
        / 255.0
    )
    return (x - mean) / std


def enc(idxs, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = m(pixel_values=prep(idxs)).last_hidden_state[:, 1:]  # drop CLS -> patches
        return h.mean(1)  # (B,768)


inv = nn.Sequential(nn.Linear(768 * 2, 512), nn.GELU(), nn.Linear(512, 14)).to(dev)
fwd = nn.Sequential(nn.Linear(768 + 14, 1024), nn.GELU(), nn.Linear(1024, 768)).to(dev)  # SimSiam predictor
opt = torch.optim.AdamW(
    [p for p in m.parameters() if p.requires_grad] + list(inv.parameters()) + list(fwd.parameters()), 1e-5
)
tri = np.where(tr)[0]
A = torch.tensor((d_act - amu) / asd, device=dev)
t0 = time.time()
for epoch in range(3):
    np.random.shuffle(tri)
    li, lf = [], []
    for k in range(0, len(tri), 24):
        b = tri[k : k + 24]
        z0 = enc(ts[b], True)
        z1 = enc(t1s[b], True)
        li_ = F.mse_loss(inv(torch.cat([z0, z1], 1)), A[b])
        p1 = fwd(torch.cat([z0, A[b]], 1))
        lf_ = -F.cosine_similarity(p1, z1.detach(), dim=-1).mean()  # SimSiam stop-grad
        loss = li_ + lf_
        opt.zero_grad()
        loss.backward()
        opt.step()
        li.append(li_.item())
        lf.append(lf_.item())
    print(
        f"epoch {epoch}: L_inv {np.mean(li):.3f} L_fwd {np.mean(lf):.3f} ({time.time() - t0:.0f}s)",
        flush=True,
    )
# encode all frames with adapted encoder
m.eval()
Zft = np.zeros((len(imgs), 768), np.float32)
for k in range(0, len(imgs), 64):
    Zft[k : k + 64] = enc(np.arange(k, min(k + 64, len(imgs))), False).cpu().numpy()
np.savez_compressed(OUT + "/dynamo_z.npz", Z=Zft)
# GATE
Zfz = np.load(OUT + "/dino_cache.npz")["M"].astype(np.float64)  # FROZEN DINOv2 baseline


def rank(Z):
    C = np.cov((Z - Z.mean(0)).T)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    return (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)


fe = np.zeros(len(imgs), int)
for eid, (s, e) in enumerate(ep):
    fe[s:e] = eid
trm, tem = fe < ne - 4, fe >= ne - 4
GX = W[:, 9:12]  # R gripper xyz


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, a=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xte @ np.linalg.solve(
        Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - Ytr.mean(0))
    ) + Ytr.mean(0)


def pca(Xtr, Xte, k=200):
    mm = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mm, full_matrices=False)
    c = Vt[:k]
    return (Xtr - mm) @ c.T, (Xte - mm) @ c.T


def decode(Z):
    Z = Z.astype(np.float64)
    P, Pt = pca(Z[trm], Z[tem])
    return r2(GX[tem], ridge(P, GX[trm], Pt))


print(f"\n=== Fork-1 gate (DINOv2, top-{K} adapted via DynaMo) ===", flush=True)
print(
    f"eff-rank: frozen {rank(Zfz):.1f}  adapted {rank(Zft.astype(np.float64)):.1f}  (collapse if adapted<<frozen)",
    flush=True,
)
print(f"R-gripper-xyz decode (held-out eps): frozen {decode(Zfz):.3f}  adapted {decode(Zft):.3f}", flush=True)
print(
    "  (adapted > frozen, no collapse => Fork-1 adds embodiment info -> run ACT demo-efficiency)", flush=True
)
