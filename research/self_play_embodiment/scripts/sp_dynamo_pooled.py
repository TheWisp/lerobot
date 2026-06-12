# ruff: noqa
"""Fork-1, done right: DynaMo-adapt DINOv2 on ALL same-embodiment data — random play (body motion)
+ scripted insertion demos (peg grasp/insert). So the encoder sees objects manipulated -> keeps object
info (fixes the random-only 'shedding' worry). inverse + forward + SimSiam stop-grad. Saves
dynamo_dino_pooled.pt. Gate: decode gripper-xyz AND peg-xy (world_buffer GT) frozen vs pooled-adapted."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image
from transformers import Dinov2Model

from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT = "/tmp/selfplay_probe"  # nosec B108
dev = "cuda"
torch.manual_seed(0)
K = 2
# random play (our-sim) + GT for gate
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs_w = wb["images"]
st_w = wb["states"].astype(np.float32)
W = wb["world"].astype(np.float64)
epw = [tuple(map(int, b)) for b in wb["ep_bounds"]]
nw = len(imgs_w)
# scripted insertion demos (peg-into-hole)
ds = LeRobotDataset("lerobot/aloha_sim_insertion_scripted")
ei = np.array(ds.hf_dataset["episode_index"])
keep = np.where(ei < 30)[0]


def to_img(t):
    return np.array(Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((256, 256)))


imgs_d, st_d, ep_d = [], [], []
for i in keep:
    it = ds[int(i)]
    imgs_d.append(to_img(it["observation.images.top"]))
    st_d.append(it["observation.state"].numpy())
    ep_d.append(int(ei[i]))
imgs_d = np.stack(imgs_d)
st_d = np.array(st_d, np.float32)
ep_d = np.array(ep_d)
IMGS = np.concatenate([imgs_w, imgs_d])
STATES = np.concatenate([st_w, st_d])
# transitions within episodes (offset dataset indices by nw)
trans = []
for s, e in epw:
    for t in range(s, e - 1):
        trans.append((t, t + 1))
for ep in np.unique(ep_d):
    idx = np.where(ep_d == ep)[0] + nw
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:
            trans.append((a, b))
trans = np.array(trans)
d_act = (STATES[trans[:, 1]] - STATES[trans[:, 0]]).astype(np.float32)
amu, asd = d_act.mean(0), d_act.std(0) + 1e-6
print(
    f"POOLED: {nw} random + {len(imgs_d)} insertion = {len(IMGS)} frames, {len(trans)} transitions",
    flush=True,
)

m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(dev)
for p in m.parameters():
    p.requires_grad = False
for blk in m.encoder.layer[-K:]:
    for p in blk.parameters():
        p.requires_grad = True
m.gradient_checkpointing_enable()
m.train()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


def prep(idxs):
    x = (
        torch.from_numpy(np.stack([np.array(Image.fromarray(IMGS[i]).resize((224, 224))) for i in idxs]))
        .to(dev)
        .permute(0, 3, 1, 2)
        .float()
        / 255.0
    )
    return (x - mean) / std


def encz(idxs, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return m(pixel_values=prep(idxs)).last_hidden_state[:, 1:].mean(1)


inv = nn.Sequential(nn.Linear(768 * 2, 512), nn.GELU(), nn.Linear(512, 14)).to(dev)
fwd = nn.Sequential(nn.Linear(768 + 14, 1024), nn.GELU(), nn.Linear(1024, 768)).to(dev)
opt = torch.optim.AdamW(
    [p for p in m.parameters() if p.requires_grad] + list(inv.parameters()) + list(fwd.parameters()), 1e-5
)
A = torch.tensor((d_act - amu) / asd, device=dev)
ix = np.arange(len(trans))
t0 = time.time()
for epoch in range(3):
    np.random.shuffle(ix)
    li, lf = [], []
    for k in range(0, len(ix), 24):
        b = ix[k : k + 24]
        z0 = encz(trans[b, 0], True)
        z1 = encz(trans[b, 1], True)
        liv = F.mse_loss(inv(torch.cat([z0, z1], 1)), A[b])
        lfv = -F.cosine_similarity(fwd(torch.cat([z0, A[b]], 1)), z1.detach(), dim=-1).mean()
        (liv + lfv).backward()
        opt.step()
        opt.zero_grad()
        li.append(liv.item())
        lf.append(lfv.item())
    print(
        f"epoch {epoch}: L_inv {np.mean(li):.3f} L_fwd {np.mean(lf):.3f} ({time.time() - t0:.0f}s)",
        flush=True,
    )
m.eval()
torch.save(m.state_dict(), OUT + "/dynamo_dino_pooled.pt")
print("[saved] dynamo_dino_pooled.pt", flush=True)
# GATE on world_buffer (has gripper + peg GT)
Zad = np.zeros((nw, 768), np.float32)
for k in range(0, nw, 64):
    Zad[k : k + 64] = encz(np.arange(k, min(k + 64, nw)), False).cpu().numpy()
Zfz = np.load(OUT + "/dino_cache.npz")["M"].astype(np.float64)
fe = np.zeros(nw, int)
for eid, (s, e) in enumerate(epw):
    fe[s:e] = eid
ne = len(epw)
trm, tem = fe < ne - 4, fe >= ne - 4
peg, soc = W[:, 0:3], W[:, 3:6]
inb = (peg[:, 2] < 0.15) & (np.abs(peg[:, 0]) < 0.4) & (np.abs(peg[:, 1] - 0.55) < 0.4)


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xt, Yt, Xe, a=10.0):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt, Xe = (Xt - mu) / sd, (Xe - mu) / sd
    return Xe @ np.linalg.solve(Xt.T @ Xt + a * np.eye(Xt.shape[1]), Xt.T @ (Yt - Yt.mean(0))) + Yt.mean(0)


def pca(Xt, Xe, k=200):
    mm = Xt.mean(0)
    _, _, Vt = np.linalg.svd(Xt - mm, full_matrices=False)
    c = Vt[:k]
    return (Xt - mm) @ c.T, (Xe - mm) @ c.T


def dec(Z, Y, mask):
    f, t = trm & mask, tem & mask
    P, Pt = pca(Z[f], Z[t])
    return r2(Y[t], ridge(P, Y[f], Pt))


def rank(Z):
    C = np.cov((Z - Z.mean(0)).T)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    return (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)


allm = np.ones(nw, bool)
GX = W[:, 9:12]
PEGxy = W[:, 0:2]
print("\n=== POOLED-adapted vs FROZEN (world_buffer held-out eps) ===", flush=True)
print(f"eff-rank: frozen {rank(Zfz):.1f}  pooled-adapted {rank(Zad.astype(np.float64)):.1f}", flush=True)
print(
    f"gripper-xyz decode: frozen {dec(Zfz, GX, allm):.3f}  adapted {dec(Zad.astype(np.float64), GX, allm):.3f}",
    flush=True,
)
print(
    f"peg-xy decode:      frozen {dec(Zfz, PEGxy, inb):.3f}  adapted {dec(Zad.astype(np.float64), PEGxy, inb):.3f}  (KEY: did it KEEP object info?)",
    flush=True,
)
