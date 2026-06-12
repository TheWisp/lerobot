# ruff: noqa
"""Fork-1, fair test: SPATIAL DynaMo so the adaptation improves the PATCH tokens the ACT uses
(not the mean). Per-token forward prediction forces the encoder to PRESERVE patch detail.
z = 8x8 maxpool of DINOv2 16x16 patches (64 tokens, matches the ACT input). On POOLED data
(random + insertion):
  inverse: g_inv([mean(z_t), mean(z_{t+1})]) -> action
  forward: per-token g_fwd([z_t_i, action]) -> z_{t+1}_i  (SimSiam cosine, stop-grad target)
Saves dynamo_dino_spatial.pt. Gate: SPATIAL decode of peg-xy AND gripper-xyz, frozen vs adapted."""

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
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs_w = wb["images"]
st_w = wb["states"].astype(np.float32)
W = wb["world"].astype(np.float64)
epw = [tuple(map(int, b)) for b in wb["ep_bounds"]]
nw = len(imgs_w)
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
print(f"POOLED {len(IMGS)} frames, {len(trans)} transitions | SPATIAL objective", flush=True)
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


def encsp(idxs, grad):  # -> (B,64,768) 8x8 maxpool patches
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = m(pixel_values=prep(idxs)).last_hidden_state[:, 1:]
        b = h.shape[0]
        g = int(round(h.shape[1] ** 0.5))
        hh = h.reshape(b, g, g, 768)
        return hh.reshape(b, 8, g // 8, 8, g // 8, 768).amax(dim=(2, 4)).reshape(b, 64, 768)


inv = nn.Sequential(nn.Linear(768 * 2, 512), nn.GELU(), nn.Linear(512, 14)).to(dev)
fwd = nn.Sequential(nn.Linear(768 + 14, 1024), nn.GELU(), nn.Linear(1024, 768)).to(dev)  # per-token predictor
opt = torch.optim.AdamW(
    [p for p in m.parameters() if p.requires_grad] + list(inv.parameters()) + list(fwd.parameters()), 1e-5
)
A = torch.tensor((d_act - amu) / asd, device=dev)
ix = np.arange(len(trans))
t0 = time.time()
for epoch in range(3):
    np.random.shuffle(ix)
    li, lf = [], []
    for k in range(0, len(ix), 16):
        b = ix[k : k + 16]
        z0 = encsp(trans[b, 0], True)
        z1 = encsp(trans[b, 1], True)  # (B,64,768)
        liv = F.mse_loss(inv(torch.cat([z0.mean(1), z1.mean(1)], 1)), A[b])
        ab = A[b][:, None, :].expand(-1, 64, -1)  # broadcast action per token
        p1 = fwd(torch.cat([z0, ab], -1))  # (B,64,768)
        lfv = -F.cosine_similarity(p1, z1.detach(), dim=-1).mean()
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
torch.save(m.state_dict(), OUT + "/dynamo_dino_spatial.pt")
print("[saved] dynamo_dino_spatial.pt", flush=True)


# GATE: spatial decode (flatten 8x8) peg + gripper, frozen vs adapted (world_buffer)
def encsp_all(model, n):
    out = np.zeros((n, 64 * 768), np.float32)
    for k in range(0, n, 48):
        ii = np.arange(k, min(k + 48, n))
        with torch.no_grad():
            out[k : k + len(ii)] = encsp(ii, False).reshape(len(ii), -1).cpu().numpy()
    return out


Zad = encsp_all(m, nw)
mf = Dinov2Model.from_pretrained("facebook/dinov2-base").to(dev).eval()
m_bak = m
m = mf
Zfz = encsp_all(mf, nw)
m = m_bak
fe = np.zeros(nw, int)
for eid, (s, e) in enumerate(epw):
    fe[s:e] = eid
ne = len(epw)
trm, tem = fe < ne - 4, fe >= ne - 4
peg = W[:, 0:3]
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
    P, Pt = pca(Z[f].astype(np.float64), Z[t].astype(np.float64))
    return r2(Y[t], ridge(P, Y[f], Pt))


GX = W[:, 9:12]
PEGxy = W[:, 0:2]
allm = np.ones(nw, bool)
print("\n=== SPATIAL gate (8x8 patches, world_buffer held-out) frozen vs spatial-adapted ===", flush=True)
print(f"gripper-xyz: frozen {dec(Zfz, GX, allm):.3f}  adapted {dec(Zad, GX, allm):.3f}", flush=True)
print(
    f"peg-xy:      frozen {dec(Zfz, PEGxy, inb):.3f}  adapted {dec(Zad, PEGxy, inb):.3f}  (KEY: patch object info preserved/improved?)",
    flush=True,
)
