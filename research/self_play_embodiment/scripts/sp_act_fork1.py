# ruff: noqa
"""Fork-1 PAYOFF: does a DynaMo-ADAPTED encoder let ACT reach with FEWER demos than FROZEN?
Same small ACT, same scripted-insertion demos, same sim eval — only the DINOv2 encoder differs
(frozen vs DynaMo-adapted-on-self-play, dynamo_dino.pt). Patch tokens (8x8 maxpool of DINOv2 16x16)
-> ACT (transformer + action chunk + L1). Sweep N_EP {3,6,12,30}; eval reach (grasp-center-to-peg)
closed-loop in our insertion sim with the MATCHING live encoder. Headline: adapted SR > frozen SR at low demos."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sp_lib import Vec
from transformers import Dinov2Model

from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT = "/tmp/selfplay_probe"  # nosec B108
dev = "cuda"
SEED = int(os.environ.get("SEED", "0"))
CHUNK = 30
NG = 20
H = 400
Dm = 256
N_EP_MAX = 30
DEMOS = [int(x) for x in os.environ.get("DEMOS", "3,6,12,30").split(",")]
torch.manual_seed(SEED)
np.random.seed(SEED)


def log(m):
    print(m, flush=True)


# --- load scripted insertion demos (images for DINOv2) ---
ds = LeRobotDataset("lerobot/aloha_sim_insertion_scripted")
ei = np.array(ds.hf_dataset["episode_index"])
keep = np.where(ei < N_EP_MAX)[0]


def to_img(t):
    return np.array(Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((256, 256)))


imgs, states, actions, epid = [], [], [], []
for i in keep:
    it = ds[int(i)]
    imgs.append(to_img(it["observation.images.top"]))
    states.append(it["observation.state"].numpy())
    actions.append(it["action"].numpy())
    epid.append(int(ei[i]))
imgs = np.stack(imgs)
states = np.array(states, np.float32)
actions = np.array(actions, np.float32)
epid = np.array(epid)
log(f"loaded {len(imgs)} demo frames, {N_EP_MAX} eps")

# --- DINOv2 encoders: frozen vs adapted ---

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


def load_dino(adapted):
    m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(dev)
    if adapted:
        m.load_state_dict(torch.load(OUT + "/dynamo_dino_pooled.pt"))
    m.eval()
    return m


@torch.no_grad()
def enc_patches(m, ims, bs=64):  # -> (N,64,768) : 8x8 maxpool of DINOv2 16x16 patch tokens
    out = np.zeros((len(ims), 64, 768), np.float16)
    for k in range(0, len(ims), bs):
        x = (
            torch.from_numpy(
                np.stack([np.array(Image.fromarray(im).resize((224, 224))) for im in ims[k : k + bs]])
            )
            .to(dev)
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        x = (x - mean) / std
        h = m(pixel_values=x).last_hidden_state[:, 1:]
        b = h.shape[0]
        g = int(round(h.shape[1] ** 0.5))
        hh = h.reshape(b, g, g, 768)
        sp = hh.reshape(b, 8, g // 8, 8, g // 8, 768).amax(dim=(2, 4)).reshape(b, 64, 768)
        out[k : k + b] = sp.float().cpu().numpy().astype(np.float16)
    return out


class SmallACT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(768, Dm)
        self.patch_pos = nn.Parameter(torch.randn(1, 64, Dm) * 0.02)
        self.state_proj = nn.Linear(14, Dm)
        el = nn.TransformerEncoderLayer(Dm, 8, 4 * Dm, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(el, 2)
        self.query = nn.Parameter(torch.randn(1, CHUNK, Dm) * 0.02)
        dl = nn.TransformerDecoderLayer(Dm, 8, 4 * Dm, 0.1, batch_first=True)
        self.dec = nn.TransformerDecoder(dl, 2)
        self.head = nn.Linear(Dm, 14)

    def forward(self, patches, state):
        toks = torch.cat([self.state_proj(state)[:, None], self.patch_proj(patches) + self.patch_pos], 1)
        return self.head(self.dec(self.query.expand(patches.shape[0], -1, -1), self.enc(toks)))


smu, ssd = states.mean(0), states.std(0) + 1e-6
amu, asd = actions.mean(0), actions.std(0) + 1e-6


def ns(x):
    return (x - smu) / ssd


def na(x):
    return (x - amu) / asd


def da(x):
    return x * asd + amu


VEC = Vec(NG)
_lf = VEC.vec.envs[0].unwrapped._env._physics.model.body("vx300s_right/left_finger_link").id
_rf = VEC.vec.envs[0].unwrapped._env._physics.model.body("vx300s_right/right_finger_link").id


def gcen():
    o = np.zeros((NG, 3))
    for i in range(NG):
        p = VEC.vec.envs[i].unwrapped._env._physics
        o[i] = (np.asarray(p.data.xpos[_lf]) + np.asarray(p.data.xpos[_rf])) / 2
    return o


def run(patches_all, live_model, n_ep):
    # build chunk samples from first n_ep episodes
    keep_ep = np.unique(epid)[:n_ep]
    samples = []
    for e in keep_ep:
        idx = np.where(epid == e)[0]
        for j in range(len(idx) - CHUNK):
            samples.append((idx[j], idx[j + 1 : j + 1 + CHUNK]))
    ti = np.array([s[0] for s in samples])
    ci = np.stack([s[1] for s in samples])
    Pt = torch.tensor(patches_all.astype(np.float32), device=dev)
    St = torch.tensor(ns(states), device=dev)
    At = torch.tensor(na(actions), device=dev)
    net = SmallACT().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-4, weight_decay=1e-4)
    n = len(ti)
    idx = np.arange(n)
    cut = int(n * 0.9)
    best, bad, bSD = 1e9, 0, None
    for ep in range(120):
        net.train()
        np.random.shuffle(idx)
        tr = idx[:cut]
        for k in range(0, len(tr), 128):
            b = tr[k : k + 128]
            loss = (net(Pt[ti[b]], St[ti[b]]) - At[ci[b]]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = idx[cut:]
            vl = sum(
                (net(Pt[ti[va[k : k + 256]]], St[ti[va[k : k + 256]]]) - At[ci[va[k : k + 256]]])
                .abs()
                .mean()
                .item()
                * len(va[k : k + 256])
                for k in range(0, len(va), 256)
            ) / max(len(va), 1)
        if vl < best - 1e-4:
            best, bad, bSD = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= 12:
            break
    net.load_state_dict(bSD)
    # eval reach
    img, prop = VEC.reset(range(40000, 40000 + NG))
    bestd = np.full(NG, 1e9)
    t = 0
    while t < H:
        P = torch.tensor(enc_patches(live_model, img).astype(np.float32), device=dev)
        st = torch.tensor(ns(prop), device=dev)
        with torch.no_grad():
            chunk = da(net(P, st).cpu().numpy())
        for j in range(CHUNK):
            (img, prop), _, _ = VEC.step(chunk[:, j].astype(np.float32))
            bestd = np.minimum(bestd, np.linalg.norm(gcen() - VEC.obj_xyz()[:, :3], axis=1))
            t += 1
            if t >= H:
                break
    return bestd


log("encoding demo patches (frozen + adapted)...")
m_fz = load_dino(False)
Pf = enc_patches(m_fz, imgs)
m_ad = load_dino(True)
Pa = enc_patches(m_ad, imgs)
log("Pf,Pa ready")
res = {}
for n_ep in DEMOS:
    for name, P, lm in [("frozen", Pf, m_fz), ("adapted", Pa, m_ad)]:
        t0 = time.time()
        b = run(P, lm, n_ep)
        res[(name, n_ep)] = b
        log(
            f"  {name:8s} n_ep={n_ep:2d}: reach minDist {b.mean():.3f}m | SR@0.05={float((b < 0.05).mean()):.2f} @0.1={float((b < 0.1).mean()):.2f}  [{time.time() - t0:.0f}s]"
        )
np.savez(
    OUT + f"/sp_fork1_s{SEED}.npz", **{f"{k[0]}|{k[1]}": v for k, v in res.items()}, demos=np.array(DEMOS)
)
log("[ok] sp_fork1 done")
