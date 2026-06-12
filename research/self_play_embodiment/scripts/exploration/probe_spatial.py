# ruff: noqa
"""Does the SPATIAL patch grid (not the global mean) encode the objects?

V-JEPA returns one token per 2D image patch (16x16 grid). Compare decoding
object/body position from:
  - global mean-pool over tokens  (what we used -> objects vanished)
  - a 4x4 max-pool of the patch grid (keeps coarse 2D location + salient features)
held-out-episode split; objects filtered to in-workspace frames (flung ones removed)."""

import os
import time

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
OUT, REPO = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256"  # nosec B108
dev, dt = "cuda", torch.bfloat16
d = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs = d["images"]
W = d["world"].astype(np.float64)
ep = [tuple(map(int, b)) for b in d["ep_bounds"]]
N = len(imgs)
from transformers import AutoVideoProcessor, VJEPA2Model

model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, dt)
Tc, G = 2, 4

reported = {}


def encode_all(bs=32):
    M = np.zeros((N, 1408), np.float32)
    S = np.zeros((N, G * G * 1408), np.float32)
    for k in range(0, N, bs):
        clips = np.stack([np.repeat(imgs[f : f + 1], Tc, 0) for f in range(k, min(k + bs, N))])
        x = torch.from_numpy(clips).to(dev).permute(0, 1, 4, 2, 3).to(dt) / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            o = model(pixel_values_videos=x)
        h = (o.last_hidden_state if getattr(o, "last_hidden_state", None) is not None else o[0]).float()
        b, ntok, C = h.shape
        g = int(round(ntok**0.5))
        assert g * g == ntok, f"ntok={ntok} not square"
        if "grid" not in reported:
            reported["grid"] = (ntok, g)
            print(f"  tokens per frame = {ntok} = {g}x{g} patches, dim={C}", flush=True)
        hh = h.reshape(b, g, g, C)
        M[k : k + b] = hh.mean((1, 2)).cpu().numpy()
        bsz = g // G
        hp = hh.reshape(b, G, bsz, G, bsz, C).amax(dim=(2, 4))  # 4x4 max-pool of the patch grid
        S[k : k + b] = hp.reshape(b, -1).cpu().numpy()
    return M, S


t0 = time.time()
M, S = encode_all()
print(f"encoded {N} frames in {time.time() - t0:.0f}s", flush=True)

frame_ep = np.zeros(N, int)
for eid, (s, e) in enumerate(ep):
    frame_ep[s:e] = eid
ne = len(ep)
fit = frame_ep < ne - 4
test = frame_ep >= ne - 4
peg, soc = W[:, 0:3], W[:, 3:6]
inb = (
    (peg[:, 2] < 0.15)
    & (soc[:, 2] < 0.15)
    & (np.abs(peg[:, 0]) < 0.4)
    & (np.abs(soc[:, 0]) < 0.4)
    & (np.abs(peg[:, 1] - 0.55) < 0.4)
    & (np.abs(soc[:, 1] - 0.55) < 0.4)
)
print(f"in-workspace frames: {inb.sum()}/{N} ({100 * inb.mean():.0f}%)", flush=True)


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    Wt = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym))
    return Xte @ Wt + ym


def decode(F, target, fmask):
    f, t = fit & fmask, test & fmask
    k = min(200, F.shape[1])
    mu = F[f].mean(0)
    _, _, Vt = np.linalg.svd(F[f] - mu, full_matrices=False)
    comp = Vt[:k]
    return r2(target[t], ridge((F[f] - mu) @ comp.T, target[f], (F[t] - mu) @ comp.T))


obj_xy = W[:, [0, 1, 3, 4]]  # peg_xy, socket_xy
body_xy = W[:, [6, 7, 9, 10]]  # Lgrip_xy, Rgrip_xy
allm = np.ones(N, bool)
print("\n=== decode position from frozen latent (held-out episodes) ===")
print(f"{'target':16s} {'global-mean':>12s} {'4x4 max-pool':>14s}")
print(f"{'OBJECTS xy':16s} {decode(M, obj_xy, inb):12.3f} {decode(S, obj_xy, inb):14.3f}")
print(f"{'BODY xy':16s} {decode(M, body_xy, allm):12.3f} {decode(S, body_xy, allm):14.3f}")
