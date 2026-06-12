# ruff: noqa
"""Comparative G2: does the action->Delta-z gap JUMP in contact-rich scripted demos
vs our contact-poor random play? Same spatial pipeline for both. (Confounded by demo
action statistics, but a quick existing-data look at whether manipulation surfaces
action->world signal.)"""

import os
import time

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image

OUT, REPO = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256"  # nosec B108
dev, dt = "cuda", torch.bfloat16
from transformers import AutoVideoProcessor, VJEPA2Model

model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, dt)
Tc, G = 2, 4


def encode(imgs):
    n = len(imgs)
    M = np.zeros((n, 1408), np.float32)
    S = np.zeros((n, G * G * 1408), np.float32)
    for k in range(0, n, 32):
        clips = np.stack([np.repeat(imgs[i : i + 1], Tc, 0) for i in range(k, min(k + 32, n))])
        x = torch.from_numpy(clips).to(dev).permute(0, 1, 4, 2, 3).to(dt) / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            o = model(pixel_values_videos=x)
        h = (o.last_hidden_state if getattr(o, "last_hidden_state", None) is not None else o[0]).float()
        b, nt, C = h.shape
        g = int(round(nt**0.5))
        hh = h.reshape(b, g, g, C)
        M[k : k + b] = hh.mean((1, 2)).cpu().numpy()
        S[k : k + b] = hh.reshape(b, G, g // G, G, g // G, C).amax(dim=(2, 4)).reshape(b, -1).cpu().numpy()
    return M, S


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, a=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xte @ np.linalg.solve(
        Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - Ytr.mean(0))
    ) + Ytr.mean(0)


def gap(F, actions, ts, t1s, epids):
    ne = epids.max() + 1
    test = epids >= ne - 4
    fit = ~test
    Z0, Z1 = F[ts], F[t1s]
    m = Z0[fit].mean(0)
    _, _, Vt = np.linalg.svd(Z0[fit] - m, full_matrices=False)
    c = Vt[:64]
    P0, P1 = (Z0 - m) @ c.T, (Z1 - m) @ c.T
    dZ = P1 - P0
    A = actions[ts]
    af = r2(dZ[test], ridge(P0[fit], dZ[fit], P0[test]))
    ac = r2(dZ[test], ridge(np.hstack([P0, A])[fit], dZ[fit], np.hstack([P0, A])[test]))
    return af, ac


def pairs_from_bounds(ep_of_frame, nframes):
    ts, t1s, epids = [], [], []
    for f in range(nframes - 1):
        if ep_of_frame[f] == ep_of_frame[f + 1]:
            ts.append(f)
            t1s.append(f + 1)
            epids.append(ep_of_frame[f])
    return np.array(ts), np.array(t1s), np.array(epids)


# ---- RANDOM (cached features) ----
z = np.load(OUT + "/feat_cache.npz")
Mr, Sr = z["M"], z["S"]
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
rand_act = wb["actions"].astype(np.float64)
ep_of = np.zeros(len(Mr), int)
for eid, (s, e) in enumerate([tuple(map(int, b)) for b in wb["ep_bounds"]]):
    ep_of[s:e] = eid
rt, rt1, rep = pairs_from_bounds(ep_of, len(Mr))

# ---- DEMOS (encode 20 episodes) ----
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_scripted")
ep_idx = np.array(ds.hf_dataset["episode_index"])
keep = np.where(ep_idx < 20)[0]
print(f"decoding {len(keep)} demo frames (20 episodes)...", flush=True)
t0 = time.time()
imgs, acts, depids = [], [], []
for j, i in enumerate(keep):
    it = ds[int(i)]
    im = (it["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    imgs.append(np.asarray(Image.fromarray(im).resize((256, 256))))
    acts.append(it["action"].numpy())
    depids.append(int(ep_idx[i]))
    if (j + 1) % 2000 == 0:
        print(f"  {j + 1}/{len(keep)} ({(j + 1) / (time.time() - t0):.0f}/s)", flush=True)
imgs = np.stack(imgs)
demo_act = np.array(acts, np.float64)
depids = np.array(depids)
print(f"decoded in {time.time() - t0:.0f}s; encoding...", flush=True)
Md, Sd = encode(imgs)
dep_of = depids  # per-frame episode id
dt_, dt1, dep = pairs_from_bounds(dep_of, len(Md))

print("\n=== action->dz gap: RANDOM (contact-poor) vs SCRIPTED DEMO (contact-rich) ===")
for name, (Fm, Fs, act, a, b, c) in {
    "RANDOM": (Mr, Sr, rand_act, rt, rt1, rep),
    "DEMO  ": (Md, Sd, demo_act, dt_, dt1, dep),
}.items():
    af_m, ac_m = gap(Fm, act, a, b, c)
    af_s, ac_s = gap(Fs, act, a, b, c)
    print(
        f"  {name}  mean: free {af_m:.3f} +act {ac_m:.3f} gap {ac_m - af_m:+.3f}  |  spatial: free {af_s:.3f} +act {ac_s:.3f} gap {ac_s - af_s:+.3f}"
    )
