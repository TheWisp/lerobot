"""Stream random play, keep ALL contact transitions + a sample of non-contacts,
with object ground-truth. Stores z_t (mean + 4x4-spatial latent), action_t, and the
world change (peg/socket/gripper deltas) per transition -> contact_buffer.npz.
Target for the test = Δobject (GT), so we only need z_t, not z_{t+1}."""

import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image

OUT, REPO = "/tmp/selfplay_probe", "facebook/vjepa2-vitg-fpc64-256"
dev, dt = "cuda", torch.bfloat16
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
CONTACT, SAMPLE, CAP = 0.002, 0.15, 12000


def log(m):
    print(m, flush=True)


def walk(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(walk(v, f"{p}{k}/"))
    else:
        out[p.rstrip("/")] = np.asarray(o)
    return out


from lerobot.envs.configs import AlohaEnv

cfg = AlohaEnv(obs_type="pixels_agent_pos", observation_height=480, observation_width=640)
vec = cfg.create_envs(n_envs=1)[cfg.type][0]
base = vec.envs[0]
phys = base.unwrapped._env._physics
KEYS = ["peg", "socket", "vx300s_left/gripper_link", "vx300s_right/gripper_link"]


def world():
    return np.concatenate([np.asarray(phys.named.data.xpos[k]) for k in KEYS]).astype(np.float64)


low = np.asarray(vec.action_space.low, np.float32)
high = np.asarray(vec.action_space.high, np.float32)
obs, _ = vec.reset(seed=0)
leaves = walk(obs)
imgk = next(k for k, v in leaves.items() if v.ndim >= 3 and v.shape[-1] == 3 and v.dtype == np.uint8)
rng = np.random.default_rng(0)
a = vec.action_space.sample().astype(np.float32)
alpha = 0.85
img_prev = leaves[imgk][0]
w_prev = world()
eid = 0
imgs, acts, wdel, epids = [], [], [], []
n_contact = 0
t0 = time.time()
for t in range(N):
    a = np.clip(alpha * a + (1 - alpha) * vec.action_space.sample(), low, high).astype(np.float32)
    obs, r, term, trunc, info = vec.step(a)
    leaves = walk(obs)
    img_next = leaves[imgk][0]
    w_next = world()
    if not (bool(term[0]) or bool(trunc[0])):
        dW = w_next - w_prev
        dobj = np.linalg.norm(dW[:6].reshape(2, 3), axis=1).max()
        is_c = dobj > CONTACT
        if (is_c or rng.random() < SAMPLE) and len(imgs) < CAP:
            imgs.append(np.asarray(Image.fromarray(img_prev).resize((256, 256))))
            acts.append(a[0].copy())
            wdel.append(dW.astype(np.float32))
            epids.append(eid)
            n_contact += int(is_c)
    else:
        eid += 1
    img_prev = img_next
    w_prev = w_next
    if (t + 1) % 5000 == 0:
        log(
            f"  step {t + 1}/{N} ({(t + 1) / (time.time() - t0):.0f}/s) stored={len(imgs)} contacts={n_contact} eps={eid}"
        )
log(
    f"collected {len(imgs)} transitions ({n_contact} contacts) over {eid + 1} episodes in {time.time() - t0:.0f}s"
)
imgs = np.stack(imgs)
acts = np.array(acts, np.float32)
wdel = np.stack(wdel)
epids = np.array(epids)

from transformers import AutoVideoProcessor, VJEPA2Model

model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, dt)
Tc, G = 2, 4


def encode(ims, bs=32):
    M = np.zeros((len(ims), 1408), np.float32)
    S = np.zeros((len(ims), G * G * 1408), np.float32)
    for k in range(0, len(ims), bs):
        clips = np.stack([np.repeat(ims[i : i + 1], Tc, 0) for i in range(k, min(k + bs, len(ims)))])
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


t1 = time.time()
M, S = encode(imgs)
log(f"encoded {len(imgs)} z_t in {time.time() - t1:.0f}s")
np.savez_compressed(
    OUT + "/contact_buffer.npz", Zmean=M, Zspatial=S, action=acts, world_delta=wdel, epid=epids
)
log("[ok] contact_buffer.npz saved")
