# ruff: noqa
"""OOD degradation-slope eval (paper's object-layout-OOD axis, graded).
Loads a saved stage-2 model; evals closed-loop on insertion with peg+socket spawned in ranges WIDENED
by factor f around the original sampler's centers (f=1 reproduces the in-dist ranges, via the SAME
override path so the protocol is identical across f). Same 48 env seeds for every (arm, f) -> paired.
Verdict statistic across arms = degradation slope SR(f) — paper claims JEPA degrades less.
Usage: sp_ood_eval.py <s2_models/TAG.pt> <f> <n> [tag]"""

import os
import sys

import mujoco
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/tmp/selfplay_probe")  # nosec B108
from sp_lib import Vec, _walk
from sp_vj_act import CHUNK, ACTJepa, resize224

dev = "cuda"
MODEL = sys.argv[1]
F = float(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 48
TAG = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(MODEL).replace(".pt", "")
T_EVAL = 500
ck = torch.load(MODEL, map_location=dev, weights_only=False)
model = ACTJepa().to(dev)
model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
model.eval()
smu, ssd, amu, asd = ck["smu"], ck["ssd"], ck["amu"], ck["asd"]
imn = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
isd_ = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


def prep(x):
    return (resize224(x).to(dev) - imn) / isd_


vec = Vec(N)
ph0 = vec.vec.envs[0].unwrapped._env._physics
padr = int(ph0.named.model.jnt_qposadr["red_peg_joint"])
sadr = int(ph0.named.model.jnt_qposadr["blue_socket_joint"])


def yr(f):
    h = min(0.10 * f, 0.15)
    return 0.5 - h, 0.5 + h  # y in [0.35,0.65] cap (gate-verified reachable)


rng = np.random.RandomState(12345 + int(F * 10))  # positions fixed per f (same for every arm)
obs, _ = vec.vec.reset(seed=list(range(400000, 400000 + N)))
ylo, yhi = yr(F)
peg_xy = np.c_[rng.uniform(0.15 - 0.05 * F, 0.15 + 0.05 * F, N), rng.uniform(ylo, yhi, N)]
soc_xy = np.c_[rng.uniform(-0.15 - 0.05 * F, -0.15 + 0.05 * F, N), rng.uniform(ylo, yhi, N)]
for i in range(N):
    p = vec.vec.envs[i].unwrapped._env._physics
    p.data.qpos[padr : padr + 2] = peg_xy[i]
    p.data.qpos[padr + 2] = 0.05
    p.data.qpos[padr + 3 : padr + 7] = [1, 0, 0, 0]
    p.data.qpos[sadr : sadr + 2] = soc_xy[i]
    p.data.qpos[sadr + 2] = 0.05
    p.data.qpos[sadr + 3 : sadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(p.model.ptr, p.data.ptr)


def get(obs):
    L = _walk(obs)
    return L[vec.imgk], L[vec.statek].astype(np.float32)


# re-render after override: take obs from a zero-ish settle step? No-op step renders the new layout.
img, prop = get(obs)
m_te = 0.01
preds = np.full((T_EVAL, N, CHUNK, 14), np.nan, np.float32)
maxr = np.zeros(N)
for t in range(T_EVAL):
    if t == 0:  # refresh obs post-override with a hold-position step
        obs, r, te_, tr_, _ = vec.vec.step(prop.astype(np.float32))
        img, prop = get(obs)
    st = torch.tensor((prop - smu) / ssd, dtype=torch.float32, device=dev)
    with torch.no_grad():
        mem = model.encode(prep(img), st)
        out, _, _ = model.act(mem, None)
        preds[t] = out.cpu().numpy() * asd + amu
    acts = np.zeros((N, 14))
    wsum = 0.0
    for rt in range(max(0, t - CHUNK + 1), t + 1):
        off = t - rt
        w = np.exp(-m_te * off)
        acts += w * preds[rt, :, off, :]
        wsum += w
    obs, r, term, trunc, info = vec.vec.step(np.clip((acts / wsum).astype(np.float32), -1, 1))
    maxr = np.maximum(maxr, np.asarray(r, float))
    img, prop = get(obs)
hist = [int((np.round(maxr) == k).sum()) for k in range(5)]
print(
    f"[OOD {TAG} f={F}] n={N}: mean {maxr.mean():.2f} | SR>=1 {(maxr >= 1).mean() * 100:.0f}% | SR>=2 {(maxr >= 2).mean() * 100:.0f}% | SR>=3 {(maxr >= 3).mean() * 100:.0f}% | hist {hist}",
    flush=True,
)
