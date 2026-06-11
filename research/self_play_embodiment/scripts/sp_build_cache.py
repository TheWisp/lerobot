# ruff: noqa
"""Unified transition cache for the embodiment-injection experiment.
Sources (same-embodiment bimanual ViperX): real aloha insertion + transfer (human+scripted) + our self-play.
HELD OUT of EVERYTHING: insertion eps 40:50 (human+scripted) -> eval set (closed-loop, via env).
Stores img@224 uint8 (memmap), state(14), action(14), global epid, taskid. -> cache/"""

import os, numpy as np, torch

os.environ.setdefault("MUJOCO_GL", "egl")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch.nn.functional as F

OUT = "/tmp/selfplay_probe/cache"  # nosec B108
os.makedirs(OUT, exist_ok=True)
RES = 224  # nosec B108
# (repo, ep_lo, ep_hi exclusive, taskid, stride) ; insertion 0:40 train(+pretrain), 40:50 held out (not cached)
specs = [
    ("lerobot/aloha_sim_insertion_human", 0, 40, 0, 1),
    ("lerobot/aloha_sim_insertion_scripted", 0, 40, 0, 1),
    ("lerobot/aloha_sim_transfer_cube_human", 0, 50, 1, 2),
    ("lerobot/aloha_sim_transfer_cube_scripted", 0, 50, 1, 2),
]
plan = []  # (kind, ref, idx, taskid, epid)
gid = 0
dss = {}
for repo, lo, hi, tid, stride in specs:
    ds = LeRobotDataset(repo)
    dss[repo] = ds
    ei = np.asarray(ds.hf_dataset["episode_index"])
    for e in range(lo, hi):
        fr_idx = np.where(ei == e)[0]
        for i in fr_idx[::stride]:
            plan.append(("hf", repo, int(i), tid, gid))
        gid += 1
# NOTE: self-play omitted from cache for now (5.9GB npz caused memory pressure; low-quality single-arm).
# Real bimanual datasets are the core; can add self-play back later via memmap if action-diversity is needed.
N = len(plan)
print(f"cache N={N} frames, {gid} episodes", flush=True)
imgs = np.memmap(OUT + "/imgs.u8", mode="w+", dtype=np.uint8, shape=(N, RES, RES, 3))
states = np.zeros((N, 14), np.float32)
actions = np.zeros((N, 14), np.float32)
epid = np.zeros(N, np.int32)
taskid = np.zeros(N, np.int8)
framepos = np.zeros(N, np.int32)


def to224(t):  # t: (3,H,W) float[0,1] -> (224,224,3) uint8
    x = F.interpolate(t[None], size=(RES, RES), mode="bilinear", align_corners=False)[0]
    return (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


import time

t0 = time.time()
cur_ep = -1
fp = 0
for n, (kind, ref, i, tid, eid) in enumerate(plan):
    if eid != cur_ep:
        cur_ep = eid
        fp = 0
    if kind == "hf":
        fr = dss[ref][i]
        imgs[n] = to224(fr["observation.images.top"])
        states[n] = fr["observation.state"].numpy()
        actions[n] = fr["action"].numpy()
    else:
        im = sp["images"][i]
        t = torch.from_numpy(im).permute(2, 0, 1).float() / 255
        imgs[n] = to224(t)
        states[n] = sp["states"][i]
        actions[n] = sp["actions"][i]
    epid[n] = eid
    taskid[n] = tid
    framepos[n] = fp
    fp += 1
    if n % 5000 == 0:
        print(f"  {n}/{N} ({n / max(1, time.time() - t0):.0f}/s)", flush=True)
imgs.flush()
np.savez(
    OUT + "/meta.npz",
    states=states,
    actions=actions,
    epid=epid,
    taskid=taskid,
    framepos=framepos,
    N=N,
    RES=RES,
)
print(
    f"[ok] cache built {N} frames | taskid counts: insertion={int((taskid == 0).sum())} transfer={int((taskid == 1).sum())} selfplay={int((taskid == 2).sum())} ({time.time() - t0:.0f}s)",
    flush=True,
)
