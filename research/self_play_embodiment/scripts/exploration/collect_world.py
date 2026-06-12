# ruff: noqa
"""Self-play buffer WITH world-state labels for the reframed (world-centric) G1 and G2.

Logs per step: image@256, agent_pos(14), action(14), and world=(peg, socket,
left-gripper, right-gripper) Cartesian positions (12). ~20 episodes for a proper
held-out-episode split. npz only (video/dataset capability already proven).
Run: python collect_world.py [N_STEPS]
"""

import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from PIL import Image

OUT, IMG = "/tmp/selfplay_probe", 256  # nosec B108
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


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
    return np.concatenate([np.asarray(phys.named.data.xpos[k]) for k in KEYS]).astype(np.float32)


low = np.asarray(vec.action_space.low, np.float32)
high = np.asarray(vec.action_space.high, np.float32)
obs, _ = vec.reset(seed=0)
leaves = walk(obs)
imgk = next(k for k, v in leaves.items() if v.ndim >= 3 and v.shape[-1] == 3 and v.dtype == np.uint8)
statek = next(k for k in leaves if k != imgk)

imgs, states, actions, worlds, ep_bounds = [], [], [], [], []
a = vec.action_space.sample().astype(np.float32)
alpha = 0.85
t0 = time.time()
ep_start = 0
for t in range(N):
    leaves = walk(obs)
    img_full = leaves[imgk][0]
    st = leaves[statek][0].astype(np.float32)
    w = world()
    a = np.clip(alpha * a + (1 - alpha) * vec.action_space.sample(), low, high).astype(np.float32)
    imgs.append(np.asarray(Image.fromarray(img_full).resize((IMG, IMG))))
    states.append(st)
    actions.append(a[0].copy())
    worlds.append(w)
    obs, r, term, trunc, info = vec.step(a)
    if bool(term[0]) or bool(trunc[0]):
        ep_bounds.append((ep_start, len(imgs)))
        ep_start = len(imgs)
    if (t + 1) % 2000 == 0:
        log(f"  step {t + 1}/{N} ({(t + 1) / (time.time() - t0):.0f}/s), {len(ep_bounds)} eps")
if ep_start < len(imgs):
    ep_bounds.append((ep_start, len(imgs)))
imgs = np.stack(imgs)
states = np.stack(states)
actions = np.stack(actions)
worlds = np.stack(worlds)
log(
    f"collected {len(imgs)} steps, {len(ep_bounds)} episodes in {time.time() - t0:.0f}s; world dim={worlds.shape[1]}"
)
np.savez_compressed(
    OUT + "/world_buffer.npz",
    images=imgs,
    states=states,
    actions=actions,
    world=worlds,
    ep_bounds=np.array(ep_bounds),
    world_keys=np.array(KEYS),
)
log(f"[ok] world_buffer.npz ({imgs.nbytes / 1e9:.2f} GB raw images)")
