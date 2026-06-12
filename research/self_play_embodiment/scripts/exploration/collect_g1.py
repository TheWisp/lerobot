# ruff: noqa
"""G1 data collection: drive aloha self-play (smoothed random walk) and emit:
  1. g1_buffer.npz       — (images@256, states, actions, ep_bounds) for the frozen-encoder probe
  2. selfplay_aloha.mp4  — video proof of the driven sim
  3. LeRobotDataset      — proper video-encoded dataset (GUI-viewable) at ds_g1/

Phases are decoupled: the npz + mp4 are written before the dataset, so a dataset
encoding hiccup never costs us the G1 data. Scratch script, lives in /tmp.
Run: python collect_g1.py [N_STEPS]
"""

import os
import shutil
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from PIL import Image

OUT = "/tmp/selfplay_probe"  # nosec B108
DS_ROOT = os.path.join(OUT, "ds_g1")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
IMG = 256
MP4_FRAMES = 900


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
vec = cfg.create_envs(n_envs=1, use_async_envs=False)[cfg.type][0]
low = np.asarray(vec.action_space.low, dtype=np.float32)
high = np.asarray(vec.action_space.high, dtype=np.float32)

obs, _ = vec.reset(seed=0)
leaves = walk(obs)
imgk = next(k for k, v in leaves.items() if v.ndim >= 3 and v.shape[-1] == 3 and v.dtype == np.uint8)
statek = next(k for k in leaves if k != imgk)
log(
    f"img={imgk}{leaves[imgk].shape} state={statek}{leaves[statek].shape} action={tuple(vec.action_space.shape)}"
)

import imageio.v2 as imageio

mp4 = imageio.get_writer(os.path.join(OUT, "selfplay_aloha.mp4"), fps=int(cfg.fps), macro_block_size=None)

imgs, states, actions, ep_bounds = [], [], [], []
a = vec.action_space.sample().astype(np.float32)
alpha = 0.85
t0 = time.time()
ep_start = 0
for t in range(N):
    leaves = walk(obs)
    img_full = leaves[imgk][0]
    st = leaves[statek][0].astype(np.float32)
    a = np.clip(alpha * a + (1 - alpha) * vec.action_space.sample(), low, high).astype(np.float32)
    if t < MP4_FRAMES:
        mp4.append_data(img_full)
    imgs.append(np.asarray(Image.fromarray(img_full).resize((IMG, IMG))))
    states.append(st)
    actions.append(a[0].copy())
    obs, r, term, trunc, info = vec.step(a)
    if bool(term[0]) or bool(trunc[0]):
        ep_bounds.append((ep_start, len(imgs)))
        ep_start = len(imgs)
    if (t + 1) % 1000 == 0:
        log(f"  step {t + 1}/{N} ({(t + 1) / (time.time() - t0):.0f}/s), {len(ep_bounds)} episodes done")
mp4.close()
if ep_start < len(imgs):
    ep_bounds.append((ep_start, len(imgs)))
imgs = np.stack(imgs)
states = np.stack(states)
actions = np.stack(actions)
log(f"collected {len(imgs)} steps, {len(ep_bounds)} episodes in {time.time() - t0:.0f}s")
np.savez_compressed(
    os.path.join(OUT, "g1_buffer.npz"),
    images=imgs,
    states=states,
    actions=actions,
    ep_bounds=np.array(ep_bounds),
)
log(f"[ok] g1_buffer.npz ({imgs.nbytes / 1e9:.2f} GB raw) + selfplay_aloha.mp4")

# --- proper LeRobotDataset (video-encoded) ---
try:
    shutil.rmtree(DS_ROOT, ignore_errors=True)
    from lerobot.datasets import LeRobotDataset, VideoEncodingManager

    feats = {
        "action": {"dtype": "float32", "shape": (actions.shape[1],), "names": None},
        "observation.state": {"dtype": "float32", "shape": (states.shape[1],), "names": None},
        "observation.images.top": {
            "dtype": "video",
            "shape": (IMG, IMG, 3),
            "names": ["height", "width", "channel"],
        },
    }
    ds = LeRobotDataset.create(
        repo_id="selfplay/aloha_g1",
        fps=int(cfg.fps),
        root=DS_ROOT,
        robot_type="aloha_sim",
        features=feats,
        use_videos=True,
    )
    with VideoEncodingManager(ds):
        for s, e in ep_bounds:
            for i in range(s, e):
                ds.add_frame(
                    {
                        "action": actions[i],
                        "observation.state": states[i],
                        "observation.images.top": imgs[i],
                        "task": "self-play random exploration",
                    }
                )
            ds.save_episode()
            log(f"  episode [{s}:{e}] saved")
    ds.finalize()
    log(f"[ok] LeRobotDataset at {DS_ROOT}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
except Exception:
    log("[FAIL] dataset write:\n" + traceback.format_exc())

log("ALL DONE")
