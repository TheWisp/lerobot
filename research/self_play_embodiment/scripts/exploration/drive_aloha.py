"""Autonomous self-play driver for the LeRobot aloha sim (plumbing + G1 substrate).

Stands up the aloha gym env, drives it with a smoothed random walk in action space
(no policy, no teleop, no hardware), and dumps:
  - frame_*.png        : RGB frames over time (visual proof the arm is being driven)
  - buffer.npz         : (agent_pos, action, next_agent_pos) for every step + strided images
  - summary.json       : shapes, steps/sec, mean frame-to-frame pixel delta (motion proof)

Scratch script (lives in /tmp, not committed) — the artifacts are the deliverable.
Run:  python drive_aloha.py [N_STEPS]
"""

import json
import os
import sys
import time

import numpy as np

OUT = "/tmp/selfplay_probe"
os.makedirs(OUT, exist_ok=True)


def walk(obs, prefix=""):
    """Flatten a (possibly nested) gym observation dict to {path: ndarray}."""
    out = {}
    if isinstance(obs, dict):
        for k, v in obs.items():
            out.update(walk(v, f"{prefix}{k}/"))
    else:
        out[prefix.rstrip("/")] = np.asarray(obs)
    return out


def is_image(arr):
    return arr.ndim >= 3 and arr.shape[-1] == 3 and arr.dtype == np.uint8


def build_env():
    """Create the aloha vec env. Returns (cfg, vec). Action space shape is (1, 14)."""
    from lerobot.envs.configs import AlohaEnv

    cfg = AlohaEnv(obs_type="pixels_agent_pos", observation_height=480, observation_width=640)
    envs = cfg.create_envs(n_envs=1, use_async_envs=False)
    return cfg, envs[cfg.type][0]


def save_png(arr, path):
    try:
        import imageio.v3 as iio

        iio.imwrite(path, arr)
    except Exception:
        from PIL import Image

        Image.fromarray(arr).save(path)


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    # mujoco offscreen render backend: try GPU (egl) first, fall back to CPU (osmesa).
    last_err = None
    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            cfg, vec = build_env()
            obs, _ = vec.reset(seed=0)
            print(f"[ok] mujoco GL backend = {backend}")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[warn] backend {backend} failed: {type(e).__name__}: {e}")
            vec = None
    if vec is None:
        raise RuntimeError(f"could not init aloha env with any GL backend: {last_err}")

    leaves = walk(obs)
    img_keys = [k for k, v in leaves.items() if is_image(v)]
    state_keys = [k for k, v in leaves.items() if not is_image(v)]
    print("action_space :", vec.action_space)
    print("image keys   :", {k: leaves[k].shape for k in img_keys})
    print("state keys   :", {k: leaves[k].shape for k in state_keys})
    assert img_keys, "no RGB image found in observation — cannot feed an encoder"

    img_key = img_keys[0]
    state_key = state_keys[0] if state_keys else None

    low = np.asarray(vec.action_space.low, dtype=np.float32)
    high = np.asarray(vec.action_space.high, dtype=np.float32)
    a = vec.action_space.sample().astype(np.float32)
    alpha = 0.85  # action smoothing -> continuous-ish motion, not pure flailing

    states, actions, next_states = [], [], []
    saved_imgs, saved_idx = [], []
    n_save = 8
    save_every = max(1, n_steps // n_save)
    prev_state = leaves[state_key][0].copy() if state_key else None

    t0 = time.time()
    n_resets = 0
    for t in range(n_steps):
        a = np.clip(alpha * a + (1 - alpha) * vec.action_space.sample(), low, high).astype(np.float32)
        obs, reward, terminated, truncated, info = vec.step(a)
        leaves = walk(obs)
        if state_key:
            cur = leaves[state_key][0].copy()
            states.append(prev_state)
            actions.append(a[0].copy())
            next_states.append(cur)
            prev_state = cur
        if bool(terminated[0]) or bool(truncated[0]):
            n_resets += 1
        if t % save_every == 0 or t == n_steps - 1:
            img = leaves[img_key][0]
            p = os.path.join(OUT, f"frame_{len(saved_imgs):03d}.png")
            save_png(img, p)
            saved_imgs.append(img.astype(np.int16))
            saved_idx.append(t)
    dt = time.time() - t0

    # motion proof: mean abs pixel delta between consecutive saved frames
    deltas = [float(np.abs(saved_imgs[i] - saved_imgs[i - 1]).mean()) for i in range(1, len(saved_imgs))]

    np.savez_compressed(
        os.path.join(OUT, "buffer.npz"),
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        next_states=np.array(next_states, dtype=np.float32),
        images=np.stack(saved_imgs).astype(np.uint8),
        image_steps=np.array(saved_idx),
    )

    summary = {
        "env": "aloha / AlohaInsertion-v0",
        "gl_backend": os.environ["MUJOCO_GL"],
        "n_steps": n_steps,
        "steps_per_sec": round(n_steps / dt, 1),
        "wall_sec": round(dt, 1),
        "n_episode_resets": n_resets,
        "action_shape": list(a.shape),
        "image_key": img_key,
        "image_shape": list(leaves[img_key][0].shape),
        "state_key": state_key,
        "state_shape": list(leaves[state_key][0].shape) if state_key else None,
        "n_transitions_logged": len(states),
        "mean_frame_delta": [round(d, 3) for d in deltas],
        "frames_saved": len(saved_imgs),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[done] artifacts in {OUT}")


if __name__ == "__main__":
    main()
