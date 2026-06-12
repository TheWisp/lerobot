# ruff: noqa
"""Replay eval episodes from a saved peak checkpoint WITH rendering: failure inspection.
Same deterministic seeds/spawns as sp_ood_eval (seeds 400000+, spawn RNG 12345+10f) -> identical episodes.
Outputs: (a) 2x3 grid mp4 per f (worst episodes first), (b) tiny LeRobotDataset for the GUI
(throwaway repo, ~5 eps per f: up to 4 failures + 1 best), labeled with f and final reward."""

import os, sys, numpy as np, torch, mujoco, imageio

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/tmp/selfplay_probe")  # nosec B108
from sp_vj_act import ACTJepa, resize224, CHUNK
from sp_lib import Vec, _walk
from PIL import Image

dev = "cuda"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/selfplay_probe/s2_models/e4_wmDS@12000.pt"  # nosec B108
REPO = sys.argv[2] if len(sys.argv) > 2 else "thewisp/e4_failures_jun12"
N = 12
T_EVAL = 450
ck = torch.load(MODEL, map_location=dev, weights_only=False)
model = ACTJepa().to(dev)
model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
model.eval()
smu, ssd, amu, asd = ck["smu"], ck["ssd"], ck["amu"], ck["asd"]
imn = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
isd_ = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


def prep(x):
    return (resize224(x).to(dev) - imn) / isd_


from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

root = Path(os.path.expanduser("~/.cache/huggingface/lerobot")) / REPO
assert not root.exists(), f"{root} exists — refusing to clobber"
feats = {
    "observation.images.top": {
        "dtype": "video",
        "shape": (240, 320, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {"dtype": "float32", "shape": (14,), "names": [f"j{i}" for i in range(14)]},
    "action": {"dtype": "float32", "shape": (14,), "names": [f"j{i}" for i in range(14)]},
}
ds = LeRobotDataset.create(REPO, fps=50, features=feats, use_videos=True)


def run_f(F):
    vec = Vec(N)
    ph0 = vec.vec.envs[0].unwrapped._env._physics
    padr = int(ph0.named.model.jnt_qposadr["red_peg_joint"])
    sadr = int(ph0.named.model.jnt_qposadr["blue_socket_joint"])
    rng = np.random.RandomState(12345 + int(F * 10))
    obs, _ = vec.vec.reset(seed=list(range(400000, 400000 + N)))
    ylo, yhi = 0.5 - min(0.10 * F, 0.15), 0.5 + min(0.10 * F, 0.15)
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

    img, prop = get(obs)
    frames = [[] for _ in range(N)]
    states_l = [[] for _ in range(N)]
    acts_l = [[] for _ in range(N)]
    preds = np.full((T_EVAL, N, CHUNK, 14), np.nan, np.float32)
    maxr = np.zeros(N)
    m_te = 0.01
    for t in range(T_EVAL):
        if t == 0:
            obs, r, te_, tr_, _ = vec.vec.step(prop.astype(np.float32))
            img, prop = get(obs)
        st = torch.tensor((prop - smu) / ssd, dtype=torch.float32, device=dev)
        with torch.no_grad():
            mem = model.encode(prep(img), st)
            out, _, _ = model.act(mem, None)
            preds[t] = out.cpu().numpy() * asd + amu
        a = np.zeros((N, 14))
        w = 0.0
        for rt in range(max(0, t - CHUNK + 1), t + 1):
            off = t - rt
            ww = np.exp(-m_te * off)
            a += ww * preds[rt, :, off, :]
            w += ww
        act = np.clip((a / w).astype(np.float32), -1, 1)
        for i in range(N):
            if t % 2 == 0:
                frames[i].append(np.array(Image.fromarray(img[i]).resize((320, 240))))
            states_l[i].append(prop[i].copy())
            acts_l[i].append(act[i].copy())
        obs, r, term, trunc, info = vec.vec.step(act)
        maxr = np.maximum(maxr, np.asarray(r, float))
        img, prop = get(obs)
    order = np.argsort(maxr)  # worst first
    grid_eps = order[:6]
    gw = imageio.get_writer(f"/tmp/selfplay_probe/e4_fail_f{F}.mp4", fps=25, macro_block_size=None)  # nosec B108
    L = min(len(frames[i]) for i in grid_eps)
    for t in range(L):
        rows = []
        for r_ in range(2):
            rows.append(np.concatenate([frames[grid_eps[r_ * 3 + c]][t] for c in range(3)], axis=1))
        gw.append_data(np.concatenate(rows, axis=0))
    gw.close()
    keep = list(order[:4]) + [order[-1]]  # 4 worst + 1 best
    for i in keep:
        for t in range(0, len(states_l[i]), 2):
            ds.add_frame(
                {
                    "observation.images.top": frames[i][t // 2],
                    "observation.state": states_l[i][t],
                    "action": acts_l[i][t],
                    "task": f"f={F} finalR={int(maxr[i])} env{i}",
                }
            )
        ds.save_episode()
    print(
        f"[f={F}] maxr per env: {maxr.astype(int).tolist()} | grid e4_fail_f{F}.mp4 | dataset +{len(keep)} eps",
        flush=True,
    )
    del vec


run_f(1.0)
run_f(2.0)
ds.finalize()
print(f"[ok] dataset {REPO}", flush=True)
