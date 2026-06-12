# ruff: noqa
"""Precise reach-to-peg labels on cached frames (no re-render), GRASP-CENTER reference
(midpoint of the two right fingers, via mj_jac at a point) to remove the link-origin offset.
For each world_buffer frame: set R-arm qpos, mj_forward, analytic Jacobian of grasp-center ->
label = clip(J⁺·(peg − graspcenter), DMAX). Saves reach_labels.npz. GT sanity: TRUE vs MEAN peg."""

import os

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
from sp_lib import Vec

OUT = "/tmp/selfplay_probe"  # nosec B108
DMAX = 0.05
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
states = wb["states"].astype(np.float64)
W = wb["world"].astype(np.float64)
peg = W[:, 0:3]
vec = Vec(1)
ph = vec.vec.envs[0].unwrapped._env._physics
jnt = [ph.model.joint(i).name for i in range(ph.model.njnt)]
r_arm = [n for n in jnt if "right" in n and "grip" not in n and "finger" not in n]
rdofs = [int(ph.named.model.jnt_dofadr[n]) for n in r_arm]
rqpos = [int(ph.named.model.jnt_qposadr[n]) for n in r_arm]
glid = ph.model.body("vx300s_right/gripper_link").id
lf = ph.model.body("vx300s_right/left_finger_link").id
rf = ph.model.body("vx300s_right/right_finger_link").id
print(f"R dofs {rdofs} qpos {rqpos}; grasp-center = midpoint of finger bodies {lf},{rf}", flush=True)


def grasp_jac(p):  # Jacobian + position of grasp-center at current physics state p
    gc = (np.asarray(p.data.xpos[lf]) + np.asarray(p.data.xpos[rf])) / 2
    jacp = np.zeros((3, p.model.nv))
    mujoco.mj_jac(p.model.ptr, p.data.ptr, jacp, None, gc, glid)
    return jacp[:, rdofs], gc


LAB = np.zeros((len(states), 14), np.float32)
for t in range(len(states)):
    ph.data.qpos[rqpos] = states[t][7:13]
    mujoco.mj_forward(ph.model.ptr, ph.data.ptr)
    Jr, gc = grasp_jac(ph)
    LAB[t, 7:13] = np.clip(np.linalg.pinv(Jr) @ (peg[t] - gc), -DMAX, DMAX)
np.savez_compressed(OUT + "/reach_labels.npz", LAB=LAB)
print("[ok] reach_labels.npz (grasp-center analytic-J labels)", flush=True)
# GT sanity
NG, H = 16, 220
vec2 = Vec(NG)


def gc_all():
    out = np.zeros((NG, 3))
    for i in range(NG):
        p = vec2.vec.envs[i].unwrapped._env._physics
        out[i] = (np.asarray(p.data.xpos[lf]) + np.asarray(p.data.xpos[rf])) / 2
    return out


def run(mode, mp=None):
    img, prop = vec2.reset(range(40000, 40000 + NG))
    best = np.full(NG, 1e9)
    for t in range(H):
        cmd = prop.copy()
        for i in range(NG):
            p = vec2.vec.envs[i].unwrapped._env._physics
            Jr, gc = grasp_jac(p)
            tgt = mp if mode == "mean" else np.asarray(p.named.data.xpos["peg"])
            cmd[i, 7:13] = np.clip(
                prop[i, 7:13] + np.clip(np.linalg.pinv(Jr) @ (tgt - gc), -DMAX, DMAX), -1, 1
            )
        (img, prop), _, _ = vec2.step(cmd.astype(np.float32))
        best = np.minimum(best, np.linalg.norm(gc_all() - vec2.obj_xyz()[:, :3], axis=1))
    return best


vec2.reset(range(40000, 40000 + NG))
mp = vec2.obj_xyz()[:, :3].mean(0)
EPS = [0.02, 0.03, 0.05]
bt, bm = run("true"), run("mean", mp)


def line(t, b):
    return f"  {t:14s} minDist {b.mean():.3f}m | " + " ".join(
        f"SR@{e}={float((b < e).mean()):.2f}" for e in EPS
    )


print(line("graspc+TRUE", bt), flush=True)
print(line("graspc+MEAN", bm), flush=True)
