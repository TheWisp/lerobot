# ruff: noqa
"""Test the user's idea: tighten threshold + PRECISE expert (mujoco analytic Jacobian) instead of
re-randomizing. (1) analytic-J GT controller servoing R-gripper to the TRUE peg -> how precise can a
sighted controller get? (2) a MEAN-peg controller (same precise servo, but to the fixed mean peg =
what proprio-memorization can do) -> does it fail at tight thresholds? If precise+true reaches ~3cm
but mean-peg fails at ~3-5cm, the task is vision-necessary at a tight threshold on existing data."""

import os

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
from sp_lib import Vec

NG, H = 16, 220
vec = Vec(NG)
phys0 = vec.vec.envs[0].unwrapped._env._physics
# map right-arm 6 joints -> qvel dof indices + action indices (aloha: [L arm6,Lgrip,R arm6,Rgrip])
jnt_names = [phys0.model.joint(i).name for i in range(phys0.model.njnt)]
r_arm = [n for n in jnt_names if "right" in n and "grip" not in n and "finger" not in n]
print("right-arm joints:", r_arm, flush=True)
rdofs = [int(phys0.named.model.jnt_dofadr[n]) for n in r_arm]  # qvel indices
act_r = list(range(7, 13))  # action dims for R arm (6)
bid = phys0.model.body("vx300s_right/gripper_link").id
print(f"R-arm dofs={rdofs} action_idx={act_r} gripper_body={bid}", flush=True)


def analytic_dq(phys, target):  # joint delta (action units≈rad) to move R-gripper toward target
    jacp = np.zeros((3, phys.model.nv))
    mujoco.mj_jacBody(phys.model.ptr, phys.data.ptr, jacp, None, bid)
    Jr = jacp[:, rdofs]  # (3,6)
    grip = np.asarray(phys.data.xpos[bid])
    dqarm = np.linalg.pinv(Jr) @ (target - grip)  # (6,)
    return dqarm


EPS = [0.02, 0.03, 0.05, 0.1]


def run(mode, mean_peg=None):
    img, prop = vec.reset(range(40000, 40000 + NG))
    best = np.full(NG, 1e9)
    for t in range(H):
        cmd = prop.copy()
        for i in range(NG):
            ph = vec.vec.envs[i].unwrapped._env._physics
            tgt = mean_peg if mode == "mean" else np.asarray(ph.named.data.xpos["peg"])
            dq = np.clip(analytic_dq(ph, tgt), -0.05, 0.05)
            cmd[i, act_r] = np.clip(prop[i, act_r] + dq, -1, 1)
        (img, prop), _, _ = vec.step(cmd.astype(np.float32))
        g = vec.gripper_xyz()[:, 3:]
        peg = vec.obj_xyz()[:, :3]
        best = np.minimum(best, np.linalg.norm(g - peg, axis=1))
    return best


# mean peg over the eval seeds (what memorization would aim at)
vec.reset(range(40000, 40000 + NG))
mean_peg = vec.obj_xyz()[:, :3].mean(0)
print(f"\npeg spread across eval seeds: std {vec.obj_xyz()[:, :3].std(0).round(3)} m", flush=True)
b_true = run("true")
b_mean = run("mean", mean_peg)


def line(tag, b):
    return f"  {tag:18s} minDist {b.mean():.3f}m | " + " ".join(
        f"SR@{e}={float((b < e).mean()):.2f}" for e in EPS
    )


print(line("analytic+TRUE peg", b_true), flush=True)
print(line("analytic+MEAN peg", b_mean), flush=True)
print(
    "\n(if TRUE reaches tight & MEAN fails tight -> vision-necessary at a tight threshold, no re-randomization)",
    flush=True,
)
