# ruff: noqa
"""Soundness gate for in-sim reach-to-peg: fit a global Jacobian J (Δgripper≈J·Δjoints) from
random play, then run a GT closed-loop controller Δjoints=clip(J⁺·(peg−gripper)) in our sim.
If it reaches the peg, the expert is good -> BC injection experiment is viable (matched domain,
peg GT, vision-necessary). Compares to no-op floor."""

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
from sp_lib import Vec, delta_command

OUT = "/tmp/selfplay_probe"  # nosec B108
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
states = wb["states"].astype(np.float64)
W = wb["world"].astype(np.float64)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
peg = W[:, 0:3]
Lg = W[:, 6:9]
Rg = W[:, 9:12]
# which gripper is the peg-reacher (closer at episode starts)?
home = [s for s, e in ep]
dL = np.linalg.norm(Lg[home] - peg[home], axis=1).mean()
dR = np.linalg.norm(Rg[home] - peg[home], axis=1).mean()
grip_all = Lg if dL < dR else Rg
which = "L" if dL < dR else "R"
print(f"peg-reacher = {which} gripper (home dist L={dL:.2f} R={dR:.2f})", flush=True)
# fit J: Δgrip(3) = J @ Δjoint(14), within-episode steps
dj, dg = [], []
for s, e in ep:
    dj.append(states[s + 1 : e] - states[s : e - 1])
    dg.append(grip_all[s + 1 : e] - grip_all[s : e - 1])
dj = np.concatenate(dj)
dg = np.concatenate(dg)
B, *_ = np.linalg.lstsq(dj, dg, rcond=None)  # (14,3): dg = dj @ B
J = B.T  # (3,14)
Jpinv = np.linalg.pinv(J)  # (14,3)
pred = dj @ B
print(
    f"J fit R2 (Δgrip from Δjoint): {1 - ((dg - pred) ** 2).sum() / ((dg - dg.mean(0)) ** 2).sum():.3f}",
    flush=True,
)
# closed-loop GT controller in sim
NG, H, DMAX = 16, 200, 0.10
vec = Vec(NG)


def reacher_idx():  # index of the chosen gripper in obj/gripper readout
    return 0 if which == "L" else 1


def run(use_ctrl):
    img, prop = vec.reset(range(40000, 40000 + NG))
    best = np.full(NG, 1e9)
    for t in range(H):
        g6 = vec.gripper_xyz()
        pegxyz = vec.obj_xyz()[:, :3]
        grip = g6[:, :3] if which == "L" else g6[:, 3:]
        if use_ctrl:
            dgoal = pegxyz - grip  # (NG,3) Cartesian error
            dj = np.clip(dgoal @ Jpinv.T, -DMAX, DMAX).astype(np.float32)  # (NG,14)
        else:
            dj = np.zeros((NG, 14), np.float32)
        (img, prop), _, _ = vec.step(delta_command(prop, dj, dmax=DMAX))
        best = np.minimum(
            best,
            np.linalg.norm(
                (vec.gripper_xyz()[:, :3] if which == "L" else vec.gripper_xyz()[:, 3:])
                - vec.obj_xyz()[:, :3],
                axis=1,
            ),
        )
    return best


b_ctrl = run(True)
b_noop = run(False)
print(
    f"\nGT Jacobian controller: minDist {b_ctrl.mean():.3f}m | SR@0.05={float((b_ctrl < 0.05).mean()):.2f} @0.1={float((b_ctrl < 0.1).mean()):.2f} @0.15={float((b_ctrl < 0.15).mean()):.2f}",
    flush=True,
)
print(
    f"no-op (floor):          minDist {b_noop.mean():.3f}m | SR@0.1={float((b_noop < 0.1).mean()):.2f}",
    flush=True,
)
