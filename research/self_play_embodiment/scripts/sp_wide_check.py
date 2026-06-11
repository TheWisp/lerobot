# ruff: noqa
"""Stage-1 gate for the cross-scene embodiment test: spawn the peg over a WIDE range and check it's
(a) REACHABLE (analytic-Jacobian GT controller reaches grasp-center -> peg) and (b) LOCALIZABLE from
vision (V-JEPA2.1 8x8 spatial decodes peg-xy on held-out wide scenes). Wide range set by env
XLO/XHI/YLO/YHI. Peg placed by setting its freejoint qpos after reset (no gym monkeypatch)."""

import os, numpy as np, torch, mujoco

os.environ.setdefault("MUJOCO_GL", "egl")
from sp_lib import Vec, delta_command, VJepa21Encoder

XLO, XHI = float(os.environ.get("XLO", "0.0")), float(os.environ.get("XHI", "0.30"))
YLO, YHI = float(os.environ.get("YLO", "0.35")), float(os.environ.get("YHI", "0.65"))
print(f"wide peg range x[{XLO},{XHI}] y[{YLO},{YHI}]", flush=True)
vec = Vec(1)
ph0 = vec.vec.envs[0].unwrapped._env._physics
jnts = [ph0.model.joint(i).name for i in range(ph0.model.njnt)]
pegj = [n for n in jnts if "peg" in n.lower()]
print("peg joints:", pegj, flush=True)
PJ = pegj[0]
padr = int(ph0.named.model.jnt_qposadr[PJ])
# defaults (z + quat) from a reset
vec.reset([0])
z0 = float(ph0.data.qpos[padr + 2])
quat0 = np.array(ph0.data.qpos[padr + 3 : padr + 7])
print(f"peg joint={PJ} qposadr={padr} z0={z0:.3f}", flush=True)
# R-arm dofs + grasp-center for the GT reach controller
r_arm = [n for n in jnts if "right" in n and "grip" not in n and "finger" not in n]
rdofs = [int(ph0.named.model.jnt_dofadr[n]) for n in r_arm]
lf = ph0.model.body("vx300s_right/left_finger_link").id
rf = ph0.model.body("vx300s_right/right_finger_link").id
glid = ph0.model.body("vx300s_right/gripper_link").id


def set_pegs(vec, xy):  # xy: (n,2)
    for i in range(vec.n):
        p = vec.vec.envs[i].unwrapped._env._physics
        p.data.qpos[padr : padr + 2] = xy[i]
        p.data.qpos[padr + 2] = z0
        p.data.qpos[padr + 3 : padr + 7] = quat0
        mujoco.mj_forward(p.model.ptr, p.data.ptr)


def gc(p):
    return (np.asarray(p.data.xpos[lf]) + np.asarray(p.data.xpos[rf])) / 2


rng = np.random.RandomState(0)
# ---- (a) reachability across wide scenes ----
NG = vec.n if vec.n > 1 else None
vecR = Vec(20)


def set_pegsR(xy):
    for i in range(20):
        p = vecR.vec.envs[i].unwrapped._env._physics
        p.data.qpos[padr : padr + 2] = xy[i]
        p.data.qpos[padr + 2] = z0
        p.data.qpos[padr + 3 : padr + 7] = quat0
        mujoco.mj_forward(p.model.ptr, p.data.ptr)


img, prop = vecR.reset(range(50000, 50020))
xyR = np.c_[rng.uniform(XLO, XHI, 20), rng.uniform(YLO, YHI, 20)]
set_pegsR(xyR)
best = np.full(20, 1e9)
for t in range(220):
    cmd = prop.copy()
    for i in range(20):
        p = vecR.vec.envs[i].unwrapped._env._physics
        jp = np.zeros((3, p.model.nv))
        mujoco.mj_jac(p.model.ptr, p.data.ptr, jp, None, gc(p), glid)
        dq = np.clip(
            np.linalg.pinv(jp[:, rdofs]) @ (np.asarray(p.named.data.xpos["peg"]) - gc(p)), -0.05, 0.05
        )
        cmd[i, 7:13] = np.clip(prop[i, 7:13] + dq, -1, 1)
    (img, prop), _, _ = vecR.step(cmd.astype(np.float32))
    # re-pin peg (controller may nudge it pre-contact); keep static target
    set_pegsR(xyR)
    best = np.minimum(
        best,
        np.array(
            [
                np.linalg.norm(
                    gc(vecR.vec.envs[i].unwrapped._env._physics)
                    - vecR.vec.envs[i].unwrapped._env._physics.named.data.xpos["peg"]
                )
                for i in range(20)
            ]
        ),
    )
print(
    f"\n(a) REACH across wide scenes: minDist {best.mean():.3f}m | SR@0.05={float((best < 0.05).mean()):.2f} @0.1={float((best < 0.1).mean()):.2f}",
    flush=True,
)
# ---- (b) localizability: collect M wide scenes (1 frame each), decode peg-xy from V-JEPA2.1 8x8 ----
M = 400
imgs = []
pegs = []
for k in range(0, M, 20):
    im, _ = vecR.reset(range(60000 + k, 60000 + k + 20))
    xy = np.c_[rng.uniform(XLO, XHI, 20), rng.uniform(YLO, YHI, 20)]
    set_pegsR(xy)
    im = vecR._unpack_imgs() if hasattr(vecR, "_unpack_imgs") else None
    # re-grab images after setting peg
    from PIL import Image as _I

    for i in range(20):
        p = vecR.vec.envs[i].unwrapped._env._physics
        rgb = np.asarray(p.render(height=480, width=640, camera_id="top"))
        imgs.append(np.array(_I.fromarray(rgb).resize((256, 256))))
        pegs.append(p.named.data.xpos["peg"][:2].copy())
imgs = np.stack(imgs)
pegs = np.array(pegs)
enc = VJepa21Encoder()
_, S = enc.encode_both(imgs, G=8)
S = S.astype(np.float64)


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xt, Yt, Xe, a=10.0):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt, Xe = (Xt - mu) / sd, (Xe - mu) / sd
    return Xe @ np.linalg.solve(Xt.T @ Xt + a * np.eye(Xt.shape[1]), Xt.T @ (Yt - Yt.mean(0))) + Yt.mean(0)


def pca(Xt, Xe, k=200):
    mm = Xt.mean(0)
    _, _, Vt = np.linalg.svd(Xt - mm, full_matrices=False)
    c = Vt[:k]
    return (Xt - mm) @ c.T, (Xe - mm) @ c.T


n = len(imgs)
tr = np.arange(n) % 5 != 0
te = ~tr
P, Pt = pca(S[tr], S[te])
print(
    f"(b) peg-xy spread std {pegs.std(0).round(3)} | V-JEPA2.1 8x8 decode (held-out scenes) R2 = {r2(pegs[te], ridge(P, pegs[tr], Pt)):.3f}",
    flush=True,
)
