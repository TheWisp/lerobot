# ruff: noqa
"""Meaningful scripted skill: PUSH across wide scenes. Per episode: wide-random peg. Right arm (only)
scripted via Jacobian IK: reach above the peg -> descend ONTO it (peg reachable per Stage-1) -> push it
~12cm in a random direction (clamped to the reachable region). Left arm + grippers held at reset.
~100% contact. Logs image(256)/proprio/action + per-frame peg + push-target. -> sp_skill_data.npz"""

import os, time, numpy as np, mujoco

os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image
from sp_lib import Vec

XLO, XHI, YLO, YHI = 0.0, 0.30, 0.35, 0.65
N, EP_PER, H = 25, 8, 150
PUSHZ = 0.022
A, B, CLIP = 40, 70, 0.08
vec = Vec(N)
ph0 = vec.vec.envs[0].unwrapped._env._physics
jnts = [ph0.model.joint(i).name for i in range(ph0.model.njnt)]
padr = int(ph0.named.model.jnt_qposadr["red_peg_joint"])
z0 = 0.05
vec.reset([0] * N)
quat0 = np.array(ph0.data.qpos[padr + 3 : padr + 7])
r_arm = [n for n in jnts if "right" in n and "grip" not in n and "finger" not in n]
rdofs = [int(ph0.named.model.jnt_dofadr[n]) for n in r_arm]
lf = ph0.model.body("vx300s_right/left_finger_link").id
rf = ph0.model.body("vx300s_right/right_finger_link").id
glid = ph0.model.body("vx300s_right/gripper_link").id


def phys(i):
    return vec.vec.envs[i].unwrapped._env._physics


def gc(p):
    return (np.asarray(p.data.xpos[lf]) + np.asarray(p.data.xpos[rf])) / 2


def pegxyz(p):
    return np.asarray(p.named.data.xpos["peg"])


def set_pegs(xy):
    for i in range(N):
        p = phys(i)
        p.data.qpos[padr : padr + 2] = xy[i]
        p.data.qpos[padr + 2] = z0
        p.data.qpos[padr + 3 : padr + 7] = quat0
        mujoco.mj_forward(p.model.ptr, p.data.ptr)


rng = np.random.RandomState(2)
imgs, states, actions, pegf, goals, finals = [], [], [], [], [], []
t0 = time.time()
for batch in range(EP_PER):
    img, prop = vec.reset(range(90000 + batch * N, 90000 + batch * N + N))
    prop_reset = prop.copy()
    xy = np.c_[rng.uniform(XLO, XHI, N), rng.uniform(YLO, YHI, N)]
    set_pegs(xy)
    peg0 = np.array([pegxyz(phys(i)) for i in range(N)])
    ang = rng.uniform(0, 2 * np.pi, N)
    dist = rng.uniform(0.10, 0.16, N)
    pv = np.c_[np.cos(ang), np.sin(ang)] * dist[:, None]
    endxy = np.clip(peg0[:, :2] + pv, [XLO, 0.38], [XHI, 0.62])  # push target, clamped reachable
    above = np.c_[peg0[:, :2], np.full(N, 0.16)]
    onpeg = np.c_[peg0[:, :2], np.full(N, PUSHZ)]
    endpt = np.c_[endxy, np.full(N, PUSHZ)]
    goals.append(endxy.copy())
    for t in range(H):
        for i in range(N):
            imgs.append(img[i])
            states.append(prop[i].copy())
            pegf.append(pegxyz(phys(i))[:2].copy())
        if t < A:
            tgt = above
        elif t < B:
            tgt = onpeg
        else:
            tgt = onpeg + (endpt - onpeg) * ((t - B) / (H - B))
        cmd = prop_reset.copy()  # hold left arm + grippers at reset
        for i in range(N):
            p = phys(i)
            jp = np.zeros((3, p.model.nv))
            mujoco.mj_jac(p.model.ptr, p.data.ptr, jp, None, gc(p), glid)
            dq = np.clip(np.linalg.pinv(jp[:, rdofs]) @ (tgt[i] - gc(p)), -CLIP, CLIP)
            cmd[i, 7:13] = np.clip(prop[i, 7:13] + dq, -1, 1)
        for i in range(N):
            actions.append(cmd[i].copy())
        (img, prop), _, _ = vec.step(cmd.astype(np.float32))
    finals.append(np.array([pegxyz(phys(i))[:2] for i in range(N)]))
print(f"collected {len(imgs)} frames ({len(imgs) / (time.time() - t0):.0f}/s)", flush=True)
imgs = np.stack([x if x.shape[0] == 256 else np.array(Image.fromarray(x).resize((256, 256))) for x in imgs])
states = np.array(states, np.float32)
actions = np.array(actions, np.float32)
pegf = np.array(pegf, np.float32)
epid = np.array([b * N + i for b in range(EP_PER) for t in range(H) for i in range(N)])
G_by_ep = np.concatenate(goals, 0)
goalf = np.array([G_by_ep[b * N + i] for b in range(EP_PER) for t in range(H) for i in range(N)])
np.savez_compressed(
    "/tmp/selfplay_probe/sp_skill_data.npz",  # nosec B108
    images=imgs,
    states=states,
    actions=actions,
    epid=epid,
    peg0=pegf,
    goal=goalf,
)  # nosec B108
fin = np.concatenate(finals, 0)
p0 = np.array([pegf[np.where(epid == e)[0][0]] for e in np.unique(epid)])
disp = np.linalg.norm(fin - p0, axis=1)
print(f"[ok] sp_skill_data.npz {imgs.shape} | {len(np.unique(epid))} scenes", flush=True)
print(
    f"PUSH: contact rate (peg moved >2cm) {(disp > 0.02).mean() * 100:.0f}% | mean peg displacement {disp.mean() * 100:.1f}cm",
    flush=True,
)
