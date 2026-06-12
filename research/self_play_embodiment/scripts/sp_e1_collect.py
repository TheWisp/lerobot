# ruff: noqa
"""E1: wide self-play at scale, VIDEO-ONLY use (WM corpus diversity; actions stored but unused by design).
1000 episodes x 150 steps. EVERY episode: peg AND socket spawned wide (layout diversity in every frame).
Behaviors alternate: m0 right-arm pushes peg | m1 left-arm pushes socket (mirror) | m2 both arms reach
their objects (bimanual co-motion). Proven contact recipe (approach-above -> onto -> push, PUSHZ=0.022).
Images stored at 224 via the canonical resize224 path. -> cache_sp/{imgs.u8, meta.npz}"""

import os, time, numpy as np, mujoco, torch

os.environ.setdefault("MUJOCO_GL", "egl")
import sys

sys.path.insert(0, "/tmp/selfplay_probe")  # nosec B108
from sp_lib import Vec
from sp_vj_act import resize224

OUT = "/tmp/selfplay_probe/cache_sp"  # nosec B108
os.makedirs(OUT, exist_ok=True)  # nosec B108
NE, BATCHES, H = 25, 40, 150  # 1000 eps
PUSHZ, A, B, CLIP = 0.022, 40, 70, 0.08
PXL, PXH, SXLO, SXHI, YL, YH = 0.0, 0.30, -0.30, 0.0, 0.35, 0.65
vec = Vec(NE)
ph0 = vec.vec.envs[0].unwrapped._env._physics
jn = [ph0.model.joint(i).name for i in range(ph0.model.njnt)]
padr = int(ph0.named.model.jnt_qposadr["red_peg_joint"])
sadr = int(ph0.named.model.jnt_qposadr["blue_socket_joint"])
r_arm = [n for n in jn if "right" in n and "grip" not in n and "finger" not in n]
l_arm = [n for n in jn if "left" in n and "grip" not in n and "finger" not in n]
rdofs = [int(ph0.named.model.jnt_dofadr[n]) for n in r_arm]
ldofs = [int(ph0.named.model.jnt_dofadr[n]) for n in l_arm]
rlf = ph0.model.body("vx300s_right/left_finger_link").id
rrf = ph0.model.body("vx300s_right/right_finger_link").id
llf = ph0.model.body("vx300s_left/left_finger_link").id
lrf = ph0.model.body("vx300s_left/right_finger_link").id
rgl = ph0.model.body("vx300s_right/gripper_link").id
lgl = ph0.model.body("vx300s_left/gripper_link").id


def phys(i):
    return vec.vec.envs[i].unwrapped._env._physics


def gc(p, a, b):
    return (np.asarray(p.data.xpos[a]) + np.asarray(p.data.xpos[b])) / 2


def obj(p, name):
    return np.asarray(p.named.data.xpos[name])


NTOT = NE * BATCHES * H
imgs = np.memmap(OUT + "/imgs.u8", mode="w+", dtype=np.uint8, shape=(NTOT, 224, 224, 3))
states = np.zeros((NTOT, 14), np.float32)
actions = np.zeros((NTOT, 14), np.float32)
epid = np.zeros(NTOT, np.int32)
framepos = np.zeros(NTOT, np.int32)
rng = np.random.RandomState(5)
t0 = time.time()
w = 0
for bt in range(BATCHES):
    img, prop = vec.reset(range(500000 + bt * NE, 500000 + bt * NE + NE))
    prop_reset = prop.copy()
    for i in range(NE):  # wide-spawn BOTH objects
        p = phys(i)
        p.data.qpos[padr : padr + 2] = [rng.uniform(PXL, PXH), rng.uniform(YL, YH)]
        p.data.qpos[padr + 2] = 0.05
        p.data.qpos[padr + 3 : padr + 7] = [1, 0, 0, 0]
        p.data.qpos[sadr : sadr + 2] = [rng.uniform(SXLO, SXHI), rng.uniform(YL, YH)]
        p.data.qpos[sadr + 2] = 0.05
        p.data.qpos[sadr + 3 : sadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(p.model.ptr, p.data.ptr)
    peg0 = np.array([obj(phys(i), "peg") for i in range(NE)])
    soc0 = np.array([obj(phys(i), "socket") for i in range(NE)])
    mode = np.array([(bt * NE + i) % 3 for i in range(NE)])
    ang = rng.uniform(0, 2 * np.pi, NE)
    dist = rng.uniform(0.10, 0.16, NE)
    pv = np.c_[np.cos(ang), np.sin(ang)] * dist[:, None]
    pend = np.c_[np.clip(peg0[:, 0] + pv[:, 0], PXL, PXH), np.clip(peg0[:, 1] + pv[:, 1], 0.38, 0.62)]
    send = np.c_[np.clip(soc0[:, 0] + pv[:, 0], SXLO, SXHI), np.clip(soc0[:, 1] + pv[:, 1], 0.38, 0.62)]
    for t in range(H):
        x224 = (resize224(img) * 255).byte().permute(0, 2, 3, 1).numpy()
        for i in range(NE):
            gi = (bt * NE + i) * H + t
            imgs[gi] = x224[i]
            states[gi] = prop[i]
            epid[gi] = bt * NE + i
            framepos[gi] = t
        cmd = prop_reset.copy()
        for i in range(NE):
            p = phys(i)

            def drive(dofs, glid, fa, fb, tgt, slot):
                jp = np.zeros((3, p.model.nv))
                mujoco.mj_jac(p.model.ptr, p.data.ptr, jp, None, gc(p, fa, fb), glid)
                dq = np.clip(np.linalg.pinv(jp[:, dofs]) @ (tgt - gc(p, fa, fb)), -CLIP, CLIP)
                cmd[i, slot : slot + 6] = np.clip(prop[i, slot : slot + 6] + dq, -1, 1)

            if mode[i] in (0, 2):  # right arm -> peg
                pz = peg0[i, :2]
                if t < A:
                    tgt = np.r_[pz, 0.16]
                elif t < B:
                    tgt = np.r_[pz, PUSHZ]
                elif mode[i] == 0:
                    tgt = np.r_[pz + (pend[i] - pz) * ((t - B) / (H - B)), PUSHZ]
                else:
                    tgt = np.r_[pz, 0.16] if t > 100 else np.r_[pz, PUSHZ]
                drive(rdofs, rgl, rlf, rrf, tgt, 7)
            if mode[i] in (1, 2):  # left arm -> socket
                sz = soc0[i, :2]
                if t < A:
                    tgt = np.r_[sz, 0.16]
                elif t < B:
                    tgt = np.r_[sz, PUSHZ]
                elif mode[i] == 1:
                    tgt = np.r_[sz + (send[i] - sz) * ((t - B) / (H - B)), PUSHZ]
                else:
                    tgt = np.r_[sz, 0.16] if t > 100 else np.r_[sz, PUSHZ]
                drive(ldofs, lgl, llf, lrf, tgt, 0)
        for i in range(NE):
            gi = (bt * NE + i) * H + t
            actions[gi] = cmd[i]
        (img, prop), _, _ = vec.step(cmd.astype(np.float32))
    if bt % 5 == 0:
        print(f"batch {bt}/{BATCHES} ({(bt + 1) * NE * H / (time.time() - t0):.0f} fr/s)", flush=True)
imgs.flush()
np.savez(OUT + "/meta.npz", states=states, actions=actions, epid=epid, framepos=framepos, N=NTOT)
disp_p = np.linalg.norm(np.array([obj(phys(i), "peg")[:2] for i in range(NE)]) - peg0[:, :2], axis=1)
print(
    f"[ok] cache_sp: {NTOT} frames, {NE * BATCHES} eps | last-batch peg displacement mean {disp_p.mean() * 100:.1f}cm",
    flush=True,
)
