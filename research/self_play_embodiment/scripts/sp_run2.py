# ruff: noqa
"""Cheap visible-reach gate v2. FIX for the joint-from-vision floor:
  - proprio is a POLICY INPUT (the self; as in real VLAs)
  - goal stays an IMAGE, but success = GRIPPER CARTESIAN L2 (the decodable r2~0.78 quantity)
  - policy = small MLP (the seen-target -> joint-move map is the Jacobian, bilinear in
    proprio -> linear ridge can't; MLP can)
Goal A: get NON-ZERO SR (oracle/base off the floor). Goal B: does e_ac help (current or goal)?

Conditions (current rep always has proprio + zc; goal = base zg image):
  oracle_xyz [proprio, target_gxyz]              pure IK ceiling (Cartesian target given)
  bothprop   [proprio, goal_proprio]             joint-IK ceiling (sanity)
  a          [proprio, zc, zg]
  noise      [proprio, zc, rand, zg]
  b_cur/c_cur   + e_free/e_ac on CURRENT zc
  b_goal/c_goal + e_free/e_ac on GOAL zg
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sp_lib import Encoder, Vec, delta_command, load_emb

OUT = "/tmp/selfplay_probe"  # nosec B108
KZ, DMAX, K_MAX, H = 200, 0.5, 30, 45
MODE = os.environ.get("MODE", "probe")
DEMOS = [int(x) for x in os.environ.get("DEMOS", "500,5000").split(",")]
EPS = [0.05, 0.1, 0.2]  # gripper Cartesian L2 (meters), 6-dim (both grippers)
SEED = int(os.environ.get("SEED", "0"))
dev = "cuda"
torch.manual_seed(SEED)
np.random.seed(SEED)


def log(m):
    print(m, flush=True)


def gerr(g, gg):
    return np.linalg.norm(g - gg, axis=-1)


# ---------- data ----------
M = np.load(OUT + "/feat_cache.npz")["M"].astype(np.float32)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
proprio = wb["states"].astype(np.float32)
GXYZ = wb["world"][:, 6:12].astype(np.float32)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
ec = np.load(OUT + "/e_cache.npz")
E_FREE, E_AC = ec["e_free"].astype(np.float32), ec["e_ac"].astype(np.float32)
# test goals = held-out PLAY frames (distribution-matched to HER goals), 32 per seed, disjoint from training
_grng = np.random.RandomState(SEED + 100)
_cand = np.concatenate([np.arange(s + 40, e - 1) for eid, (s, e) in enumerate(ep) if eid >= ne - 4])
_gi = _grng.choice(_cand, 32, replace=False)
G_Z, G_GXYZ, G_PROP = M[_gi].copy(), GXYZ[_gi].copy(), proprio[_gi].copy()
NG = len(G_GXYZ)
train_frames = np.concatenate([np.arange(s, e) for eid, (s, e) in enumerate(ep) if eid < ne - 4])
pmu = M[train_frames].mean(0)
_, _, Vt = np.linalg.svd(M[train_frames] - pmu, full_matrices=False)
PC = Vt[:KZ]


def pca(z):
    return (z - pmu) @ PC.T


ZG_PCA = pca(G_Z)

# ---------- HER demo pool ----------
rng = np.random.RandomState(SEED)
src = []
for eid, (s, e) in enumerate(ep):
    if eid >= ne - 4:
        continue
    for t in range(s, e - 1):
        k = rng.randint(1, min(K_MAX, e - 1 - t) + 1)
        src.append((t, t + k))
src = np.array(src)
rng.shuffle(src)
ti, gi = src[:, 0], src[:, 1]
NOISE = rng.randn(len(ti), 64).astype(np.float32)
LAB = np.clip(proprio[gi] - proprio[ti], -DMAX, DMAX).astype(np.float32)
ZC_d, ZG_d = pca(M[ti]), pca(M[gi])
log(f"demo pool {len(ti)} | MODE={MODE} DEMOS={DEMOS} | {NG} goals, H={H}, MLP policy")


# ---------- feature builder ----------
def feats(name, P):  # P = dict of arrays
    pr, zc, zg = P["pr"], P["zc"], P["zg"]
    if name == "oracle_xyz":
        return np.hstack([pr, P["gg"]])
    if name == "bothprop":
        return np.hstack([pr, P["gp"]])
    if name == "a":
        return np.hstack([pr, zc, zg])
    if name == "noise":
        return np.hstack([pr, zc, P["nz"], zg])
    if name == "b_cur":
        return np.hstack([pr, zc, P["ef_c"], zg])
    if name == "c_cur":
        return np.hstack([pr, zc, P["ea_c"], zg])
    if name == "b_goal":
        return np.hstack([pr, zc, zg, P["ef_g"]])
    if name == "c_goal":
        return np.hstack([pr, zc, zg, P["ea_g"]])
    raise ValueError(name)


# ---------- MLP policy (early-stop on held-out demos) ----------
class MLP:
    def fit(self, X, Y):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.xmu) / self.xsd
        n = len(X)
        cut = max(1, int(n * 0.85))
        idx = np.random.permutation(n)
        tri, vai = idx[:cut], idx[cut:]
        Xt = torch.tensor(Xn[tri], device=dev)
        Yt = torch.tensor(Y[tri], device=dev)
        Xv = torch.tensor(Xn[vai], device=dev)
        Yv = torch.tensor(Y[vai], device=dev)
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 14)
        ).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3, weight_decay=1e-4)
        best, bad, bestSD = 1e18, 0, None
        for epoch in range(300):
            self.net.train()
            perm = torch.randperm(len(Xt), device=dev)
            for k in range(0, len(perm), 256):
                bi = perm[k : k + 256]
                loss = ((self.net(Xt[bi]) - Yt[bi]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = ((self.net(Xv) - Yv) ** 2).mean().item() if len(vai) else loss.item()
            if v < best - 1e-5:
                best, bad, bestSD = v, 0, {k: x.clone() for k, x in self.net.state_dict().items()}
            else:
                bad += 1
            if bad >= 15:
                break
        self.net.load_state_dict(bestSD)
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return (
                self.net(torch.tensor((X - self.xmu) / self.xsd, device=dev, dtype=torch.float32))
                .cpu()
                .numpy()
            )


def demo_feats(name, nd):
    idx = np.arange(nd)
    P = dict(
        pr=proprio[ti[idx]],
        zc=ZC_d[idx],
        zg=ZG_d[idx],
        nz=NOISE[idx],
        gg=GXYZ[gi[idx]],
        gp=proprio[gi[idx]],
        ef_c=E_FREE[ti[idx]],
        ea_c=E_AC[ti[idx]],
        ef_g=E_FREE[gi[idx]],
        ea_g=E_AC[gi[idx]],
    )
    return feats(name, P)


def fit_policy(name, nd):
    return MLP().fit(demo_feats(name, nd), LAB[:nd])


# ---------- closed-loop eval (Cartesian gripper metric) ----------
enc = Encoder()
F_FREE, F_AC = load_emb(OUT + "/f_free.pt"), load_emb(OUT + "/f_ac.pt")


@torch.no_grad()
def emb(z, f):
    return f(torch.tensor(z, device=dev)).cpu().numpy()


VEC = Vec(NG)


def rollout(name, pol):
    img, prop = VEC.reset(range(20000, 20000 + NG))
    start_err = gerr(VEC.gripper_xyz(), G_GXYZ)
    errs = np.full((H, NG), np.nan)
    for t in range(H):
        z = enc.encode(img)
        P = dict(
            pr=prop,
            zc=pca(z),
            zg=ZG_PCA,
            nz=np.random.randn(NG, 64).astype(np.float32),
            gg=G_GXYZ,
            gp=G_PROP,
            ef_c=emb(z, F_FREE),
            ea_c=emb(z, F_AC),
            ef_g=emb(G_Z, F_FREE),
            ea_g=emb(G_Z, F_AC),
        )
        delta = pol.predict(feats(name, P))
        (img, prop), _, _ = VEC.step(delta_command(prop, delta, dmax=DMAX))
        errs[t] = gerr(VEC.gripper_xyz(), G_GXYZ)
    return errs[-5:].mean(0), start_err


def row(tag, fe, se=None):
    s = f"  {tag:10s} finalErr {fe.mean():.3f}m | " + " ".join(
        f"SR@{e}={float((fe < e).mean()):.2f}" for e in EPS
    )
    return s + (f"  (start {se.mean():.3f}m)" if se is not None else "")


# ---------- run ----------
if MODE == "probe":
    log("\n=== NON-ZERO CHECK (Cartesian gripper reach) ===")
    for nd in DEMOS:
        log(f"--- #demos={nd} ---")
        for name in ["oracle_xyz", "bothprop", "a"]:
            t0 = time.time()
            fe, se = rollout(name, fit_policy(name, nd))
            log(row(name, fe, se) + f"  [{time.time() - t0:.0f}s]")
else:
    conds = ["oracle_xyz", "bothprop", "a", "noise", "b_cur", "c_cur", "b_goal", "c_goal"]
    res = {}
    log("\n=== SR vs #demos (Cartesian gripper reach) ===")
    for nd in DEMOS:
        log(f"\n--- #demos={nd} ---")
        for name in conds:
            t0 = time.time()
            fe, _ = rollout(name, fit_policy(name, nd))
            res[(name, nd)] = fe
            log(row(name, fe) + f"  [{time.time() - t0:.0f}s]")
    np.savez(
        OUT + f"/sp_results2_s{SEED}.npz",
        **{f"{n}|{d}": v for (n, d), v in res.items()},
        demos=np.array(DEMOS),
        eps=np.array(EPS),
    )
    log(f"\n[ok] saved sp_results2_s{SEED}.npz")
