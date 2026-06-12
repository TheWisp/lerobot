# ruff: noqa
"""THE embodiment-injection experiment: few-shot goal-conditioned reaching.

Demos via HER on random self-play: for play frame t pick a future frame t+k as the goal,
label = clip(proprio[t+k]-proprio[t], +-DMAX) (bounded joint-space step toward an achieved
pose). Policy = ridge((current_rep, z_goal) -> delta). Closed-loop delta control to a
held-out SETTLED goal; success = reached within eps (joint L2).

Conditions differ ONLY in the current-state representation (goal = base PCA(z_goal) for all):
  a       [zc, zg]              vision substrate floor
  noise   [zc, rand64, zg]      capacity control (must ~= a)
  b_free  [zc, e_free, zg]      e from action-FREE objective
  c_ac    [zc, e_ac,  zg]       e from action-CONDITIONED objective  <- the injection
  oracle  [zc, zg, goal_proprio]             isolates goal-decoding (ceiling-ish)
Sanities (at max demos): bothprop [..,cur_proprio,goal_proprio] ~1.0 ; shuffle (c, mismatched goal) ~chance.
Proof = c_ac > b_free > noise ~= a, opening up in the low-demo regime.
"""

import os
import time

import numpy as np
import torch
from sp_lib import Encoder, Vec, delta_command, load_emb, reach_err

OUT = "/tmp/selfplay_probe"  # nosec B108
KZ, DMAX, K_MAX, H = 200, 0.5, 30, 45
DEMOS = [int(x) for x in os.environ.get("DEMOS", "50,150,500,1500,5000").split(",")]
EPS = [0.3, 0.5, 0.8]


def log(m):
    print(m, flush=True)


# ---------- data ----------
M = np.load(OUT + "/feat_cache.npz")["M"].astype(np.float32)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
proprio = wb["states"].astype(np.float32)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
ec = np.load(OUT + "/e_cache.npz")
E_FREE, E_AC = ec["e_free"].astype(np.float32), ec["e_ac"].astype(np.float32)
g = np.load(OUT + "/goals.npz")
G_IMG, G_PROP, G_Z = g["goal_img"], g["goal_proprio"].astype(np.float32), g["z_goal"].astype(np.float32)
NG = len(G_PROP)
train_frames = np.concatenate([np.arange(s, e) for eid, (s, e) in enumerate(ep) if eid < ne - 4])

# ---------- PCA (fit on train frames' base z) ----------
pmu = M[train_frames].mean(0)
_, _, Vt = np.linalg.svd(M[train_frames] - pmu, full_matrices=False)
PC = Vt[:KZ]


def pca(z):
    return (z - pmu) @ PC.T


ZG_PCA = pca(G_Z)  # test goals projected once

# ---------- HER demo pool (from train frames) ----------
rng = np.random.RandomState(0)
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
log(f"demo pool {len(ti)} | PCA k={KZ} | conditions on {NG} held-out goals, H={H}")


# ---------- condition feature builders ----------
def feats(name, zc, zg, ef, ea, nz, cp, gp):
    if name == "a":
        return np.hstack([zc, zg])
    if name == "noise":
        return np.hstack([zc, nz, zg])
    if name == "b_free":
        return np.hstack([zc, ef, zg])
    if name == "c_ac":
        return np.hstack([zc, ea, zg])
    if name == "oracle":
        return np.hstack([zc, zg, gp])
    if name == "bothprop":
        return np.hstack([zc, zg, cp, gp])
    if name == "shuffle":
        return np.hstack([zc, ea, zg])  # zg pre-shuffled by caller
    raise ValueError(name)


# ---------- ridge with alpha search ----------
class Ridge:
    def fit(self, X, Y):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.xmu) / self.xsd
        self.b = Y.mean(0)
        n = len(X)
        cut = int(n * 0.8)
        tr, va = slice(0, cut), slice(cut, n)
        best, bestW = 1e18, None
        for a in [1.0, 10.0, 100.0, 1000.0]:
            W = np.linalg.solve(Xn[tr].T @ Xn[tr] + a * np.eye(Xn.shape[1]), Xn[tr].T @ (Y[tr] - self.b))
            v = ((Xn[va] @ W + self.b - Y[va]) ** 2).mean()
            if v < best:
                best, bestW, self.a = v, W, a
        # refit on all with chosen alpha
        self.W = np.linalg.solve(Xn.T @ Xn + self.a * np.eye(Xn.shape[1]), Xn.T @ (Y - self.b))
        return self

    def predict(self, X):
        return ((X - self.xmu) / self.xsd) @ self.W + self.b


def fit_policy(name, nd, shuffle=False):
    idx = np.arange(nd)
    zg = ZG_d[idx].copy()
    if shuffle:
        zg = zg[rng.permutation(nd)]
    X = feats(
        name if not shuffle else "shuffle",
        ZC_d[idx],
        zg,
        E_FREE[ti[idx]],
        E_AC[ti[idx]],
        NOISE[idx],
        proprio[ti[idx]],
        proprio[gi[idx]],
    )
    return Ridge().fit(X, LAB[idx])


# ---------- closed-loop eval ----------
enc = Encoder()
F_FREE, F_AC = load_emb(OUT + "/f_free.pt"), load_emb(OUT + "/f_ac.pt")


@torch.no_grad()
def emb(z, f):
    return f(torch.tensor(z, device="cuda")).cpu().numpy()


VEC = Vec(NG)


def rollout(name, pol, shuffle=False):
    img, prop = VEC.reset(range(20000, 20000 + NG))
    zg = ZG_PCA.copy()
    if shuffle:
        zg = zg[::-1].copy()  # fixed mismatch (NG even -> no fixed point)
    errs = np.full((H, NG), np.nan)
    for t in range(H):
        z = enc.encode(img)
        zc = pca(z)
        nz = np.random.randn(NG, 64).astype(np.float32)
        X = feats(name if not shuffle else "shuffle", zc, zg, emb(z, F_FREE), emb(z, F_AC), nz, prop, G_PROP)
        delta = pol.predict(X)
        (img, prop), _, _ = VEC.step(delta_command(prop, delta, dmax=DMAX))
        errs[t] = reach_err(prop, G_PROP)
    final = errs[-5:].mean(0)  # mean of last 5 steps -> "reached and held"
    return final


def sr_row(tag, final):
    return f"  {tag:9s} finalErr {final.mean():.3f} | " + "  ".join(
        f"SR@{e}={float((final < e).mean()):.2f}" for e in EPS
    )


# ---------- run ----------
log("\n=== SR vs #demos (final-err over 32 held-out goals) ===")
results = {}
for nd in DEMOS:
    log(f"\n--- #demos = {nd} ---")
    for name in ["a", "noise", "b_free", "c_ac", "oracle"]:
        t0 = time.time()
        pol = fit_policy(name, nd)
        final = rollout(name, pol)
        results[(name, nd)] = final.copy()
        log(sr_row(name, final) + f"   [{time.time() - t0:.0f}s]")

log("\n=== sanity controls (at max demos) ===")
nd = DEMOS[-1]
for name in ["bothprop"]:
    pol = fit_policy(name, nd)
    log(sr_row(name, rollout(name, pol)))
pol = fit_policy("c_ac", nd, shuffle=True)
log(sr_row("shuffle", rollout("c_ac", pol, shuffle=True)))

np.savez(
    OUT + "/sp_results.npz",
    **{f"{n}|{d}": v for (n, d), v in results.items()},
    demos=np.array(DEMOS),
    eps=np.array(EPS),
)
log("\n[ok] saved sp_results.npz")
