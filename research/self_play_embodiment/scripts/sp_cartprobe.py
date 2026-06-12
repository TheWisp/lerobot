# ruff: noqa
"""Contrast probe: vision encodes the visible CARTESIAN world but not the joints.
Decode gripper-XY and object-XY (from world_buffer 'world') vs the 14 joints, same
linear ridge / held-out episodes. Backs the recommendation: give proprio (the self),
let vision carry the world."""

import numpy as np

OUT = "/tmp/selfplay_probe"  # nosec B108
M = np.load(OUT + "/feat_cache.npz")["M"].astype(np.float64)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
W = wb["world"].astype(np.float64)
J = wb["states"].astype(np.float64)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
fe = np.zeros(len(M), int)
for eid, (s, e) in enumerate(ep):
    fe[s:e] = eid
tr, te = fe < ne - 4, fe >= ne - 4


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, a=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xte @ np.linalg.solve(
        Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - Ytr.mean(0))
    ) + Ytr.mean(0)


def pca(Xtr, Xte, k):
    m = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
    c = Vt[:k]
    return (Xtr - m) @ c.T, (Xte - m) @ c.T


Ptr, Pte = pca(M[tr], M[te], 200)
# in-workspace filter for objects (flung outliers ruin linear fit)
peg, soc = W[:, 0:3], W[:, 3:6]
inb = (peg[:, 2] < 0.15) & (soc[:, 2] < 0.15) & (np.abs(peg[:, 0]) < 0.4) & (np.abs(soc[:, 0]) < 0.4)
tgt = {
    "L-gripper XYZ": W[:, 6:9],
    "R-gripper XYZ": W[:, 9:12],
    "object XY (peg+soc)": W[:, [0, 1, 3, 4]],
    "14 joints": J,
}
for name, Y in tgt.items():
    m = inb if "object" in name else np.ones(len(M), bool)
    f, t = tr & m, te & m
    pr, pt = pca(M[f], M[t], 200)
    print(f"  {name:22s} R2 = {r2(Y[t], ridge(pr, Y[f], pt)):.3f}")
