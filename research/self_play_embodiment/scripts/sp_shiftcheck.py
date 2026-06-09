"""Is 'a' floored by goal-image DISTRIBUTION SHIFT or by hard end-to-end learning?
Train gripper-xyz decoder on PLAY frames; test on (1) held-out PLAY frames and (2) the
actual SETTLED test goals. If test-goal R2 collapses vs play R2 -> shift (must fix train
goal distribution). If test-goal R2 stays ~0.7 -> target IS recoverable, problem is the
end-to-end policy (structure/e can help)."""

import numpy as np

OUT = "/tmp/selfplay_probe"
M = np.load(OUT + "/feat_cache.npz")["M"].astype(np.float64)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
GX = wb["world"][:, 6:12].astype(np.float64)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
fe = np.zeros(len(M), int)
for eid, (s, e) in enumerate(ep):
    fe[s:e] = eid
tr, te = fe < ne - 4, fe >= ne - 4
g = np.load(OUT + "/goals.npz")
GZ = g["z_goal"].astype(np.float64)
GG = g["goal_gxyz"].astype(np.float64)


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, a=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xte @ np.linalg.solve(
        Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - Ytr.mean(0))
    ) + Ytr.mean(0)


def pca_fit(Xtr, k):
    m = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
    return m, Vt[:k]


m, c = pca_fit(M[tr], 200)


def proj(X):
    return (X - m) @ c.T


P_tr, P_te, P_goal = proj(M[tr]), proj(M[te]), proj(GZ)
pred_play = ridge(P_tr, GX[tr], P_te)
pred_goal = ridge(P_tr, GX[tr], P_goal)
print("gripper-xyz decode (train=play):")
print(f"  held-out PLAY frames : R2={r2(GX[te], pred_play):.3f}")
print(f"  SETTLED test goals   : R2={r2(GG, pred_goal):.3f}   <- if << play R2 => distribution shift")
print(f"  test-goal decode err : {np.linalg.norm(GG - pred_goal, axis=1).mean():.3f} m (goal spread ~0.64m)")
