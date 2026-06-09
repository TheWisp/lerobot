"""Decisive diagnostic: how linearly decodable are the 14 JOINTS (agent_pos) from the
frozen latent? Closed-loop delta reaching floored even for the oracle -> suspect joints
aren't linearly accessible. Compare mean-pool M vs spatial S (4x4 maxpool), held-out
episodes, PCA->ridge. Also per-joint R2 to see which joints are lost."""

import numpy as np

OUT = "/tmp/selfplay_probe"
fc = np.load(OUT + "/feat_cache.npz")
M = fc["M"].astype(np.float64)
S = fc["S"].astype(np.float64)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
J = wb["states"].astype(np.float64)  # (8000,14) joints/agent_pos
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
fe = np.zeros(len(M), int)
for eid, (s, e) in enumerate(ep):
    fe[s:e] = eid
tr, te = fe < ne - 4, fe >= ne - 4


def r2(Y, P, axis=None):
    return 1 - ((Y - P) ** 2).sum(0) / (((Y - Y.mean(0)) ** 2).sum(0) + 1e-9)


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


for tag, Z in [("mean-pool M", M), ("spatial S ", S)]:
    for k in [64, 200]:
        Ptr, Pte = pca(Z[tr], Z[te], k)
        pred = ridge(Ptr, J[tr], Pte)
        rj = r2(J[te], pred)  # per-joint
        print(
            f"{tag} PCA{k:3d}: overall R2={r2(J[te].ravel(), pred.ravel()):.3f} | per-joint min/med/max {rj.min():.2f}/{np.median(rj):.2f}/{rj.max():.2f}"
        )
# baseline: how much does a single-step delta need? typical |joint| spread
print(f"\njoint std (per-joint, test): {J[te].std(0).round(2)}")
print(f"mean joint std = {J[te].std(0).mean():.3f}  (reach needs decode residual << this)")
