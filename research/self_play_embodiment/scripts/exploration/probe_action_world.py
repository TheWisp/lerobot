"""THE clean test: does knowing the action improve prediction of the OBJECT's motion
it causes? Predict Δobject (GT) from spatial z_t vs (z_t, a_t), on contact slices
(threshold-swept), episode-split. Random actions => no action/scene confound; object
GT => isolates world from body. Δbody is the sanity (action should clearly help there)."""

import numpy as np

OUT = "/tmp/selfplay_probe"
d = np.load(OUT + "/contact_buffer.npz")
Zs = d["Zspatial"].astype(np.float64)
A = d["action"].astype(np.float64)
wd = d["world_delta"].astype(np.float64)
epid = d["epid"]
dObj, dBody = wd[:, :6], wd[:, 6:12]
dobj_mag = np.linalg.norm(dObj.reshape(-1, 2, 3), axis=2).max(1)
valid = dobj_mag < 0.2  # drop single-step flings (degenerate outliers)
ne = epid.max() + 1
n_te_ep = max(2, ne // 5)
test = epid >= ne - n_te_ep
fit = ~test
print(f"transitions={len(A)} (valid {valid.sum()}) episodes={ne} | held-out {n_te_ep} eps", flush=True)
print(
    f"contacts: >1mm={(dobj_mag > 0.001).sum()} >2mm={(dobj_mag > 0.002).sum()} >5mm={(dobj_mag > 0.005).sum()} >10mm={(dobj_mag > 0.010).sum()}",
    flush=True,
)


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xtr, Ytr, Xte, a=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xte @ np.linalg.solve(
        Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - Ytr.mean(0))
    ) + Ytr.mean(0)


def pca(Xtr, Xte, k=128):
    m = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
    c = Vt[: min(k, Xtr.shape[0] - 1)]
    return (Xtr - m) @ c.T, (Xte - m) @ c.T


def gap(target, mask):
    f, t = fit & mask & valid, test & mask & valid
    if t.sum() < 20 or f.sum() < 50:
        return None
    Ptr, Pte = pca(Zs[f], Zs[t])
    af = r2(target[t], ridge(Ptr, target[f], Pte))
    ac = r2(target[t], ridge(np.hstack([Ptr, A[f]]), target[f], np.hstack([Pte, A[t]])))
    return af, ac, int(t.sum())


allm = np.ones(len(A), bool)
print("\n=== ACTION -> WORLD (Δobject) prediction, spatial latent, episode-split ===")
print(f"{'slice':20s}{'n_te':>6s}{'free':>8s}{'+action':>9s}{'gap':>8s}")
r = gap(dObj, allm)
if r:
    print(f"{'all transitions':20s}{r[2]:6d}{r[0]:8.3f}{r[1]:9.3f}{r[1] - r[0]:+8.3f}")
for thr in [0.001, 0.002, 0.005, 0.010]:
    r = gap(dObj, dobj_mag > thr)
    if r:
        print(
            f"{'contact >' + str(int(thr * 1000)) + 'mm':20s}{r[2]:6d}{r[0]:8.3f}{r[1]:9.3f}{r[1] - r[0]:+8.3f}"
        )
r = gap(dBody, allm)
if r:
    print(
        f"\n  [sanity] dBODY all  free {r[0]:.3f} +action {r[1]:.3f} gap {r[1] - r[0]:+.3f}  (action should clearly help)"
    )
