# ruff: noqa
"""PCA comparison on V-JEPA 2.1: raw z patches vs inverse-dynamics f_invdyn vs forward f_ac.
(A) per-token patch-PCA RGB (caveat: f trained on mean-pooled z; per-token is an approximation).
(B) the quantitative backbone -- gripper-xyz decode (held-out eps) from z / e_invdyn / e_ac /
    e_free / random-64. Does e_invdyn concentrate embodiment better than forward e_ac / random?"""

import os

import matplotlib
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sp_lib import load_emb

OUT = "/tmp/selfplay_probe"  # nosec B108
dev = "cuda"
# ---------- (B) decode probe (cached) ----------
M = np.load(OUT + "/vj21_cache.npz")["M"].astype(np.float64)
ec = np.load(OUT + "/e_cache_vj21.npz")
EI, EA, EF = ec["e_invdyn"].astype(np.float64), ec["e_ac"].astype(np.float64), ec["e_free"].astype(np.float64)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
GX = wb["world"][:, 6:12].astype(np.float64)
imgs = wb["images"]
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


rng = np.random.RandomState(0)
R = rng.randn(768, 64)
Zr = M @ R
print("=== gripper-xyz decode R2 (V-JEPA 2.1, held-out eps) ===", flush=True)
ztr, zte = pca(M[tr], M[te], 200)
for name, (Xtr, Xte) in {
    "z (PCA200)": (ztr, zte),
    "e_invdyn(64)": (EI[tr], EI[te]),
    "e_ac(64)": (EA[tr], EA[te]),
    "e_free(64)": (EF[tr], EF[te]),
    "random-64": (Zr[tr], Zr[te]),
}.items():
    print(f"  {name:14s} R2 = {r2(GX[te], ridge(Xtr, GX[tr], Xte)):.3f}", flush=True)
# ---------- (A) per-token patch-PCA ----------
enc = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", trust_repo=True)[0].to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(dev)
f_inv, f_ac = load_emb(OUT + "/f_vj21_invdyn.pt"), load_emb(OUT + "/f_vj21_ac.pt")
held = np.concatenate([np.arange(s + 40, e - 1) for eid, (s, e) in enumerate(ep) if eid >= ne - 4])
d = np.linalg.norm(GX[held] - GX[ep[0][0]], axis=1)
pick = list(held[np.argsort(d)[::-1]][[0, 8, 40, 100]])
zt, it, at = [], [], []
with torch.no_grad():
    for p in pick:
        x = (
            torch.from_numpy(np.array(Image.fromarray(imgs[p]).resize((384, 384))))
            .to(dev)
            .permute(2, 0, 1)[None, :, None]
            .float()
            / 255
        )
        x = (x - mean) / std
        h = enc(x).float()[0]  # (576,768)
        zt.append(h.cpu().numpy().reshape(24, 24, 768))
        it.append(f_inv(h).cpu().numpy().reshape(24, 24, -1))
        at.append(f_ac(h).cpu().numpy().reshape(24, 24, -1))


def pca_rgb(gs):
    N, g, _, D = gs.shape
    X = gs.reshape(-1, D)
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ Vt[:3].T
    lo, hi = np.percentile(Y, 2, 0), np.percentile(Y, 98, 0)
    Y = np.clip((Y - lo) / (hi - lo + 1e-9), 0, 1)
    return Y.reshape(N, g, g, 3)


cols = [
    ("image", None),
    ("z patches (raw)", pca_rgb(np.stack(zt))),
    ("f_invdyn /token", pca_rgb(np.stack(it))),
    ("f_ac /token", pca_rgb(np.stack(at))),
]
fig, ax = plt.subplots(len(pick), 4, figsize=(12, 3 * len(pick)))
for r in range(len(pick)):
    for c, (t, data) in enumerate(cols):
        A = ax[r, c]
        if data is None:
            A.imshow(imgs[pick[r]])
        else:
            A.imshow(
                np.array(Image.fromarray((data[r] * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST))
            )
        A.set_xticks([])
        A.set_yticks([])
        if r == 0:
            A.set_title(t, fontsize=11)
plt.tight_layout()
plt.savefig(OUT + "/vj21_invdyn_pca.png", dpi=110, bbox_inches="tight")
print("[ok] vj21_invdyn_pca.png", flush=True)
