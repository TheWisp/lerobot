# ruff: noqa
"""Is the embodiment latent 'better' than og JEPA -- visually and quantitatively?
A) PROBE: decode gripper-xyz (held-out eps) from full z vs e_ac(64) vs e_free(64) vs a
   random-64 projection of z. If e_ac > rand64 (and >= e_free), it CONCENTRATES embodiment
   info into 64 dims (quantitatively better summary).
B) VIZ: per-token PCA-RGB of JEPA tokens vs f_ac applied per-token vs f_free per-token.
   e_ac is global; applying f per-token is an approximation (input-norm is pooled) -> shows
   whether the embodiment transform changes SPATIAL structure or just inherits JEPA's.
"""

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

# ---------- A) quantitative probe (cached features) ----------
M = np.load(OUT + "/feat_cache.npz")["M"].astype(np.float64)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
GX = wb["world"][:, 6:12].astype(np.float64)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
ec = np.load(OUT + "/e_cache.npz")
EA, EF = ec["e_ac"].astype(np.float64), ec["e_free"].astype(np.float64)
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
R = rng.randn(1408, 64)
Zr = M @ R
feats = {
    "full z (PCA200)": pca(M[tr], M[te], 200),
    "e_ac (64)": (EA[tr], EA[te]),
    "e_free (64)": (EF[tr], EF[te]),
    "rand-64 proj of z": (Zr[tr], Zr[te]),
}
print("=== gripper-xyz decode R2 (held-out eps): is e_ac a better 64-d embodiment summary? ===")
for name, (Xtr, Xte) in feats.items():
    print(f"  {name:20s} R2 = {r2(GX[te], ridge(Xtr, GX[tr], Xte)):.3f}")

# ---------- B) per-token spatial PCA: JEPA vs f_ac vs f_free ----------
imgs = wb["images"]
held = np.concatenate([np.arange(s + 40, e - 1) for eid, (s, e) in enumerate(ep) if eid >= ne - 4])
d = np.linalg.norm(GX[held] - GX[ep[0][0]], axis=1)
pick = held[np.argsort(d)[::-1]][[0, 8, 40, 100]]
FR = imgs[pick]
from transformers import AutoVideoProcessor, VJEPA2Model

m = VJEPA2Model.from_pretrained("facebook/vjepa2-vitg-fpc64-256", torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, torch.bfloat16)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, torch.bfloat16)
f_ac, f_free = load_emb(OUT + "/f_ac.pt"), load_emb(OUT + "/f_free.pt")
tok_j, tok_a, tok_f = [], [], []
with torch.no_grad():
    for im in FR:
        clip = np.repeat(im[None], 2, 0)[None]
        x = torch.from_numpy(clip).to(dev).permute(0, 1, 4, 2, 3).to(torch.bfloat16) / 255.0
        x = (x - mean) / std
        h = m(pixel_values_videos=x).last_hidden_state.float()[0]  # (256,1408)
        g = int(round(len(h) ** 0.5))
        tok_j.append(h.cpu().numpy().reshape(g, g, -1))
        tok_a.append(f_ac(h).cpu().numpy().reshape(g, g, -1))
        tok_f.append(f_free(h).cpu().numpy().reshape(g, g, -1))


def pca_rgb(grids):
    N, g, _, D = grids.shape
    X = grids.reshape(-1, D)
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ Vt[:3].T
    lo, hi = np.percentile(Y, 2, 0), np.percentile(Y, 98, 0)
    Y = np.clip((Y - lo) / (hi - lo + 1e-9), 0, 1)
    return Y.reshape(N, g, g, 3)


cols = [
    ("image", None),
    ("V-JEPA tokens", pca_rgb(np.stack(tok_j))),
    ("f_ac per-token", pca_rgb(np.stack(tok_a))),
    ("f_free per-token", pca_rgb(np.stack(tok_f))),
]
fig, ax = plt.subplots(len(FR), 4, figsize=(12, 3 * len(FR)))
for r in range(len(FR)):
    for c, (t, data) in enumerate(cols):
        A = ax[r, c]
        if data is None:
            A.imshow(FR[r])
        else:
            A.imshow(
                np.array(Image.fromarray((data[r] * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST))
            )
        A.set_xticks([])
        A.set_yticks([])
        if r == 0:
            A.set_title(t, fontsize=11)
plt.tight_layout()
plt.savefig(OUT + "/sp_emb_pca.png", dpi=110, bbox_inches="tight")
print(f"[ok] {OUT}/sp_emb_pca.png")
