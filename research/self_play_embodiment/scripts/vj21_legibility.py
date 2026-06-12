# ruff: noqa
"""Does V-JEPA 2.1 fix the mushy-patch / poor-decode problem ON OUR FRAMES?
Encode all play frames (mean-pooled 24x24 patches @384, single-frame 2D path), then decode
gripper-xyz + 14 joints (held-out episodes) and PCA-viz the patches. Compare to V-JEPA2
(gripper 0.78 / joints 0.18) and DINOv2 (gripper 0.78)."""

import os
import time

import matplotlib
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "/tmp/selfplay_probe"  # nosec B108
dev = "cuda"
enc = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", trust_repo=True)[0].to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(dev)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs = wb["images"]
GX = wb["world"][:, 6:12].astype(np.float64)
J = wb["states"].astype(np.float64)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)


@torch.no_grad()
def encode(ims, bs=48, keepgrid=None):
    N = len(ims)
    M = np.zeros((N, 768), np.float32)
    grids = {}
    for k in range(0, N, bs):
        chunk = np.stack(
            [np.array(Image.fromarray(ims[i]).resize((384, 384))) for i in range(k, min(k + bs, N))]
        )
        x = torch.from_numpy(chunk).to(dev).permute(0, 3, 1, 2)[:, :, None].float() / 255.0
        x = (x - mean) / std
        h = enc(x).float()  # (b,576,768)
        M[k : k + len(h)] = h.mean(1).cpu().numpy()
        if keepgrid is not None:
            for j, gi in enumerate(range(k, min(k + bs, N))):
                if gi in keepgrid:
                    grids[gi] = h[j].reshape(24, 24, 768).cpu().numpy()
    return M, grids


t0 = time.time()
# frames to keep grids for (viz)
held = np.concatenate([np.arange(s + 40, e - 1) for eid, (s, e) in enumerate(ep) if eid >= ne - 4])
d = np.linalg.norm(GX[held] - GX[ep[0][0]], axis=1)
pick = list(held[np.argsort(d)[::-1]][[0, 8, 40, 100]])
M, grids = encode(imgs, keepgrid=set(pick))
print(f"encoded {len(M)} @384 in {time.time() - t0:.0f}s -> {M.shape}", flush=True)
np.savez_compressed(OUT + "/vj21_cache.npz", M=M)
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
print("\n=== V-JEPA 2.1 base decode (held-out eps), vs V-JEPA2 / DINOv2 ===", flush=True)
print(
    f"  gripper-xyz R2 = {r2(GX[te], ridge(Ptr, GX[tr], Pte)):.3f}   (V-JEPA2 0.78, DINOv2 0.78)", flush=True
)
print(f"  14-joints   R2 = {r2(J[te], ridge(Ptr, J[tr], Pte)):.3f}   (V-JEPA2 0.18)", flush=True)


# patch-PCA viz
def pca_rgb(gs):
    N, g, _, D = gs.shape
    X = gs.reshape(-1, D)
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ Vt[:3].T
    lo, hi = np.percentile(Y, 2, 0), np.percentile(Y, 98, 0)
    Y = np.clip((Y - lo) / (hi - lo + 1e-9), 0, 1)
    return Y.reshape(N, g, g, 3)


gs = np.stack([grids[p] for p in pick])
rgb = pca_rgb(gs)
fig, ax = plt.subplots(len(pick), 2, figsize=(6, 3 * len(pick)))
for r, p in enumerate(pick):
    ax[r, 0].imshow(imgs[p])
    ax[r, 1].imshow(
        np.array(Image.fromarray((rgb[r] * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST))
    )
    for c in range(2):
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
    if r == 0:
        ax[r, 0].set_title("image")
        ax[r, 1].set_title("V-JEPA 2.1 patch-PCA")
plt.tight_layout()
plt.savefig(OUT + "/vj21_pca.png", dpi=110, bbox_inches="tight")
print("[ok] vj21_pca.png + vj21_cache.npz", flush=True)
