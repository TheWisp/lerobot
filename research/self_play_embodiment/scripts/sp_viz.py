# ruff: noqa
"""Put the embodiment latent back on pixels, and show where it DIFFERS from raw JEPA.
e = f(mean_pool(JEPA(img))) is a global vector -> use OCCLUSION SENSITIVITY: slide a gray
patch; sens[pos] = ||rep(full) - rep(occluded)||. Compute for raw z, e_ac, e_free from the
SAME occluded forwards. Normalize each to [0,1]; diff maps show where e_ac re-weights
attention vs the substrate (red = e_ac cares more, blue = less).
Montage cols: image | JEPA z-sens | e_ac-sens | (e_ac - z) | (e_ac - e_free).
"""

import os

import matplotlib
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sp_lib import Encoder, load_emb

OUT = "/tmp/selfplay_probe"  # nosec B108
dev = "cuda"
P, S = 56, 16  # occluder patch size / stride on 256px
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs = wb["images"]
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
GX = wb["world"][:, 6:12]
# pick frames from held-out episodes where the arm/gripper has moved (interesting body pose)
held = np.concatenate([np.arange(s + 40, e - 1) for eid, (s, e) in enumerate(ep) if eid >= ne - 4])
home = GX[ep[0][0]]
d = np.linalg.norm(GX[held] - home, axis=1)
pick = held[np.argsort(d)[::-1]][[0, 8, 30, 80]]  # a few high-displacement frames
enc = Encoder()
f_ac, f_free = load_emb(OUT + "/f_ac.pt"), load_emb(OUT + "/f_free.pt")


@torch.no_grad()
def emb(z, f):
    return f(torch.tensor(z, device=dev)).cpu().numpy()


pos = [(y, x) for y in range(0, 256 - P + 1, S) for x in range(0, 256 - P + 1, S)]
ny = len(range(0, 256 - P + 1, S))


def occ_maps(img):
    variants = [img.copy()]
    for y, x in pos:
        v = img.copy()
        v[y : y + P, x : x + P] = 128
        variants.append(v)
    Z = enc.encode(np.stack(variants))  # (1+npos, 1408)
    z0, EA, EF = Z[0], emb(Z, f_ac), emb(Z, f_free)
    ea0, ef0 = EA[0], EF[0]
    sz = np.linalg.norm(Z[1:] - z0, axis=1)
    sa = np.linalg.norm(EA[1:] - ea0, axis=1)
    sf = np.linalg.norm(EF[1:] - ef0, axis=1)
    g = lambda v: v.reshape(ny, ny)
    return g(sz), g(sa), g(sf)


def up(m):  # normalize [0,1] + upsample to 256 (nearest)
    m = (m - m.min()) / (np.ptp(m) + 1e-9)
    return np.kron(m, np.ones((256 // ny + 1, 256 // ny + 1)))[:256, :256]


rows = len(pick)
fig, ax = plt.subplots(rows, 5, figsize=(15, 3 * rows))
titles = ["image", "JEPA z sensitivity", "e_ac sensitivity", "e_ac − z (re-weight)", "e_ac − e_free"]
for r, idx in enumerate(pick):
    sz, sa, sf = occ_maps(imgs[idx])
    Z, A, F = up(sz), up(sa), up(sf)
    ax[r, 0].imshow(imgs[idx])
    ax[r, 1].imshow(imgs[idx])
    ax[r, 1].imshow(Z, cmap="jet", alpha=0.55)
    ax[r, 2].imshow(imgs[idx])
    ax[r, 2].imshow(A, cmap="jet", alpha=0.55)
    ax[r, 3].imshow(imgs[idx])
    ax[r, 3].imshow(A - Z, cmap="RdBu_r", alpha=0.6, vmin=-1, vmax=1)
    ax[r, 4].imshow(imgs[idx])
    ax[r, 4].imshow(A - F, cmap="RdBu_r", alpha=0.6, vmin=-1, vmax=1)
    for c in range(5):
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(titles[c], fontsize=11)
plt.tight_layout()
plt.savefig(OUT + "/sp_latent_viz.png", dpi=110, bbox_inches="tight")
print(f"[ok] {OUT}/sp_latent_viz.png  ({rows} frames, occluder {P}px/{S}stride, grid {ny}x{ny})")
