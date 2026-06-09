"""DINO-style PATCH-FEATURE PCA -> RGB on aloha frames, per encoder. Shows how each frozen
substrate spatially parses the robot scene (the segmentation-like view from DINOv2/v3 papers).
Cols: image | V-JEPA patch-PCA | DINOv2 patch-PCA (if downloadable). Same 16x16 grid.
Recipe: per encoder, stack all frames' patch tokens, PCA->3 comps, percentile-normalize -> RGB."""

import os

import matplotlib
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "/tmp/selfplay_probe"
dev = "cuda"
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs = wb["images"]
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
GX = wb["world"][:, 6:12]
held = np.concatenate([np.arange(s + 40, e - 1) for eid, (s, e) in enumerate(ep) if eid >= ne - 4])
d = np.linalg.norm(GX[held] - GX[ep[0][0]], axis=1)
pick = held[np.argsort(d)[::-1]][[0, 5, 20, 50, 120]]
FR = imgs[pick]  # (5,256,256,3) uint8


def pca_rgb(grids):
    """grids: (N,g,g,D) -> (N,g,g,3) in [0,1] via shared PCA + percentile norm."""
    N, g, _, D = grids.shape
    X = grids.reshape(-1, D).astype(np.float64)
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ Vt[:3].T
    lo, hi = np.percentile(Y, 2, 0), np.percentile(Y, 98, 0)
    Y = np.clip((Y - lo) / (hi - lo + 1e-9), 0, 1)
    return Y.reshape(N, g, g, 3)


panels = [("image", None)]

# ---- V-JEPA patch tokens ----
from transformers import AutoVideoProcessor, VJEPA2Model

m = VJEPA2Model.from_pretrained("facebook/vjepa2-vitg-fpc64-256", torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(dev, torch.bfloat16)
std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(dev, torch.bfloat16)
grids = []
with torch.no_grad():
    for im in FR:
        clip = np.repeat(im[None], 2, 0)[None]  # (1,2,256,256,3)
        x = torch.from_numpy(clip).to(dev).permute(0, 1, 4, 2, 3).to(torch.bfloat16) / 255.0
        x = (x - mean) / std
        h = m(pixel_values_videos=x).last_hidden_state.float()[0].cpu().numpy()  # (256,1408)
        g = int(round(len(h) ** 0.5))
        grids.append(h.reshape(g, g, -1))
panels.append(("V-JEPA ViT-g patch-PCA", pca_rgb(np.stack(grids))))
del m
torch.cuda.empty_cache()

# ---- DINOv2 / DINOv3 (drop CLS+register prefix via "last g*g tokens" rule) ----
from transformers import AutoImageProcessor


def dino_grids(ModelCls, rid):
    dm = ModelCls.from_pretrained(rid).to(dev).eval()
    dp = AutoImageProcessor.from_pretrained(rid)
    grids = []
    with torch.no_grad():
        for im in FR:
            x = dp(images=Image.fromarray(im), return_tensors="pt").to(dev)
            o = dm(**x).last_hidden_state[0].float().cpu().numpy()  # (n_tok, D)
            g = int(np.sqrt(len(o)))
            grids.append(o[len(o) - g * g :].reshape(g, g, -1))  # last g*g = patches
    del dm
    torch.cuda.empty_cache()
    return np.stack(grids)


from transformers import Dinov2Model

DINOS = [("DINOv2-base patch-PCA", Dinov2Model, "facebook/dinov2-base")]
try:
    from transformers import DINOv3ViTModel

    DINOS.append(("DINOv3-ViTB16 patch-PCA", DINOv3ViTModel, "facebook/dinov3-vitb16-pretrain-lvd1689m"))
except Exception:
    pass
for label, Cls, rid in DINOS:
    try:
        panels.append((label, pca_rgb(dino_grids(Cls, rid))))
        print(f"{label} ok", flush=True)
    except Exception as e:
        print(f"{label} skipped: {repr(e)[:110]}", flush=True)

# ---- montage ----
nc = len(panels)
fig, ax = plt.subplots(len(FR), nc, figsize=(3 * nc, 3 * len(FR)))
for r in range(len(FR)):
    for c, (title, data) in enumerate(panels):
        A = ax[r, c]
        if data is None:
            A.imshow(FR[r])
        else:
            up = np.array(Image.fromarray((data[r] * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST))
            A.imshow(up)
        A.set_xticks([])
        A.set_yticks([])
        if r == 0:
            A.set_title(title, fontsize=11)
plt.tight_layout()
plt.savefig(OUT + "/sp_dino_pca.png", dpi=110, bbox_inches="tight")
print(f"[ok] {OUT}/sp_dino_pca.png", flush=True)
