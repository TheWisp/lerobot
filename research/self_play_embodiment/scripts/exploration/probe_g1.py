"""G1 probe: is the FROZEN V-JEPA latent meaningful on our sim frames?

Test: can a linear probe read the arm's own 14-D joint state out of the frozen
ViT-g latent of a single observation? Baseline: same probe on raw-pixel PCA.
If V-JEPA can't beat raw pixels, there's an internet-video -> sim-render gap and
the whole plan stalls here (the cheapest, most likely silent failure).

numpy-only stats (closed-form ridge + SVD-PCA) so there's no sklearn dependency.
Run: python probe_g1.py
"""

import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

OUT = "/tmp/selfplay_probe"
REPO = "facebook/vjepa2-vitg-fpc64-256"
dev, dt = "cuda", torch.bfloat16

d = np.load(os.path.join(OUT, "g1_buffer.npz"))
imgs, states, ep = d["images"], d["states"], d["ep_bounds"]
print(f"buffer: images={imgs.shape} states={states.shape} episodes={len(ep)}", flush=True)

from transformers import AutoVideoProcessor, VJEPA2Model

print("loading ViT-g...", flush=True)
model = VJEPA2Model.from_pretrained(REPO, torch_dtype=dt).to(dev).eval()
proc = AutoVideoProcessor.from_pretrained(REPO)
mean = torch.tensor(getattr(proc, "image_mean", [0.485, 0.456, 0.406])).view(1, 1, 3, 1, 1).to(dev, dt)
std = torch.tensor(getattr(proc, "image_std", [0.229, 0.224, 0.225])).view(1, 1, 3, 1, 1).to(dev, dt)
print("mean/std:", mean.flatten().tolist(), std.flatten().tolist(), flush=True)


def encode(clips_uint8):  # [B,T,H,W,3] uint8 -> [B,hidden] float32 (mean-pooled tokens)
    x = torch.from_numpy(clips_uint8).to(dev).permute(0, 1, 4, 2, 3).to(dt) / 255.0
    x = (x - mean) / std
    with torch.no_grad():
        out = model(pixel_values_videos=x)
    h = out.last_hidden_state if getattr(out, "last_hidden_state", None) is not None else out[0]
    return h.float().mean(dim=1).cpu().numpy()


# pick a clip length the pretrained model accepts (temporal pos-embed may be fixed)
T = None
for cand in [16, 8, 4, 64, 2]:
    try:
        encode(np.repeat(imgs[:1][None], cand, axis=1))
        T = cand
        print("clip length T =", T, flush=True)
        break
    except Exception as e:
        print(f"  T={cand} rejected: {type(e).__name__}: {str(e)[:90]}", flush=True)
assert T is not None, "no clip length accepted by the model"
B = 8 if T <= 16 else 2

# probe frames: every frame that has T-1 predecessors within its episode
idxs = [f for (s, e) in ep for f in range(int(s) + T - 1, int(e))]
rng = np.random.default_rng(0)
rng.shuffle(idxs)
idxs = idxs[:1200]
print(f"encoding {len(idxs)} observations (T={T}, B={B})...", flush=True)

lat, Y, t0 = [], [], time.time()
for k in range(0, len(idxs), B):
    sel = idxs[k : k + B]
    clips = np.stack([imgs[f - T + 1 : f + 1] for f in sel])  # [b,T,H,W,3]
    lat.append(encode(clips))
    Y.append(states[sel])
    if (k // B) % 20 == 0:
        print(
            f"  {k + len(sel)}/{len(idxs)} ({(k + len(sel)) / (time.time() - t0 + 1e-9):.0f}/s)", flush=True
        )
lat = np.concatenate(lat).astype(np.float64)
Y = np.concatenate(Y).astype(np.float64)
print(f"latents {lat.shape} in {time.time() - t0:.0f}s", flush=True)


def ridge_pred(Xtr, Ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - ym))
    return Xte @ W + ym


def r2(Yte, pred):
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9
    return 1 - ss_res / ss_tot


n = len(lat)
ntr = int(0.8 * n)
Ytr, Yte = Y[:ntr], Y[ntr:]

r2_vj = r2(Yte, ridge_pred(lat[:ntr], Ytr, lat[ntr:]))

# baseline: raw-pixel PCA (downsample 256->64 for a fair, cheap raw-pixel features)
flat = (imgs[idxs][:, ::4, ::4, :].reshape(len(idxs), -1).astype(np.float64)) / 255.0
mu = flat[:ntr].mean(0)
Xc = flat[:ntr] - mu
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
comp = Vt[:256]
Ptr, Pte = (flat[:ntr] - mu) @ comp.T, (flat[ntr:] - mu) @ comp.T
r2_pca = r2(Yte, ridge_pred(Ptr, Ytr, Pte))

verdict = (
    "PASS"
    if (r2_vj.mean() > r2_pca.mean() + 0.05 and r2_vj.mean() > 0.3)
    else "WEAK"
    if r2_vj.mean() > r2_pca.mean()
    else "FAIL"
)
res = {
    "clip_T": T,
    "n_samples": n,
    "hidden": int(lat.shape[1]),
    "vjepa_meanR2": round(float(r2_vj.mean()), 3),
    "vjepa_R2_min": round(float(r2_vj.min()), 3),
    "vjepa_R2_max": round(float(r2_vj.max()), 3),
    "pixelPCA_meanR2": round(float(r2_pca.mean()), 3),
    "verdict": verdict,
}
print("\n=== G1 RESULT: read 14-D joint state from a single frozen-encoder latent ===", flush=True)
print(
    f"  V-JEPA ViT-g   mean R2 = {res['vjepa_meanR2']:.3f}  (per-dim {res['vjepa_R2_min']:.2f}..{res['vjepa_R2_max']:.2f})",
    flush=True,
)
print(f"  raw-pixel PCA  mean R2 = {res['pixelPCA_meanR2']:.3f}", flush=True)
print(
    f"  VERDICT: {verdict}  (V-JEPA {'beats' if r2_vj.mean() > r2_pca.mean() else 'does NOT beat'} raw pixels)",
    flush=True,
)
with open(os.path.join(OUT, "g1_result.json"), "w") as f:
    json.dump(res, f, indent=2)
