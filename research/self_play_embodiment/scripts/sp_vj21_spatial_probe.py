"""Gate F1b: does V-JEPA 2.1 SPATIAL (4x4 max-pool of the 24x24 tokens) recover the object?
Mean-pool diluted small objects (0.19); spatial should be far better (V-JEPA2 spatial was 0.525).
Encodes play frames -> vj21_spatial.npz (S, 4x4x768=12288); decodes object-xy + gripper-xy
(held-out eps, in-workspace). If object decodes well -> reach-to-object task is viable on 2.1-spatial.
"""
import os, time, numpy as np, torch
os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image
OUT = "/tmp/selfplay_probe"; dev = "cuda"
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
imgs = wb["images"]; W = wb["world"].astype(np.float64)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]; ne = len(ep)
enc = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", trust_repo=True)[0].to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(dev)
G = int(os.environ.get("GRID", "4"))  # spatial grid (max-pool of 24x24)
@torch.no_grad()
def encode(ims, bs=48):
    N = len(ims); S = np.zeros((N, G * G * 768), np.float32)
    for k in range(0, N, bs):
        chunk = np.stack([np.array(Image.fromarray(ims[i]).resize((384, 384))) for i in range(k, min(k + bs, N))])
        x = torch.from_numpy(chunk).to(dev).permute(0, 3, 1, 2)[:, :, None].float() / 255.0
        x = (x - mean) / std
        h = enc(x).float()  # (b,576,768)
        b, nt, C = h.shape; g = int(round(nt ** 0.5)); hh = h.reshape(b, g, g, C)
        sp = hh.reshape(b, G, g // G, G, g // G, C).amax(dim=(2, 4)).reshape(b, -1)  # (b, 16*768)
        S[k:k + b] = sp.cpu().numpy()
    return S
t0 = time.time(); S = encode(imgs); print(f"GRID={G} encoded {len(S)} spatial in {time.time()-t0:.0f}s -> {S.shape}", flush=True)
np.savez_compressed(OUT + f"/vj21_spatial_g{G}.npz", S=S)
fe = np.zeros(len(S), int)
for eid, (s, e) in enumerate(ep): fe[s:e] = eid
peg, soc = W[:, 0:3], W[:, 3:6]
inb = (peg[:, 2] < .15) & (soc[:, 2] < .15) & (np.abs(peg[:, 0]) < .4) & (np.abs(soc[:, 0]) < .4) & (np.abs(peg[:, 1] - .55) < .4) & (np.abs(soc[:, 1] - .55) < .4)
def r2(Y, P): return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)
def ridge(Xtr, Ytr, Xte, a=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xte @ np.linalg.solve(Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - Ytr.mean(0))) + Ytr.mean(0)
def pca(Xtr, Xte, k):
    m = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - m, full_matrices=False); c = Vt[:k]
    return (Xtr - m) @ c.T, (Xte - m) @ c.T
objxy = W[:, [0, 1, 3, 4]]; gripxy = W[:, [6, 7, 9, 10]]
trm, tem = (fe < ne - 4) & inb, (fe >= ne - 4) & inb
Ptr, Pte = pca(S[trm], S[tem], 200)
print(f"object-xy  decode (V-JEPA2.1 spatial 4x4): R2 = {r2(objxy[tem], ridge(Ptr, objxy[trm], Pte)):.3f}  (V-JEPA2 spatial 0.525)", flush=True)
trg, teg = fe < ne - 4, fe >= ne - 4
Ptr2, Pte2 = pca(S[trg], S[teg], 200)
print(f"gripper-xy decode (V-JEPA2.1 spatial 4x4): R2 = {r2(gripxy[teg], ridge(Ptr2, gripxy[trg], Pte2)):.3f}", flush=True)
print("[ok] vj21_spatial.npz", flush=True)
