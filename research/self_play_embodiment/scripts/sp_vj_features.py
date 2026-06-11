# ruff: noqa
import os, time, numpy as np, torch

os.environ.setdefault("MUJOCO_GL", "egl")
from sp_lib import VJepa21Encoder

C = "/tmp/selfplay_probe/cache"  # nosec B108
m_ = np.load(C + "/meta.npz")
N = int(m_["N"])
RES = int(m_["RES"])
imgs = np.memmap(C + "/imgs.u8", mode="r", dtype=np.uint8, shape=(N, RES, RES, 3))
enc = VJepa21Encoder()
_, S = enc.encode_both(imgs[:4], G=8)
print("vj feat check:", S.shape, "-> per-frame 64x768", flush=True)
out = np.memmap(C + "/vj_feats.f16", mode="w+", dtype=np.float16, shape=(N, 64, 768))
t0 = time.time()
for k in range(0, N, 256):
    j = min(k + 256, N)
    _, Sp = enc.encode_both(imgs[k:j], G=8)
    out[k:j] = Sp.reshape(-1, 64, 768).astype(np.float16)
    if k % 5120 == 0:
        print(f"{k}/{N} ({k / max(1, time.time() - t0):.0f}/s)", flush=True)
out.flush()
print(f"[ok] vj_feats.f16 {out.shape} ({time.time() - t0:.0f}s)", flush=True)
