# ruff: noqa
"""E1b: V-JEPA2.1 teacher features for cache_sp (same encode path as the main cache)."""

import os, time, numpy as np, sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/tmp/selfplay_probe")  # nosec B108
from sp_lib import VJepa21Encoder

C = "/tmp/selfplay_probe/cache_sp"  # nosec B108
m_ = np.load(C + "/meta.npz")
N = int(m_["N"])
imgs = np.memmap(C + "/imgs.u8", mode="r", dtype=np.uint8, shape=(N, 224, 224, 3))
enc = VJepa21Encoder()
out = np.memmap(C + "/vj_feats.f16", mode="w+", dtype=np.float16, shape=(N, 64, 768))
t0 = time.time()
for k in range(0, N, 256):
    j = min(k + 256, N)
    _, S = enc.encode_both(imgs[k:j], G=8)
    out[k:j] = S.reshape(-1, 64, 768).astype(np.float16)
    if k % 12800 == 0:
        print(f"{k}/{N} ({k / max(1, time.time() - t0):.0f}/s)", flush=True)
out.flush()
print(f"[ok] cache_sp/vj_feats.f16 ({time.time() - t0:.0f}s)", flush=True)
