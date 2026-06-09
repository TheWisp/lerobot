"""Encode all play frames with DINOv2 (mean-pooled patches, 768-d) -> dino_cache.npz.
Gate A: swap the substrate, reuse everything else."""
import numpy as np, time
from sp_lib import DinoEncoder
OUT="/tmp/selfplay_probe"
imgs=np.load(OUT+"/world_buffer.npz",allow_pickle=True)["images"]
enc=DinoEncoder(); t0=time.time()
M=enc.encode(imgs); print(f"encoded {len(M)} frames -> {M.shape} in {time.time()-t0:.0f}s",flush=True)
np.savez_compressed(OUT+"/dino_cache.npz", M=M)
print("[ok] dino_cache.npz",flush=True)
