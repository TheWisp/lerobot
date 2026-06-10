"""Clean domain check (no noop-collapse confound): same CONTENT (home pose), different render
source. Our-sim home (Vec reset) vs the dataset's per-episode home frames. If our-sim home is
far from ALL dataset-home frames vs the dataset-home intra-spread -> render-domain gap."""
import os, numpy as np, torch
os.environ.setdefault("MUJOCO_GL", "egl")
from sp_lib import Vec, VJepa21Encoder
OUT = "/tmp/selfplay_probe"
d = np.load(OUT + "/manip_cache.npz"); M = d["M"].astype(np.float32); epid = d["epid"]
# dataset home features = first frame of each episode
home_idx = [np.where(epid == e)[0][0] for e in range(int(epid.max()) + 1)]
Mh = M[home_idx]  # (n_ep, 768) dataset-home features
# our-sim home frames (several seeds; arm home is fixed, peg/socket vary)
vec = Vec(8); img, _ = vec.reset(range(30000, 30008))
enc = VJepa21Encoder(); sim_h = enc.encode(img)  # (8,768) our-sim home
def nn_dist(x, bank): return np.linalg.norm(bank - x, axis=1).min()
intra = np.mean([nn_dist(Mh[i], np.delete(Mh, i, 0)) for i in range(len(Mh))])  # dataset-home NN spread
cross = np.mean([nn_dist(sim_h[i], Mh) for i in range(len(sim_h))])             # our-sim-home -> dataset-home
print(f"dataset-home intra-NN dist (raw feat L2):   {intra:.1f}", flush=True)
print(f"our-sim-home -> nearest dataset-home dist:  {cross:.1f}", flush=True)
print(f"ratio cross/intra = {cross/intra:.2f}  (~1 => same domain; >>1 => render gap)", flush=True)
# cosine too (scale-invariant)
def cos_nn(x, bank): n = bank/np.linalg.norm(bank,axis=1,keepdims=True); xx=x/np.linalg.norm(x); return 1-(n@xx).max()
ci = np.mean([cos_nn(Mh[i], np.delete(Mh,i,0)) for i in range(len(Mh))])
cc = np.mean([cos_nn(sim_h[i], Mh) for i in range(len(sim_h))])
print(f"cosine: dataset-home intra {ci:.3f} | our-sim->dataset {cc:.3f}", flush=True)
