"""Diagnose the reach-to-peg floor: is there a DOMAIN GAP between the dataset's renders
(BC trained on these) and our-sim renders (eval)? Compare a dataset frame vs an our-sim reset
frame visually, and check whether our-sim V-JEPA2.1 features fall inside the dataset feature cloud."""
import os, numpy as np, torch, matplotlib
os.environ.setdefault("MUJOCO_GL", "egl"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sp_lib import Vec, VJepa21Encoder
OUT = "/tmp/selfplay_probe"
ds = LeRobotDataset("lerobot/aloha_sim_insertion_scripted")
def to_img(t): return np.array(Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((256, 256)))
ei = np.array(ds.hf_dataset["episode_index"])
ds_frame0 = to_img(ds[int(np.where(ei == 0)[0][0])]["observation.images.top"])   # dataset, episode start (home)
ds_frame_mid = to_img(ds[int(np.where(ei == 0)[0][200])]["observation.images.top"])  # mid-episode
vec = Vec(2); img, _ = vec.reset([0, 1]); sim_frame0 = img[0]                       # our sim reset (home)
# step a bit so the arm leaves home
for _ in range(40): (img, _), _, _ = vec.step(np.tile(np.zeros(14, np.float32), (2, 1)))
sim_frame_mid = img[0]
fig, ax = plt.subplots(2, 2, figsize=(7, 7))
for a, im, t in [(ax[0, 0], ds_frame0, "DATASET frame (home)"), (ax[0, 1], ds_frame_mid, "DATASET frame (mid)"),
                 (ax[1, 0], sim_frame0, "OUR-SIM reset (home)"), (ax[1, 1], sim_frame_mid, "OUR-SIM +40 noop")]:
    a.imshow(im); a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(OUT + "/sp_domain_check.png", dpi=110); print("[ok] sp_domain_check.png", flush=True)
# feature-cloud check
M = np.load(OUT + "/manip_cache.npz")["M"].astype(np.float32)   # dataset features
enc = VJepa21Encoder()
sim_M = enc.encode(np.stack([sim_frame0, sim_frame_mid]))
mu, sd = M.mean(0), M.std(0) + 1e-6
def z(x): return (x - mu) / sd
dd = np.linalg.norm(z(M[:2000]) - z(M[:2000]).mean(0), axis=1)   # typical dataset radius (normalized)
sim_r = np.linalg.norm(z(sim_M) - z(M).mean(0), axis=1)
print(f"normalized feature radius: dataset typical = {dd.mean():.1f} +- {dd.std():.1f}", flush=True)
print(f"  our-sim frames radius = {sim_r.round(1)}  (>> dataset typical => OOD/domain gap)", flush=True)
# nearest dataset neighbor distance for a sim frame
nn = np.linalg.norm(z(M) - z(sim_M[1]), axis=1).min()
print(f"  our-sim(mid) nearest dataset-feature dist = {nn:.1f}  (vs dataset typical {dd.mean():.1f})", flush=True)
