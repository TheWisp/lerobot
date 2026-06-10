"""Encode aloha_sim_insertion_scripted (first N_EP episodes) with V-JEPA 2.1:
mean(768) + spatial 8x8 (64*768, float16) + states(14) + actions(14) + epid.
This is the reach-to-object data (arm deliberately reaches/inserts the peg -> contact-rich,
vision-necessary). -> manip_cache.npz"""
import os, time, numpy as np, torch
os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
OUT = "/tmp/selfplay_probe"; dev = "cuda"; G = 8
N_EP = int(os.environ.get("N_EP", "30"))
ds = LeRobotDataset("lerobot/aloha_sim_insertion_scripted")
ei = np.array(ds.hf_dataset["episode_index"])
keep = np.where(ei < N_EP)[0]
print(f"encoding {len(keep)} frames from {N_EP} episodes", flush=True)
enc = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", trust_repo=True)[0].to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(dev)
def to_img(t):  # CHW float[0,1] -> HWC uint8 256
    a = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.array(Image.fromarray(a).resize((256, 256)))
imgs, states, actions, epid = [], [], [], []
t0 = time.time()
for j, i in enumerate(keep):
    it = ds[int(i)]
    imgs.append(to_img(it["observation.images.top"]))
    states.append(it["observation.state"].numpy()); actions.append(it["action"].numpy()); epid.append(int(ei[i]))
    if (j + 1) % 2000 == 0: print(f"  loaded {j+1}/{len(keep)} ({(j+1)/(time.time()-t0):.0f}/s)", flush=True)
imgs = np.stack(imgs); states = np.array(states, np.float32); actions = np.array(actions, np.float32); epid = np.array(epid)
print(f"loaded imgs {imgs.shape} in {time.time()-t0:.0f}s; encoding...", flush=True)
@torch.no_grad()
def encode(ims, bs=48):
    N = len(ims); M = np.zeros((N, 768), np.float32); S = np.zeros((N, G * G * 768), np.float16)
    for k in range(0, N, bs):
        chunk = ims[k:k + bs]
        x = torch.from_numpy(np.stack([np.array(Image.fromarray(im).resize((384, 384))) for im in chunk])).to(dev).permute(0, 3, 1, 2)[:, :, None].float() / 255.0
        x = (x - mean) / std
        h = enc(x).float(); b, nt, C = h.shape; g = int(round(nt ** 0.5)); hh = h.reshape(b, g, g, C)
        M[k:k + b] = hh.mean((1, 2)).cpu().numpy()
        S[k:k + b] = hh.reshape(b, G, g // G, G, g // G, C).amax(dim=(2, 4)).reshape(b, -1).cpu().numpy().astype(np.float16)
        if (k // bs) % 40 == 0: print(f"  enc {k}/{N}", flush=True)
    return M, S
t1 = time.time(); M, S = encode(imgs); print(f"encoded in {time.time()-t1:.0f}s -> M{M.shape} S{S.shape}", flush=True)
np.savez_compressed(OUT + "/manip_cache.npz", M=M, S=S, states=states, actions=actions, epid=epid)
print("[ok] manip_cache.npz", flush=True)
