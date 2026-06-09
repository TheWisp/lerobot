"""Shared harness for the embodiment-injection experiment.

Env: aloha insertion sim (position-setpoint action, identity to agent_pos).
We drive it with DELTA control: command = clip(proprio + delta, -1, 1). This is the
crux that makes the CURRENT-STATE representation matter -- the optimal per-step delta
is (goal - proprio), so a policy must read its current pose (from vision) to act. The
true proprio used to integrate the delta is the robot's own joint encoder (legitimate,
available on real robots); it is NEVER fed to the policy as an input.

Encoder: frozen V-JEPA2 ViT-g, mean-pooled last_hidden_state -> z in R^1408.
"""
import os, numpy as np, torch
os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image

REPO = "facebook/vjepa2-vitg-fpc64-256"
DEV, DT = "cuda", torch.bfloat16
IMG = 256
ALOHA_LOW, ALOHA_HIGH = -1.0, 1.0  # action_space box

# ---------------- env harness ----------------
def _walk(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items(): out.update(_walk(v, f"{p}{k}/"))
    else: out[p.rstrip("/")] = np.asarray(o)
    return out

class Vec:
    """Thin wrapper over the lerobot aloha vector env with clean (N,14) plumbing."""
    def __init__(self, n_envs):
        from lerobot.envs.configs import AlohaEnv
        self.cfg = AlohaEnv(obs_type="pixels_agent_pos", observation_height=480, observation_width=640)
        self.vec = self.cfg.create_envs(n_envs=n_envs, use_async_envs=False)[self.cfg.type][0]
        self.n = n_envs
        obs, _ = self.vec.reset(seed=[0] * n_envs)
        leaves = _walk(obs)
        self.imgk = next(k for k, v in leaves.items() if v.ndim >= 3 and v.shape[-1] == 3 and v.dtype == np.uint8)
        self.statek = next(k for k in leaves if k != self.imgk)

    GRIP_KEYS = ["vx300s_left/gripper_link", "vx300s_right/gripper_link"]
    def gripper_xyz(self):
        """(n,6) Cartesian [L-xyz, R-xyz] read from each sub-env's physics (eval metric)."""
        out = np.zeros((self.n, 6), np.float32)
        for i in range(self.n):
            phys = self.vec.envs[i].unwrapped._env._physics
            out[i] = np.concatenate([np.asarray(phys.named.data.xpos[k]) for k in self.GRIP_KEYS])
        return out

    def reset(self, seeds):
        seeds = list(map(int, seeds)); assert len(seeds) == self.n
        obs, _ = self.vec.reset(seed=seeds)
        return self._unpack(obs)

    def step(self, action):
        action = np.asarray(action, np.float32)
        assert action.shape == (self.n, 14), action.shape
        obs, r, term, trunc, info = self.vec.step(action)
        return self._unpack(obs), np.asarray(term), np.asarray(trunc)

    def _unpack(self, obs):
        leaves = _walk(obs)
        imgs = leaves[self.imgk]                         # (N,480,640,3) uint8
        proprio = leaves[self.statek].astype(np.float32) # (N,14)
        imgs256 = np.stack([np.asarray(Image.fromarray(imgs[i]).resize((IMG, IMG))) for i in range(self.n)])
        return imgs256, proprio

# ---------------- encoder ----------------
class Encoder:
    def __init__(self):
        from transformers import VJEPA2Model, AutoVideoProcessor
        self.model = VJEPA2Model.from_pretrained(REPO, torch_dtype=DT).to(DEV).eval()
        proc = AutoVideoProcessor.from_pretrained(REPO)
        self.mean = torch.tensor(proc.image_mean).view(1, 1, 3, 1, 1).to(DEV, DT)
        self.std = torch.tensor(proc.image_std).view(1, 1, 3, 1, 1).to(DEV, DT)

    @torch.no_grad()
    def encode(self, imgs256, bs=64):
        """imgs256: (N,256,256,3) uint8 -> (N,1408) float32 mean-pooled latent."""
        N = len(imgs256); out = np.zeros((N, 1408), np.float32)
        for k in range(0, N, bs):
            chunk = imgs256[k:k + bs]
            clips = np.stack([np.repeat(chunk[i:i + 1], 2, 0) for i in range(len(chunk))])  # Tc=2
            x = torch.from_numpy(clips).to(DEV).permute(0, 1, 4, 2, 3).to(DT) / 255.0
            x = (x - self.mean) / self.std
            o = self.model(pixel_values_videos=x)
            h = (o.last_hidden_state if getattr(o, "last_hidden_state", None) is not None else o[0]).float()
            out[k:k + len(chunk)] = h.mean(1).cpu().numpy()
        return out

# ---------------- DINOv2 encoder (Gate A substrate swap) ----------------
class DinoEncoder:
    """DINOv2-base: mean-pooled patch tokens -> 768-d (same interface as Encoder.encode)."""
    DIM = 768
    def __init__(self, rid="facebook/dinov2-base"):
        from transformers import Dinov2Model, AutoImageProcessor
        self.model = Dinov2Model.from_pretrained(rid).to(DEV).eval()
        self.proc = AutoImageProcessor.from_pretrained(rid)
    @torch.no_grad()
    def encode(self, imgs256, bs=64):
        N = len(imgs256); out = np.zeros((N, self.DIM), np.float32)
        for k in range(0, N, bs):
            x = self.proc(images=[Image.fromarray(im) for im in imgs256[k:k + bs]], return_tensors="pt").to(DEV)
            h = self.model(**x).last_hidden_state[:, 1:].float()  # drop CLS
            out[k:k + len(h)] = h.mean(1).cpu().numpy()
        return out

# ---------------- V-JEPA 2.1 encoder (Gate A/B substrate; dense 2D tokenizer) ----------------
class VJepa21Encoder:
    """V-JEPA 2.1 (torch.hub): single-frame 2D path, mean-pooled 24x24 patches -> 768-d."""
    DIM = 768
    def __init__(self, variant="vjepa2_1_vit_base_384"):
        self.enc = torch.hub.load("facebookresearch/vjepa2", variant, trust_repo=True)[0].to(DEV).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(DEV)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(DEV)
    @torch.no_grad()
    def encode(self, imgs256, bs=48):
        N = len(imgs256); out = np.zeros((N, self.DIM), np.float32)
        for k in range(0, N, bs):
            chunk = np.stack([np.array(Image.fromarray(im).resize((384, 384))) for im in imgs256[k:k + bs]])
            x = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2)[:, :, None].float() / 255.0
            x = (x - self.mean) / self.std
            h = self.enc(x).float()  # (b,576,768)
            out[k:k + len(h)] = h.mean(1).cpu().numpy()
        return out

# ---------------- embodiment encoder (loaded at eval to compute e_cur live) ----------------
import torch.nn as nn
class EmbEnc(nn.Module):
    """z -> e bottleneck, input-norm baked in as buffers. Structure must match the trainer."""
    def __init__(self, in_dim=1408, D=64):
        super().__init__()
        self.register_buffer("zmu", torch.zeros(in_dim))
        self.register_buffer("zsd", torch.ones(in_dim))
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, D))
    def forward(self, z_raw):
        return self.net((z_raw - self.zmu) / self.zsd)

def load_emb(path):
    ck = torch.load(path, map_location=DEV); f = EmbEnc(ck.get("in_dim", 1408), ck["D"]).to(DEV)
    f.load_state_dict(ck["state"]); f.eval()
    return f

# ---------------- delta control ----------------
def delta_command(proprio, delta, dmax=None):
    """command = clip(proprio + clip(delta, +-dmax), -1, 1)."""
    if dmax is not None:
        delta = np.clip(delta, -dmax, dmax)
    return np.clip(proprio + delta, ALOHA_LOW, ALOHA_HIGH).astype(np.float32)

# ---------------- success metric ----------------
def reach_err(proprio, goal_proprio):
    """per-env joint-space L2 distance (14-dim)."""
    return np.linalg.norm(proprio - goal_proprio, axis=-1)
