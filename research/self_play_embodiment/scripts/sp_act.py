# ruff: noqa
"""Small ACT on a FROZEN substrate — the 'is the network correct?' test, then the injection test.
Frozen V-JEPA2.1 8x8 patch tokens (64) + proprio (+ optional e token) -> 2-layer Transformer encoder
-> 30 learned action queries -> 2-layer decoder -> ACTION CHUNK (absolute joints), L1 loss.
Trained on scripted insertion demos (manip_cache); eval closed-loop in our insertion sim with
action chunking. Metric: min R-gripper-to-peg + insertion success (env is_success).
Env: USE_E=0/1 (inject embodiment token), N_EP (#demo episodes), CHUNK, EXEC."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sp_lib import Vec, VJepa21Encoder, load_emb

OUT = "/tmp/selfplay_probe"  # nosec B108
dev = "cuda"
USE_E = int(os.environ.get("USE_E", "0"))
N_EP = int(os.environ.get("N_EP", "30"))
CHUNK = int(os.environ.get("CHUNK", "30"))
EXEC = int(os.environ.get("EXEC", "30"))
NG, H = 20, 400
D = 256
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)
np.random.seed(SEED)


def log(m):
    print(m, flush=True)


d = np.load(OUT + "/manip_cache.npz")
S = d["S"].astype(np.float32).reshape(len(d["S"]), 64, 768)  # 8x8 V-JEPA2.1 patch tokens
M = d["M"].astype(np.float32)
states = d["states"].astype(np.float32)
actions = d["actions"].astype(np.float32)
epid = d["epid"]
_finv = load_emb(
    OUT + "/f_vj21_invdyn.pt"
)  # TASK-AGNOSTIC e (random-play inverse-dynamics); same f at train+eval
with torch.no_grad():
    E_INV = _finv(torch.tensor(M, device=dev)).cpu().numpy().astype(np.float32)
keep_ep = np.unique(epid)[:N_EP]
mask = np.isin(epid, keep_ep)
smu, ssd = states[mask].mean(0), states[mask].std(0) + 1e-6
amu, asd = actions[mask].mean(0), actions[mask].std(0) + 1e-6
# build full-chunk training samples within episodes
samples = []
for e in keep_ep:
    idx = np.where(epid == e)[0]
    for j in range(len(idx) - CHUNK):
        samples.append((idx[j], idx[j + 1 : j + 1 + CHUNK]))  # obs at t, action chunk t+1..t+CHUNK
ti = np.array([s[0] for s in samples])
ci = np.stack([s[1] for s in samples])
log(f"USE_E={USE_E} N_EP={N_EP} CHUNK={CHUNK} | {len(ti)} chunk-samples")


class SmallACT(nn.Module):
    def __init__(self, use_e):
        super().__init__()
        self.use_e = use_e
        self.patch_proj = nn.Linear(768, D)
        self.patch_pos = nn.Parameter(torch.randn(1, 64, D) * 0.02)
        self.state_proj = nn.Linear(14, D)
        if use_e:
            self.e_proj = nn.Linear(64, D)
        el = nn.TransformerEncoderLayer(D, 8, 4 * D, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(el, 2)
        self.query = nn.Parameter(torch.randn(1, CHUNK, D) * 0.02)
        dl = nn.TransformerDecoderLayer(D, 8, 4 * D, 0.1, batch_first=True)
        self.dec = nn.TransformerDecoder(dl, 2)
        self.head = nn.Linear(D, 14)

    def forward(self, patches, state, e=None):
        B = patches.shape[0]
        toks = [self.state_proj(state)[:, None], self.patch_proj(patches) + self.patch_pos]
        if self.use_e:
            toks = [self.e_proj(e)[:, None]] + toks
        ctx = self.enc(torch.cat(toks, 1))
        return self.head(self.dec(self.query.expand(B, -1, -1), ctx))  # [B,CHUNK,14]


def norm_s(x):
    return (x - smu) / ssd


def norm_a(x):
    return (x - amu) / asd


def denorm_a(x):
    return x * asd + amu


net = SmallACT(USE_E).to(dev)
opt = torch.optim.AdamW(net.parameters(), 1e-4, weight_decay=1e-4)
Spt = torch.tensor(S, device=dev)
St = torch.tensor(norm_s(states), device=dev)
At = torch.tensor(norm_a(actions), device=dev)
Et = torch.tensor(E_INV, device=dev)
n = len(ti)
idx = np.arange(n)
cut = int(n * 0.9)
log(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M; training...")
t0 = time.time()
best = 1e9
bad = 0
bSD = None
for ep in range(120):
    net.train()
    np.random.shuffle(idx)
    tr = idx[:cut]
    for k in range(0, len(tr), 128):
        b = tr[k : k + 128]
        patches = Spt[ti[b]]
        st = St[ti[b]]
        e = Et[ti[b]] if USE_E else None
        tgt = At[ci[b]]  # [B,CHUNK,14]
        pred = net(patches, st, e)
        loss = (pred - tgt).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        va = idx[cut:]
        vl = 0
        for k in range(0, len(va), 256):
            b = va[k : k + 256]
            e = Et[ti[b]] if USE_E else None
            vl += (net(Spt[ti[b]], St[ti[b]], e) - At[ci[b]]).abs().mean().item() * len(b)
        vl /= len(va)
    if vl < best - 1e-4:
        best, bad, bSD = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
    if bad >= 12:
        break
net.load_state_dict(bSD)
log(f"trained {ep} ep, val L1 {best:.4f}, {time.time() - t0:.0f}s")

# eval closed-loop with action chunking
enc = VJepa21Encoder()
F_INV = load_emb(OUT + "/f_vj21_invdyn.pt") if USE_E else None

VEC = Vec(NG)
_lf = VEC.vec.envs[0].unwrapped._env._physics.model.body("vx300s_right/left_finger_link").id
_rf = VEC.vec.envs[0].unwrapped._env._physics.model.body("vx300s_right/right_finger_link").id


def gc():
    o = np.zeros((NG, 3))
    for i in range(NG):
        p = VEC.vec.envs[i].unwrapped._env._physics
        o[i] = (np.asarray(p.data.xpos[_lf]) + np.asarray(p.data.xpos[_rf])) / 2
    return o


@torch.no_grad()
def rollout():
    img, prop = VEC.reset(range(40000, 40000 + NG))
    best = np.full(NG, 1e9)
    succ = np.zeros(NG, bool)
    t = 0
    while t < H:
        Mn, Sn = enc.encode_both(img, G=8)
        patches = torch.tensor(Sn.reshape(NG, 64, 768), device=dev)
        st = torch.tensor(norm_s(prop), device=dev)
        e = F_INV(torch.tensor(Mn, device=dev)) if USE_E else None
        chunk = denorm_a(net(patches, st, e).cpu().numpy())  # [NG,CHUNK,14]
        for j in range(min(EXEC, CHUNK)):
            (img, prop), term, trunc = VEC.step(chunk[:, j].astype(np.float32))
            best = np.minimum(best, np.linalg.norm(gc() - VEC.obj_xyz()[:, :3], axis=1))
            succ |= np.asarray(term)
            t += 1
            if t >= H:
                break
    return best, succ


b, s = rollout()
log(
    f"\n[USE_E={USE_E}] reach minDist {b.mean():.3f}m | SR@0.05={float((b < 0.05).mean()):.2f} @0.1={float((b < 0.1).mean()):.2f} | insert success {float(s.mean()):.2f}"
)
np.savez(OUT + f"/sp_act_e{USE_E}_n{N_EP}_s{SEED}.npz", best=b, succ=s)
