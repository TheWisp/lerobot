# ruff: noqa
"""Are the 14 joints recoverable NONLINEARLY from the frozen latent? (linear was R2~0.16)
Small MLP, held-out episodes, early stop. Decides: keep joint-reach (use MLP policy) vs
joints genuinely poorly encoded by vision (must give proprio / use a Cartesian goal)."""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
OUT = "/tmp/selfplay_probe"  # nosec B108
M = np.load(OUT + "/feat_cache.npz")["M"].astype(np.float32)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
J = wb["states"].astype(np.float32)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
ne = len(ep)
fe = np.zeros(len(M), int)
for eid, (s, e) in enumerate(ep):
    fe[s:e] = eid
tr, te = fe < ne - 4, fe >= ne - 4
dev = "cuda"
mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-6
Xtr = torch.tensor((M[tr] - mu) / sd, device=dev)
Xte = torch.tensor((M[te] - mu) / sd, device=dev)
Ytr = torch.tensor(J[tr], device=dev)
Yte = torch.tensor(J[te], device=dev)
net = nn.Sequential(nn.Linear(1408, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 14)).to(
    dev
)
opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


best = -9
bad = 0
for ep_i in range(300):
    net.train()
    perm = torch.randperm(len(Xtr), device=dev)
    for k in range(0, len(perm), 256):
        bi = perm[k : k + 256]
        loss = ((net(Xtr[bi]) - Ytr[bi]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        v = r2(Yte, net(Xte)).item()
    if v > best + 1e-4:
        best, bad = v, 0
    else:
        bad += 1
    if bad >= 15:
        break
print(f"MLP joint decode (held-out eps): best R2 = {best:.3f}  (linear ridge was 0.18)", flush=True)
