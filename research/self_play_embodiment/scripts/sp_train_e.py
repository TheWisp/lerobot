"""Train the embodiment encoder f: z -> e (bottleneck D=64), two variants that differ
ONLY in whether the body's achieved action is in the prediction objective:
  (b) e_free: g(f(z_t))          -> dz   (action-FREE)
  (c) e_ac:   g(f(z_t), d_t)     -> dz   (action-CONDITIONED; d_t = achieved displacement)
where dz = z_{t+1}-z_t (mean-pooled JEPA latent change), d_t = proprio_{t+1}-proprio_t.

e = f(z) adds NO information beyond z (deterministic fn) -- it concentrates the
action-relevant/affordance directions so a small policy can read them with fewer demos.
That is the whole bet; it can only show in the low-data regime.

Saves f_free.pt / f_ac.pt (with input-norm baked in) and e_cache.npz (e for all frames).
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
OUT = "/tmp/selfplay_probe"
D = 64

M = np.load(OUT + "/feat_cache.npz")["M"].astype(np.float32)  # (8000,1408) base z
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
proprio = wb["states"].astype(np.float32)  # (8000,14)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
N = len(M)
ne = len(ep)
pairs = np.array([(t, t + 1, eid) for eid, (s, e) in enumerate(ep) for t in range(s, e - 1)])
ts, t1s, pep = pairs[:, 0], pairs[:, 1], pairs[:, 2]
test_ep = set(range(ne - 4, ne))
tr = ~np.isin(pep, list(test_ep))
va = np.isin(pep, list(test_ep))
print(f"{N} frames, {len(pairs)} transitions, train {tr.sum()} val {va.sum()}", flush=True)

dz = M[t1s] - M[ts]  # target
d_act = proprio[t1s] - proprio[ts]  # achieved displacement
zt = M[ts]
# normalization stats (train only)
zmu, zsd = zt[tr].mean(0), zt[tr].std(0) + 1e-6
dzmu, dzsd = dz[tr].mean(0), dz[tr].std(0) + 1e-6
amu, asd = d_act[tr].mean(0), d_act[tr].std(0) + 1e-6
dev = "cuda"


def T(x):
    return torch.tensor(x, device=dev)


ZT = T((zt - zmu) / zsd)
DZ = T((dz - dzmu) / dzsd)
A = T((d_act - amu) / asd)
trI, vaI = T(np.where(tr)[0].astype(np.int64)), T(np.where(va)[0].astype(np.int64))


class EmbEnc(nn.Module):
    """z -> e, with input normalization baked in as buffers (so eval matches training)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("zmu", torch.tensor(zmu))
        self.register_buffer("zsd", torch.tensor(zsd))
        self.net = nn.Sequential(nn.Linear(1408, 256), nn.GELU(), nn.Linear(256, D))

    def forward(self, z_raw):  # z_raw: unnormalized base latent
        return self.net((z_raw - self.zmu) / self.zsd)


def train(use_action, tag):
    torch.manual_seed(0)
    f = EmbEnc().to(dev)
    g_in = D + (14 if use_action else 0)
    g = nn.Sequential(nn.Linear(g_in, 512), nn.GELU(), nn.Linear(512, 1408)).to(dev)
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), lr=1e-3)
    Zin = T(zt)  # raw; f normalizes internally
    best, best_state, patience = 1e9, None, 0
    for epoch in range(200):
        f.train()
        g.train()
        perm = trI[torch.randperm(len(trI), device=dev)]
        for k in range(0, len(perm), 256):
            bi = perm[k : k + 256]
            e = f(Zin[bi])
            gin = torch.cat([e, A[bi]], 1) if use_action else e
            loss = ((g(gin) - DZ[bi]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        f.eval()
        g.eval()
        with torch.no_grad():
            e = f(Zin[vaI])
            gin = torch.cat([e, A[vaI]], 1) if use_action else e
            vl = ((g(gin) - DZ[vaI]) ** 2).mean().item()
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in f.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 12:
            break
    f.load_state_dict(best_state)
    torch.save({"state": f.state_dict(), "D": D}, f"{OUT}/f_{tag}.pt")
    with torch.no_grad():
        e_all = f(T(M)).cpu().numpy()
    print(f"  [{tag}] best val MSE {best:.4f} (epoch {epoch}) | e std {e_all.std():.3f}", flush=True)
    return e_all


print("training e_free (action-FREE)...", flush=True)
e_free = train(False, "free")
print("training e_ac (action-CONDITIONED)...", flush=True)
e_ac = train(True, "ac")
np.savez_compressed(OUT + "/e_cache.npz", e_free=e_free, e_ac=e_ac)
print("[ok] saved f_free.pt, f_ac.pt, e_cache.npz", flush=True)
