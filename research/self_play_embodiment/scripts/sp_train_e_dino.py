"""Train embodiment encoder f: z -> e (D=64) on DINOv2 features, THREE objectives:
  free   : g(f(z_t))            -> dz        (action-FREE forward)
  ac     : g(f(z_t), d_t)       -> dz        (action-COND forward; Gate A analog of e_ac)
  invdyn : h(f(z_t), f(z_{t+1}))-> d_t        (INVERSE DYNAMICS; Gate B -- targets controllability)
d_t = proprio[t+1]-proprio[t] (achieved displacement), dz = z_{t+1}-z_t (DINOv2 latent change).
Saves f_dino_{tag}.pt (in_dim baked) + e_cache_dino.npz (e for all frames, each objective)."""
import os, numpy as np, torch, torch.nn as nn
from sp_lib import EmbEnc
torch.manual_seed(0); OUT = "/tmp/selfplay_probe"; D = 64; dev = "cuda"
SUB = os.environ.get("SUBSTRATE", "dino")
CACHE = {"dino": "dino_cache.npz", "vj21": "vj21_cache.npz"}[SUB]
M = np.load(OUT + "/" + CACHE)["M"].astype(np.float32); IN = M.shape[1]
print(f"SUBSTRATE={SUB} cache={CACHE}", flush=True)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True); proprio = wb["states"].astype(np.float32)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]; ne = len(ep)
pairs = np.array([(t, t + 1, eid) for eid, (s, e) in enumerate(ep) for t in range(s, e - 1)])
ts, t1s, pep = pairs[:, 0], pairs[:, 1], pairs[:, 2]
tr = ~np.isin(pep, list(range(ne - 4, ne))); va = ~tr
print(f"DINOv2 feats {M.shape} | {len(pairs)} transitions | train {tr.sum()} val {va.sum()}", flush=True)
dz = M[t1s] - M[ts]; d_act = proprio[t1s] - proprio[ts]; zt = M[ts]; zt1 = M[t1s]
def std(x, m): mu, sd = x[m].mean(0), x[m].std(0) + 1e-6; return mu, sd
zmu, zsd = std(zt, tr); dzmu, dzsd = std(dz, tr); amu, asd = std(d_act, tr)
def T(x): return torch.tensor(x, device=dev)
ZT, ZT1, DZ, A = T(zt), T(zt1), T((dz - dzmu) / dzsd), T((d_act - amu) / asd)
trI, vaI = T(np.where(tr)[0].astype(np.int64)), T(np.where(va)[0].astype(np.int64))

def train(obj):
    torch.manual_seed(0)
    f = EmbEnc(IN, D).to(dev)
    with torch.no_grad(): f.zmu.copy_(T(zmu)); f.zsd.copy_(T(zsd))
    if obj == "invdyn": head = nn.Sequential(nn.Linear(2 * D, 512), nn.GELU(), nn.Linear(512, 14)).to(dev); tgtdim = 14
    else: head = nn.Sequential(nn.Linear(D + (14 if obj == "ac" else 0), 512), nn.GELU(), nn.Linear(512, IN)).to(dev)
    opt = torch.optim.Adam(list(f.parameters()) + list(head.parameters()), lr=1e-3)
    best, bad, bestSD = 1e18, 0, None
    for epoch in range(200):
        f.train(); head.train(); perm = trI[torch.randperm(len(trI), device=dev)]
        for k in range(0, len(perm), 256):
            bi = perm[k:k + 256]
            if obj == "invdyn":
                pred = head(torch.cat([f(ZT[bi]), f(ZT1[bi])], 1)); loss = ((pred - A[bi]) ** 2).mean()
            else:
                gin = torch.cat([f(ZT[bi]), A[bi]], 1) if obj == "ac" else f(ZT[bi])
                loss = ((head(gin) - DZ[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        f.eval(); head.eval()
        with torch.no_grad():
            if obj == "invdyn": vl = ((head(torch.cat([f(ZT[vaI]), f(ZT1[vaI])], 1)) - A[vaI]) ** 2).mean().item()
            else:
                gin = torch.cat([f(ZT[vaI]), A[vaI]], 1) if obj == "ac" else f(ZT[vaI])
                vl = ((head(gin) - DZ[vaI]) ** 2).mean().item()
        if vl < best - 1e-5: best, bad, bestSD = vl, 0, {k: v.clone() for k, v in f.state_dict().items()}
        else: bad += 1
        if bad >= 12: break
    f.load_state_dict(bestSD)
    torch.save({"state": f.state_dict(), "D": D, "in_dim": IN}, f"{OUT}/f_{SUB}_{obj}.pt")
    with torch.no_grad(): e_all = f(T(M)).cpu().numpy()
    print(f"  [{obj}] best val {best:.4f} (ep {epoch}) | e std {e_all.std():.3f}", flush=True)
    return e_all

es = {obj: train(obj) for obj in ["free", "ac", "invdyn"]}
np.savez_compressed(OUT + f"/e_cache_{SUB}.npz", e_free=es["free"], e_ac=es["ac"], e_invdyn=es["invdyn"])
print(f"[ok] f_{SUB}_*.pt + e_cache_{SUB}.npz", flush=True)
