"""Train embodiment e (free/ac/invdyn) on the MANIPULATION (insertion-scripted) mean features.
Contact-rich, deliberate motion -> tests if action is more recoverable here (invdyn val) than in
random play. Saves f_manip_{obj}.pt + e_cache_manip.npz. e on mean (768); spatial z handles objects."""
import os, numpy as np, torch, torch.nn as nn
from sp_lib import EmbEnc
torch.manual_seed(0); OUT = "/tmp/selfplay_probe"; D = 64; dev = "cuda"
d = np.load(OUT + "/manip_cache.npz")
M = d["M"].astype(np.float32); proprio = d["states"].astype(np.float32); epid = d["epid"]; IN = M.shape[1]
ne = int(epid.max()) + 1
# transitions within episode
pairs = []
for e in range(ne):
    idx = np.where(epid == e)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1: pairs.append((a, b, e))
pairs = np.array(pairs); ts, t1s, pep = pairs[:, 0], pairs[:, 1], pairs[:, 2]
tr = pep < ne - 4; va = ~tr
print(f"manip feats {M.shape} | {len(pairs)} transitions | train {tr.sum()} val {va.sum()}", flush=True)
dz = M[t1s] - M[ts]; d_act = proprio[t1s] - proprio[ts]
def stq(x, m): return x[m].mean(0), x[m].std(0) + 1e-6
zmu, zsd = stq(M[ts], tr); dzmu, dzsd = stq(dz, tr); amu, asd = stq(d_act, tr)
def T(x): return torch.tensor(x, device=dev)
ZT, ZT1, DZ, A = T(M[ts]), T(M[t1s]), T((dz - dzmu) / dzsd), T((d_act - amu) / asd)
trI, vaI = T(np.where(tr)[0].astype(np.int64)), T(np.where(va)[0].astype(np.int64))
def train(obj):
    torch.manual_seed(0)
    f = EmbEnc(IN, D).to(dev)
    with torch.no_grad(): f.zmu.copy_(T(zmu)); f.zsd.copy_(T(zsd))
    if obj == "invdyn": head = nn.Sequential(nn.Linear(2 * D, 512), nn.GELU(), nn.Linear(512, 14)).to(dev)
    else: head = nn.Sequential(nn.Linear(D + (14 if obj == "ac" else 0), 512), nn.GELU(), nn.Linear(512, IN)).to(dev)
    opt = torch.optim.Adam(list(f.parameters()) + list(head.parameters()), lr=1e-3)
    best, bad, bSD = 1e18, 0, None
    for epoch in range(200):
        f.train(); head.train(); perm = trI[torch.randperm(len(trI), device=dev)]
        for k in range(0, len(perm), 256):
            bi = perm[k:k + 256]
            if obj == "invdyn": loss = ((head(torch.cat([f(ZT[bi]), f(ZT1[bi])], 1)) - A[bi]) ** 2).mean()
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
        if vl < best - 1e-5: best, bad, bSD = vl, 0, {k: v.clone() for k, v in f.state_dict().items()}
        else: bad += 1
        if bad >= 12: break
    f.load_state_dict(bSD)
    torch.save({"state": f.state_dict(), "D": D, "in_dim": IN}, f"{OUT}/f_manip_{obj}.pt")
    with torch.no_grad(): e_all = f(T(M)).cpu().numpy()
    print(f"  [{obj}] val {best:.4f} (ep {epoch})", flush=True); return e_all
es = {o: train(o) for o in ["free", "ac", "invdyn"]}
np.savez_compressed(OUT + "/e_cache_manip.npz", e_free=es["free"], e_ac=es["ac"], e_invdyn=es["invdyn"])
print("[ok] f_manip_*.pt + e_cache_manip.npz", flush=True)
