"""IN-SIM reach-to-peg injection test (matched domain, peg GT, VISION-NECESSARY).
Expert = global-Jacobian servo Δjoints=clip(J⁺·(peg−Rgrip),DMAX) (validated: GT ctrl SR@0.1=0.88).
BC it from vision; eval closed-loop in our sim; metric = min R-gripper-to-peg over rollout.
proprio_only must floor (peg only in image -> cap broken). e_invdyn = the Jacobian -> should inject.
Conditions: proprio_only / a[pr,zc] / noise / b_free / c_invdyn / oracle[pr,peg_xyz].
Data: world_buffer (random play, peg GT) + vj21 mean/spatial + e_cache_vj21.  SR vs #demos."""
import os, time, numpy as np, torch, torch.nn as nn
from sp_lib import Vec, VJepa21Encoder, load_emb, delta_command
OUT = "/tmp/selfplay_probe"; dev = "cuda"
KZ, DMAX, H, NG = 200, 0.10, 130, 24
DEMOS = [int(x) for x in os.environ.get("DEMOS", "50,200,1000").split(",")]
EPS = [0.05, 0.10, 0.15]; SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED); np.random.seed(SEED)
def log(m): print(m, flush=True)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
states = wb["states"].astype(np.float32); W = wb["world"].astype(np.float64); ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]
peg = W[:, 0:3]; Rg = W[:, 9:12]
M = np.load(OUT + "/vj21_cache.npz")["M"].astype(np.float32)
S = np.load(OUT + "/vj21_spatial_g8.npz")["S"].astype(np.float32)
ec = np.load(OUT + "/e_cache_vj21.npz"); E_FREE, E_INV = ec["e_free"].astype(np.float32), ec["e_invdyn"].astype(np.float32)
# fit global Jacobian J⁺ (R gripper) from play; expert labels
dj, dg = [], []
for s, e in ep:
    dj.append(states[s + 1:e].astype(np.float64) - states[s:e - 1]); dg.append(Rg[s + 1:e] - Rg[s:e - 1])
dj = np.concatenate(dj); dg = np.concatenate(dg)
B, *_ = np.linalg.lstsq(dj, dg, rcond=None); Jpinv = np.linalg.pinv(B.T)  # (14,3)
LAB = np.clip((peg - Rg) @ Jpinv.T, -DMAX, DMAX).astype(np.float32)  # expert Δjoint toward peg
# spatial PCA (GPU)
Sg = torch.tensor(S, device=dev); smean = Sg.mean(0); _, _, Vv = torch.pca_lowrank(Sg - smean, q=KZ, niter=5); Vv = Vv[:, :KZ].contiguous()
ZC = ((Sg - smean) @ Vv).cpu().numpy(); del Sg; torch.cuda.empty_cache()
log(f"in-sim reach-to-peg | {len(M)} frames | SEED={SEED} DEMOS={DEMOS} | KZ={KZ} H={H}")

def feats(name, P):
    pr, zc = P["pr"], P["zc"]
    if name == "proprio_only": return pr
    if name == "a":            return np.hstack([pr, zc])
    if name == "noise":        return np.hstack([pr, zc, P["nz"]])
    if name == "b_free":       return np.hstack([pr, zc, P["ef"]])
    if name == "c_invdyn":     return np.hstack([pr, zc, P["ei"]])
    if name == "oracle":       return np.hstack([pr, P["pg"]])
    raise ValueError(name)
class MLP:
    def fit(self, X, Y):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-6; Xn = (X - self.xmu) / self.xsd
        n = len(X); cut = max(1, int(n * 0.85)); idx = np.random.permutation(n); tri, vai = idx[:cut], idx[cut:]
        Xt, Yt = torch.tensor(Xn[tri], device=dev), torch.tensor(Y[tri], device=dev)
        Xv, Yv = torch.tensor(Xn[vai], device=dev), torch.tensor(Y[vai], device=dev)
        self.net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 14)).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3, weight_decay=1e-4); best, bad, bSD = 1e18, 0, None
        for epc in range(300):
            self.net.train(); perm = torch.randperm(len(Xt), device=dev)
            for k in range(0, len(perm), 256):
                bi = perm[k:k + 256]; loss = ((self.net(Xt[bi]) - Yt[bi]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad(): v = ((self.net(Xv) - Yv) ** 2).mean().item() if len(vai) else loss.item()
            if v < best - 1e-5: best, bad, bSD = v, 0, {k: x.clone() for k, x in self.net.state_dict().items()}
            else: bad += 1
            if bad >= 15: break
        self.net.load_state_dict(bSD); return self
    def predict(self, X):
        self.net.eval()
        with torch.no_grad(): return self.net(torch.tensor((X - self.xmu) / self.xsd, device=dev, dtype=torch.float32)).cpu().numpy()
order = np.random.permutation(len(M))
def fit_policy(name, nd):
    i = order[:nd]
    P = dict(pr=states[i], zc=ZC[i], nz=np.random.randn(nd, 64).astype(np.float32), ef=E_FREE[i], ei=E_INV[i], pg=peg[i].astype(np.float32))
    return MLP().fit(feats(name, P), LAB[i])
enc = VJepa21Encoder(); F_FREE, F_INV = load_emb(OUT + "/f_vj21_free.pt"), load_emb(OUT + "/f_vj21_invdyn.pt")
@torch.no_grad()
def emb(m, f): return f(torch.tensor(m, device=dev)).cpu().numpy()
def spatial_pca(Sb): t = torch.tensor(Sb, device=dev); return ((t - smean) @ Vv).cpu().numpy()
VEC = Vec(NG)
def rollout(name, pol):
    img, prop = VEC.reset(range(40000, 40000 + NG)); best = np.full(NG, 1e9)
    for t in range(H):
        Mn, Sn = enc.encode_both(img, G=8)
        P = dict(pr=prop, zc=spatial_pca(Sn), nz=np.random.randn(NG, 64).astype(np.float32),
                 ef=emb(Mn, F_FREE), ei=emb(Mn, F_INV), pg=VEC.obj_xyz()[:, :3])
        (img, prop), _, _ = VEC.step(delta_command(prop, pol.predict(feats(name, P)), dmax=DMAX))
        best = np.minimum(best, np.linalg.norm(VEC.gripper_xyz()[:, 3:] - VEC.obj_xyz()[:, :3], axis=1))
    return best
def row(tag, b): return f"  {tag:13s} minDist {b.mean():.3f}m | " + " ".join(f"SR@{e}={float((b<e).mean()):.2f}" for e in EPS)
conds = ["proprio_only", "a", "noise", "b_free", "c_invdyn", "oracle"]
res = {}
log("\n=== in-sim reach-to-peg (min R-gripper-to-peg) ===")
for nd in DEMOS:
    log(f"\n--- #demos={nd} ---")
    for name in conds:
        t0 = time.time(); b = rollout(name, fit_policy(name, nd)); res[(name, nd)] = b
        log(row(name, b) + f"  [{time.time()-t0:.0f}s]")
np.savez(OUT + f"/sp_reach_insim_s{SEED}.npz", **{f"{n}|{d}": v for (n, d), v in res.items()}, demos=np.array(DEMOS), eps=np.array(EPS))
log(f"\n[ok] sp_reach_insim_s{SEED}.npz")
