"""Reach-to-peg via BC of scripted insertion demos, eval closed-loop in our insertion sim.
VISION-NECESSARY (peg location only in image -> proprio-only must floor = cap broken, unlike
free-space reaching where bothprop=1.0). Tests if inverse-dynamics e injects on a task that
actually needs vision. Spatial z (object+gripper) + e on mean; delta control; metric = min
gripper-to-peg distance over the rollout.
Conditions: proprio_only / a[pr,zc] / noise / b_free / c_invdyn. SR vs #demos.
"""
import os, time, numpy as np, torch, torch.nn as nn
from sp_lib import Vec, VJepa21Encoder, load_emb, delta_command
OUT = "/tmp/selfplay_probe"; dev = "cuda"
KZ, DMAX, H, NG = 200, 0.5, 110, 24
MODE = os.environ.get("MODE", "full")
DEMOS = [int(x) for x in os.environ.get("DEMOS", "100,500,2000").split(",")]
EPS = [0.05, 0.10, 0.15]; SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED); np.random.seed(SEED)
def log(m): print(m, flush=True)

d = np.load(OUT + "/manip_cache.npz")
M = d["M"].astype(np.float32); S = d["S"].astype(np.float32); states = d["states"].astype(np.float32)
actions = d["actions"].astype(np.float32); epid = d["epid"]
ec = np.load(OUT + "/e_cache_manip.npz"); E_FREE, E_INV = ec["e_free"].astype(np.float32), ec["e_invdyn"].astype(np.float32)
ne = int(epid.max()) + 1
# spatial PCA on GPU (49152-d) — fit on training frames (all here; eval is in-sim, not held-out frames)
Sg = torch.tensor(S, device=dev); smean = Sg.mean(0)
_, _, V = torch.pca_lowrank(Sg - smean, q=KZ, niter=5); V = V[:, :KZ].contiguous()
ZC = ((Sg - smean) @ V).cpu().numpy(); del Sg; torch.cuda.empty_cache()
LAB = np.clip(actions - states, -DMAX, DMAX).astype(np.float32)  # commanded delta
log(f"manip BC | {len(M)} frames | SEED={SEED} DEMOS={DEMOS} | KZ={KZ} H={H} NG={NG}")

def feats(name, P):
    pr, zc = P["pr"], P["zc"]
    if name == "proprio_only": return pr
    if name == "a":            return np.hstack([pr, zc])
    if name == "noise":        return np.hstack([pr, zc, P["nz"]])
    if name == "b_free":       return np.hstack([pr, zc, P["ef"]])
    if name == "c_invdyn":     return np.hstack([pr, zc, P["ei"]])
    raise ValueError(name)

class MLP:
    def fit(self, X, Y):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-6; Xn = (X - self.xmu) / self.xsd
        n = len(X); cut = max(1, int(n * 0.85)); idx = np.random.permutation(n); tri, vai = idx[:cut], idx[cut:]
        Xt, Yt = torch.tensor(Xn[tri], device=dev), torch.tensor(Y[tri], device=dev)
        Xv, Yv = torch.tensor(Xn[vai], device=dev), torch.tensor(Y[vai], device=dev)
        self.net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 14)).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3, weight_decay=1e-4); best, bad, bSD = 1e18, 0, None
        for ep in range(300):
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

# fixed demo subset per seed
order = np.random.permutation(len(M))
def fit_policy(name, nd):
    idx = order[:nd]
    P = dict(pr=states[idx], zc=ZC[idx], nz=np.random.randn(nd, 64).astype(np.float32), ef=E_FREE[idx], ei=E_INV[idx])
    return MLP().fit(feats(name, P), LAB[idx])

enc = VJepa21Encoder()
F_FREE, F_INV = load_emb(OUT + "/f_manip_free.pt"), load_emb(OUT + "/f_manip_invdyn.pt")
Vt = V  # on GPU
@torch.no_grad()
def emb(m, f): return f(torch.tensor(m, device=dev)).cpu().numpy()
def spatial_pca(Sbatch):
    t = torch.tensor(Sbatch, device=dev); return ((t - smean) @ Vt).cpu().numpy()
VEC = Vec(NG)
def rollout(name, pol):
    img, prop = VEC.reset(range(30000, 30000 + NG))
    best = np.full(NG, 1e9)
    for t in range(H):
        Mn, Sn = enc.encode_both(img, G=8)
        P = dict(pr=prop, zc=spatial_pca(Sn), nz=np.random.randn(NG, 64).astype(np.float32),
                 ef=emb(Mn, F_FREE), ei=emb(Mn, F_INV))
        (img, prop), _, _ = VEC.step(delta_command(prop, pol.predict(feats(name, P)), dmax=DMAX))
        g = VEC.gripper_xyz(); peg = VEC.obj_xyz()[:, :3]
        dmin = np.minimum(np.linalg.norm(g[:, :3] - peg, axis=1), np.linalg.norm(g[:, 3:] - peg, axis=1))
        best = np.minimum(best, dmin)
    return best
def row(tag, b): return f"  {tag:13s} minDist {b.mean():.3f}m | " + " ".join(f"SR@{e}={float((b<e).mean()):.2f}" for e in EPS)

conds = ["proprio_only", "a", "noise", "b_free", "c_invdyn"]
res = {}
log("\n=== reach-to-peg (min gripper-to-peg over rollout) ===")
for nd in DEMOS:
    log(f"\n--- #demos={nd} ---")
    for name in conds:
        t0 = time.time(); b = rollout(name, fit_policy(name, nd)); res[(name, nd)] = b
        log(row(name, b) + f"  [{time.time()-t0:.0f}s]")
np.savez(OUT + f"/sp_manip_results_s{SEED}.npz", **{f"{n}|{d}": v for (n, d), v in res.items()}, demos=np.array(DEMOS), eps=np.array(EPS))
log(f"\n[ok] sp_manip_results_s{SEED}.npz")
