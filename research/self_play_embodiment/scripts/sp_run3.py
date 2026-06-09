"""Gate A + B: injection test on DINOv2 substrate, with an inverse-dynamics e condition.
Identical task/metric/HER/MLP/seeds to sp_run2 (V-JEPA) -- only the substrate + e-objectives change.
Conditions: oracle_xyz, bothprop, a[pr,zc,zg], noise, b_free, c_ac, c_invdyn (e on CURRENT).
Proof of injection would be c_ac or c_invdyn > b_free ~= noise ~= a in the low-demo regime."""
import os, time, numpy as np, torch, torch.nn as nn
from sp_lib import DinoEncoder, VJepa21Encoder, load_emb, delta_command
OUT = "/tmp/selfplay_probe"
KZ, DMAX, K_MAX, H = 200, 0.5, 30, 45
MODE = os.environ.get("MODE", "full")
DEMOS = [int(x) for x in os.environ.get("DEMOS", "100,200,500,1000").split(",")]
EPS = [0.05, 0.1, 0.2]; SEED = int(os.environ.get("SEED", "0"))
SUB = os.environ.get("SUBSTRATE", "dino")
CACHE = {"dino": "dino_cache.npz", "vj21": "vj21_cache.npz"}[SUB]
dev = "cuda"; torch.manual_seed(SEED); np.random.seed(SEED)
def log(m): print(m, flush=True)
def gerr(g, gg): return np.linalg.norm(g - gg, axis=-1)

M = np.load(OUT + "/" + CACHE)["M"].astype(np.float32)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
proprio = wb["states"].astype(np.float32); GXYZ = wb["world"][:, 6:12].astype(np.float32)
ep = [tuple(map(int, b)) for b in wb["ep_bounds"]]; ne = len(ep)
ec = np.load(OUT + f"/e_cache_{SUB}.npz")
E_FREE, E_AC, E_INV = ec["e_free"].astype(np.float32), ec["e_ac"].astype(np.float32), ec["e_invdyn"].astype(np.float32)
_grng = np.random.RandomState(SEED + 100)
_cand = np.concatenate([np.arange(s + 40, e - 1) for eid, (s, e) in enumerate(ep) if eid >= ne - 4])
_gi = _grng.choice(_cand, 32, replace=False)
G_GXYZ, G_PROP, G_Zraw = GXYZ[_gi].copy(), proprio[_gi].copy(), M[_gi].copy()
NG = len(G_GXYZ)
train_frames = np.concatenate([np.arange(s, e) for eid, (s, e) in enumerate(ep) if eid < ne - 4])
pmu = M[train_frames].mean(0); _, _, Vt = np.linalg.svd(M[train_frames] - pmu, full_matrices=False); PC = Vt[:KZ]
def pca(z): return (z - pmu) @ PC.T
ZG_PCA = pca(G_Zraw)

rng = np.random.RandomState(SEED); src = []
for eid, (s, e) in enumerate(ep):
    if eid >= ne - 4: continue
    for t in range(s, e - 1):
        k = rng.randint(1, min(K_MAX, e - 1 - t) + 1); src.append((t, t + k))
src = np.array(src); rng.shuffle(src); ti, gi = src[:, 0], src[:, 1]
NOISE = rng.randn(len(ti), 64).astype(np.float32)
LAB = np.clip(proprio[gi] - proprio[ti], -DMAX, DMAX).astype(np.float32)
ZC_d, ZG_d = pca(M[ti]), pca(M[gi])
log(f"SUBSTRATE={SUB} | demo pool {len(ti)} | SEED={SEED} | {NG} goals, H={H}")

def feats(name, P):
    pr, zc, zg = P["pr"], P["zc"], P["zg"]
    if name == "oracle_xyz": return np.hstack([pr, P["gg"]])
    if name == "bothprop":   return np.hstack([pr, P["gp"]])
    if name == "a":          return np.hstack([pr, zc, zg])
    if name == "noise":      return np.hstack([pr, zc, P["nz"], zg])
    if name == "b_free":     return np.hstack([pr, zc, P["ef"], zg])
    if name == "c_ac":       return np.hstack([pr, zc, P["ea"], zg])
    if name == "c_invdyn":   return np.hstack([pr, zc, P["ei"], zg])
    raise ValueError(name)

class MLP:
    def fit(self, X, Y):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-6; Xn = (X - self.xmu) / self.xsd
        n = len(X); cut = max(1, int(n * 0.85)); idx = np.random.permutation(n); tri, vai = idx[:cut], idx[cut:]
        Xt, Yt = torch.tensor(Xn[tri], device=dev), torch.tensor(Y[tri], device=dev)
        Xv, Yv = torch.tensor(Xn[vai], device=dev), torch.tensor(Y[vai], device=dev)
        self.net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 14)).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3, weight_decay=1e-4); best, bad, bSD = 1e18, 0, None
        for epoch in range(300):
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
        with torch.no_grad():
            return self.net(torch.tensor((X - self.xmu) / self.xsd, device=dev, dtype=torch.float32)).cpu().numpy()

def demo_feats(name, nd):
    i = np.arange(nd)
    P = dict(pr=proprio[ti[i]], zc=ZC_d[i], zg=ZG_d[i], nz=NOISE[i], gg=GXYZ[gi[i]], gp=proprio[gi[i]],
             ef=E_FREE[ti[i]], ea=E_AC[ti[i]], ei=E_INV[ti[i]])
    return feats(name, P)
def fit_policy(name, nd): return MLP().fit(demo_feats(name, nd), LAB[:nd])

enc = VJepa21Encoder() if SUB == "vj21" else DinoEncoder()
F_FREE, F_AC, F_INV = load_emb(OUT + f"/f_{SUB}_free.pt"), load_emb(OUT + f"/f_{SUB}_ac.pt"), load_emb(OUT + f"/f_{SUB}_invdyn.pt")
@torch.no_grad()
def emb(z, f): return f(torch.tensor(z, device=dev)).cpu().numpy()
from sp_lib import Vec
VEC = Vec(NG)
def rollout(name, pol):
    img, prop = VEC.reset(range(20000, 20000 + NG)); errs = np.full((H, NG), np.nan)
    for t in range(H):
        z = enc.encode(img)
        P = dict(pr=prop, zc=pca(z), zg=ZG_PCA, nz=np.random.randn(NG, 64).astype(np.float32),
                 gg=G_GXYZ, gp=G_PROP, ef=emb(z, F_FREE), ea=emb(z, F_AC), ei=emb(z, F_INV))
        (img, prop), _, _ = VEC.step(delta_command(prop, pol.predict(feats(name, P)), dmax=DMAX))
        errs[t] = gerr(VEC.gripper_xyz(), G_GXYZ)
    return errs[-5:].mean(0)
def row(tag, fe): return f"  {tag:10s} finalErr {fe.mean():.3f}m | " + " ".join(f"SR@{e}={float((fe<e).mean()):.2f}" for e in EPS)

conds = ["oracle_xyz", "bothprop", "a", "noise", "b_free", "c_ac", "c_invdyn"]
res = {}
log("\n=== DINOv2 injection sweep (Cartesian reach) ===")
for nd in DEMOS:
    log(f"\n--- #demos={nd} ---")
    for name in conds:
        t0 = time.time(); fe = rollout(name, fit_policy(name, nd)); res[(name, nd)] = fe
        log(row(name, fe) + f"  [{time.time()-t0:.0f}s]")
np.savez(OUT + f"/sp_{SUB}_results_s{SEED}.npz", **{f"{n}|{d}": v for (n, d), v in res.items()}, demos=np.array(DEMOS), eps=np.array(EPS))
log(f"\n[ok] sp_{SUB}_results_s{SEED}.npz")
