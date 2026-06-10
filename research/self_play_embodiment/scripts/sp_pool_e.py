"""Test the user's point: does training e on ALL same-embodiment data (random play + scripted
insertion, ~20k transitions) give a BETTER embodiment encoder than random-only? Metric =
gripper-xyz decode (held-out eps) on the our-sim random-play frames. random-only e_invdyn was 0.876.
Trains inverse-dynamics e on {random}, {scripted}, {pooled}; decodes on the same our-sim held-out set."""
import numpy as np, torch, torch.nn as nn
from sp_lib import EmbEnc
torch.manual_seed(0); OUT = "/tmp/selfplay_probe"; D = 64; dev = "cuda"
# random play (our-sim)
wb = np.load(OUT + "/world_buffer.npz", allow_pickle=True)
Mr = np.load(OUT + "/vj21_cache.npz")["M"].astype(np.float32); Pr = wb["states"].astype(np.float32)
GXr = wb["world"][:, 9:12].astype(np.float64)  # R gripper xyz (our-sim GT)
epr = wb["ep_bounds"]; ner = len(epr)
fe = np.zeros(len(Mr), int)
for eid, (s, e) in enumerate([tuple(map(int, b)) for b in epr]): fe[s:e] = eid
def pairs_eb(ebounds):
    P = []
    for s, e in [tuple(map(int, b)) for b in ebounds]:
        for t in range(s, e - 1): P.append((t, t + 1))
    return np.array(P)
pr_pairs = pairs_eb(epr)
# scripted insertion (dataset render)
md = np.load(OUT + "/manip_cache.npz"); Ms = md["M"].astype(np.float32); Ps = md["states"].astype(np.float32); eps_ = md["epid"]
ps_pairs = []
for e in range(int(eps_.max()) + 1):
    idx = np.where(eps_ == e)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1: ps_pairs.append((a, b))
ps_pairs = np.array(ps_pairs)
IN = Mr.shape[1]
def T(x): return torch.tensor(x, device=dev)
def make(Mz, Pz, prs):
    z0, z1 = Mz[prs[:, 0]], Mz[prs[:, 1]]; da = Pz[prs[:, 1]] - Pz[prs[:, 0]]; return z0, z1, da
def train_e(sources):  # list of (M,P,pairs); first source's stats used for norm
    z0 = np.concatenate([make(M, P, pr)[0] for M, P, pr in sources])
    z1 = np.concatenate([make(M, P, pr)[1] for M, P, pr in sources])
    da = np.concatenate([make(M, P, pr)[2] for M, P, pr in sources]).astype(np.float32)
    zmu, zsd = z0.mean(0), z0.std(0) + 1e-6; amu, asd = da.mean(0), da.std(0) + 1e-6
    Z0, Z1, A = T(z0), T(z1), T((da - amu) / asd); n = len(z0)
    f = EmbEnc(IN, D).to(dev)
    with torch.no_grad(): f.zmu.copy_(T(zmu)); f.zsd.copy_(T(zsd))
    head = nn.Sequential(nn.Linear(2 * D, 512), nn.GELU(), nn.Linear(512, 14)).to(dev)
    opt = torch.optim.Adam(list(f.parameters()) + list(head.parameters()), 1e-3)
    idx = np.arange(n); cut = int(n * 0.9)
    for epc in range(60):
        np.random.shuffle(idx); tr = idx[:cut]
        for k in range(0, len(tr), 256):
            bi = tr[k:k + 256]; loss = ((head(torch.cat([f(Z0[bi]), f(Z1[bi])], 1)) - A[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    f.eval(); return f
def r2(Y, P): return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)
def ridge(Xtr, Ytr, Xte, a=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xte @ np.linalg.solve(Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ (Ytr - Ytr.mean(0))) + Ytr.mean(0)
tr_m, te_m = fe < ner - 4, fe >= ner - 4
def decode(f):
    with torch.no_grad(): e = f(T(Mr)).cpu().numpy()  # e on our-sim random frames
    return r2(GXr[te_m], ridge(e[tr_m], GXr[tr_m], e[te_m]))
RP = (Mr, Pr, pr_pairs); SP = (Ms, Ps, ps_pairs)
print("R-gripper-xyz decode from e_invdyn (held-out our-sim eps); higher = richer embodiment:", flush=True)
print(f"  random-only   ({len(pr_pairs)} trans): R2 = {decode(train_e([RP])):.3f}", flush=True)
print(f"  scripted-only ({len(ps_pairs)} trans): R2 = {decode(train_e([SP])):.3f}", flush=True)
print(f"  POOLED        ({len(pr_pairs)+len(ps_pairs)} trans): R2 = {decode(train_e([RP, SP])):.3f}", flush=True)
