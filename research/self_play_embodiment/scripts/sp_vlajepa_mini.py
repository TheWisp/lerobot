# ruff: noqa
"""VLA-JEPA mini-repro: V-JEPA2.1 teacher -> ACT student, in aloha sim.
Frozen V-JEPA2.1 features (precomputed). Trainable shared encoder S + action head A (SmallACT).
WM-aux predictor P: (pooled S(f_t), action) -> predict V-JEPA2.1 FUTURE latent f_{t+H} (stop-grad).
Joint: L = L_action(K insertion demos) + beta*L_WM(ALL data). WM dropped at inference.
TEST: demo-efficiency, WM-taught (beta=1) vs action-only (beta=0). Held-out insertion L1, K sweep, 3 seeds."""

import os, numpy as np, torch, torch.nn as nn

os.environ.setdefault("MUJOCO_GL", "egl")
C = "/tmp/selfplay_probe/cache"  # nosec B108
dev = "cuda"
CHUNK = 16
H = 8  # nosec B108
m_ = np.load(C + "/meta.npz")
N = int(m_["N"])
vj = np.memmap(C + "/vj_feats.f16", mode="r", dtype=np.float16, shape=(N, 64, 768))
states = m_["states"].astype(np.float32)
actions = m_["actions"].astype(np.float32)
epid = m_["epid"]
taskid = m_["taskid"]
framepos = m_["framepos"]
ins_eps = np.unique(epid[taskid == 0])
amu, asd = actions[taskid == 0].mean(0), actions[taskid == 0].std(0) + 1e-6
smu, ssd = states[taskid == 0].mean(0), states[taskid == 0].std(0) + 1e-6
stn = ((states - smu) / ssd).astype(np.float32)
achunk_all = np.zeros((N, CHUNK, 14), np.float32)
for e in np.unique(epid):
    fr = np.where(epid == e)[0]
    fr = fr[np.argsort(framepos[fr])]
    for j in range(len(fr)):
        ac = actions[fr[j : j + CHUNK]]
        if len(ac) < CHUNK:
            ac = np.vstack([ac, np.repeat(ac[-1:], CHUNK - len(ac), 0)])
        achunk_all[fr[j]] = (ac - amu) / asd
samp = np.sort(np.random.RandomState(0).choice(N, 4000, replace=False))
vm = np.asarray(vj[samp]).mean(1)
VJM = torch.tensor(vm.mean(0).astype(np.float32), device=dev)
VJS = torch.tensor((vm.std(0) + 1e-6).astype(np.float32), device=dev)


def feat(idx):
    f = torch.tensor(np.asarray(vj[idx]), dtype=torch.float32, device=dev)
    return (f - VJM) / VJS


class Student(nn.Module):
    def __init__(s):
        super().__init__()
        s.pp = nn.Linear(768, 256)
        s.pos = nn.Parameter(torch.randn(1, 64, 256) * 0.02)
        s.sp = nn.Linear(14, 256)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 512, batch_first=True), 2)
        s.q = nn.Parameter(torch.randn(1, CHUNK, 256) * 0.02)
        s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(256, 4, 512, batch_first=True), 2)
        s.head = nn.Linear(256, 14)

    def shared(s, f, state):
        return s.enc(torch.cat([s.sp(state)[:, None], s.pp(f) + s.pos], 1))

    def act(s, mem):
        return s.head(s.dec(s.q.expand(len(mem), -1, -1), mem))


class WMHead(nn.Module):
    def __init__(s):
        super().__init__()
        s.ae = nn.Linear(14 * CHUNK, 256)
        s.net = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 768))

    def forward(s, pooled, ach):
        return s.net(torch.cat([pooled, s.ae(ach.reshape(len(ach), -1))], -1))


rng = np.random.RandomState(0)
ins = ins_eps.copy()
rng.shuffle(ins)
held = ins[55:75]
pool = ins[:40]
heldset = set(held.tolist())
held_frames = np.concatenate([np.where(epid == e)[0] for e in held])
WMp = []
for e in np.unique(epid):
    if e in heldset:
        continue
    fr = np.where(epid == e)[0]
    fr = fr[np.argsort(framepos[fr])]
    for j in range(len(fr) - H):
        WMp.append((fr[j], fr[j + H]))
WMp = np.array(WMp)
print(f"N={N} | WM pairs={len(WMp)} | held frames={len(held_frames)} | pool eps={len(pool)}", flush=True)


def run(eps, beta, seed, iters=4000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fa = np.concatenate([np.where(epid == e)[0] for e in eps])
    st = Student().to(dev)
    wm = WMHead().to(dev) if beta > 0 else None
    opt = torch.optim.AdamW(list(st.parameters()) + (list(wm.parameters()) if wm else []), 3e-4)
    st.train()
    for it in range(iters):
        b = fa[np.random.randint(0, len(fa), 64)]
        mem = st.shared(feat(b), torch.tensor(stn[b], device=dev))
        loss = (st.act(mem) - torch.tensor(achunk_all[b], device=dev)).abs().mean()
        if wm is not None:
            wb = WMp[np.random.randint(0, len(WMp), 64)]
            mem2 = st.shared(feat(wb[:, 0]), torch.tensor(stn[wb[:, 0]], device=dev))
            pred = wm(mem2.mean(1), torch.tensor(achunk_all[wb[:, 0]], device=dev))
            with torch.no_grad():
                tgt = feat(wb[:, 1]).mean(1)
            loss = loss + beta * ((pred - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    st.eval()
    with torch.no_grad():
        l = 0.0
        for k in range(0, len(held_frames), 256):
            b = held_frames[k : k + 256]
            mem = st.shared(feat(b), torch.tensor(stn[b], device=dev))
            l += (st.act(mem) - torch.tensor(achunk_all[b], device=dev)).abs().sum().item()
    return l / (len(held_frames) * CHUNK * 14)


print(
    "=== VLA-JEPA mini-repro: WM-taught vs action-only (held-out insertion L1, lower=better) ===", flush=True
)
res = {}
for K in [5, 10, 20, 40]:
    for tag, beta in [("action-only", 0.0), ("WM-taught ", 1.0)]:
        ls = [run(ins[:K], beta, s) for s in range(3)]
        res[(K, tag.strip())] = np.mean(ls)
        print(f"K={K:2d} {tag}: L1 {np.mean(ls):.4f} ± {np.std(ls):.4f}", flush=True)
print("\n=== SUMMARY (held-out insertion L1) ===", flush=True)
for K in [5, 10, 20, 40]:
    a, w = res[(K, "action-only")], res[(K, "WM-taught")]
    print(
        f"  K={K:2d}: action-only {a:.4f}  WM-taught {w:.4f}  delta {100 * (a - w) / a:+.1f}% (+=WM helps)",
        flush=True,
    )
