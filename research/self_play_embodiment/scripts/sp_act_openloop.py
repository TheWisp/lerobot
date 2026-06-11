# ruff: noqa
"""Low-variance demo-efficiency signal: open-loop action-chunk L1 on HELD-OUT insertion episodes.
For encoder in {frozen, forward-pretrained}: train SmallACT on K episodes (3 seeds), measure L1 of the
predicted action chunk vs ground truth on a fixed held-out set. Lower = better demo-efficiency.
No env rollout -> clean signal (the closed-loop SR was floored)."""

import os, sys, numpy as np, torch, torch.nn as nn

os.environ.setdefault("MUJOCO_GL", "egl")
from transformers import Dinov2Model

OUT = "/tmp/selfplay_probe"  # nosec B108
C = OUT + "/cache"
dev = "cuda"
CHUNK = 16  # nosec B108
CKPT = sys.argv[1] if len(sys.argv) > 1 else OUT + "/real_fwd_ep2.pt"
m_ = np.load(C + "/meta.npz")
N = int(m_["N"])
RES = int(m_["RES"])
imgs = np.memmap(C + "/imgs.u8", mode="r", dtype=np.uint8, shape=(N, RES, RES, 3))
states = m_["states"].astype(np.float32)
actions = m_["actions"].astype(np.float32)
epid = m_["epid"]
taskid = m_["taskid"]
framepos = m_["framepos"]
ins_eps = np.unique(epid[taskid == 0])
amu, asd = actions[taskid == 0].mean(0), actions[taskid == 0].std(0) + 1e-6
smu, ssd = states[taskid == 0].mean(0), states[taskid == 0].std(0) + 1e-6
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


def load_enc(ckpt):
    m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(dev).eval()
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=dev))
    for p in m.parameters():
        p.requires_grad = False
    return m


def encsp(m, x255):
    x = (x255 / 255.0 - mean) / std
    with torch.no_grad():
        h = m(pixel_values=x).last_hidden_state[:, 1:]
        b = h.shape[0]
        g = int(round(h.shape[1] ** 0.5))
        return (
            h.reshape(b, g, g, 768).reshape(b, 8, g // 8, 8, g // 8, 768).amax(dim=(2, 4)).reshape(b, 64, 768)
        )


def encode_eps(m, eps, bs=64):
    frs = np.concatenate([np.where(epid == e)[0][np.argsort(framepos[np.where(epid == e)[0]])] for e in eps])
    fb = {}
    for k in range(0, len(frs), bs):
        ii = frs[k : k + bs]
        x = torch.from_numpy(np.ascontiguousarray(imgs[ii])).to(dev).permute(0, 3, 1, 2).float()
        f = encsp(m, x).cpu().numpy().astype(np.float16)
        for j, fr in enumerate(ii):
            fb[int(fr)] = f[j]
    return fb


class SmallACT(nn.Module):
    def __init__(s):
        super().__init__()
        s.pp = nn.Linear(768, 256)
        s.pos = nn.Parameter(torch.randn(1, 64, 256) * 0.02)
        s.sp = nn.Linear(14, 256)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 512, batch_first=True), 2)
        s.q = nn.Parameter(torch.randn(1, CHUNK, 256) * 0.02)
        s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(256, 4, 512, batch_first=True), 2)
        s.head = nn.Linear(256, 14)

    def forward(s, feat, state):
        tok = s.pp(feat.float()) + s.pos
        st = s.sp(state)[:, None]
        mem = s.enc(torch.cat([st, tok], 1))
        h = s.dec(s.q.expand(len(feat), -1, -1), mem)
        return s.head(h)


def samples(fb, eps):
    F, S, A = [], [], []
    for e in eps:
        fr = np.where(epid == e)[0]
        fr = fr[np.argsort(framepos[fr])]
        for j in range(len(fr)):
            ac = actions[fr[j : j + CHUNK]]
            if len(ac) < CHUNK:
                ac = np.vstack([ac, np.repeat(ac[-1:], CHUNK - len(ac), 0)])
            F.append(fb[int(fr[j])])
            S.append((states[fr[j]] - smu) / ssd)
            A.append((ac - amu) / asd)
    return (
        torch.tensor(np.stack(F), device=dev),
        torch.tensor(np.stack(S).astype(np.float32), device=dev),
        torch.tensor(np.stack(A).astype(np.float32), device=dev),
    )


def train_l1(fb, tr_eps, te_F, te_S, te_A, seed, iters=4000):
    torch.manual_seed(seed)
    Ff, Ss, Aa = samples(fb, tr_eps)
    net = SmallACT().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4)
    net.train()
    for it in range(iters):
        b = torch.randint(0, len(Ff), (64,), device=dev)
        loss = (net(Ff[b], Ss[b]) - Aa[b]).abs().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    net.eval()
    with torch.no_grad():
        l = 0.0
        for k in range(0, len(te_F), 256):
            l += (net(te_F[k : k + 256], te_S[k : k + 256]) - te_A[k : k + 256]).abs().sum().item()
    return l / (len(te_F) * CHUNK * 14)


rng = np.random.RandomState(0)
ins = ins_eps.copy()
rng.shuffle(ins)
held = ins[55:75]
pool = ins[:40]
KS = [5, 10, 20, 40]
print(f"=== open-loop demo-efficiency (held-out insertion L1, 3 seeds) | ckpt={CKPT} ===", flush=True)
res = {}
for tag, ckpt in [("frozen", None), ("pretrained", CKPT)]:
    m = load_enc(ckpt)
    fb = encode_eps(m, list(pool) + list(held))
    te_F, te_S, te_A = samples(fb, held)
    for K in KS:
        ls = [train_l1(fb, ins[:K], te_F, te_S, te_A, s) for s in range(3)]
        res[(tag, K)] = (np.mean(ls), np.std(ls))
        print(f"[{tag}] K={K:2d}: L1 {np.mean(ls):.4f} ± {np.std(ls):.4f}", flush=True)
print("\n=== SUMMARY (held-out action-chunk L1, lower=better) ===", flush=True)
for K in KS:
    f, p = res[("frozen", K)][0], res[("pretrained", K)][0]
    print(f"  K={K:2d}: frozen {f:.4f}  pretrained {p:.4f}  delta {100 * (f - p) / f:+.1f}%", flush=True)
