# ruff: noqa
"""Stage 3 on REAL data: forward/affordance pretraining (encoder-direct, DINOv2 top-2 blocks).
Predict next FROZEN features from (adapted z_t, action). Trains on all cached transitions EXCEPT a
held-out set of insertion episodes reserved for the gate. Per epoch -> (1) action-decodability on
held-out insertion (frozen vs adapted: >= frozen = genuine state-repr gain, not baking) (2) proprio
decode. Saves per-epoch checkpoints. Knee = pick by held-out action-decode, not forward loss."""

import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

os.environ.setdefault("MUJOCO_GL", "egl")
from transformers import Dinov2Model

OUT = "/tmp/selfplay_probe"  # nosec B108
C = OUT + "/cache"
dev = "cuda"
torch.manual_seed(0)
K = 2
EPOCHS = 5
PER_EP = 20000  # nosec B108
m_ = np.load(C + "/meta.npz")
N = int(m_["N"])
RES = int(m_["RES"])
imgs = np.memmap(C + "/imgs.u8", mode="r", dtype=np.uint8, shape=(N, RES, RES, 3))
states = m_["states"]
actions = m_["actions"].astype(np.float32)
epid = m_["epid"]
taskid = m_["taskid"]
# transitions: consecutive cache indices within an episode
pairs = np.where(epid[:-1] == epid[1:])[0]
trans = np.c_[pairs, pairs + 1]
# gate: held-out insertion episodes (excluded from forward training)
ins_eps = np.unique(epid[taskid == 0])
gate_eps = set(ins_eps[-10:].tolist())
is_gate_tr = np.array([epid[a] in gate_eps for a in trans[:, 0]])
train_trans = trans[~is_gate_tr]
gate_frames = np.where((taskid == 0) & np.array([e in gate_eps for e in epid]))[0]
print(
    f"N={N} | transitions={len(trans)} train={len(train_trans)} | gate insertion frames={len(gate_frames)}",
    flush=True,
)
A = actions
amu, asd = A[train_trans[:, 0]].mean(0), A[train_trans[:, 0]].std(0) + 1e-6
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


def prep(idxs):
    x = torch.from_numpy(np.ascontiguousarray(imgs[idxs])).to(dev).permute(0, 3, 1, 2).float() / 255.0
    return (x - mean) / std


def encsp(model, idxs, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = model(pixel_values=prep(idxs)).last_hidden_state[:, 1:]
        b = h.shape[0]
        g = int(round(h.shape[1] ** 0.5))
        return (
            h.reshape(b, g, g, 768).reshape(b, 8, g // 8, 8, g // 8, 768).amax(dim=(2, 4)).reshape(b, 64, 768)
        )


def encflat(model, idxs):
    out = np.zeros((len(idxs), 64 * 768), np.float32)
    for k in range(0, len(idxs), 48):
        ii = idxs[k : k + 48]
        with torch.no_grad():
            out[k : k + len(ii)] = encsp(model, ii, False).reshape(len(ii), -1).cpu().numpy()
    return out


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-9)


def ridge(Xt, Yt, Xe, a=10.0):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt, Xe = (Xt - mu) / sd, (Xe - mu) / sd
    return Xe @ np.linalg.solve(Xt.T @ Xt + a * np.eye(Xt.shape[1]), Xt.T @ (Yt - Yt.mean(0))) + Yt.mean(0)


def pca(Xt, Xe, k=200):
    mm = Xt.mean(0)
    _, _, Vt = np.linalg.svd(Xt - mm, full_matrices=False)
    c = Vt[:k]
    return (Xt - mm) @ c.T, (Xe - mm) @ c.T


def gate(model, tag):
    g = gate_frames[::2]
    ntr = len(g) * 7 // 10
    Z = encflat(model, g).astype(np.float64)
    tr, te = np.arange(len(g)) < ntr, np.arange(len(g)) >= ntr
    Pa, Pat = pca(Z[tr], Z[te])
    aR = r2(A[g][te], ridge(Pa, A[g][tr], Pat))  # action decode (held-out insertion)
    Ps, Pst = pca(Z[tr], Z[te])
    sR = r2(states[g][te], ridge(Ps, states[g][tr], Pst))  # proprio decode
    print(f"  [{tag}] action-decode R2={aR:.3f} | proprio-decode R2={sR:.3f}", flush=True)
    return aR


m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(dev)
for p in m.parameters():
    p.requires_grad = False
for blk in m.encoder.layer[-K:]:
    for p in blk.parameters():
        p.requires_grad = True
m.gradient_checkpointing_enable()
mf = Dinov2Model.from_pretrained("facebook/dinov2-base").to(dev).eval()
fwd = nn.Sequential(nn.Linear(768 + 14, 1024), nn.GELU(), nn.Linear(1024, 768)).to(dev)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad] + list(fwd.parameters()), 1e-5)
print("per-epoch gate (held-out insertion) — knee by action-decode generalization:", flush=True)
m.eval()
gate(mf, "frozen")
for epoch in range(EPOCHS):
    m.train()
    sel = train_trans[np.random.choice(len(train_trans), PER_EP, replace=False)]
    ls = []
    for k in range(0, len(sel), 16):
        b = sel[k : k + 16]
        z0 = encsp(m, b[:, 0], True)
        with torch.no_grad():
            tgt = encsp(mf, b[:, 1], False)
        a = torch.tensor((A[b[:, 0]] - amu) / asd, device=dev, dtype=torch.float32)[:, None, :].expand(
            -1, 64, -1
        )
        loss = F.mse_loss(fwd(torch.cat([z0, a], -1)), tgt)
        loss.backward()
        opt.step()
        opt.zero_grad()
        ls.append(loss.item())
    m.eval()
    print(f"epoch {epoch} L_fwd {np.mean(ls):.4f}", flush=True)
    gate(m, f"ep{epoch} ")
    torch.save(m.state_dict(), OUT + f"/real_fwd_ep{epoch}.pt")
print("[done]", flush=True)
