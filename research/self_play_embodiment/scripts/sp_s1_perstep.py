# ruff: noqa
"""Per-step z-gap on the stride-16 best ckpt: is z's signal diluted by teacher-forced context at later k?
gap@k=0 (predict s_t1 from s_t0 alone) is where z has the most room; gap should shrink with k."""

import sys, numpy as np, torch

sys.path.insert(0, "/tmp/selfplay_probe")  # nosec B108
from sp_vj_act import ACTJepa, Predictor, NLAT, T, STRIDE, C, dev

m_ = np.load(C + "/meta.npz")
N = int(m_["N"])
RES = int(m_["RES"])
imgs = np.memmap(C + "/imgs.u8", mode="r", dtype=np.uint8, shape=(N, RES, RES, 3))
vj = np.memmap(C + "/vj_feats.f16", mode="r", dtype=np.float16, shape=(N, 64, 768))
states = m_["states"].astype(np.float32)
epid = m_["epid"]
taskid = m_["taskid"]
framepos = m_["framepos"]
ck = torch.load("/tmp/selfplay_probe/s1/s1_6000.pt", map_location=dev, weights_only=False)  # nosec B108
smu, ssd, tmu, tsd = ck["smu"], ck["ssd"], ck["tmu"], ck["tsd"]
stn = ((states - smu) / ssd).astype(np.float32)
TL = np.zeros((N, 768), np.float32)
for k in range(0, N, 4096):
    TL[k : k + 4096] = np.asarray(vj[k : k + 4096]).mean(1)
TL = (TL - tmu) / tsd
TLt = torch.tensor(TL, device=dev)
model = ACTJepa().to(dev)
model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["model"].items()})
model.eval()
pred = Predictor(True).to(dev)
pred.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["pred"].items()})
pred.eval()
# twin is not saved -> retrain quickly? No: approximate no-z by zeroing z? Not equivalent. Instead compare per-step
# main loss vs per-step copy: the SHAPE tells dilution; and z-contribution via z-zero ablation (lower bound).
ins = np.unique(epid[taskid == 0])
tr_ = np.unique(epid[taskid == 1])
val = set(ins[-4:].tolist()) | set(tr_[-4:].tolist())
A = []
F = []
for e in sorted(val):
    fr = np.where(epid == e)[0]
    fr = fr[np.argsort(framepos[fr])]
    for j in range(0, len(fr) - STRIDE * T, 7):
        A.append(fr[j])
        F.append([fr[j + STRIDE * k] for k in range(1, T + 1)])
A = np.array(A)
F = np.array(F)
imn = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
isd = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
Lm = np.zeros(T)
Lz = np.zeros(T)
Cp = np.zeros(T)
n = 0
with torch.no_grad():
    for k in range(0, len(A), 128):
        ab = A[k : k + 128]
        fb = F[k : k + 128]
        x = torch.from_numpy(np.ascontiguousarray(imgs[ab])).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x = (x - imn) / isd
        mem = model.encode(x, torch.tensor(stn[ab], device=dev))
        z = mem[:, -NLAT:]
        S = torch.cat([TLt[ab][:, None], torch.stack([TLt[fb[:, j]] for j in range(T - 1)], 1)], 1)
        tgt = torch.stack([TLt[fb[:, j]] for j in range(T)], 1)
        Lm += ((pred(S, z) - tgt) ** 2).mean(-1).sum(0).cpu().numpy()
        Lz += (
            ((pred(S, torch.zeros_like(z)) - tgt) ** 2).mean(-1).sum(0).cpu().numpy()
        )  # z zeroed (lower-bound ablation)
        Cp += ((S - tgt) ** 2).mean(-1).sum(0).cpu().numpy()
        n += len(ab)
Lm /= n
Lz /= n
Cp /= n
print("k:        ", " ".join(f"{k:7d}" for k in range(T)))
print("copy/k:   ", " ".join(f"{v:7.3f}" for v in Cp))
print("main/k:   ", " ".join(f"{v:7.3f}" for v in Lm))
print("z-zero/k: ", " ".join(f"{v:7.3f}" for v in Lz))
print("zgain%/k: ", " ".join(f"{100 * (z - mn) / c:+6.1f}%" for mn, z, c in zip(Lm, Lz, Cp)))

# shuffle-z: input-dependence check (bias-tokens vs real per-sample info)
Ls = np.zeros(T)
n2 = 0
with torch.no_grad():
    for k in range(0, len(A), 128):
        ab = A[k : k + 128]
        fb = F[k : k + 128]
        x = torch.from_numpy(np.ascontiguousarray(imgs[ab])).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x = (x - imn) / isd
        mem = model.encode(x, torch.tensor(stn[ab], device=dev))
        z = mem[:, -NLAT:]
        zperm = z[torch.randperm(len(z), device=dev)]
        S = torch.cat([TLt[ab][:, None], torch.stack([TLt[fb[:, j]] for j in range(T - 1)], 1)], 1)
        tgt = torch.stack([TLt[fb[:, j]] for j in range(T)], 1)
        Ls += ((pred(S, zperm) - tgt) ** 2).mean(-1).sum(0).cpu().numpy()
        n2 += len(ab)
Ls /= n2
print("zshuf/k:  ", " ".join(f"{v:7.3f}" for v in Ls))
print("shufgain%/k:", " ".join(f"{100 * (s - mn) / c:+6.1f}%" for mn, s, c in zip(Lm, Ls, Cp)))
