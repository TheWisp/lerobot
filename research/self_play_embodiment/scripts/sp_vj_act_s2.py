# ruff: noqa
"""VLA-JEPA mini-repro — Stage 2 [ROUND 2]: action fine-tune the ACT student.
[S3.3 E9] L = L_act + beta*L_WM on the SAME demo data; predictor continues from stage 1 [E3, spatial];
WM dropped at inference (<latent> toks remain; decoder cross-attends them = E6 analogue). E8 SWAPPED
for ACT L1+10*KL + temporal ensembling (Zhao'23) per the ACT-student decision.
ROUND-2 FIXES: (B) action stats from the K DEMO EPISODES ONLY (control also proprio); jepa arm proprio
stats = stage-1 ckpt provenance (its allowed corpus). (C) eval images via the SAME resize224 path as the
cache. Banner prints all provenance. beta default 0.1 (upstream world_model_loss_weight).
Usage: sp_vj_act_s2.py <ckpt|''> <K> <iters> <beta> <n_eval> <tag> [seed]"""

import os, sys, time, numpy as np, torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/tmp/selfplay_probe")  # nosec B108
from sp_vj_act import ACTJepa, Predictor, resize224, NLAT, T, STRIDE, CHUNK, NTOK
from sp_lib import Vec, _walk

C = "/tmp/selfplay_probe/cache"  # nosec B108
dev = "cuda"  # nosec B108
CKPT = sys.argv[1] if len(sys.argv) > 1 else ""
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
BETA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
NEVAL = int(sys.argv[5]) if len(sys.argv) > 5 else 24
TAG = sys.argv[6] if len(sys.argv) > 6 else "run"
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 1
EVAL_AT = (
    sorted(int(x) for x in sys.argv[8].split(",")) if len(sys.argv) > 8 and sys.argv[8] else []
)  # mid-train evals (SR-vs-steps; peak comparison per user)
TASK = sys.argv[9] if len(sys.argv) > 9 else "ins"  # "ins" | "cube" (universality probe)
TID = 0 if TASK == "ins" else 1
ENV_TASK = None if TASK == "ins" else "AlohaTransferCube-v0"
T_EVAL = 500
KL_W = 10.0
torch.manual_seed(SEED)
np.random.seed(SEED)
m_ = np.load(C + "/meta.npz")
N = int(m_["N"])
RES = int(m_["RES"])
imgs = np.memmap(C + "/imgs.u8", mode="r", dtype=np.uint8, shape=(N, RES, RES, 3))
vj = np.memmap(C + "/vj_feats.f16", mode="r", dtype=np.float16, shape=(N, NTOK, 768))
states = m_["states"].astype(np.float32)
actions = m_["actions"].astype(np.float32)
epid = m_["epid"]
taskid = m_["taskid"]
framepos = m_["framepos"]
task_eps = np.unique(epid[taskid == TID])
shuf = task_eps.copy()
np.random.RandomState(0).shuffle(shuf)
demo_eps = shuf[:K]  # SAME eps for all arms
demo_mask = np.isin(epid, demo_eps)
# --- ROUND-2 (B): stats provenance ---
amu, asd = actions[demo_mask].mean(0), actions[demo_mask].std(0) + 1e-6  # demo-only ALWAYS
if CKPT:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert ck.get("spatial"), "round-2 requires a spatial stage-1 ckpt (s1sp_*.pt)"
    smu, ssd, tmu, tsd = ck["smu"], ck["ssd"], ck["tmu"], ck["tsd"]
    sprov = "stage-1 ckpt (its corpus)"
else:
    smu, ssd = states[demo_mask].mean(0), states[demo_mask].std(0) + 1e-6
    tmu = tsd = None
    sprov = f"K={K} demos only"
stn = ((states - smu) / ssd).astype(np.float32)
frames, achunks, wm_pairs = [], [], []
for e in demo_eps:
    fr = np.where(epid == e)[0]
    fr = fr[np.argsort(framepos[fr])]
    for j in range(len(fr)):
        ac = actions[fr[j : j + CHUNK]]
        if len(ac) < CHUNK:
            ac = np.vstack([ac, np.repeat(ac[-1:], CHUNK - len(ac), 0)])
        frames.append(fr[j])
        achunks.append((ac - amu) / asd)
        if j + STRIDE * T < len(fr):
            wm_pairs.append([fr[j]] + [fr[j + STRIDE * k] for k in range(1, T + 1)])
frames = np.array(frames)
achunks = np.stack(achunks).astype(np.float32)
wm_pairs = np.array(wm_pairs)
print(
    f"=== Stage 2 R2 task={TASK} [{TAG}] ckpt={'CONTROL' if not CKPT else os.path.basename(CKPT)} K={K} beta={BETA if CKPT else 0.0} seed={SEED} iters={ITERS}"
    f"\n BANNER: demo eps={sorted(demo_eps.tolist())} | action-stats: K demos only | proprio-stats: {sprov} | eval path: resize224 (cache-identical)",
    flush=True,
)
model = ACTJepa().to(dev)
pred = None
if CKPT:
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    pred = Predictor(True).to(dev)
    pred.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["pred"].items()})
    TMU = torch.tensor(tmu, device=dev)
    TSD = torch.tensor(tsd, device=dev)

    def feats(idx):
        return (torch.tensor(np.asarray(vj[idx], dtype=np.float32), device=dev) - TMU) / TSD
else:
    BETA = 0.0
imn = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
isd_ = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


def prep(x_uint8):
    return (resize224(x_uint8).to(dev) - imn) / isd_  # ROUND-2 (C): one path


def closed_eval(step):
    """Closed-loop SR at a checkpoint (same protocol every call; WM dropped). Prints [RESULT TAG@step]."""
    vec = Vec(NEVAL, task=ENV_TASK)
    obs, _ = vec.vec.reset(seed=list(range(300000, 300000 + NEVAL)))

    def get(obs):
        L = _walk(obs)
        return L[vec.imgk], L[vec.statek].astype(np.float32)

    img, prop = get(obs)
    m_te = 0.01
    preds = np.full((T_EVAL, NEVAL, CHUNK, 14), np.nan, np.float32)
    maxr = np.zeros(NEVAL)
    for t in range(T_EVAL):
        st = torch.tensor((prop - smu) / ssd, dtype=torch.float32, device=dev)
        with torch.no_grad():
            mem = model.encode(prep(img), st)
            out, _, _ = model.act(mem, None)
            preds[t] = out.cpu().numpy() * asd + amu
        acts = np.zeros((NEVAL, 14))
        wsum = 0.0
        for rt in range(max(0, t - CHUNK + 1), t + 1):
            off = t - rt
            w = np.exp(-m_te * off)
            acts += w * preds[rt, :, off, :]
            wsum += w
        obs, r, term, trunc, info = vec.vec.step(np.clip((acts / wsum).astype(np.float32), -1, 1))
        maxr = np.maximum(maxr, np.asarray(r, float))
        img, prop = get(obs)
    hist = [int((np.round(maxr) == k).sum()) for k in range(5)]
    print(
        f"[RESULT {TAG}@{step}] K={K} beta={BETA} seed={SEED}: mean {maxr.mean():.2f} | SR>=1 {(maxr >= 1).mean() * 100:.0f}% | SR>=3 {(maxr >= 3).mean() * 100:.0f}% | SR=4 {(maxr >= 4).mean() * 100:.0f}% | hist {hist}",
        flush=True,
    )
    del vec


params = list(model.parameters()) + (list(pred.parameters()) if pred else [])
opt = torch.optim.AdamW(params, 1e-4)
t0 = time.time()
for it in range(ITERS):
    b = np.random.randint(0, len(frames), 48)
    fb = frames[b]
    mem = model.encode(prep(np.asarray(imgs[fb])), torch.tensor(stn[fb], device=dev))
    out, mu, lv = model.act(mem, torch.tensor(achunks[b], device=dev))
    l1 = (out - torch.tensor(achunks[b], device=dev)).abs().mean()
    kl = -0.5 * (1 + lv - mu**2 - lv.exp()).mean()
    loss = l1 + KL_W * kl
    if BETA > 0 and len(wm_pairs):
        wb = wm_pairs[np.random.randint(0, len(wm_pairs), 16)]
        mem2 = model.encode(prep(np.asarray(imgs[wb[:, 0]])), torch.tensor(stn[wb[:, 0]], device=dev))
        z = mem2[:, -NLAT:]
        S = torch.stack([feats(wb[:, j]) for j in range(T)], 1)  # [E3] GT spatial grids
        tgt = torch.stack([feats(wb[:, j]) for j in range(1, T + 1)], 1)
        loss = loss + BETA * ((pred(S, z) - tgt) ** 2).mean()  # [E9] + beta*L_WM [E5]
    loss.backward()
    opt.step()
    opt.zero_grad()
    if it % 3000 == 0:
        print(f"  it {it}: L1 {l1.item():.4f} KL {kl.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    if (it + 1) in EVAL_AT and (it + 1) < ITERS:
        model.eval()
        os.makedirs("/tmp/selfplay_probe/s2_models", exist_ok=True)  # nosec B108
        _sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
        torch.save(
            {"model": _sd, "smu": smu, "ssd": ssd, "amu": amu, "asd": asd, "task": TASK},
            f"/tmp/selfplay_probe/s2_models/{TAG}@{it + 1}.pt",  # nosec B108
        )  # nosec B108
        closed_eval(it + 1)
        model.train()
model.eval()
print(f"[trained] ({time.time() - t0:.0f}s) — closed-loop eval (WM dropped)", flush=True)
os.makedirs("/tmp/selfplay_probe/s2_models", exist_ok=True)  # nosec B108
sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
torch.save(
    {"model": sd, "smu": smu, "ssd": ssd, "amu": amu, "asd": asd, "task": TASK},
    f"/tmp/selfplay_probe/s2_models/{TAG}.pt",  # nosec B108
)  # nosec B108
print(f"[saved] s2_models/{TAG}.pt", flush=True)
closed_eval(ITERS)
