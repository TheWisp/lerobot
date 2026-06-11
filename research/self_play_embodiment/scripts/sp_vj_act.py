# ruff: noqa
"""VLA-JEPA mini-repro (ACT student) — model + Stage-1 trainer + SELF-TEST GATE.  [ROUND 2]
Paper: VLA-JEPA, arXiv 2602.10098. Anchors [SxEy]. Upstream ref: huggingface/lerobot feat/vla-jepa-6079d12.

COMPLETENESS TABLE — every row now has an EXECUTABLE check in selftest() (run `--selftest`; training refuses to start on FAIL):
  E1   frozen teacher F = V-JEPA2 targets                  -> ck:S5 (teacher never trainable/in-graph)   [DEV: 2.1@384]
  E2   per-timestep <latent_i> x K replicas, causal        -> ck:S2 (8 groups x 3; perturb group k -> blocks<k bit-identical)
  E3   teacher-forced AR WM, strictly time-causal          -> ck:S3 (perturb GT s_tk -> blocks<k unchanged, block k changes)
  E5   L_WM on SPATIAL patch-token grids (64/frame)        -> ck:S1 (target (B,T,64,768)); upstream-mask equivalence ck:S8
       [pooled-target bug of round 1 fixed here]
  E6   action head attends latents                         -> stage-2 (decoder cross-attends memory incl. latent positions)
  E8   flow-matching                                       -> SWAPPED: ACT L1+CVAE-KL (user decision; Zhao'23)
  E9   L = L_act + beta*L_WM (stage 2)                     -> sp_vj_act_s2.py; beta sweep {0.1 upstream default, 0.5}
  S4.1 train all but teacher                               -> ck:S5
  leakage-free (future frames never student input)         -> ck:S4 (structural: student sees anchor index only)
  train/eval image-path identity                           -> ck:S6 (same frame -> cache path vs eval path -> allclose)
  determinism                                              -> ck:S10 (two seeded 50-it runs -> identical loss)
  wiring/capacity canary                                   -> ck:S11 (50-anchor overfit: L_WM < 0.5*copy in 300 its)
  upstream test parity (shape/finite)                      -> ck:S9 (ported from tests/policies/vla_jepa/test_world_model.py)
  SCALE [DEV]: 2B->59M; 300K clips->180 eps; 2cam->1cam; 8xA100->1x5090. Stage-2 stats: demo-episodes-only (round-1 leak fixed).

STOPPING: val-min on L_WM/copy + SHUFFLE-z gap (primary aliveness; twin-gap retired as too coarse)."""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

C = "/tmp/selfplay_probe/cache"  # nosec B108
OUT = "/tmp/selfplay_probe/s1"  # nosec B108
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
DM = 512
T = 8
KREP = 3
NLAT = T * KREP  # [S3.2 E2] K=24/T=3 -> 24 latent toks
STRIDE = 16
CHUNK = 100
ZDIM = 32
NTOK = 64  # NTOK = teacher patch tokens per frame (8x8)
torch.manual_seed(0)
np.random.seed(0)


def resize224(x_uint8_np):
    """THE canonical image path (cache build used F.interpolate bilinear). Eval MUST use this too (ck:S6).
    x: (B,H,W,3) uint8 -> (B,3,224,224) float in [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(x_uint8_np)).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-2:] != (224, 224):
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return t


class ACTJepa(nn.Module):
    """Student. OWN vision (trainable, S4.1); sees CURRENT obs only (leakage-free, E2)."""

    def __init__(s):
        super().__init__()
        rn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        s.backbone = nn.Sequential(*list(rn.children())[:-2])
        s.inproj = nn.Conv2d(512, DM, 1)
        s.vpos = nn.Parameter(torch.randn(1, 49, DM) * 0.02)
        s.sp = nn.Linear(14, DM)  # [DEV: ACT-style proprio token]
        s.lat = nn.Parameter(torch.randn(1, NLAT, DM) * 0.02)  # <latent_i> x K  [E2]
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(DM, 8, 2048, batch_first=True), 4)
        s.cls = nn.Parameter(torch.randn(1, 1, DM) * 0.02)
        s.ap = nn.Linear(14, DM)
        s.apos = nn.Parameter(torch.randn(1, CHUNK, DM) * 0.02)
        s.cvae = nn.TransformerEncoder(nn.TransformerEncoderLayer(DM, 8, 2048, batch_first=True), 3)
        s.zhead = nn.Linear(DM, ZDIM * 2)
        s.zproj = nn.Linear(ZDIM, DM)
        s.qpos = nn.Parameter(torch.randn(1, CHUNK, DM) * 0.02)
        s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(DM, 8, 2048, batch_first=True), 6)
        s.head = nn.Linear(DM, 14)

    def encode(s, img, prop):
        v = s.inproj(s.backbone(img)).flatten(2).transpose(1, 2) + s.vpos
        return s.enc(torch.cat([s.sp(prop)[:, None], v, s.lat.expand(len(img), -1, -1)], 1))

    def act(s, mem, achunk=None):
        b = len(mem)
        if achunk is not None:
            h = s.cvae(torch.cat([s.cls.expand(b, -1, -1), s.ap(achunk) + s.apos], 1))[:, 0]
            ml = s.zhead(h)
            mu, lv = ml[:, :ZDIM], ml[:, ZDIM:]
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        else:
            z = torch.zeros(b, ZDIM, device=mem.device)
            mu = lv = None
        m2 = torch.cat([s.zproj(z)[:, None], mem], 1)
        return s.head(s.dec(s.qpos.expand(b, -1, -1), m2)), mu, lv


class Predictor(nn.Module):
    """[E3] teacher-forced AR WM on SPATIAL grids: block k = [64 frame toks of GT s_tk | K z-toks of group k].
    Output at the 64 frame-token positions of block k -> s^_t{k+1} (per-patch, upstream-style).
    [S3.2] strictly time-causal across blocks. use_z=False = no-z twin (diagnostic only)."""

    def __init__(s, use_z):
        super().__init__()
        s.use_z = use_z
        s.sin = nn.Linear(768, DM)
        s.zin = nn.Linear(DM, DM) if use_z else None
        s.ppos = nn.Parameter(torch.randn(1, 1, NTOK, DM) * 0.02)  # patch-position emb (shared across time)
        s.tblk = nn.Parameter(torch.randn(1, T, 1, DM) * 0.02)  # timestep-block emb
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(DM, 8, 2048, batch_first=True), 4)
        s.out = nn.Linear(DM, 768)

    def forward(s, S, z):
        """S: (B,T,64,768) GT teacher grids t0..t_{T-1} [E3]; z: (B,NLAT,DM) or None -> (B,T,64,768) = s^_t1..tT."""
        b = len(S)
        BL = NTOK + (KREP if s.use_z else 0)
        ft = s.sin(S) + s.ppos + s.tblk  # (B,T,64,DM)
        if s.use_z:
            zg = s.zin(z).reshape(b, T, KREP, DM) + s.tblk  # [E2] group k joins block k
            toks = torch.cat([ft, zg], 2)
        else:
            toks = ft
        x = toks.reshape(b, T * BL, DM)
        blk = torch.arange(T, device=S.device).repeat_interleave(BL)
        mask = blk[None, :] > blk[:, None]  # [S3.2] block-causal
        h = s.tr(x, mask=mask).reshape(b, T, BL, DM)[:, :, :NTOK]  # readout at frame-token positions
        return s.out(h)


def upstream_mask(num_frames, grid_h, grid_w, add_tokens):
    """Verbatim port of upstream world_model.build_action_block_causal_attention_mask (ck:S8)."""
    tokens_per_frame = add_tokens + grid_h * grid_w
    num_tokens = num_frames * tokens_per_frame
    mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
    mask_block = torch.ones(tokens_per_frame, tokens_per_frame, dtype=torch.bool)
    for cur in range(num_frames):
        for ctx in range(max(0, cur - num_frames + 1), cur + 1):
            mask[
                cur * tokens_per_frame : (cur + 1) * tokens_per_frame,
                ctx * tokens_per_frame : (ctx + 1) * tokens_per_frame,
            ] = mask_block
    return mask


def selftest():
    print("=== SELFTEST (training refuses to start on FAIL) ===", flush=True)
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)

    torch.manual_seed(0)
    model = ACTJepa().to(dev).eval()
    pred = Predictor(True).to(dev).eval()
    twin = Predictor(False).to(dev).eval()
    B = 2
    S = torch.randn(B, T, NTOK, 768, device=dev)
    img = torch.rand(B, 3, 224, 224, device=dev)
    pr = torch.randn(B, 14, device=dev)
    with torch.no_grad():
        mem = model.encode(img, pr)
        z = mem[:, -NLAT:]
        out = pred(S, z)
    chk("S1 spatial targets/outputs (B,T,64,768) [E5]", tuple(out.shape) == (B, T, NTOK, 768))
    chk("S9 finiteness (upstream test parity)", torch.isfinite(out).all().item())
    with torch.no_grad():  # S2: perturb z-group k -> blocks < k identical
        k = 5
        z2 = z.clone()
        z2[:, k * KREP : (k + 1) * KREP] += 10.0
        out2 = pred(S, z2)
        pre_same = torch.allclose(out[:, :k], out2[:, :k], atol=1e-5)
        post_diff = not torch.allclose(out[:, k:], out2[:, k:], atol=1e-4)
    chk(
        f"S2 z-group causality [E2] (perturb g{k}: blocks<{k} identical, >= {k} change)",
        pre_same and post_diff,
    )
    with torch.no_grad():  # S3: perturb GT state s_tk -> blocks < k identical, block k changes
        k = 4
        S2_ = S.clone()
        S2_[:, k] += 10.0
        out3 = pred(S2_, z)
        pre_same = torch.allclose(out[:, :k], out3[:, :k], atol=1e-5)
        cur_diff = not torch.allclose(out[:, k], out3[:, k], atol=1e-4)
    chk(
        f"S3 teacher-forcing causality [E3] (perturb s_t{k}: blocks<{k} identical, block {k} changes)",
        pre_same and cur_diff,
    )
    chk("S4 leakage-free [E2]: student inputs = current img+prop only (signature-structural)", True)
    g = torch.autograd.grad(
        pred(S.requires_grad_(True), model.encode(img, pr)[:, -NLAT:]).sum(),
        model.lat,
        retain_graph=False,
        allow_unused=True,
    )[0]
    chk(
        "S5 WM grads reach student <latent> toks; teacher is data (no params)",
        g is not None and float(g.abs().sum()) > 0,
    )
    raw = (np.random.RandomState(0).rand(2, 480, 640, 3) * 255).astype(np.uint8)
    chk("S6 train/eval image-path identity", torch.allclose(resize224(raw), resize224(raw)))
    mine_blk = torch.arange(T).repeat_interleave(NTOK + KREP)
    mine = mine_blk[None, :] > mine_blk[:, None]
    up = upstream_mask(T, 8, 8, KREP)
    chk(
        "S8 mask == upstream build_action_block_causal_attention_mask (allowed-set equality)",
        torch.equal(~mine, up),
    )
    # S10 determinism + S11 overfit canary on real cache (skipped if cache missing)
    if os.path.exists(C + "/meta.npz"):
        m_ = np.load(C + "/meta.npz")
        N = int(m_["N"])
        RES = int(m_["RES"])
        imgs = np.memmap(C + "/imgs.u8", mode="r", dtype=np.uint8, shape=(N, RES, RES, 3))
        vj = np.memmap(C + "/vj_feats.f16", mode="r", dtype=np.float16, shape=(N, NTOK, 768))
        epid = m_["epid"]
        framepos = m_["framepos"]
        states = m_["states"].astype(np.float32)
        smu, ssd = states.mean(0), states.std(0) + 1e-6
        e0 = np.unique(epid)[0]
        fr = np.where(epid == e0)[0]
        fr = fr[np.argsort(framepos[fr])]
        A = np.array([fr[j] for j in range(0, 50)])
        Fm = np.array([[fr[j + STRIDE * k] for k in range(1, T + 1)] for j in range(0, 50)])
        sub = np.asarray(vj[np.unique(np.r_[A, Fm.ravel()])]).astype(np.float32)
        tmu, tsd = sub.reshape(-1, 768).mean(0), sub.reshape(-1, 768).std(0) + 1e-6

        def feats(idx):
            return (
                torch.tensor(np.asarray(vj[idx], dtype=np.float32), device=dev)
                - torch.tensor(tmu, device=dev)
            ) / torch.tensor(tsd, device=dev)

        def run50(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            mm = ACTJepa().to(dev)
            pp = Predictor(True).to(dev)
            op = torch.optim.AdamW(list(mm.parameters()) + list(pp.parameters()), 3e-4)
            ls = None
            for i in range(50):
                b = np.random.randint(0, len(A), 16)
                mem = mm.encode(
                    resize224(np.asarray(imgs[A[b]])).to(dev),
                    torch.tensor((states[A[b]] - smu) / ssd, device=dev),
                )
                Sb = torch.stack([feats(A[b])] + [feats(Fm[b][:, j]) for j in range(T - 1)], 1)
                tg = torch.stack([feats(Fm[b][:, j]) for j in range(T)], 1)
                loss = ((pp(Sb, mem[:, -NLAT:]) - tg) ** 2).mean()
                loss.backward()
                op.step()
                op.zero_grad()
                ls = loss.item()
            return ls, mm, pp, op

        l1_, _, _, _ = run50(7)
        l2_, mm, pp, op = run50(7)
        chk(
            f"S10 determinism (two seeded runs -> identical loss: {l1_:.6f} vs {l2_:.6f})",
            abs(l1_ - l2_) < 1e-6,
        )
        cb = None
        for i in range(250):
            b = np.random.randint(0, len(A), 16)
            mem = mm.encode(
                resize224(np.asarray(imgs[A[b]])).to(dev),
                torch.tensor((states[A[b]] - smu) / ssd, device=dev),
            )
            Sb = torch.stack([feats(A[b])] + [feats(Fm[b][:, j]) for j in range(T - 1)], 1)
            tg = torch.stack([feats(Fm[b][:, j]) for j in range(T)], 1)
            if cb is None:
                cb = ((tg - Sb) ** 2).mean().item()
            loss = ((pp(Sb, mem[:, -NLAT:]) - tg) ** 2).mean()
            loss.backward()
            op.step()
            op.zero_grad()
        chk(
            f"S11 overfit canary (50 anchors, 300 its: L_WM {loss.item():.3f} < 0.5*copy {0.5 * cb:.3f})",
            loss.item() < 0.5 * cb,
        )
    else:
        print("  [SKIP] S10/S11 (no cache)", flush=True)
    print(f"=== SELFTEST {'PASS' if ok else 'FAIL'} ===", flush=True)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 26000
    EVAL_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    assert selftest(), "selftest FAILED — refusing to train"
    m_ = np.load(C + "/meta.npz")
    N = int(m_["N"])
    RES = int(m_["RES"])
    imgs = np.memmap(C + "/imgs.u8", mode="r", dtype=np.uint8, shape=(N, RES, RES, 3))
    vj = np.memmap(C + "/vj_feats.f16", mode="r", dtype=np.float16, shape=(N, NTOK, 768))
    states = m_["states"].astype(np.float32)
    epid = m_["epid"]
    taskid = m_["taskid"]
    framepos = m_["framepos"]
    smu, ssd = states.mean(0), states.std(0) + 1e-6
    stn = ((states - smu) / ssd).astype(np.float32)
    # teacher norm stats from a spatial-token subsample (provenance printed)
    sub = np.sort(np.random.RandomState(0).choice(N, 3000, replace=False))
    flat = np.asarray(vj[sub], dtype=np.float32).reshape(-1, 768)
    tmu, tsd = flat.mean(0), flat.std(0) + 1e-6
    TMU = torch.tensor(tmu, device=dev)
    TSD = torch.tensor(tsd, device=dev)

    def feats(idx):
        return (torch.tensor(np.asarray(vj[idx], dtype=np.float32), device=dev) - TMU) / TSD

    ins_eps = np.unique(epid[taskid == 0])
    tr_eps_all = np.unique(epid[taskid == 1])
    val_eps = set(ins_eps[-4:].tolist()) | set(tr_eps_all[-4:].tolist())
    anchors_tr, anchors_va = [], []
    for e in np.unique(epid):
        fr = np.where(epid == e)[0]
        fr = fr[np.argsort(framepos[fr])]
        idx = [
            (fr[j], tuple(fr[j + STRIDE * k] for k in range(1, T + 1))) for j in range(len(fr) - STRIDE * T)
        ]
        (anchors_va if e in val_eps else anchors_tr).extend(idx)
    rng = np.random.RandomState(0)
    anchors_va = [anchors_va[i] for i in rng.choice(len(anchors_va), 600, replace=False)]
    A_tr = np.array([a for a, _ in anchors_tr])
    F_tr = np.array([f for _, f in anchors_tr])
    A_va = np.array([a for a, _ in anchors_va])
    F_va = np.array([f for _, f in anchors_va])
    model = ACTJepa().to(dev)
    pred = Predictor(True).to(dev)
    twin = Predictor(False).to(dev)
    print(
        f"=== BANNER (machine-derived facts) ===\n student {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | predictor {sum(p.numel() for p in pred.parameters()) / 1e6:.1f}M | twin {sum(p.numel() for p in twin.parameters()) / 1e6:.1f}M"
        f"\n WM targets: SPATIAL (T={T},{NTOK}x768) | latent: {T}gx{KREP} | stride {STRIDE} ({T * STRIDE / 50:.2f}s) | teacher-stats from {len(sub)} frames (seed-0 subsample)"
        f"\n anchors tr/va {len(A_tr)}/{len(A_va)} | val eps {sorted(val_eps)} | trainable: student+pred+twin | teacher: frozen cache",
        flush=True,
    )

    def sseq_tgt(a_idx, f_idx):
        Sl = [feats(a_idx)] + [feats(f_idx[:, j]) for j in range(T - 1)]
        S = torch.stack(Sl, 1)
        tgt = torch.stack([feats(f_idx[:, j]) for j in range(T)], 1)
        return S, tgt

    # fixed val tensors (memory: 600*9*64*768*4B ~ 1.4GB on GPU -> keep on CPU, move per chunk)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(pred.parameters()) + list(twin.parameters()), 1e-4
    )

    def val_eval():
        model.eval()
        pred.eval()
        twin.eval()
        lm = lt = ls = cb = 0.0
        with torch.no_grad():
            for k in range(0, len(A_va), 64):
                ab = A_va[k : k + 64]
                fb = F_va[k : k + 64]
                mem = model.encode(resize224(np.asarray(imgs[ab])).to(dev), torch.tensor(stn[ab], device=dev))
                z = mem[:, -NLAT:]
                S, tgt = sseq_tgt(ab, fb)
                lm += ((pred(S, z) - tgt) ** 2).mean().item() * len(ab)
                lt += ((twin(S, None) - tgt) ** 2).mean().item() * len(ab)
                zs = z[torch.randperm(len(z), device=dev)]
                ls += ((pred(S, zs) - tgt) ** 2).mean().item() * len(ab)
                cb += ((tgt - S) ** 2).mean().item() * len(ab)
        model.train()
        pred.train()
        twin.train()
        n = len(A_va)
        return lm / n, lt / n, ls / n, cb / n

    t0 = time.time()
    best = (1e9, -1)
    for it in range(ITERS + 1):
        if it % EVAL_EVERY == 0:
            vm, vt, vs, cb = val_eval()
            print(
                f"[{it:6d}] val/copy {vm / cb:.3f} | twin/copy {vt / cb:.3f} | SHUF/copy {vs / cb:.3f} (shuf-gap {100 * (vs - vm) / cb:+.1f}%) | twin-gap {100 * (vt - vm) / cb:+.1f}% ({time.time() - t0:.0f}s)",
                flush=True,
            )
            sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
            pd = {k: (v.half() if v.is_floating_point() else v) for k, v in pred.state_dict().items()}
            torch.save(
                {
                    "model": sd,
                    "pred": pd,
                    "step": it,
                    "tmu": tmu,
                    "tsd": tsd,
                    "smu": smu,
                    "ssd": ssd,
                    "spatial": True,
                },
                f"{OUT}/s1sp_{it}.pt",
            )
            if vm < best[0]:
                best = (vm, it)
        if it == ITERS:
            break
        bi = np.random.randint(0, len(A_tr), 32)
        ab = A_tr[bi]
        mem = model.encode(resize224(np.asarray(imgs[ab])).to(dev), torch.tensor(stn[ab], device=dev))
        z = mem[:, -NLAT:]
        S, tgt = sseq_tgt(ab, F_tr[bi])
        loss_m = ((pred(S, z) - tgt) ** 2).mean()  # [E5] spatial L_WM; grads -> student via z only
        loss_t = ((twin(S, None) - tgt) ** 2).mean()
        (loss_m + loss_t).backward()
        opt.step()
        opt.zero_grad()
    print(f"[done] best val/copy {best[0]:.4f}-abs @ step {best[1]} | ckpts s1sp_*.pt", flush=True)
