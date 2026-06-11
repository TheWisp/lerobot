# Self-Play Embodiment Injection — Findings

**Question:** Can _embodiment_ be **injected** into an (arbitrary) frozen world-model
representation — so that a goal-conditioned policy needs fewer teleop demos?

**Setup:** frozen **V-JEPA2 ViT-g** (`facebook/vjepa2-vitg-fpc64-256`) as the substrate ·
**aloha** insertion sim (`MUJOCO_GL=egl`, ~268 steps/s) · cheap **random self-play** buffers ·
few-shot **goal-conditioned reaching** via **HER**, scored as **success-rate vs #demos**.

The mechanism under test = **Fork-2**: a small bottleneck `e = f(z)` trained on self-play with
an **action-conditioned** objective, vs an **action-free** control. The headline causal test is
`c (z+e_ac)` vs `b (z+e_free)` (both pretrained on the _same_ self-play, differing only in whether
the action is in the objective), with `noise (z+rand)` and `a (z)` as floors and `oracle`/`bothprop`
as ceilings.

## TL;DR

**The Fork-2 global-bottleneck injection does NOT work on V-JEPA.** Three independent, consistent
lines of evidence; properly error-barred (4 seeds). The single most useful positive finding is that
the **substrate is the real lever** (DINOv2 segments the scene where V-JEPA cannot).

---

## What the experiment established (sound, validated)

The closed-loop harness is trustworthy — the brackets all land where they must:

| control                                       | result               | meaning                                          |
| --------------------------------------------- | -------------------- | ------------------------------------------------ |
| GT-action replay                              | joint L2 = **0.000** | sim is deterministic / seeded reset reproducible |
| hand oracle (Δ = goal−proprio)                | SR = **1.00**        | metric + delta-control ceiling                   |
| learned `bothprop` (both proprios)            | SR@0.1 = **1.00**    | task is learnable; IK from proprio works         |
| learned `oracle_xyz` (Cartesian target given) | SR@0.1 = **1.00**    | Cartesian servoing works                         |
| `shuffle` (mismatched goal)                   | SR = **0.00**        | policy actually uses the goal                    |
| `no-op` (Δ=0)                                 | SR = **0.00**        | floor                                            |

## The three findings

### 1. Vision encodes the _world_, not the _self_ (v1 joint-reach floored)

Decoding from frozen V-JEPA (held-out episodes, linear ridge **and** MLP):

| target                | R²                                 |
| --------------------- | ---------------------------------- |
| gripper Cartesian XYZ | **0.78**                           |
| 14 joint angles       | **0.18** (linear) / **0.18** (MLP) |

Joint configuration is essentially **absent** from the latent (many joint configs → near-identical
images; V-JEPA trained on natural video has no proprioception). A first reaching task defined in
**joint space from vision** therefore floored at SR 0 for _every_ condition incl. the oracle.
**Camera angle does not rescue it** (joint-R²: top 0.17, angle 0.02, front_close 0.22). This is the
design principle "proprio is a model input → vision's job is the world" — the task wrongly asked
vision to be the proprio channel.

**Fix:** give **proprio as a policy input** (the self) and define the goal/metric in the **visible
Cartesian space** (reach the gripper to a position, decodable at 0.78). After also fixing a
**goal-image distribution shift** (a play-frame-trained gripper decoder gets 0.78 on play frames but
only **0.22** on _settled_ goal frames → test goals must be held-out **play** frames), vision-goal
reaching lifts off the floor: `a` (target from goal image) → SR@0.1 0→0.56 over 100→5000 demos.

### 2. The injection signal is NULL across seeds

4 seeds (fresh goals + MLP init + demo subset each). Injection deltas, SR@0.2, mean±std:

| #demos | c_cur−b_cur | c_cur−noise | c_goal−b_goal | c_goal−a   |
| ------ | ----------- | ----------- | ------------- | ---------- |
| 100    | +0.04±0.05  | +0.02±0.06  | +0.01±0.03    | +0.02±0.03 |
| 200    | −0.04±0.14  | +0.00±0.04  | +0.10±0.12    | +0.02±0.22 |
| 500    | −0.02±0.24  | −0.16±0.27  | +0.31±0.30    | −0.10±0.09 |
| 1000   | −0.04±0.03  | +0.03±0.04  | −0.05±0.02    | −0.04±0.01 |

Every delta is within ~1σ of zero, signs flip across seeds, variance in the 200–500 transition is
huge. **`c_ac` ≈ `b_free` ≈ `noise` ≈ `a`.** At 500 demos plain `z` (a) is the _best_ learned
condition (finalErr 0.171 vs e-conditions 0.19–0.25) — adding `e` mildly _hurts_. A single seed had
shown "c_ac halves error at 200 demos"; that was MLP-init/goal noise (pre-registered "need c−b > std"
— it isn't).

### 3. `e = f(z)` is a lossy reweight, not a richer latent

Gripper-xyz decode R² (held-out eps), is `e_ac` a better 64-d embodiment summary?

| representation                  | R²       |
| ------------------------------- | -------- |
| full z (PCA-200)                | 0.78     |
| **random** 64-d projection of z | **0.57** |
| e_ac (64)                       | 0.44     |
| e_free (64)                     | 0.42     |

`e_ac` preserves _less_ embodiment info than a **random** 64-d sketch, and ≈ `e_free`
(action-conditioning barely changes the representation). The Δz objective discards absolute state.
Per-token PCA of `f_ac` is no cleaner than (slightly worse than) JEPA's already-mushy tokens.
**A deterministic `f(z)` adds no information — it can only reweight — and reweighting a mushy latent
buys nothing.**

### Bonus: the substrate IS the lever (DINO-style patch-PCA)

Patch-feature PCA→RGB on aloha frames: **V-JEPA tokens are mushy/blocky** (no clean arm/object
segmentation); **DINOv2 segments the scene cleanly** (table/arms/background). This explains
V-JEPA≈pixels, joints-R²0.18, weak object decode — V-JEPA is a _video_ model fed static clips, its
per-frame spatial semantics are poor. See `figures/sp_dino_pca.png`.

---

## Why Fork-2 can't work here (the mechanism, not a hyperparameter)

`e=f(z)` is a deterministic function of a **frozen** encoder → it carries **no information beyond z**.
It can only make existing directions more linearly accessible. Two reasons that buys nothing here:

- **No accessibility gap to fill:** the policy already has full `z` (1408-d) and a flexible MLP; in
  the regime where it can solve the task at all, it can read `z` directly.
- **Bounded by the frozen front-end:** if the encoder doesn't extract embodiment (V-JEPA's mushy
  patches), no bottleneck around it can. → Predicts bottleneck **size** won't fix it, and Fork-2
  _might_ only work on a substrate that already has clean structure (DINO).
- **Task may be too easy for embodiment-from-vision:** `bothprop` (proprios only, no vision) = 1.0,
  so reaching's embodiment/IK is solved by proprio; vision only supplies the _target_ (a perception
  task). There is little room for an embodiment feature to help.

## Update (2026-06-10) — Gate A/B + V-JEPA 2.1: first positive signal

Two follow-up levers, same harness.

**Objective (Gate B): inverse dynamics, not forward Δz.** Three e-objectives on the same bottleneck `f(z)`:
`free` g(f(z*t))→Δz · `ac` g(f(z_t),a_t)→Δz (forward, = original e_ac) · `invdyn` h(f(z_t),f(z*{t+1}))→a*t (predict the action).
Forward Δz is \_distracted* (encodes appearance variance, much action-irrelevant) — why `e_ac` was a lossy
reweight. Inverse dynamics is _targeted_: to name the action, `f` must keep the controllable/embodiment
directions and can ignore uncontrollable distractors (classic inverse-model property, cf. ICM/poke).

**DINOv2 substrate (Gate A), 4 seeds — the first non-null.** At **200 demos** `c_invdyn` is the best learned
condition: finalErr **0.29±0.03** vs a 0.39±0.11, e*ac 0.42, noise 0.54, e_free 0.62; SR@0.2
`c_invdyn−a`=**+0.18±0.09**, `−noise`=**+0.28±0.11** (~2σ). Ordering `invdyn>ac>free` as predicted.
**Narrow:** plain `a` overtakes by 500 demos, all converge by 1000. A real but modest low-data-only
demo-efficiency gain — the \_objective* drives it (forward `e_ac` on the same legible substrate stayed meh).

**V-JEPA 2.1 substrate — the article's dense-feature claim holds on our frames.** Obtained via torch.hub
(`vjepa2_1_vit_base_384`; needs `timm`; restore the weight URL off the dead `localhost:8300` mirror).
Single-frame 2D path, 24×24 tokens. **Patch-PCA dramatically cleaner than V-JEPA2's mush** (arm/table/bg
segmented — `figures/vj21_pca.png`). But mean-pool decode only marginally better: gripper **0.83** (vs 0.78),
joints **0.20** (still floor — non-visual in any substrate). 2.1's gain lives in the _dense tokens_ that
mean-pooling discards; inverse-dyn pretext-val is best on 2.1 (0.66 vs DINOv2 0.72).

**V-JEPA 2.1 × inverse-dynamics — the clearest injection result (4 seeds, demos 50–500).** At low data the
inverse-dynamics feature rescues the policy where plain z floors (finalErr, lower=better):

| #demos | a (z only) | z+e_ac (fwd) | **z+e_invdyn** | SR@0.2 `c_invdyn−a`    |
| ------ | ---------- | ------------ | -------------- | ---------------------- |
| 50     | 0.529±0.05 | 0.647        | **0.383±0.08** | +0.09±0.07             |
| 100    | 0.509±0.11 | 0.488        | **0.309±0.01** | **+0.24±0.10**         |
| 200    | 0.409±0.11 | 0.347        | **0.235±0.04** | **+0.27±0.21**         |
| 500    | 0.149      | 0.244        | 0.164          | −0.07±0.08 (converged) |

Ordering `invdyn > ac > free`, bigger + lower-data than DINOv2 (which peaked +0.18 @200). Tight `c_invdyn`
std (0.01–0.04) = seed-robust. **Still low-data-only** (plain z catches up by 500). This is the
proof-of-concept: embodiment injection works given a **legible substrate** × the **inverse-dynamics
objective**. See `figures/sp_inject_curve.png`.

**Why it works (mechanism) — inverse dynamics _distills_, forward _reweights_.** gripper-xyz decode R²
(V-JEPA 2.1, held-out eps): **`e_invdyn`(64-d) = 0.876 — beats full z(PCA-200) at 0.826** — while
`e_ac`(64) = 0.623 ≈ random-64 = 0.634 > `e_free` = 0.560. So the inverse-dynamics bottleneck makes the
embodiment signal _more linearly accessible than z itself_ in 64 dims (a genuine distillation a tiny
low-data policy can exploit); the forward objective is a lossy reweight (≈ a random sketch), consistent
with the V-JEPA2 null. `f(z)` adds no Shannon information either way — but the inverse objective concentrates
the controllable directions, the forward one chases appearance. See `scripts/vj21_invdyn_pca.py` +
`figures/vj21_invdyn_pca.png` (per-token PCA of z vs f_invdyn vs f_ac; the decode numbers are the solid part —
per-token is an approximation since `f` is trained on mean-pooled z).

**Takeaways:** (1) the **objective** (inverse dynamics) is a bigger lever than architecture (Fork-1/2) or
bottleneck size; (2) the effect is **real but small and low-data-only**; (3) `bothprop`=1.0 means reaching is
proprio-solved, capping how much _any_ vision-embodiment feature can help — a **contact/manipulation** task is
likely needed for a large effect; (4) to exploit V-JEPA 2.1, go **spatial/patch-based**, not mean-pool.

## Next steps (ranked) — see also the design memory

1. **Substrate swap → DINOv2 (then DINOv3 when access clears), re-run the _same_ injection test.**
   Cheapest, highest-value: tests whether substrate legibility _gates_ injectability. Harness is
   substrate-agnostic — swap the encoder in `sp_lib.py`. DINOv3 (`DINOv3ViTModel`, transformers 5.3)
   is gated + pending Meta review.
2. **Fork-1 (backprop into the encoder)** with an **inverse-dynamics** objective (predict action from
   `z_t,z_{t+1}`), judged by the same few-shot Cartesian SR, monitoring prior washout (object/patch
   legibility). Fork-1 _can_ add information (unlike Fork-2). The earlier `g3_lite.py` was a weak
   Fork-1 attempt on the floored joint task; redo it properly on the Cartesian task.
3. **A task that _needs_ embodiment-from-vision** — manipulation (push/place to a visible target)
   where proprio is insufficient and vision-affordance is necessary. Needs contact-rich data (the
   action→object signal was ~0 in random play).
4. **Bottleneck-size sweep** — low value (the issue is the mechanism, not the hyperparameter), but
   cheap to rule out.

## Manipulation track (2026-06-10) — reach-to-peg attempt, blocked + diagnosed

User chose the manipulation direction (reaching is proprio-capped: `bothprop`=1.0). Goal: a
**vision-necessary** task — reach the gripper to the peg, whose location is only in the image
(`proprio_only` must floor). Progress + the wall:

- **Object now legible (spatial):** V-JEPA2.1 mean-pool dilutes the tiny peg/socket (object-xy 0.19),
  but **8×8 spatial** recovers it: **object-xy 0.71, gripper-xy 0.93** (4×4 was too coarse — 0.36 —
  because 4×4 over the 24×24 grid averages 36 tokens/cell). Lesson: small objects need a fine grid.
- **Strong action signal on contact-rich data:** inverse-dynamics val on insertion-scripted demos =
  **0.22** vs **0.66** on random play — deliberate motion makes the action far more recoverable from
  the latent pair (promising for `e_invdyn`).
- **BUT reach-to-peg FLOORS** (BC of scripted demos, eval in-sim): minDist ~0.25–0.32 m, SR≈0, and
  _worse_ with more demos. **Diagnosed:** render-domain is fine (our-sim vs dataset home frames cosine
  0.002), so the cause is (1) **open-loop BC of scripted absolute trajectories drifting closed-loop**
  (no self-correction — the HER reaching worked precisely because its label is goal-relative), and
  (2) the external dataset has **no peg GT**, so no robust "servo-to-peg" label and no way to convey a
  goal for a novel peg pose. (First-pass "domain gap" reading was a measurement artifact — a
  gravity-collapsed noop eval frame + wrong reference mean.)
- **Fix / next:** generate reach demos **in our own sim** (matched render + free peg GT) with a
  goal-relative / Jacobian-IK servoing label, then re-run the injection conditions. The encode/e/BC/eval
  scaffolding is built and substrate-agnostic; only the data source changes.

Scripts: `sp_manip_inspect/encode/train_e/run.py`, `sp_domain_check{,2}.py`; `sp_lib.Vec.obj_xyz`,
`VJepa21Encoder.encode_both`. Figure `sp_domain_check.png`.

**In-sim reach-to-peg (matched domain, peg GT) — built, but the TASK isn't vision-necessary.** Fixed a
global Jacobian from random play (Δgrip≈J·Δjoints, R²=0.93); the GT servo `Δjoints=clip(J⁺·(peg−grip))`
reaches (SR@0.1=0.88) — sound expert. BC'd it from vision (`sp_reach_jcheck.py`, `sp_reach_insim.py`).
Result exposed the real blocker: **`proprio_only` reaches** (200 demos: minDist 0.11, SR@0.15=0.96)
while every spatial-`z` condition floors. Cause: **aloha's peg varies only ~0.02–0.05 m across
episodes**, so proprio memorizes the near-fixed peg — the task has the _same proprio shortcut_ as
free-space reaching, and the irrelevant 200-d spatial input _hurts_ (overfit). **A valid
vision-necessary manipulation test needs WIDE object randomization** (peg spread ≫ arm precision) +
re-collected play + a J-expert that generalizes across peg positions — a defined but larger redo.
Net manipulation status: scaffolding + two diagnoses done; no positive injection result yet (the
positive result remains the V-JEPA2.1 × inverse-dynamics _reaching_ one).

**Principle validated — pool ALL same-embodiment data for `e` (`sp_pool_e.py`).** `e` is meant to be
pretrained on all cheap task-agnostic data of the same embodiment; only the policy head uses task
demos. R-gripper decode (held-out our-sim): random-only 0.907, scripted-only 0.577, **pooled 0.918**.
Pooling is best + robust (scripted-only transfers poorly alone; pooling rescues it). The reaching
positive already trained `e` task-agnostically (on random play) — fair; the manip attempts wrongly used
task-specific `e`. Going forward: one pooled `e`, reused across tasks; task must still be vision-necessary.

**Tighten-the-threshold makes reach-to-peg vision-necessary on EXISTING data (no re-randomization).**
The aloha peg spread is std 2.7×5.6 cm (range 10×20 cm) — _memorizable within a loose 0.1–0.15 m
threshold_, which is why `proprio_only` "reached." With a **precise expert** (mujoco analytic Jacobian,
`sp_jac_analytic.py`) and **SR@0.05**: a controller that sees the **true** peg gets SR@0.05=**0.94**,
one that memorizes the **mean** peg gets **0.06**. So tighten threshold + precise expert ⇒
vision-necessary, no re-collection. Caveat: `gripper_link` ref has a ~4 cm geometric offset to the peg
(perfect reach floors at 0.042 m) → measure to the **fingertip** so vision's ~3 cm decode error still
clears 0.05. Next: regenerate precise analytic-J labels for cached frames (replay configs via
`mj_forward`, no re-render) + rerun the injection conditions at SR@0.05.

**Corrected run (precise labels + grasp-center + SR@0.05) — vision branch collapses (the deepest
blocker).** With precise per-frame analytic-J labels, the task IS learnable from low-dim inputs
(`proprio_only` reaches ~10 cm by memorizing the mean; `oracle`=proprio+true-peg reaches at 1000
demos). But **every condition that includes the 200-d spatial `z` floors near home** (~0.28 m) —
_below_ `proprio_only`. The small MLP can't learn an object→action servo from frozen spatial features
in low data; the high-dim vision input distracts rather than helps, so `a≈noise≈b_free≈c_invdyn` (all
floored) and `e` can't be evaluated (its host, vision, doesn't work). **Reach-to-object is blocked not
just by task-validity but by few-shot visuomotor learnability from frozen spatial features.** Contrast:
the reaching positive worked because the goal was a _mean-pooled_ whole-scene latent the MLP could use;
extracting+servoing a small object from spatial tokens is a harder visuomotor map this setup doesn't
crack. **Recommendation: consolidate the reaching positive (the solid contribution); reach-to-object
needs either a vision pipeline that pre-decodes the object (give the policy a low-d target) or a
stronger visuomotor learner — a separate research problem.**

## DECISIVE: ACT validation + Fork-2 is a dead end for competent policies (2026-06-10)

User pushed: "is the network even correct? ACT/diffusion succeed here." Right call — my policy was a
toy (frozen features → PCA-200 → ~100K-param MLP → single-step MSE delta), nothing like ACT.

**Built a small ACT** (`sp_act.py`): frozen V-JEPA2.1 8×8 patch tokens + proprio → 2-layer Transformer
encoder → learned action queries → 2-layer decoder → **action chunk + L1** (3.9M params). Differs from
canonical ACT only by: frozen encoder (intended — testing the substrate), smaller size, no CVAE
(fine for scripted/unimodal demos), open-loop chunks instead of temporal ensembling.

**Network validated:** small ACT REACHES the peg — SR@0.1 **0.75** @30 demo-eps (minDist 0.085 m) —
where the toy MLP floored at ~0.28. The earlier failures (and the reaching "positive") were the
**weak policy**, not the task/substrate.

**Injection on ACT: `e`-token HURTS, consistently** (reach SR@0.1, e=0 vs e=1): 3ep .45/.15, 6ep
.25/.25, 12ep .50/.10, 30ep **.75/.10**. Not noise (consistent, present even at 30 eps).

**Why — the real result:** `e = f(frozen features)` is a deterministic function of features the ACT
**already consumes** (the patch tokens) → it adds **zero information** → can only be neutral or harmful
(harmful here: the ACT leans on the salient token, train/eval drift in it then misleads). The toy-MLP
"injection helps" was a **weak-policy artifact** — the MLP used a lossy PCA-mean and couldn't extract
embodiment, so a pre-concentrated `e` helped _it_; a competent extractor (ACT) erases the benefit, as
information theory requires.

**Conclusion for the whole project:** **Fork-2 (a bottleneck `e`-token around a FROZEN encoder) cannot
inject embodiment into a competent policy.** The only mechanism that can add information is **Fork-1 —
train the ENCODER with the embodiment (inverse-dynamics) objective** so the features themselves become
embodiment-aware. Next experiment: inverse-dynamics encoder-pretraining vs none, measured by ACT
demo-efficiency (does it cut #demos). Scripts: `sp_act.py`.

## Literature check before Fork-1 (2026-06-10) — our idea ≈ DynaMo (works), with a recipe fix

Lit review (full digest in git history of this commit) — directly de-risks/reframes Fork-1:

- **Fork-1 ≈ DynaMo (Cui & Pinto, NeurIPS 2024, arXiv:2409.12192):** encoder trained with **latent
  inverse + forward dynamics** in-domain → feeds standard imitation heads → beats R3M/MVP/VC-1/ImageNet
  & from-scratch **at low demos**, across heads. Our real action labels make it strictly easier.
- **Brandfonbrener et al. NeurIPS 2023 (2305.16985):** inverse-dynamics is the BEST pretraining objective
  and the margin is **largest at small finetune-data** — predicts our demo-efficiency win.
- **LAPA (ICLR 2025, 2410.11758):** latent-inverse-dyn pretraining beats GT-action OpenVLA, ~30× cheaper.
- **RECIPE FIX — pure inverse-dynamics COLLAPSES** (ICM; Levine-Stone-Zhang RLC 2024 "Multistep Inverse Is
  Not All You Need" 2403.11940; DynaMo's "constant-embedding solution"). Our `e_invdyn` was single-head →
  at risk. **Must add forward-dynamics/consistency + SimSiam stop-gradient (DynaMo).**
- **Fork-2 failure is well-grounded:** Hansen ICML 2023 (2212.05749) & Dasari CoRL 2023 (2310.09289) —
  frozen generic encoder + small head doesn't beat augmented from-scratch; Diffusion Policy (2303.04137
  Table 5) & OpenVLA (2406.09246) — **finetune encoder > freeze** for BC; no added bottleneck helps.
  DynaMo wins frozen-downstream only because the encoder was **adapted in-domain** first.
- **Warnings:** sim↔real R²≈32% (Dasari) → validate Fork-1 on REAL/HVLA-S1, not just aloha sim; and
  self-play must exercise task-relevant DoF (random play barely touches objects → body-only embodiment).
- **Plan:** build Fork-1 as DynaMo (inverse+forward+stop-grad encoder adaptation on self-play) → feed
  ACT/S1 head, no bottleneck → validate in real HVLA-S1 (DINOv2). Read DynaMo + ACDF first.

## Fork-1 cheap gate PASSES (2026-06-10) — adapting the encoder adds info Fork-2 couldn't

DynaMo-style adaptation of **DINOv2** (fine-tune top-2 blocks, inverse + forward + SimSiam stop-grad,
on random-play self-play; `sp_dynamo.py`). Gate (held-out eps, R-gripper-xyz decode):
**frozen 0.78 → adapted 0.90**, no full collapse (eff-rank 18→6.4), L*inv 0.99→0.74. So **Fork-1
measurably adds embodiment info** (same data/substrate; only the encoder's extraction changed) — the
clean empirical complement to "Fork-2 (redundant-vision bottleneck) is info-theoretically doomed for a
competent policy." (Fork-2 \_is* saveable only by making the token carry NON-redundant info — privileged
distillation, or the HVLA-S2-token pattern where the token is a separate system's reasoning, not a
re-encoding of the policy's own vision.)
Caveats: rank 18→6.4 = strong concentration on controllable dims (sheds scene/object info — re-tune for
richer tasks; matches the inverse-dynamics "discards irrelevant-but-controllable" warning). And this is
a decode PROXY — the ACT uses patch tokens (gripper ~0.93 even frozen), so the real verdict is ACT
demo-efficiency with adapted vs frozen encoder (next build: save adapted encoder → re-encode patches →
ACT #demos sweep). Sim-first per plan; real (HVLA-S1) only after sim works.

## Files

- `scripts/sp_lib.py` — env harness (delta control, gripper-xyz metric) + V-JEPA encoder + `EmbEnc` loader
- `scripts/sp_goals_play.py` · `sp_train_e.py` · `sp_run2.py` · `sp_agg.py` — the v2 Cartesian experiment + aggregation
- `scripts/sp_jointprobe.py · sp_mlpprobe.py · sp_cartprobe.py · sp_shiftcheck.py` — the diagnostics behind finding #1 and the goal-shift fix
- `scripts/sp_viz.py · sp_dinoviz.py · sp_embviz.py` — occlusion / DINO patch-PCA / latent-quality visualizations
- `scripts/sp_dino_encode.py · sp_train_e_dino.py · sp_run3.py · sp_agg_dino.py` — Gate A/B: substrate-parametrized (`SUBSTRATE=dino|vj21`) encode / 3-objective e-training / injection sweep / aggregate
- `scripts/vj21_legibility.py · vj21_probe.py` — V-JEPA 2.1 bring-up + legibility (decode + patch-PCA)
- `scripts/sp_run.py · sp_check.py · sp_goals.py` — v1 joint-reach (the floored version)
- `scripts/exploration/` — G1/G2/G3-era probes & data collection (`collect_world.py`, `g3_lite.py`, `probe_*`)
- `scripts/sp_agg_dino.py · sp_plot_inject.py` — substrate-aware aggregate (`SUBSTRATE=dino|vj21`) + the injection comparison figure
- `figures/` — `sp_curve2.png` (V-JEPA2 SR, 4 seeds), `sp_dino_pca.png`, `sp_emb_pca.png`, `sp_latent_viz.png`, `vj21_pca.png` (2.1 patch-PCA), `sp_inject_curve.png` (DINOv2 vs 2.1 injection)
- `results/` — `sp_results2_s{0..3}.npz` (V-JEPA2), `sp_dino_results_s{0..3}.npz` (DINOv2), `sp_vj21_results_s{0..3}.npz` (V-JEPA 2.1) — all with `c_invdyn`

**Scratch caveat:** scripts hardcode `/tmp/selfplay_probe/` and depend on local data buffers
(`world_buffer.npz` ~153 MB, `feat_cache.npz` ~766 MB) that are **not** committed (size). They are the
method-of-record, not turnkey. Conda python: `/home/feit/miniforge3/envs/lerobot/bin/python`.

## PROJECT CONCLUSION (2026-06-10) — Fork-1 injects the BODY but evicts the OBJECT; injecting embodiment into a vision encoder is misdirected

Ran Fork-1 properly (DynaMo: inverse + forward + stop-grad, fine-tune DINOv2 top-2 blocks), on POOLED
self-play (random play + scripted insertion), with both a mean and a spatial (per-token forward) objective.

- **ACT demo-efficiency, frozen vs pooled-adapted (mean objective):** adapted WORSE at every demo level
  (SR@0.1 frozen 0.15/0.45/0.50/0.60 vs adapted 0.00/0.25/0.40/0.50). Not a win — a consistent loss.
- **Spatial gate (the right metric — 8x8 patch decode, frozen vs spatial-adapted):**
  gripper-xyz 0.91→**0.95** (body UP) but peg-xy 0.38→**−0.38** (object DESTROYED).
- **Mechanism (textbook inverse-dynamics failure; ICM / Levine-Stone-Zhang / exo-noise):** the objective
  keeps action-coupled features (the arm) and discards the rest. The peg is static until grasped, so
  neither inverse (arm-dominated) nor forward (peg static within episode) needs to encode the peg's
  across-episode position → it's dropped. Pooling/insertion data didn't save it (peg still mostly static
  per frame).

**Ties back to Finding #1 (vision encodes the WORLD, proprio the SELF):** the project tried to inject
embodiment (the self/body) into a vision (world) encoder. But the body is proprio's job (bothprop=1.0:
proprio alone solves reaching), and vision's unique value is the object — which the dynamics objective
evicts. So dynamics-injection pushes vision toward the self → REDUNDANT where the body matters (proprio
has it) and DESTRUCTIVE where the object matters (sheds the peg). That's why Fork-1 hurt reach-to-object
and frozen vision won.

**Verdict:** Fork-2 = redundant (no added info to a competent policy). Fork-1 = genuinely injects _body_
embodiment (gripper 0.78→0.95) but it's the wrong thing for a vision encoder (redundant w/ proprio,
evicts the object). For manipulation, the right combo is the one that already worked: **frozen vision
(keeps the object) + proprio (the body)**. No free lunch: an object-preserving anchor just pulls back
toward frozen. Scripts: sp*dynamo*{pooled,spatial}.py, sp_act_fork1.py.

---

## Chapter: VLA-JEPA mini-repro (ACT student) — the WM-teacher route, tested faithfully (2026-06-10/11)

**Pivot rationale.** After the encoder-direct negatives, the user redirected to the literature-backed
mechanism: a JEPA world model _teaching_ a policy (VLA-JEPA, arXiv 2602.10098 — Qwen3-VL-2B + frozen
V-JEPA2 teacher + latent-action world model, WM dropped at inference; LeRobot upstream port exists:
`feat/vla-jepa-*`, `lerobot/VLA-JEPA-Pretrain`). Full repro infeasible (8×A100, 300K-clip corpus);
mini-repro = same structure with an ACT student (59M, own ResNet18) + V-JEPA2.1 teacher, aloha sim.

**Method discipline (the lasting contribution).** Per-line paper anchors (`[S3.2 E3]`) + completeness
table + **executable selftest gate** (10 checks: spatial shapes, z-group & teacher-forcing causality via
perturbation, leakage, trainability, train/eval path identity, mask equality against upstream's
`build_action_block_causal_attention_mask` verbatim port, determinism, overfit canary). Training refuses
to start on FAIL. This caught: missing teacher forcing (E3), ungrouped/non-causal latent tokens (E2),
**pooled WM targets** (paper+upstream predict per-frame patch grids — an unflagged "DONE" in the v1
audit), a stage-2 stats leak (few-shot claim using all-80-episode normalization), and a train/eval
resize mismatch. Independent-implementation diffing > self-review, every time.

**Stage-1 protocol** (paper has none — fixed 50K steps, downstream-only): stop at val-min of
L_WM/copy-baseline; aliveness via **shuffle-z gap** (swap z across samples). Twin-gap (separately
trained no-z predictor) RETIRED — it measures redundancy vs the teacher-forced GT context, not whether
the student learned to anticipate (pooled run: twin-gap ≈0 while shuffle/z-zero showed z alive, +33–45%
/ +15→83% with horizon). Spatial round: val 5.6→0.42 no overfit upturn, shuffle-gap +41–42% from the
first checkpoint; ckpt = s1sp_24000, stride 16 (0.32 s/step — stride 6 starved z; our ambiguity lever,
the paper's comes from corpus diversity).

**Stage-2 results (insertion, K=10 unless noted; proper ACT = CVAE + chunk100 + temporal ensembling —
earlier "floors" were the toy policy's fault):**

| test                                                                 | result                                                                                                                                            |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| fixed 12k, 2 seeds, n=24                                             | ctrl 33.5% > β0.1 23% > β0.5 19% (ordering both seeds)                                                                                            |
| decomposition (init-only, β=0)                                       | 29% ≈ ctrl — init neutral; aux is the suspect                                                                                                     |
| **SR-vs-steps to 36k** (fair test per user: peak, not matched steps) | ctrl 33→38→**42%** rising; β0.1 21→21→12; β0.5 17→12→8 — aux arms **decline**; "slower convergence" rejected; β dose-response in peak AND decline |
| **fresh n=48 re-eval** (same models, new seeds)                      | ctrl 35 / β0.1 31 / β0.5 31 / init 38% — **gaps evaporate**; n=24 inflated the harm (±14 pt swings per model across eval sets)                    |
| OOD object-layout (f=2×,3× spawn, n=48 paired)                       | **all arms floor (2–6%)** — degradation slope unmeasurable at this policy strength                                                                |
| transfer_cube K=10                                                   | ctrl 29–33 / β0.1 29 / init 29% — neutral; deficit does not generalize to the dynamic task                                                        |
| K=3                                                                  | both 4% (floor, undiscriminating)                                                                                                                 |
| K=25                                                                 | ctrl 67% vs β0.1 62% (within noise; both reach 8% full insertion)                                                                                 |

**Verdict.** At 59M/180-episode scale the WM-teacher mechanism is **neutral-to-mildly-harmful
in-distribution, neutral cross-task, untestable OOD** (policies too weak off-distribution). The only
robust harm signal is the within-arm overtraining decline with the aux on (objective conflict compounds);
the only robust safety signal is that stage-1 _initialization_ costs nothing. Consistent with the paper's
own fine print: WM adds <1% in-distribution at 2B scale (LIBERO 97.2 vs 96.1); its gains are OOD/robustness
(LIBERO-Plus +10–18 pts) — a regime our base policies cannot yet reach.

**Methodological lessons (now standing practice).** (1) Fixed-budget comparisons can establish _gains_
but not _deficits_ — a losing arm may be slower; compare peaks (user's correction). (2) n=24 closed-loop
SR has ±10 pt σ and ±14 pt cross-set swings — never headline single-set gaps. (3) Self-certification
fails; executable selftests + upstream diffing catch what re-reading does not.

**Go/no-go on large-model VLA-JEPA: NO-GO for now.** Nothing measured justifies the 2B build as a
continuation of this line. The exposed levers are (a) base-policy competence (K=25 → 67%/8-pt full
insertion shows data scaling works; OOD needs a much stronger base before the slope question is posable)
and (b) stage-1 corpus diversity (scene-determined self-play) — relevant only after (a).

Figure: `figures/vlajepa_sr_vs_steps_ood.png`. Scripts: `scripts/sp_vj_act.py` (model + stage-1 +
selftest), `scripts/sp_vj_act_s2.py` (stage-2 + arms), `scripts/sp_ood_eval.py` (graded OOD), queue
`scripts/sp_queue2.sh`. Logs/ckpts in /tmp/selfplay_probe (scratch).

---

## Bridge chapter (2026-06-10, retro-committed): affordance reframe → wide-scene gate → data quality → bimanual pivot → real-data negatives

Recorded after the fact for completeness; chronologically this precedes the VLA-JEPA chapter.

- **Reframe (user):** embodiment = action→outcome/affordance, NOT proprio (source vs outcome); prior
  negatives tested in-distribution where VLAs overfit the scene — the bet lives in cross-scene/OOD.
- **Wide-scene gate PASSED:** peg spawned x∈[0,0.3] y∈[0.35,0.65] (std ~8.7 cm) is reachable (analytic-
  Jacobian GT SR@0.05=0.95) and localizable (V-JEPA2.1 8×8 decode R²=0.993 on held-out scenes). `sp_wide_check.py`.
- **Data-quality audit (user spotted the chaos; we measured it):** random/jitter self-play = left-arm
  flailing |Δ|≈0.084/step + only 8% episodes with object contact → junk for affordance learning.
  Scripted push (approach-above → descend-onto-peg → push, PUSHZ=0.022 — peg settles to z≈0.01, not its
  0.05 spawn) = 84–100% contact, left arm still. `sp_data_check.py`, `sp_skill_collect.py`.
- **Task is BIMANUAL** (AlohaInsertion reward: R-arm grasps peg AND L-arm grasps socket → lift → insert)
  → single-arm synthetic data misaligned; real same-embodiment datasets exist and match our env exactly
  (`lerobot/aloha_sim_{insertion,transfer_cube}_{human,scripted}`, 200 eps, 14-dim, top cam, 50 fps).
  Pivot: unified cache (`sp_build_cache.py` → 56k frames @224 + `sp_vj_features.py` → V-JEPA2.1 64×768/frame;
  insertion eval eps 40:50 held out of everything).
- **Real-data in-dist negatives (pre-VLA-JEPA):** (a) encoder-direct forward FT of DINOv2: action-decode
  flat 0.825→0.816 (no action-baking — verified), proprio +0.015, but ACT open-loop L1 frozen BEATS
  adapted at every K (`sp_real_forward.py`, `sp_act_openloop.py`); (b) joint (single-stage) WM-aux mini:
  hurts at K=5–20, neutral K=40 (`sp_vlajepa_mini.py`) — later understood as a structural deviation
  (paper is two sequential stages) and superseded by the faithful chapter above.
- **Per-step diagnostics that resolved the twin-gap confusion:** `sp_s1_perstep.py` (z-zero +15→83% with
  horizon; shuffle-z +33–45% — z alive and sample-specific while aggregate twin-gap read ≈0).

Raw result lines for the VLA-JEPA chapter archived in `results/vlajepa_results.txt`.
