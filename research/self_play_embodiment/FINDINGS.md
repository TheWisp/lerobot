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
`free` g(f(z_t))→Δz · `ac` g(f(z_t),a_t)→Δz (forward, = original e_ac) · `invdyn` h(f(z_t),f(z_{t+1}))→a_t (predict the action).
Forward Δz is _distracted_ (encodes appearance variance, much action-irrelevant) — why `e_ac` was a lossy
reweight. Inverse dynamics is _targeted_: to name the action, `f` must keep the controllable/embodiment
directions and can ignore uncontrollable distractors (classic inverse-model property, cf. ICM/poke).

**DINOv2 substrate (Gate A), 4 seeds — the first non-null.** At **200 demos** `c_invdyn` is the best learned
condition: finalErr **0.29±0.03** vs a 0.39±0.11, e_ac 0.42, noise 0.54, e_free 0.62; SR@0.2
`c_invdyn−a`=**+0.18±0.09**, `−noise`=**+0.28±0.11** (~2σ). Ordering `invdyn>ac>free` as predicted.
**Narrow:** plain `a` overtakes by 500 demos, all converge by 1000. A real but modest low-data-only
demo-efficiency gain — the _objective_ drives it (forward `e_ac` on the same legible substrate stayed meh).

**V-JEPA 2.1 substrate — the article's dense-feature claim holds on our frames.** Obtained via torch.hub
(`vjepa2_1_vit_base_384`; needs `timm`; restore the weight URL off the dead `localhost:8300` mirror).
Single-frame 2D path, 24×24 tokens. **Patch-PCA dramatically cleaner than V-JEPA2's mush** (arm/table/bg
segmented — `figures/vj21_pca.png`). But mean-pool decode only marginally better: gripper **0.83** (vs 0.78),
joints **0.20** (still floor — non-visual in any substrate). 2.1's gain lives in the _dense tokens_ that
mean-pooling discards; inverse-dyn pretext-val is best on 2.1 (0.66 vs DINOv2 0.72).

**V-JEPA 2.1 × inverse-dynamics — the clearest injection result (4 seeds, demos 50–500).** At low data the
inverse-dynamics feature rescues the policy where plain z floors (finalErr, lower=better):

| #demos | a (z only) | z+e_ac (fwd) | **z+e_invdyn** | SR@0.2 `c_invdyn−a` |
| ------ | ---------- | ------------ | -------------- | ------------------- |
| 50     | 0.529±0.05 | 0.647        | **0.383±0.08** | +0.09±0.07          |
| 100    | 0.509±0.11 | 0.488        | **0.309±0.01** | **+0.24±0.10**      |
| 200    | 0.409±0.11 | 0.347        | **0.235±0.04** | **+0.27±0.21**      |
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
  *worse* with more demos. **Diagnosed:** render-domain is fine (our-sim vs dataset home frames cosine
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
episodes**, so proprio memorizes the near-fixed peg — the task has the *same proprio shortcut* as
free-space reaching, and the irrelevant 200-d spatial input *hurts* (overfit). **A valid
vision-necessary manipulation test needs WIDE object randomization** (peg spread ≫ arm precision) +
re-collected play + a J-expert that generalizes across peg positions — a defined but larger redo.
Net manipulation status: scaffolding + two diagnoses done; no positive injection result yet (the
positive result remains the V-JEPA2.1 × inverse-dynamics *reaching* one).

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
