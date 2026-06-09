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

## Files

- `scripts/sp_lib.py` — env harness (delta control, gripper-xyz metric) + V-JEPA encoder + `EmbEnc` loader
- `scripts/sp_goals_play.py` · `sp_train_e.py` · `sp_run2.py` · `sp_agg.py` — the v2 Cartesian experiment + aggregation
- `scripts/sp_jointprobe.py · sp_mlpprobe.py · sp_cartprobe.py · sp_shiftcheck.py` — the diagnostics behind finding #1 and the goal-shift fix
- `scripts/sp_viz.py · sp_dinoviz.py · sp_embviz.py` — occlusion / DINO patch-PCA / latent-quality visualizations
- `scripts/sp_run.py · sp_check.py · sp_goals.py` — v1 joint-reach (the floored version)
- `scripts/exploration/` — G1/G2/G3-era probes & data collection (`collect_world.py`, `g3_lite.py`, `probe_*`)
- `figures/` — `sp_curve2.png` (SR vs demos, 4 seeds), `sp_dino_pca.png`, `sp_emb_pca.png`, `sp_latent_viz.png`
- `results/` — `sp_results2_s{0..3}.npz` (per-seed final errors)

**Scratch caveat:** scripts hardcode `/tmp/selfplay_probe/` and depend on local data buffers
(`world_buffer.npz` ~153 MB, `feat_cache.npz` ~766 MB) that are **not** committed (size). They are the
method-of-record, not turnkey. Conda python: `/home/feit/miniforge3/envs/lerobot/bin/python`.
