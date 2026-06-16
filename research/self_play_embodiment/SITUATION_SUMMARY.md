# Self-Play World-Model (VLA-JEPA) — Situation Summary for Investigation

_Snapshot 2026-06-16. Companion to WM_GRAFT_JOURNAL.md (full detail)._

## Goal

Test whether teaching a robot policy from **cheap actionless self-play video** (via a V-JEPA world-model
objective) makes it **more data-efficient** — fewer teleop demos for the same task success.

## What VLA-JEPA is (1 paragraph)

A frozen **V-JEPA2** video encoder (teacher) produces latent embeddings of frames. A student ACT policy
carries special ⟨latent⟩ tokens; in **stage 1** we train those tokens + a predictor to forecast the teacher's
**future latents** from the current frame (self-supervised, no actions) — the bet being this instills
object/dynamics ("embodiment") knowledge. In **stage 2** we fine-tune on K teleop demos with
`L = L_action + β·L_WM`; the action head cross-attends the ⟨latent⟩ tokens. World-model head is dropped at
inference. Two arms compared: **control** (plain ACT, from scratch) vs **WM** (stage-1 pretrained + β·L_WM).

## What we found (after fixing eval bugs)

The original "win" was an eval artifact. With a corrected eval (no action clip, dynamic episode length;
9-check audit gate, none tautological):

| task (grasp rung SR≥1, 3 seeds, n=96, peak)                      | control  | WM(demos+SP) |
| ---------------------------------------------------------------- | -------- | ------------ |
| Insertion K=25 (the PDF result, re-evaluated on the SAME models) | **71.3** | 67.0         |
| — was, under the BUGGY eval                                      | 54.0     | 68.3         |
| Cube K=10 (mean reward)                                          | **1.43** | 0.86         |

- The buggy eval **suppressed control by ~17 pts**; fixing it erases the WM advantage on both tasks.
- WM = **low-variance, low-ceiling** (behaves like a regularizer); control = high-variance, high-ceiling.
- **Net: with a clean eval, WM does NOT improve data efficiency on insertion or cube.**

## The behavioral clue (human visual inspection of rollouts, n=96 datasets in GUI)

- **cube WM**: gripper rarely lands near the cube; continues unrelated motion (reaches to the wrong side).
  Looks like it **ignores / never learned the cube's importance**. cube **control**: good sense of where to
  place the gripper, fails only on fine accuracy.
- **insert WM**: broad failures — can't reliably pick up either object, can't insert.
- Datasets: `thewisp/eval_rollout_{cube,insert}_{control,wm}` (96 eps each, same seeds across arms → episode i
  is the identical scene in both).

## Hypothesis under test

The WM representation **under-encodes object position** (the perception sub-skill data-efficiency depends on).
Two mechanisms, likely compounding:

1. **JEPA's future-prediction objective rewards predictable dynamics (smooth self-caused arm motion) over
   hard-to-predict object location** → ⟨latent⟩ tokens carry a motion prior, not a localizer.
2. **The self-play corpus was 100% `cube_handover`** (cube almost always already-grasped / in-air), so stage-1
   never learned to localize a cube **resting on the table** — exactly the initial-reach phase that fails.
   (Note: `cube_1k` was listed but never used — the SP frame budget was filled by handover first.)

## Proposed diagnostic: linear probe for object position

Freeze each policy's vision encoder, extract features on labeled frames (object xyz from sim physics), fit a
**linear (Ridge) readout** features→object-xy, report held-out R². Compare **WM vs control** (+ V-JEPA teacher
as upper bound, ImageNet/random as floor). Linear (not MLP) on frozen features measures whether object
position is **present and usable** by the (linear-attention) action head.

- WM R² ≪ control R² → confirms representation-level perception failure (the suspected mechanism).
- WM R² ≈ control R² → failure is downstream (action head), look elsewhere.
- Optional: probe "cube-on-table" vs "cube-grasped" frames separately to test mechanism #2 directly.

## Literature anchors (method is standard)

- Alain & Bengio 2016, _Understanding intermediate layers using linear classifier probes_ (arXiv:1610.01644) — canonical linear probe.
- Linear-evaluation protocol for SSL: SimCLR (Chen 2020), MoCo (He 2020), BYOL (Grill 2020).
- I-JEPA (Assran 2023), V-JEPA (Bardes 2024) — our teacher family; evaluated via frozen-feature probes.
- Probing for world state: Li et al. 2023 _Emergent World Representations_ (Othello-GPT, ICLR); Nanda et al. 2023 (linear probe suffices with right basis).
- Visual pretraining for manipulation (frozen-feature eval): R3M (Nair 2022), MVP (Xiao 2022).
- Caveat: these support the _method_; no paper ran this exact handover-pretrained-ACT probe — it's the standard tool applied to our question.

## Open questions to investigate

1. Does the probe confirm WM under-encodes object position (mechanism), or is the gap downstream?
2. Is mechanism #2 (handover-only SP → no table-localization) the dominant cause? Would task-matched SP
   (cube sitting on table, being picked up) change it?
3. Is there any regime (lower data? harder task where control floors?) where the WM prior actually helps?
4. Data hygiene: one self-play dataset was locally corrupt (never actually used in training) — recover or discard?

---

# Follow-up (2026-06-16): what changed + current plan

## What changed since the summary above

1. **Read the reference papers.** The closest published method (a vision-language-action model with a JEPA latent world model) and the V-JEPA 2 world-model line differ from our instantiation on three axes: (a) **base policy** — they build on a multi-billion-parameter pretrained vision-language model with an object-grounded vision encoder, whereas ours is a small from-scratch imitation policy; (b) **pretraining data** — they use large, diverse human video plus high-quality expert robot demonstrations, whereas ours is a small, narrow self-play corpus; (c) their reported world-model benefit is mainly **out-of-distribution robustness**, "minimal for in-distribution," yet we evaluated almost entirely in-distribution.

2. **Recovered the project's origin premise.** The original intent is **self-exploration for state _coverage_ (including failures), made embodiment-aware by an _action-conditioned_ dynamics objective** — i.e., the representation should encode "how my actions change the world," not merely "what the scene looks like." We treat that plan as unvalidated, but it sharpened the key distinction below.

3. **Confirmed our world-model objective is _action-free_.** It predicts future scene latents from the current observation alone, with no action input. By the action-conditioned/action-free distinction, this is a **"spectator" objective** — it learns to forecast the scene, not the action→outcome map that control needs.

4. **Ran the object probe — and it REFUTED our working hypothesis** (the most important new fact). The world-model representation encodes the object's position **as well as or better than** the control baseline (held-out R² ≈ 0.93 vs 0.89; ~1.7 cm vs 2.4 cm error), including in the pre-grasp, object-on-table phase (so not explainable by reading the gripper). **Perception is therefore not the bottleneck** — the world-model policy fails to act on the object even though it localizes it accurately. The deficit is **representation → action**, not seeing.

5. **Method corrections.** Replaced an uninformative coarse spatial heatmap with an intuitive predicted-vs-true position overlay; and switched the probe to use the **full (un-pooled) representation** so it preserves the spatial/relational structure the policy actually relies on, rather than averaging it away.

6. **Process & hygiene.** Adopted a human dataset quality-check before any training run (after finding a corrupt, never-used self-play dataset and a corpus-composition surprise).

## Hard facts now (independent of any plan)

- With a corrected evaluation (3 training seeds), the world-model policy is **≤ the control baseline** on both tasks — no data-efficiency gain.
- The world-model policy is **low-variance / low-ceiling** (behaves like a regularizer); control is high-variance / high-ceiling.
- The world-model representation **perceives the object at least as well as control** → the failure is **downstream of perception**.
- Our world-model objective is **action-free (spectator)**.

## Current plan (fact-first)

1. **Action decodability probe** (the cheap, decisive next test): measure whether the _expert action_ is **less** linearly recoverable from the world-model representation than from control's, using the full representation. If so, the representation — though it localizes objects well — is geometrically less _control-relevant_; that is the direct "good spectator, poor controller" evidence and it localizes the failure to the representation rather than the action decoder.
2. **Out-of-distribution evaluation** (reuses existing models): the one untested regime where a low-variance representation could plausibly help, and where the reference work says the benefit actually lives.
3. **Decision fork** after (1)+(2): if the representation is control-hostile **and** there is no OOD win, the action-free + narrow-data configuration is a dead end as built → either write up the negative result, or treat the original mechanism (**action-conditioned objective + coverage-oriented self-play**) as a new, larger bet. If instead the representation is controllable, the issue is in how the action stage uses it, not the representation.

**Deliberately not pursuing:** training an additional action decoder (expensive, low-confidence), or tuning in-distribution data scale (the world model already loses there).

---

# Follow-up 2 (2026-06-16): the representation and the policy are both innocent open-loop

We ran the action-side probes and a falsifiability audit. Three open-loop refutations now point to one mechanism.

## What we found

1. **Action decodability — "control-aliasing" refuted.** A linear probe of the representation for the expert action (controlling for the fact that, in this action space, the action is ~0.98 decodable from proprioception alone — so we isolate the _vision-added_ part). The world-model representation decodes the expert action **better** than control (vision-added R² 0.71 vs 0.63), **including in the contact phase** (0.60 vs 0.48) where aliasing was predicted to bite. The world-model representation is _more_ control-relevant, not less.

2. **Trained-policy open-loop accuracy — "bad routing/decoder" refuted.** Comparing, on held-out expert frames, the best-possible linear use of the representation vs what the _trained_ policy actually outputs: the trained world-model policy is **more accurate open-loop** than control (action MSE 0.0065 vs 0.0102; R² 0.956 vs 0.931; better in both reach and contact). The learned decoder routes the geometry _better_, not worse. (Architectural note: our action decoder cross-attends the full representation, so there was never an input-level bottleneck discarding it.)

3. **The pattern → covariate shift / narrow recovery basin.** The world-model policy is _better_ open-loop (lower expert-imitation error, lower seed variance) yet _worse_ closed-loop. Lower open-loop error + worse closed-loop is the textbook signature of **covariate shift**: the world-model regularizer fits the expert manifold **tighter** but **narrower**, so it imitates beautifully on the expert path and recovers poorly once its own small errors push it off-path. The looser, blurrier control policy recovers more gracefully.

## Probe trustworthiness (falsifiability audit)

To make these (negative) conclusions trustworthy, we audited the probe methodology itself — and it passes: the solver is exactly ridge (matches the closed-form primal); the R² metric is correct; **a label-permutation test makes held-out R² collapse to ~0** (so a probe _can_ fail, and does on shuffled labels — the positive numbers are real signal, not p≫n overfitting); and an episode-split-vs-frame-split check shows no leakage inflation. So the "representation is innocent" result is not a probe artifact.

## Updated plan

The open-loop chain is exhausted (representation good; trained policy good open-loop). The remaining live hypothesis is closed-loop **covariate shift / narrow basin**, testable two ways:

1. **Off-trajectory (recovery-basin) probe** — perturb expert states with action noise and measure how fast each policy's action error grows off-manifold (prediction: world-model steeper). Also a target-free sensitivity version. Open-loop, cheap, decisive for the mechanism.
2. **Out-of-distribution / closed-loop evaluation** — the regime where any world-model benefit (robustness) would actually appear.
