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
4. Data hygiene: `cube_1k` is locally corrupt (truncated parquet footers, not on Hub) — recover or discard?
