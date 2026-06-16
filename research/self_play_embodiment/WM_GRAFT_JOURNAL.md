# WM-on-LeRobot-ACT graft — experiment journal (Jun 15 2026)

Goal: test whether world-model (VLA-JEPA) pretraining on free self-play video lets LeRobot's ACT reach
the same TransferCube success with **fewer demos** (demo-reduction ratio), read horizontally off an
SR-vs-#demos curve. TransferCube chosen because it saturates only near 50 demos → headroom at low K.

## Datasets

HF uploads (224×224, robot_type bimanual_viperx_aloha_sim):

- `thewisp/aloha_sim_selfplay_corpus` — 1,200 eps / 312,515 fr = wide_1k peg/socket SP (1000) + cube_handover SP (200).
- `thewisp/aloha_sim_wm_corpus` — 1,900 eps / 556,172 fr = selfplay_corpus + 4 aloha demos (transfer/insertion × human/scripted, 50 each, resized to 224) + 500 ACT transfer rollouts (75% success).

### Stage-1 pretraining corpus (IMPORTANT for writeup)

Stage-1 used `cache_wm` = a **random ~150k-frame SUBSET (504 episodes) of `aloha_sim_wm_corpus`**,
capped at 150k to keep V-JEPA encode + pretrain tractable (full corpus is 556k fr). Spans ALL sources
(sampled, not complete — episodes dropped at random):

| source                                        | eps used / total | frames      |
| --------------------------------------------- | ---------------- | ----------- |
| wide_1k peg/socket self-play                  | 264 / 1000       | 39,600      |
| cube_handover self-play (long retry episodes) | 50 / 200         | 40,056      |
| transfer_cube_human demo                      | 15 / 50          | 6,000       |
| transfer_cube_scripted demo                   | 15 / 50          | 6,000       |
| insertion_human demo                          | 17 / 50          | 8,500       |
| insertion_scripted demo                       | 12 / 50          | 4,800       |
| policy rollouts (transfer, 75%)               | 131 / 500        | 42,178      |
| **total**                                     | **504 / 1900**   | **147,134** |

Balance: self-play ~80k fr (54%), rollouts ~42k (29%), demos ~25k (17%). Subset is deterministic
(np.random.RandomState(0) episode shuffle in build_wm_cache.py). NOT all demos; includes rollouts.

## Architecture (proper graft) — `src/lerobot/policies/act/` (this branch)

- Config flag `wm_num_latents` (default 0 = vanilla ACT, control path byte-identical).
- `wm_num_latents=24` (=T·KREP=8·3) learnable ⟨latent⟩ tokens appended to the transformer-encoder input;
  decoder cross-attends them (LeRobot decoder attends full encoder memory).
- `ACT.wm_encode()` returns encoder output at the latent positions (z) for stage-1.
- Teacher: frozen V-JEPA2.1 ViT-B/384, 8×8=64×768 spatial grids; T=8, stride=16.
- Predictor: copied VERBATIM from prototype `sp_vj_act.py` (teacher-forced AR, block-causal, z→(B,8,3,512)).
- Stage-1 runs INSIDE the real ACT policy → FrozenBatchNorm + demo-stat normalization identical to stage-2.
- Stage-2 transfer: inject ENCODER PATH only (157 tensors: backbone + encoder + latents + input projs);
  decoder / VAE / action-head left at fresh seed-100000 init (same as control). `stage2_wm_train.py`
  monkeypatches `make_policy` so both arms use the identical lerobot-train pipeline.

## Resolution note (caveat for writeup)

Stage-1 = 224 (self-play native). Stage-2 train + eval = 480×640 (demos native; reuses control + eval
pipeline). Transfer across resolutions is sound for conv + FrozenBN + sinusoidal pos, but the latents
were pretrained attending ~49 img tokens (224) and attend ~300 at stage-2 (480) — a real context shift;
suspect #1 if low-K shows no gain. (gym_kwargs now passes observation_width/height so an all-224 rerun is
possible if needed.)

## Results so far

- **Control K-sweep (no WM, ImageNet backbone, 480, n=200, seed 1000, 400-step eps, fixed 100k steps):**
  K5 = 21.0% · K10 = 29.0% · K25 = 52.5% · K50 = 73.0%. (loss curves: converged ~70-80k; losscurves.png)
- **Backbone-only graft (WRONG — discards latents): K5 = 8.0%** (< control 21%; identical train loss to
  control K5 → worse _prior/generalization_, not under/over-training). Abandoned.
- **Stage-1 representation gate (proper graft, union subset, final @30k steps):**

  | stage-1 corpus                    | val/copy ↓  | shuffle-z gap ↑ |
  | --------------------------------- | ----------- | --------------- |
  | demos only (prototype Result 1)   | 2.04 (dead) | +1.3%           |
  | demos + self-play (prototype)     | 0.574       | +36.2%          |
  | self-play only (prototype)        | 0.560       | +34.1%          |
  | **union subset (this run, @30k)** | **0.446**   | **+43%**        |

  shuffle-z climbed 0→13→22→26→35→43% over training; val/copy fell to 0.446 (beats persistence ~55%).
  Both metrics BEAT the prototype's best → stage-1 learns transferable anticipation, latents informative.
  (gate code: stage1_wm_proper.py shuffle_z_gate; copy baseline: compute_copy_baseline.py.
  val = random in-distribution held-out from the union subset — NOT the wide-scene OOD probe; see TODO.)

- **WM-proper K-sweep:** RUNNING (K=50 gate first → 25 → 10 → 5; eval 480/seed1000/n=200). Pending.

## Open / optional (see TODO.md)

- Wide-scene (OOD) representation gate (hold out wide self-play eps).
- 3-corpus stage-1 ablation (demos-only / +SP / SP-only) incl. demos-only "dead" negative control.
- All-224 rerun if resolution gap suspected.
- Stage-2 aux WM loss (β·L_WM, prototype E9) — current stage-2 is transfer-only.

---

# CONSOLIDATION (Jun 15, paused) — corrected findings + disciplined plan

## Hard knowns (verified this session)

- **Control K-sweep, TransferCube, full success (pc_success=rung4), n=200 seed1000 480px 100k steps:**
  K5=21.0 · K10=29.0 · K25=52.5 · K50=73.0.
- **TransferCube K=50 rung-rates (from eval max_rewards):**
  | rung | control | WM-full | WM-bbonly |
  |---|---|---|---|
  | >=1 grasp | 99.0 | 83.0 | 88.5 |
  | >=2 lift | 81.0 | 69.0 | 75.5 |
  | =4 full | 73.0 | 54.5 | 61.0 |
  WM regresses at EVERY rung incl. grasp. KEY: TransferCube grasp is **saturated** (control 99%) -> no
  perception headroom for WM to help; remaining difficulty (rung3->4) is precision, not perception.
- **Prototype, INSERTION, grasp rung (rung>=1), pooled across spawn scales (the real prototype Result):**
  | corpus | in-dist | 1.25x | 1.5x | 2x |
  |---|---|---|---|---|
  | control (no WM) | 54.0 | 43.0 | 26.3 | 19.7 |
  | WM demos-only | 59.3 | 35.3 | 24.7 | 19.3 |
  | WM demos+self-play | **68.3** | **46.7** | **30.7** | **21.7** |
  | WM self-play-only | 47.7 | 34.3 | 18.7 | 21.0 |
  -> WM(demos+SP) BEATS control on the grasp rung WITH headroom (control only 54%), and holds an OOD edge.
- **Stage-1 representation gate passes in both** (current: val/copy 0.446, shuffle-z +43%; prototype Result 1
  similar). Good anticipation features are NOT the issue.
- **WM is NOT undertrained** at 100k: train-loss converges same as control, reaches slightly LOWER train loss
  (0.038 vs 0.042) yet evals worse -> mild overfit / worse generalization, not too-few-steps.

## Corrected interpretation

- Earlier WRONG claim "WM never won downstream" — RETRACTED. Prototype WM(demos+SP) won the insertion grasp
  rung (68.3 vs 54.0). The signal is real, on a **perception-bottlenecked rung WITH headroom**.
- TransferCube full-success was the WRONG vehicle: grasp saturated (99%), remaining difficulty = precision.
- So the WM benefit shows up where perception is the bottleneck AND there's headroom (hard-task grasp / OOD).

## VERDICT OVERTURNED — corrected eval kills the WM win (2026-06-16)

The PDF/insertion result above (and the "corrected interpretation") were measured with a BUGGY eval:
(1) actions clipped to [-1,1] though env/demos use ~±1.6; (2) fixed 500-step episode > env cap -> autoreset
2nd episode + stale temporal-ensemble. After fixing the eval (no clip, dynamic per-env episode, freeze-at-done,
early-stop; gated by eval_audit.py 9 checks, none tautological), we RE-EVALUATED THE EXACT SAME trained models
(e4/e4s2/e4s3 ctrl & wmDS, K=25, 3 seeds, n=96, peak-over-ckpts) via reeval.py (eval core diffed byte-identical
to the cube eval):

| insertion grasp SR>=1 | PDF (buggy) | corrected eval              |
| --------------------- | ----------- | --------------------------- |
| control               | 54.0        | **71.3** (76/65/73) (+17.3) |
| WM demos+self-play    | 68.3        | **67.0** (66/68/67) (-1.3)  |

WM barely moved; CONTROL jumped +17pts. The entire PDF "win" was the buggy eval SUPPRESSING CONTROL (the
trained control policy's eval-time actions clipped/diverged more than WM's). Corrected: control >= WM at EVERY
rung (mean reward 1.65 vs 1.44; SR=4 12.3 vs 8.3).

Cube K=10, 3 seeds, corrected eval (independent confirmation): control mean 1.43 vs WM 0.86; transfer 27% vs 11%.
(Seed-0's apparent WM lift-win was a fluke where control collapsed; control = high-variance/high-ceiling,
WM = low-variance/low-ceiling, i.e. WM behaves as a REGULARIZER that caps performance.)

NET: with a correct eval, WM(demos+SP) ~= or < control on BOTH insertion and cube. No data-efficiency benefit.
The "self-play world model helps" result does NOT survive a clean eval.

CORRECTION OF MY OWN EARLIER CLAIM: I had said insertion was "largely robust" to the bugs because only ~2.2%
of DEMO actions exceed ±1. Wrong axis — eval uses the trained POLICY's actions, which clipped enough to cost
control ~17pts. The bugs materially affected insertion.

Next: visual inspection of rollouts (4 datasets (cube,insert)x(control,wm), recorded via record_eval.py from
the corrected eval, in the GUI) to understand the behavioral pattern before deciding whether any WM variant/regime
is worth pursuing.

## PROBE: object IS encoded — failure is downstream of perception (2026-06-16)

Hypothesis (from rollout videos): the cube WM "ignores the object." **REFUTED by direct measurement.**
Linear probe (closed-form ridge, held out by seed) of each TRAINED student's pooled representation for the
sim ground-truth cube xy, frames from a control rollout fed to BOTH encoders:

| feature (cube xy) | ALL R²    | RMSE       | TABLE (pre-grasp) R² | EE xy R² |
| ----------------- | --------- | ---------- | -------------------- | -------- |
| raw pixels        | 0.72–0.79 | 3.3 cm     | 0.76                 | 0.995    |
| control (mem)     | 0.89      | 2.3 cm     | 0.895                | 0.995    |
| **WM (mem)**      | **0.93**  | **1.7 cm** | **0.935**            | 0.994    |

WM encodes the cube **as well as / better than** control — including the on-table pre-grasp stratum (cube
independent of arm, so not kinematic leakage). So the WM policy whiffs NOT because it can't see the cube
(it localizes it to ~1.7 cm) but because it doesn't convert that into action. **The deficit is downstream of
perception (representation→action), not perception.** Figure: ![probe](probe_object_viz.png)

Consistent with our WM being action-FREE (predictor `forward(S,z)` takes no action — verified): an action-free
WM learns a strong _spectator_ representation (knows where things are, needed to predict the scene) but not the
action→outcome map control needs. Good spectator, poor controller.

Next options (fact-driven; "train a fresh action head" is NOT cheap — the LeRobot ACT port already showed that):

- **Action linear-probe** (cheap, no head training): probe features→expert action on demos. If WM encodes
  position better but action worse than control, that's the direct "good spectator, poor controller" evidence.
- **OOD eval** (reuses existing models): WM is low-variance; the only untested regime where it might win.
- Accept the in-dist negative and write up.

## Divergences prototype -> current LeRobot graft (the confounds)

1. stage-2 aux loss beta\*L_WM DROPPED (current transfer-only). Prototype HAD it — but note prototype's win
   was with aux on; not a proven silver bullet.
2. decoder depth 1 (LeRobot default) vs 6 (prototype).
3. resolution: stage-1 224 -> stage-2/eval 480 (49->300 tokens). Prototype 224 everywhere.
4. latents transferred vs co-trained from scratch.
5. VAE latent as encoder-token (LeRobot) vs decoder-memory (prototype).
6. control = strong LeRobot ACT (73%) vs weak custom ACT (54%).
7. metric: full pc_success vs grasp-rung-rate + OOD scales.
8. task: TransferCube (current) vs insertion (prototype).

## DISCIPLINED PLAN (one change at a time, FROM the working prototype)

Start at the known-good prototype (ACTJepa, insertion + peg/socket SP, WM>control on grasp rung). Change ONE
variable at a time toward the LeRobot graft; after each change re-run the SAME tests (stage-1 representation
gate AND stage-2 grasp-rung comparison at a sensible high K, incl. OOD scales). Lock each variable before the
next.

- [ ] CHANGE 1: task insertion->TransferCube AND swap peg/socket SP -> cube SP (~same #frames). Everything
      else = prototype. Does WM still beat control at some rung/K? (find a sensible K.)
- [ ] CHANGE 2..n: then swap remaining variables one at a time (corpus mix, decoder depth, resolution, aux
      loss, transfer-vs-cotrain, finally LeRobot ACT itself).

## Caveat / cleanup debt

- Prototype training cache `cache_sp` (insertion+peg/socket imgs+vjepa feats, 35GB) was DELETED during the
  storage cleanup. Re-running the prototype requires rebuilding its cache. Prototype scripts (sp_vj_act.py,
  sp_vj_act_s2.py, sp_build_cache.py) live in /tmp/selfplay_probe (ephemeral) — back up before use.

---

# DECISION LOG (Jun 15) — CHANGE-1 first-signal

Verified against selfplay_wm_summary.pdf (authoritative prior result):

- Prototype task = **AlohaInsertion**, K=25, decoder×6, stage-2 loss L1+10·KL+**0.1·L_WM** (aux on), 224px.
- Result protocol = **3 training seeds × n=96 per cell, peak-over-checkpoints** (per-arm val-plateau).
  **n=24 was explicitly REJECTED** (±14-pt swings) — it is only the dev default in sp_vj_act_s2.py.
- Headline metric = **grasp rung (≥1)**, full ladder reported; rung4 (insertion) near-floor 0–20%.
- Result 2 (insertion, K=25, peak, 3×96): control 54.0 / WM(demos) 59.3 / **WM(demos+SP) 68.3** / WM(SP-only) 50.3 (in-dist grasp).
- Prototype stage-1 demo video = 2-task (insertion+transfer-cube, 180 eps); SP = 1000 peg/socket.

CHANGE-1 = swap **task→TransferCube** + **SP→cube SP** (~same frames), everything else = prototype.
Note: TransferCube grasp **saturates** (control ~99% at K=50) -> chose **K=10** to give the grasp rung
headroom (the prototype used K=25 because insertion grasp had headroom at 54%).

**First-signal run (decided):** K=10, **1 seed**, arms = control vs **WM(cube-demos+cube-SP)**, **in-dist only**,
n=96, peak-over-checkpoints. ~2.5-3 h. Isolation: cache*proto_cube/, s2_cube*_ outputs, proto_cube_results.txt;
prototype scripts (~/wm_graft_backup/prototype/) untouched, duplicated as _\_cube.py.
GATE: if WM >= control at any rung -> expand to full faithful protocol (4 arms × 3 seeds × n=96 × OOD f=1/1.25/1.5/2),
sweeping K to find the headroom operating point. If WM < control at all rungs -> task-swap doesn't preserve the
effect; investigate next variable.

---

# CHANGE-1 RESULT (Jun 15) — task->TransferCube + data->cube-SP, prototype code otherwise

Setup: faithful prototype (ACTJepa decoder6, aux beta0.1, 224px, self-test PASSED), ONE variable changed
(task insertion->TransferCube + SP peg/socket->cube SP). K=10, seed0, n=96 in-dist, peak-over-checkpoints.
Isolated: cache_proto_cube (171k fr: 20k cube demos + 151k cube SP); sp_vj_act_cube.py / sp_vj_act_s2_cube.py.

Stage-1 representation gate REPRODUCES prototype on cube: WIDE val/copy 0.55 (proto 0.56-0.57), shuf-gap +23% (proto +34%).

Stage-2 policy (peak over checkpoints 3k/6k/9k/12k):
| rung | control | WM(demos+SP) |
|---|---|---|
| >=1 grasp | 44% | 46% (~tie) |
| >=2 lift | 18.8% | **40.6%** |
| =4 full transfer | 6% | **11%** |
Lift-rung win is consistent at EVERY checkpoint (ctrl mean ~14% vs WM ~32%, ~2x) -> robust, not peak-picking.
K=10 chosen well: control grasp ~44% (headroom), NOT saturated like K=50's 99%.

VERDICT: CHANGE-1 PASSES. The WM benefit survives the task+data swap, converting to policy gains (esp. lift rung)
exactly as the prototype showed on insertion. Confirms the earlier LeRobot-graft K=50 regression (54.5 vs 73)
was a PORTING artifact (resolution 224->480 / decoder 1 vs 6 / aux-loss dropped), NOT the method.

CAVEATS: 1 seed x n=96 is noisy (PDF: +-14-20pt). Lift rung is the robust signal; grasp/full within noise.
NEXT: (a) confirm with full 3-seed protocol + OOD scales; (b) continue one-change-at-a-time toward LeRobot ACT
(next variables: decoder 6->1, resolution 224->480, transfer-vs-cotrain, then the LeRobot ACT itself).
