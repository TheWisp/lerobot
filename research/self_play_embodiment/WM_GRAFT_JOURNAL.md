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
