# ACT baseline reproduction & sim-eval methodology — learnings (2026-06-13)

Context: before comparing any world-model arm, we tried to reproduce the canonical ACT full-insertion
baseline on `gym_aloha/AlohaInsertion-v0`. The published number is **20.6%** (lerobot model card,
n=500). What we found, and the methodology it forces.

## The number decomposition (full insertion, n=500 unless noted)

| pipeline                                                | SR                     | what it isolates                                |
| ------------------------------------------------------- | ---------------------- | ----------------------------------------------- |
| published card                                          | 20.6%                  | their sim version + their training              |
| **published ckpt, migrated, OUR eval (seed 1000)**      | **17.2%**              | our eval is SOUND; residual = sim-version drift |
| our retrain (LeRobot defaults) on current v3.0/AV1 data | 13–15% (peak @25k)     | + training-data quality / recipe                |
| our custom ACT (`sp_vj_act`)                            | ~10% (peak ckpt, n=96) | our reimplementation                            |

**No single big bug — three ~3-pt drifts stack:**

1. **Sim-version drift (~3.4 pt):** published ckpt scores 17.2% in our env vs 20.6% on the card, on the
   SAME checkpoint AND the SAME eval seed-set (both seed=1000, incrementing — confirmed from the ckpt's
   `train_config.json` + `eval_info.json` seed column). So the gap is purely mujoco/gym-aloha version
   change since early-2025; not seed variance, not episode-set. We cannot erase it without their 2025 sim.
2. **Training-data quality (~3 pt):** gold ckpt (trained on old lossless/h264 data) 17.2% vs our retrain
   on v3.0 AV1 ~14%. See AV1 below.
3. **Our reimpl (~4 pt):** custom ACT ~10% vs canonical ~14%; also overtrains (peaks early, declines).

## AV1 / video-compression finding (the train-vs-eval mismatch)

- LeRobot stores ALL datasets as **AV1 video, CRF≈30** (~70× reduction). Measured loss on our frames:
  **PSNR 29 dB**, concentrated at edges (max pixel diff 196) — exactly the cue fine insertion needs.
- Codec history of `aloha_sim_insertion_human`: **v1.0 = raw parquet (lossless), v1.4 = h264, v1.6→v3.0 =
  AV1**. The 20.6% ckpt is from **2025-01-29**, trained on the older lossless/h264 data.
- **Structural train-inference mismatch (ubiquitous to LeRobot):** training reads COMPRESSED stored
  frames; inference (sim render OR real camera) sees PRISTINE frames. Every LeRobot policy has this gap.
  Usually negligible on coarse tasks (Robo-DM: 75× compression, ~no loss), but bites on PRECISION tasks
  (insertion) — our measured ~3 pt. Community treats compression as "fine"; under-discussed for fine tasks.
- This is NOT the 17-vs-20 gap (that's pure sim-version; eval is pristine in both). AV1 only explains the
  training-side 17→14.

## Checkpoint format break + the fix

- Old checkpoints (pre-~May-2025) store normalization as in-model buffers. A LeRobot refactor moved
  normalization into a separate **processor pipeline** (`policy_preprocessor.json` /
  `policy_postprocessor.json`); the loader now HARD-REQUIRES them → old ckpts fail with
  `Could not find 'policy_preprocessor.json'`. Weights themselves are fine.
- **Official fix:** `src/lerobot/processor/migrate_policy_normalization.py --pretrained-path <repo>
--output-dir <dir>` — extracts norm stats from the old state_dict, writes the processor JSONs, saves a
  loadable ckpt. This is how we recovered & evaluated the published checkpoints.
- Old DATASET versions also won't load with current LeRobot (`v2.0` → `NotImplementedError: contact
maintainer`; `v1.x` → path errors). There is a dataset converter `convert_dataset_v21_to_v30.py`.

## Eval methodology (now standing practice)

- **Fixed seed = deterministic starting conditions.** `lerobot-eval --eval.seed=1000` → per-episode seeds
  1000,1001,… → `sample_insertion_pose(seed)` → exact spawn poses. Default is already 1000.
- **The original 20.6% eval set = seeds 1000…1499 (500 eps), n_obs=1, NO temporal ensemble, NO image
  transforms** (`train_config.json`: `image_transforms.enable=false`). We match it by default.
- **PAIRED fixed-seed eval for arm comparisons:** eval control & WM arms on the SAME seed → identical
  spawns → the wmDS-vs-control _difference_ is measured on the same episodes, collapsing spawn variance.
  This is the fix for the ±10-14 pt small-n swings that bit us (E4/E5, the n=24→n=96 corrections).
- **Eval all checkpoints at full n; never select a checkpoint on the noisy in-training n=24 signal then
  report a tight eval of the pick** (that's selection-on-noise — it mislocated the K=50 peak: 24k-grasp
  vs 12k-insertion).
- Caveat: fixed seed pins the ENV; GPU cuBLAS nondeterminism can still flip rare borderline episodes.
  Spawns are identical regardless; full determinism needs torch deterministic mode (slower), not needed
  for paired comparisons.

## eval_info.json anomaly (their bug, noted)

The insertion repo's `eval_info.json` actually contains the TRANSFER-cube eval (pc_success 83.0, video
paths `act_aloha_transfer/080000`). So the published per-episode insertion rewards aren't recoverable
from it; only the README's 20.6% and the seed scheme are trustworthy.

## Task reference

- **AlohaInsertion** (our task): R-arm grasps peg, L-arm grasps socket, lift both, insert. Bottleneck =
  sub-mm alignment. Canonical full-success ~20% @50 demos; caps ~15-32% across methods (data-starved /
  precision-limited; nobody saturates at 50 demos). Best-2025 ~24.8%.
- **AlohaTransferCube**: R-arm grasps cube, hands it to L-arm (bimanual handover). Coarser — no precision
  insert. Canonical ~83% (ckpt 80k). Saturates → supports a data-efficiency RATIO claim that insertion
  cannot; but near-ceiling at 50 demos, so WM headroom lives at LOW K.

## Implications for the WM study

- Our entire WM study ran on AV1 frames + custom ACT → absolute numbers low, but arms equally affected,
  so RELATIVE wmDS-vs-control claims survive. For the strongest baseline, train on lossless data.
- For a data-efficiency RATIO (not just vertical gap), TransferCube (saturates) is a better task than
  Insertion (capped). Consider both: TransferCube for the ratio, Insertion for vertical-gap + OOD.
- Always: paired fixed-seed eval, all-checkpoint-at-full-n peak, lossless training data.
