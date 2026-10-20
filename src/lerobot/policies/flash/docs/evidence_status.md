# FLASH evidence from recorded data

Audited 2026-10-20. This distinguishes preparation, offline pose replay,
physical execution and independently verified task success. The local datasets
listed here are not in git.

## Demonstration extraction

The prepared `insertion-migrated-v6` bundle under
`~/.cache/huggingface/lerobot/demos/insertion/flash/v1/prepared/` contains a
591-frame source episode, 11 execution intervals, a grasp sample at 228 and a
release sample at 513. Its preparation report counts 188 directly observed
dowel poses and zero fully observed ring poses. The ring count requires care:
the cached evidence has 575 `predicted` and 16 `lost` ring frames. The axial
symmetry wrapper changes a successful RGB-D pose to `predicted` when its
orientation is locked to the declared support plane; it retains the direct
translation marker. Thus zero does **not** mean the ring was unseen in all
frames, nor does 575 establish independent six-axis accuracy. The earlier v3
bundle, before this orientation constraint, counted 575 ring frames as
`observed`.

The selected dowel grasp reference is a pre-occlusion frame. The bundle labels
the grasp pose and held contact pose as predicted from that frame and recorded
calibrated fingertip motion. There is no independently measured held dowel
pose during insertion in this prepared skill. Extraction has produced a
replayable spatial skill, but has not established that its contact geometry is
accurate enough for a live insertion.

## Offline perception replay

The [initial RGB-D replay](rgbd_replay.md) used a SAM3 mask, DINOv2 matches and
aligned depth. On static ring frames 100–130 it fitted 31/31 frames, with
0.22 mm median reference-centre displacement. That measures consistency of
the fitted poses, not pose error against ground truth. For the turning and
occluded dowel it fitted 19/45 sampled frames, without reliable reacquisition
after lift. A held-view reference restored one fit but later disagreed with
the group proposal by 3.6 cm at the object centre; the visible-depth checks
did not identify a clear winner.

Recorded live RGB-D from 2026-10-20 supplies a second counterexample: a
stationary dowel produced low-residual shape fits whose object-frame origins
differed by over 20 mm, then a partial view led to a roughly 90 mm candidate
jump. The [fixture note](../../../../../tests/policies/flash/fixtures/README.md)
explains which frames were retained for regression. A surface residual alone
therefore cannot certify the full pose.

## Physical rollouts

The sealed prototype has completed real insertion rollouts. For example,
`~/.cache/huggingface/lerobot/demos/insertion/acts/20261016_112149/act.json`
reports `ok: true`, and its final camera frame visibly shows the dowel upright
in the ring. This supports a visually successful physical result for the
**sealed prototype**. The runner's `ok` flag means its action sequence
finished; the recording has no independent insertion verifier or calibrated
contact-pose ground truth.

The **new FLASH redesign** has no recorded successful physical insertion.
Its LeRobot episode manifests under
`~/.cache/huggingface/lerobot/demos/flash_insertion_trial_20261020*` all report
`failure` or `interrupted`, including the latest four trials:

| Trial time (2026-10-20) | Recorded frames | Manifest result                                                     |
| ----------------------- | --------------: | ------------------------------------------------------------------- |
| 11:34:39                |              19 | Motor deadline continuity boundary missing                          |
| 11:39:11                |              84 | Dowel usable pose lost during motion                                |
| 11:45:32                |              79 | Dowel goal changed beyond replanning envelope after a bad pose jump |
| 11:52:44                |              13 | Dowel usable pose lost during motion                                |

The actual recordings establish that the hardware path launched, commanded
the arm and wrote standard LeRobot episodes with FLASH traces. They also show
why no task-success claim is justified yet. The current architecture work
targets the pose-evidence boundary exposed by those trials before another
physical insertion attempt.
