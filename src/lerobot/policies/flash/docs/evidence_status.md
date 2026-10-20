# FLASH evidence from recorded data

Audited 2026-10-20. This distinguishes preparation, offline pose replay,
physical execution and independently verified task success. The local datasets
listed here are not in git.

## Demonstration extraction

The prepared `insertion-migrated-v7` bundle under
`~/.cache/huggingface/lerobot/demos/insertion/flash/v1/prepared/` contains a
591-frame source episode, 11 execution intervals, a grasp sample at 228 and a
release sample at 513. The demonstrated pickup/lift through sample 290 is
anchored to the achieved grasp; target-relative transport begins after that.
The previous v6 bundle mistakenly made the 242–290 lift relative to the ring
because a 9.7-second keyframe rounded to arm sample 290 at 9.71 seconds, then
was compared against the unrounded annotation. The preparation report counts 188 directly observed
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

Recorded live RGB-D from 2026-10-20 initially seemed to supply a second
counterexample: a stationary dowel produced low-residual shape fits whose
object-frame origins differed by 21.5 mm. Rechecking the corresponding modeled
surface points showed at most 1.44 mm change, with a 0.46 mm change at their
centre. The reference origin is arbitrary and remote from the visible object;
rejecting on origin motion was a false rejection. A later partial view did
produce a roughly 90 mm candidate jump, which still needs separate assessment.
The [fixture note](../../../../../tests/policies/flash/fixtures/README.md)
explains which frames were retained for regression. A surface residual alone
does not certify every task-relevant orientation or hidden contact pose.

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

The v7 bundle's first physical launch failed during startup before arm motion:
only 10 of 12 required appearance correspondences were found, and the fallback
shape fit was rejected on the old origin-motion check. This is not a failed
insertion or evidence about the corrected control trajectory. The subsequent
surface-only continuity fix has a two-frame real-RGB-D regression check; a
fresh physical outcome remains unverified. The next v7 trial recorded 70
stationary scene frames and 71 motor ticks, issued zero arm commands, then
reported `dowel usable pose unavailable at interval entry`. Its dowel pose was
last directly observed about 1.5 seconds before termination, beyond the
configured 0.8-second age limit. The worker delivered 43 pose results with no
reported per-role fit failures, yet the accepted scene state published only
seven fresh dowel poses during the recorded interval. Replaying all 70 stored
RGB-D frames through the same backend recovered a dowel pose on every frame;
replaying every fourth or eighth frame also succeeded. That evidence narrows
this failure to the live producer-to-scene-estimator path or earlier streaming
state. It does not prove the final insertion pose would be accurate.

The later v7 approach trials issued real commands but did not reach the
grasp. In the 16:28 trial, a dowel appearance fit at camera frame 206 passed
the matched-point RANSAC despite disagreeing with the remaining segmented
depth: its visible-surface 90th percentile was 6.6 mm, versus 3.7 mm for the
preceding pose and a configured 5 mm tolerance. The bad fit changed the
planned grasp goal and the planner stopped. Reprocessing that exact recorded
frame reproduced the fit. The backend now checks every matched pose against
the visible RGB-D surface and falls back to the independent shape fit when
that check fails. A fixture from the real frame locks in this regression.
The sealed runner's corresponding rule checked a candidate against camera
depth; its other tracking limitations remain relevant.

Subsequent physical trials did not reach the grasp. The plain dowel cannot
seed enough object-specific optical-flow points, so the slower direct RGB-D
updates need a causal prediction between fits. Enabling the existing bounded
stationary prior exposed an event-time bug: its newer predicted timestamp
prevented later-arriving direct measurements from refreshing the belief. The
scene estimator now rebases a prediction at the measurement's source frame,
and startup readiness checks the latest direct-support time rather than the
published prediction's evidence label. This made the approach start, but its
first ring point-track anchor replayed historical frames serially on the
scene worker (about 0.8 s), making the dowel prediction exceed the planner's
0.8 s source-age bound mid-motion. Replaying one historical frame at a time
between new camera events reduced the next trial's maximum scene-processing
lag to 0.25 s; it still ended on a RealSense cache/notification race that
returned one capture twice. The camera now publishes its frame and
notification under the same lock, with a forced-interleaving regression test.

The following GUI-launched attempt on 2026-10-20 recorded 246 frames and
reached the approach to grasp, but stopped before contact: the dowel had no
usable pose at the next interval entry. Its last direct view was about six
seconds old, all fourteen object-point seeds lacked enough tracks, and no
environment group had been established. The bounded stationary prediction
expired as designed. This is a missing occlusion-support mechanism, not
evidence that the dowel moved. The arm was parked afterward. The GUI now
displays live RGB-D and motor state from FLASH through LeRobot's optional
observation stream, and local episode links open its ordinary dataset replay.
The replay has no FLASH-specific pose outline yet. No successful redesign
insertion has been recorded.
