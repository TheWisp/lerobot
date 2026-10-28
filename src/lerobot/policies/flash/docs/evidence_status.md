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

The repeatable [prepared-skill audit](../../../../../benchmarks/flash_audit_legacy_skill.py)
now checks every retained arm sample against the sealed demo's time, calibrated
fingertip pose and gripper command. The local
`insertion-migrated-v10-preferred-hover` bundle has eleven contiguous intervals,
grasp/release events at arm samples 228/513, and a constant demonstrated held
attachment; all recorded sample values match exactly. A visual contact sheet
confirms the grasp, lift, insertion and release boundaries. The audit caught
one wrong relation in v7: the final world-relative retreat was made
ring-relative; v10 retains it as a motion relative to the interval entry.
The demo's first hover at sample 179 is annotated `best_effort` and is now a
preferred connection endpoint; the grasp at sample 228 remains required.
When a preferred endpoint cannot be reached, the planner advances only along
its checked IK prefix and records the achieved fraction. This is not collision
avoidance: the current skill still lacks general per-axis required/preferred/free
constraints and a scene-aware path planner.

On 2026-10-21, the Run → FLASH inline editor regenerated this demo through its
own asynchronous preparation action into a review candidate. The new reference
masks and appearance model points/descriptors are byte-for-byte identical to
the selected v10 bundle's v7 reference. The fresh pose trace has 172 observed
dowel frames, 567 predicted ring frames and 16/8 additional lost frames for
dowel/ring compared with the older retained trace. Ring poses agree exactly
where both traces are valid. The new grasp reference is displaced about 3.45 mm
from v10, while the reference-frame dowel surface residual is essentially
unchanged (2.116 vs 2.125 mm, 90th percentile). The dowel uses a new axial
canonicalization in this regeneration, so a raw pose-origin delta alone does
not establish physical error. This candidate is available for visual review
but is not the default live skill. No physical trial or independent contact
accuracy test has validated it.

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

## Two-demo check (2026-10-28)

The same Run → FLASH demo replay now lists the insertion and gamepad-on-cube
recordings. Their locally prepared bundles are
`insertion/flash/v1/prepared/flash-scene-20261028-causal` and
`pick_place/flash/v1/prepared/flash-gamepad-cube-20261028-causal` under the
LeRobot demo cache. Both use the demo's recorded RGB-D and scene/group
configuration copied from a matching trial config. The latter is a second
task case, not a special branch in the tracker.

| Demo and pose source                  | Directly observed held-object frames | Predicted | Lost |
| ------------------------------------- | -----------------------------------: | --------: | ---: |
| Dowel insertion, appearance + scene   |                                  169 |        84 |  338 |
| Gamepad onto cube, appearance + scene |                                  178 |         3 |  322 |
| Gamepad onto cube, surface + scene    |                                  247 |       128 |  128 |

These are the 591 insertion and 503 gamepad frames processed by the respective
prepared skills; they are **coverage counts, not accuracy scores**. The
surface pass used the same demo annotation in the isolated tracking worker and
is saved as `pick_place/flash/v1/prepared/flash-gamepad-cube-20261028-surface`.
It finds new gamepad poses during some partially visible carry frames where
appearance matching loses it, but still loses other carry frames. The
pregrasp reference surface agrees with its frame-98 mask and depth (523 of 531
projected reference points inside the mask, 0.9 mm 90th-percentile depth
residual). At carried frames 290 and 300 the direct surface estimate is about
20 mm from the grasp-attachment prediction at the reference-cloud centre;
which is more accurate in 3D is not established. Some direct fits have large
same-frame reference-surface residuals when the gripper occludes the object.
The cube's 503 processed frames have 322 direct and 181 predicted poses in the
appearance pass, with no lost frames. Both demos therefore need an
observability-aware comparison of vision, group motion and attachment before
we use a recovered pose to correct contact motion.

The next candidate surface pass checks both directions of the fit: current
object pixels against the saved reference surface, and the nearest saved
reference point on each camera ray against _clean_ depth pixels inside the
current mask. It uses the existing
surface-fit threshold and minimum inlier count. A reference point hidden behind
the currently visible surface can fail this conservative check; failure means
the current view does not certify the pose, not that the pose is proven wrong.
Rejected candidates remain in the scene-evidence sidecar and can be inspected
in Run → FLASH → demo replay → 3D scene. The live action skill is unchanged.

| Candidate prepared bundle (2026-10-28) | Held-object observed | Predicted | Lost |
| -------------------------------------- | -------------------: | --------: | ---: |
| Gamepad/cube, original surface pass    |                  247 |       128 |  128 |
| Gamepad/cube, clean-depth check        |                  226 |       110 |  167 |
| Dowel/ring, clean-depth surface pass   |                  272 |       128 |  191 |

The gamepad keeps direct poses at carried frames 290 and 300, where the clean
visible reference points agree with depth. At frames 400 and 410 it withholds
the direct pose. At frame 430, the first version rejected a candidate at 75.7
mm depth p90; the pixel-counted version accepts the _same candidate pose and
same reference model_ at 3.4 mm p90. Several reference points projected onto
the same depth pixel at different depths, so the first version counted hidden
rear points as separate depth disagreements. The closest point per pixel is
the visible predicted surface. This corrects that score, but does not prove
the candidate pose is accurate against independent ground truth. The remaining
coverage change is a coverage-versus-certainty trade-off, not a measured
accuracy gain.
The final candidates are `flash-gamepad-cube-20261028-depth-pixels` and
`flash-dowel-ring-20261028-depth-pixels` in their respective demo datasets.
No insertion surface pass without this check has been prepared, so there is no
same-backend pre-check comparison. Its default appearance-based skill has not
been replaced. Neither candidate has been validated in a physical rollout.

## Physical rollouts

The sealed prototype has completed real insertion rollouts. For example,
`~/.cache/huggingface/lerobot/demos/insertion/acts/20261016_112149/act.json`
reports `ok: true`, and its final camera frame visibly shows the dowel upright
in the ring. This supports a visually successful physical result for the
**sealed prototype**. The runner's `ok` flag means its action sequence
finished; the recording has no independent insertion verifier or calibrated
contact-pose ground truth.

At the time of the earlier trials below, the **new FLASH redesign** had no
recorded visually successful physical insertion.
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
The Run → FLASH tracker panel now shows recorded accepted poses and the
separate gripper-predicted held outline for recordings that include it. No
visually successful redesign insertion had been recorded at that point.

In the 19:11 live trial, the arm grasped and lifted the dowel before the next
generated carry connection failed its IK check. The retiming code tried the
far endpoint first and rejected the path when that single local solve failed;
it now subdivides failed IK steps. The recorded start and goal pass offline
with the existing 3 mm and 3° limits. A physical rerun remains outstanding:
after that stop, the robot bus found arm motors 1–6 but not gripper motor 7 on
two connection attempts, so the return path did not move and the dowel remains
in the gripper.

The [offline IK preview](../../../../../benchmarks/flash_preview_ik.py) of the
revised bundle completes all eleven intervals from the demonstration's initial
joint state. Beginning at the last trial's measured lift pose, it also
completes the remaining carry, dense insertion, release and retreat: 75 ticks
for the first carry, 15 for the next approach and 138 for the 4.6 s insertion
interval. This checks calibrated IK geometry and event timing only. The trial's
grasp-time pose supplies an approximate attachment; the preview has no live
perception, collision or motor feedback and cannot establish task success.

## Later same-day attempts and held-pose audit

The 20:18:38 physical attempt visually placed the dowel upright in the ring,
but its manifest says `interrupted`: the next demonstrated retreat failed a
joint-speed check after release. The 20:33:10 attempt completed the command
sequence without a planner error, but the dowel entered tilted and fell out
after release; its manifest says `unknown` because the skill has no independent
success milestone. Neither manifest alone expresses the physical outcome.

Among the 19 attempts from 17:22:16 through 20:33:10, six issued an
`attach-held` event. Four later recordings contain held-pose telemetry and
instance masks; none contains an accepted post-grasp held observation. The
gripper attachment remained an unverified prediction. The
[held-pose evidence plan](held_pose_plan.md) records the source audit, the
grasp-closure timing finding and the partial-observation experiments to run
before enabling live attachment corrections.
