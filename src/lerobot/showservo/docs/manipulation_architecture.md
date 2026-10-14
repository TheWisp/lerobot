# Key-frame manipulation runtime

The manipulation API has one execution architecture. Every act validates its demo
reference poses, then executes the key-frame runner. `line`, `follow` and `lift`
are key-frame movement modes within that runner. The old marks runner and the
separate grasp/place/world planners have been removed.

Older recordings can still contain named marks such as `pregrasp`, `grasp_end`
and `preplace`. `showservo.motion.key_frames` converts that saved representation
into key frames. This is data-format compatibility, not another execution path.
Act requests cannot select a runner; an obsolete `runner` field is rejected.

## Object-independent interaction sequences

Objects are designated on the RGB-D recording and identified by user labels.
The recorder, worker, tracker and act runner do not select a policy by object
name or class. Shapes, poses and symmetry declarations belong to the demo.
Names in historical tests and failure comments describe evidence, not runtime
dispatch rules. The old `benchmarks/showservo_reset/reset_policy.py` experiment
is a separate, scene-specific reset program; it is not imported or invoked by
the demonstration framework and must not be used to reset a different demo.

`demo_holdings` reads every recorded attachment from gripper commands and
observations. Each grasp/release interval is executed in the same key-frame
runner, with its own measured or explicitly predicted attachment correction.
A held tool can follow several interaction segments before release. Completing
the first such segment does not detach it from the gripper in group tracking.
Later grasps repeat the reference, grasp and held-pose checks. A missing or
failed second grasp cannot inherit the first grasp's correction. The editor
shows every attachment interval and replay records each selection separately.

For a subsequent attachment interval, an object used again is referenced at its
visible pose before that interaction, after the preceding release. This avoids
applying the object's earlier demonstrated displacement twice. That reference
passes the same depth-surface verification as the initial reference. An unseen
later reference is refused, rather than silently reusing the original pose.
Grasp and release transitions need separate recorded follow segments; an
attachment is to the object that segment names. A held key frame must reference
another object or the world.

**New demo** clears the previous demo's live object state, labels and injected
error settings, without moving the arm or deleting saved data. A new recording
cannot overwrite a different saved demo with the same name. Record with the
leader or gizmo, save, designate the objects, then author the key frames. RGB-D
recording does not require choosing a tracking object beforehand.

This supports recording general hand-object and object-object motion sequences
on the single gripper, with rigid pose-based replay. It does not infer contact
forces, friction, deformation, liquid motion, or success for an unseen task.
Several contacts inside a streamed segment remain a replayed demonstrated path;
there is no general contact-dynamics solver or per-contact replanning. Occluded
held-object accuracy remains limited by the documented visual checks and the
explicit no-slip fallback. The sequence regressions use a simulated arm and
supplied poses; a new physical task still needs its own observed trial.

## Code ownership

`lerobot.gui.api.manipulation` assembles the endpoint routers. Code imports the
owning module explicitly; there is no `pregrasp.py` forwarding facade.

| Module                  | Responsibility                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------- |
| `runtime`               | Shared session data, configuration and dedicated executors                             |
| `session`, `camera`     | State readout, frames and calibration from the GUI-owned camera                        |
| `acquisition`, `worker` | Object registration, reacquisition and worker job transport                            |
| `tracking`              | Geometry verification, acceptance, accepted pose history and following                 |
| `holds`, `held_views`   | Checked demo references, object-to-gripper relationships and direct held-view matching |
| `planning`              | Object-relative key-frame composition, reach checks and landing choices                |
| `execution`             | Reference preflight, the key-frame runner and manual approach controls                 |
| `demos`                 | Demo persistence, object designation and key-frame editing                             |
| `recording`, `replay`   | Raw/control evidence, trial records, playback and exports                              |
| `rendering`, `groups`   | Overlays and conditional environment-group propagation                                 |

Pure registration, rigid-motion calculations and key-frame sample planning live
in `lerobot.showservo.motion`, outside the GUI API. The worker entrypoint is
`benchmarks/manipulation_worker.py`. The HTTP prefix remains `/api/pregrasp` for
compatibility with existing clients and saved scripts; this name does not select
an execution architecture. Camera and arm ownership remain with the GUI.

## Measured-surface correction

P2P proposes a rigid pose. Before control accepts it, `tracking` compares the
reference object's visible surface with the current masked, aligned depth.
`showservo.surface_constraint` can align a supported matching planar patch's
normal and normal offset. This measures two orientation components and one
translation component; tangential position and twist remain from the proposal.
The correction rotates around the predicted reference patch centre and then
moves only along the observed normal. It does not assume a level table or an
upright object and does not change P2P's internal map.

The planar check first projects reference points through the proposed pose into
the live object mask. Points outside the mask, without depth, or behind measured
foreground do not participate in the reference plane fit. This uses the same
visibility rule and noise allowance as the general surface comparison. Otherwise
a gripper hiding one face can make the independently fitted majority planes
describe different physical faces, producing a false rotation disagreement.
Conversely, live plane fitting uses only points within the existing local
association radius of the transformed visible reference subset. This prevents
newly revealed material from replacing the common patch with another face.
The result records the visible sample counts and the reference patch's material
anchor. Held-view correction uses that same anchor in demo and live fingertip
coordinates; it does not extrapolate a partial-view rotation to the centre of
hidden reference geometry. Cross-reference/temporal agreement is still required,
and inconsistent rotations retain the explicitly permitted grasp prediction.

Split-sample agreement, surface span, curvature and local overlap checks decide
whether that plane supports a correction. When a reference plane is supported
but the current plane is not verified, the observation is rejected. For a
non-planar reference, the general surface check may retain an unchanged proposal
when it agrees. Existing visibility, depth and centre-support gates still apply.
Group-motion comparison remains diagnostic; it is not a blanket pose override.

A non-planar elongated surface can also constrain its principal direction when
the leading covariance eigenvalue is separated beyond the point-noise budget.
Independent split fits must agree, the measured surface extents must match, and
both surfaces must explain each other within the existing depth tolerance. This
observes two angular components; axial roll and translation remain the proposal.
The replay explicitly labels axial roll as unobserved. It is not a slip check or
a substitute for a missing/shortened held-object view.

Reference acquisition tries the measured image-axis alignment and its opposite
in addition to quarter turns. These are appearance proposals only: pixels map
back to the original RGB-D coordinates before the unchanged 3D consensus gate.
Subsequent surface refinement is local and must preserve the matching-point
consensus. A symmetric shape's slightly better surface fit cannot reverse the
ends that the appearance match identified. Explicit user-declared symmetry
continues to choose equivalent turns after registration.

The same surface policy checks acquisition, secondary targets, observations used
for held-object correction, and the saved demo reference poses inverted during
planning. Raw proposals and actual control decisions remain separate in records
and replay. This prevents a corrected live pose from being compared against an
uncorrected, tilted demo reference.

See [pose verification](pose_verification.md) for measured results and limitations.
Visible surface agreement is not independent full-pose ground truth. In-hand
ambiguity and active inspection remain unresolved. The adaptation mechanisms
below distinguish accepted measurements from predictions and proposed work.

## Adaptation mechanisms and their limits

This is the review map for consolidating adaptation within the single key-frame
architecture. Status describes the reactive implementation of October 14, 2026. Object names in the evidence identify experiments, not policy branches.

| Mechanism                              | Implemented behavior                                                                                                                                | Boundary or unresolved limitation                                                                                                                         |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Object-relative targets                | Compose demo key frames with the current object's motion. Pre-grasp line targets update while approaching.                                          | Follow/contact samples are recomposed each control cycle; accepted poses are required, and missing observations remain predictions.                       |
| Reacquisition and surface correction   | Fresh segmentation/registration recovers identity; verified surface geometry constrains supported pose components.                                  | Visibility is not a verified full pose. Rejection can leave only prediction available.                                                                    |
| Environment and attachment propagation | Groups carry established relationships through missing observations; a held group can be driven by arm kinematics.                                  | An arm-driven held group is not independent evidence that the object stayed fixed in the fingers.                                                         |
| Held-object correction                 | Compare demo and live attachments, including direct matching to demo held views; replan the remaining path before applying a correction.            | After the initial checkpoint, a background observer retries held matching through contact until opening starts; accepted results update the next command. |
| Equivalent landing search              | Search the rotations permitted by the demo, choose a feasible full path, and search again if the current landing fails after grasp/hold correction. | This is a bounded, demo-seeded search, not maximum reachability or general collision planning; details below.                                             |
| Playback progress                      | Joint-lag guards slow/pause the demo clock and stop a stalled arm.                                                                                  | A changed pose inserts bounded correction samples before phase progress; there is no rewind or independent terminal task-success check.                   |
| Repeated interactions                  | Rebase a reused object's reference before its next attachment and repeat the checks.                                                                | Corrections and observations must belong to that interaction; a previous grasp cannot authorize the next.                                                 |

### Equivalent landings: current selection rule

The target is inferred from the current attachment and the objects referenced by
its held key frames. There are no object-name or category-specific exceptions.
The operator's saved rule determines which turns are equivalent: `exact` allows
only the demonstrated relation, `symmetry` uses the declared order, and `turn`
samples the full circle every 5 degrees (72 candidates). The latter is a sampling
resolution, not a proof that the best feasible angle is within 2.5 degrees of the
selected one.

Each turn rotates about the **base-frame vertical** through the target's measured
top-surface centre. It is not a declared object-local symmetry axis, a detected
hole centre, or permission to change arbitrary position/orientation components.
This distinction matters for tilted targets, asymmetric surfaces and other tasks.

`motion.rank_landings` solves sampled target-relative poses within calibrated
joint limits, starting from the demo's arm posture and using its joint changes
as hints. It rejects a candidate at the first position/orientation tolerance
failure. Among survivors, the cost is the mean squared difference from the demo's
arm joint angles. `_plan_act` tries full paths from the current arm state for at
most the first four ranked candidates and accepts the first that passes its
reach, joint-step and workspace/floor checks.

Consequently, "reachable" means reachable by this solve and these checks. The
chosen candidate is not necessarily the one with the largest joint-limit margin,
best obstacle clearance, smallest physical placement error or shortest duration.
Other IK configurations are not exhaustively searched, and the four-candidate
cutoff can leave a feasible later candidate untried. The current full-path checks
are not a general arm/held-object collision check.

The real insertion run `20261014_075438` illustrates the adaptation: the remaining
path with its previous landing failed (worst IK residual 14.95 mm). Reselection
found six sampled-feasible turns, chose 355 degrees (-5 degrees), and produced a
full path with worst IK residual 0.50 mm. The physical trial inserted, released
and withdrew without a freeze. These are computed reach residuals and one observed
outcome, not measured placement accuracy or a reliability rate. The optimization
in `2c6953410` only avoids further solves after a candidate fails; it preserves
the feasible ranking. The diagnosed-case profile fell from 11.64 to 1.22 seconds.

**Proposed, not implemented:** represent required, preferred and free components
on the existing interaction/key-frame constraints. Specify the axis, centre and
allowed range of a symmetry, rather than assuming base vertical or inferring
task freedom from one demonstration. Solve a feasible upcoming contact relation
and arm configuration, then plan a collision-checked free-space entry into it.
Free transit position may change while the required contact relation remains
fixed. Evaluate joint-limit margin, clearance and execution cost; retain demo
similarity as a preference. The tradeoffs and search budget need review, rather
than another unexplained weighted score. Budget exhaustion must be distinguished
from an exhaustive infeasibility result.

For review, records should include the constraint declaration, candidate angles
and configurations, scores/margins, full-path rejection reasons, search duration
and any exhausted budget. Currently they retain the selected turn, survivor
count, selected cost and plan summaries, not the entire candidate search.

### Held-pose evidence: what the earlier comparison established

A held pose is the object's attachment relative to the fingers. Given synchronized
observations, `T_tip_object(t) = inverse(T_base_tip(t)) @ T_base_object(t)`.
Comparing this with the demonstrated attachment lets the controller reproduce
the desired object relation despite a different grasp. Predicting a fixed
attachment from grasp-time estimates cannot observe subsequent slip or a wrong
initial estimate.

The comparison from recorded insertion run `20261013_223811` is retained. Offline
direct SAM3/DINO/RGB-D matching to demo held views 350 and 370 independently
accepted live frames 38, 41, 42, 43 and 44. They implied additional gripper-target
corrections of 7.06 and 7.15 mm relative to the saved fallback. The resulting target
positions agreed within 0.15 mm across references, but orientation corrections
differed by 3.99 degrees. This supports repeatable positional evidence, not a
ground-truth object-centre error, measured slip, or verified full orientation.
Earlier hover frames 31-35 failed verification. The first three accepted later
views preceded observed ring motion, but were already in the recorded contact
follow segment; absence of visible target motion does not prove free-space clearance.

Commit `972403e6b` installed direct held-view matching, including its position-only
result when reference orientations disagree and prediction is explicitly allowed.
It was not reverted. That version tried it at the first held line, before
later approach/contact lines, without collecting later moving views. In real run
`20261014_075438`, P2P lost the object during grasping; six completed direct matches
failed common-surface verification and the seventh expired with the shared
five-second budget. Zero observations were accepted, so the recorded correction
remained **grasp prediction** through release. A visible object and an attempted
match are not themselves an accepted attachment measurement.

Local evidence is in the experiment archive under
`20261014/held-pose-replay/summary_sam3.json`,
`20261014/held-pose-replay/held_comparison_sam3.png`, and
`20261014/rotation-trial/{report,mechanism-audit}.json`; the original acts retain
RGB-D frames, arm samples and measurement decisions. These local artifacts are
not bundled with this document.

### Per-cycle execution and attachment updates

Every planned sample retains its object frame, local pose, held flag, gripper
command, joint hint and floor reference. The live executor composes
`object_motion @ local_pose @ attachment_correction` (the correction only when
held) on each control cycle. A new accepted receiver pose or attachment estimate
therefore changes the next command during a follow, including contact. The
controller keeps the current demonstration phase; it does not rewind. Bounded
correction steps precede phase advancement and its gripper event. Terminal
settling also reads updated targets and uses the existing reach and servo
settling tolerances, rather than requiring numerical-zero IK error at a limit.

The motor loop owns the bus and receives at most one pending sample. A compare
and swap prevents a slow solve from replacing an already consumed command.
Updating the pending command preserves the playback clock and lag guards;
a temporary empty producer queue cannot clear a hard-lag timeout. Release and
withdrawal remain ordered. Sliced paths start from current joints, because their
old endpoints may have changed during live execution.

Every new step passes IK, calibrated joint bounds, joint-step limits and
workspace/floor checks. An infeasible step holds phase and gripper while the
executor searches the remaining path under the demo's permitted landing turns.
This live search tries all sampled-feasible candidates in posture-cost order,
including a bounded entry from the current command; it has no four-candidate
cutoff. Changed observations invalidate a search beyond the existing solver
accuracy; the next step is checked again before commanding it. Initial and
checkpoint planning still use the four-candidate limit described above. Neither
path provides general collision checking or exhaustive IK-branch search.

After the initial hold checkpoint, direct held-view matching runs beside the
trajectory with one worker request at a time. It uses the shared camera's frame
publication timestamp and interpolated, checked encoder history, never FK at
job completion. Missing timestamp brackets or serial gaps longer than two
motor periods reject the frame. The host publication timestamp is **not a
calibrated exposure timestamp**: sensor/processing latency remains a measurement
uncertainty. This is not multi-camera fusion.

The existing segmentation, surface checks and consensus rules remain required.
An accepted full or permitted position-only correction updates live targets and
the arm-driven group attachment. Prediction remains labelled as such; copying
the correction into a group is not independent verification. Each observation
window is bounded by the existing hold budget, and each result records its
observation and acceptance times. Release and new interactions invalidate old
results. Opening freezes further attachment updates; it does not remove the
remaining release/withdrawal commands. Records retain correction revisions,
per-command goals, and live landing-search attempts. More informative later
views may still fail verification; live execution alone does not make a pose
observable.

Each reactive command also records the frame transform, relative goal, attachment
correction and revision used before its IK solve. These inputs stay with the
queued sample through replacement and consumption. A revision accepted while IK
is running must not retrospectively label an older command as using the update.
The recorded inputs reconstruct the desired goal; encoder FK remains the
separate evidence for actual arm motion.

The ring outline in replay is the saved controller pose, not proof of a fresh
measurement. In run `20261014_075438`, the last accepted ring view was at +10.12 s.
Later group records refused its own motion when visibility was below the
tracker's 97% requirement and carried it with group 0 (the table). The resulting
outline stayed almost fixed even if the object moved independently. Replanning
from that estimate cannot repair the missing observation. Replays retain the
last visual timestamp and own-group-fit refusal reason.

## Waiting, parallel work and servo freezes

The main stationary command gap in `20261014_075438` was 7.18 seconds: approximately
five seconds of held-view matching, then 1.64 seconds between hold selection and
the next command, including correction planning and landing reselection, plus
settling/processing. An earlier 2.39-second gap measured the demo's attachment.
Neither was a latched arm freeze. The run also recorded individual serial-read
delays, including about 845 ms before initial motion; their cause is separate from
the explicit measurement wait and is not established by this trial.

Some work already runs concurrently: the jog control loop owns the motor bus on
its own thread, the planner has a dedicated executor, and perception runs in a
separate worker process. Normal demo grip/carry measurements already start beside
the approach and cache completed results, including verified failures. Direct
held-view reference preparation is still performed within its measurement call.
The act coroutine nevertheless waits for a held estimate and its resulting plan.
The perception worker consumes one queued job at a time,
so launching more asynchronous jobs does not create more GPU capacity and can
delay tracking. `holds._held_view` explicitly assumes a stationary arm: it reads
the fingertip immediately before fetching the frame, without a capture-time joint
interpolation contract. Moving that helper into a background task unchanged
would introduce image/arm timing error.

**Remaining concurrency work, not implemented:** prepare and cache direct
held-view geometry before the checkpoint; introduce priority/coalescing in the
shared perception worker; calibrate exposure time against encoder reads; and
recognize attachment changes that invalidate an accumulating consensus. The
initial stationary checkpoint and its bounded wait remain. The new observer
starts after that checkpoint, rather than claiming to have removed the earlier
7.18-second pause. Active inspection and general collision-aware transitions
also remain unimplemented.

Parallelizing matching hides computation only if the carry provides usable
evidence. Repeated frames of the same uninformative view can still yield no pose.
If profiling shows CPU contention despite dedicated threads, an isolated planner
process with bounded numerical-library threads is an option; the last trial does
not establish that this is the source of the serial delays.

Joint-lag, overload and invalid-read freezes remain a separate control problem.
Do not remove those guards to eliminate a perception wait. Timing records should
separate intentional observation waits, queue/compute time, plan search,
measurement age, servo catch-up and hardware faults. Validation must cover late
results during rotation, changed target/attachment, cancellation/release, worker
saturation and a real rollout demonstrating that fresh evidence was applied
during motion without degrading control-loop latency. Replay accuracy alone
does not validate this scheduling or the physical correction.

## Regression coverage

The key-frame runner is tested against expected grasp, carry, placement, drop,
reach, occlusion and hold-correction behavior. Planner tests verify reference
poses, recorded follow samples, timing, gripper commands, joint hints and floor
references directly. They no longer require a second implementation as an oracle.
Persisted mark-to-key-frame conversion, old replay records and the HTTP routes
retain their own compatibility checks.

### Physical validation of reactive execution (2026-10-14)

Commit `2d2f16b74` passed 307 targeted regression tests. Real insertion act
`20261014_140613` completed pickup, insertion, release and withdrawal without a
servo freeze; the final RGB frame shows the dowel in the ring. No error was
injected. The first attempt had stopped before motion because a saved click now
hit the tray; clicking the visible dowel restored a strong match without changing
the demo or acceptance gates.

During contact, a ring observation was accepted at +23.923 s. The next recorded
command at +23.974 s had a goal displaced by 8.45 mm, with phase 42 retained while
the controller inserted correction steps. That is goal displacement (including
any nominal sample increment), not ground-truth ring motion. It demonstrates
that the physical run consumed a new target observation during an active follow.
The initial allowed landing was 355 degrees; no live landing reselection was
needed. Median command spacing during the contact follow was 33.6 ms. The initial
stationary held check still produced a roughly six-second command gap.

The moving held observer made fresh matching attempts, so acquisition/timestamp
plumbing ran on hardware. Every completed match was rejected for insufficient
common surface; no fresh attachment correction was installed. The successful
insertion used grasp prediction throughout. Therefore this trial validates live
target retargeting and execution, **not** measured in-hand pose recovery or general
reliability. The later held-view matches that supported an older offline case
were insufficient in that trial. The scene ended with the object
inserted and needs separation before another ordinary trial.

Durable local evidence: `/home/feit/Documents/showservo-experiments/20261014/reactive-trial/`
contains the report, tests, deployment/commit logs, polling images and complete
act record. It is separate from the source repository.

### Fresh held-position correction on hardware (2026-10-14, 16:39)

After autonomous separation of the objects, act `20261014_163916` ran commit
`8a5897446` with the common-surface constraint and command provenance. It completed
pickup, insertion, release and withdrawal without a servo freeze or injected
error; raw final frame 149 shows the dowel standing in the ring. The reset left
the ring in a different location from the previous run, and the existing allowed
landing search supplied the carry plan. The reset itself used inspected,
calibrated gizmo motions; it was not validation of a general collision planner.

The initial stationary check still rejected weak appearance matches. During
subsequent motion, five observations against demo references 350, 370 and 373
established a held-position correction, with 1.217 mm inter-view position spread.
Reference rotations disagreed by 90.37°, so rotation explicitly remained the
permitted grasp prediction. This is partial measured recovery, not full pose
observability or independent millimetre accuracy.

The new attachment was accepted at Unix time 1791992385.086707. Its first consumed
command followed 33.516 ms later; recomposition began 22.262 ms after acceptance.
Holding the nominal sample and target motion fixed, replacing only the old
attachment changes that command's desired position by 4.987 mm and leaves its
orientation unchanged. The recorded arm sample 1.831 ms after consumption has
exactly the same commanded joints. All 677 recorded compositions reconstruct
their goals exactly; 135 held commands use the new attachment revision. This
establishes that fresh in-hand position evidence affected the running action.

The latest contributing camera publication preceded command consumption by
745.724 ms. The 33.516 ms figure is acceptance-to-command latency, not camera-to-
command latency; publication time is still not calibrated exposure time. One
successful insertion does not establish reliability under arbitrary occlusion,
slip, initial grasps or target motion. Full held orientation remains unresolved.

Durable local evidence: `/home/feit/Documents/showservo-experiments/20261014/held-reset/`
contains reset telemetry/images, the actual retest, acquisition attempts and the
command reconstruction audit in `retest/report.json`.
