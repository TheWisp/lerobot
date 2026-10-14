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
architecture. Status describes the implementation at `2c6953410` (October 14,
2026). Object names in the evidence identify experiments, not policy branches.

| Mechanism                              | Implemented behavior                                                                                                                                | Boundary or unresolved limitation                                                                                                                   |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Object-relative targets                | Compose demo key frames with the current object's motion. Pre-grasp line targets update while approaching.                                          | A streamed follow/contact segment uses a precomputed path; new observations do not continuously retarget it.                                        |
| Reacquisition and surface correction   | Fresh segmentation/registration recovers identity; verified surface geometry constrains supported pose components.                                  | Visibility is not a verified full pose. Rejection can leave only prediction available.                                                              |
| Environment and attachment propagation | Groups carry established relationships through missing observations; a held group can be driven by arm kinematics.                                  | An arm-driven held group is not independent evidence that the object stayed fixed in the fingers.                                                   |
| Held-object correction                 | Compare demo and live attachments, including direct matching to demo held views; replan the remaining path before applying a correction.            | Direct matching currently occurs while stopped at the first held line. Later clearer views do not update that correction during the contact stream. |
| Equivalent landing search              | Search the rotations permitted by the demo, choose a feasible full path, and search again if the current landing fails after grasp/hold correction. | This is a bounded, demo-seeded search, not maximum reachability or general collision planning; details below.                                       |
| Playback progress                      | Joint-lag guards slow/pause the demo clock and stop a stalled arm.                                                                                  | Joint tracking is not object-relative convergence. There is no automatic rewind after target motion or general terminal task-success check.         |
| Repeated interactions                  | Rebase a reused object's reference before its next attachment and repeat the checks.                                                                | Corrections and observations must belong to that interaction; a previous grasp cannot authorize the next.                                           |

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
It was not reverted. The deployed runner tries it at the first held line, before
later approach/contact lines. It does not collect a moving carry window or retry
at the later views that supported the offline comparison. In real run
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

**Proposed, not implemented:** use supported observations gathered throughout
free carry, expressed in the correct fingertip frame at each capture time. Keep
measured components separate from unobservable ones, and retain prediction only
for the latter under the declared fallback policy. Verify material-patch identity
and account for partial visibility; removing the whole-shape extent or overlap
gate alone would not establish a valid partial match. A changed attachment or
slip invalidates the previous consensus. At a verified pre-contact boundary,
choose the supported correction and replan from the current state. If the useful
view only occurs later, plan a clear inspection pose that exposes it before
contact; do not simply descend into contact to obtain a better image.

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

**Proposed concurrency contract, not implemented:**

- Extend reference preparation and caching to the direct held-view masks and
  geometry before the hold checkpoint. Cache by recording/reference, calibration
  and verification-policy identity, and invalidate it when those inputs change.
- Capture immutable RGB-D, mask/reference identity, capture time and synchronized
  joint/FK evidence during a checked free-space carry. Interpolate arm state at
  capture time, reject unsupported timestamp gaps, and never attach the arm pose
  at job completion to an older image.
- Give tracking and fresh held measurements bounded scheduling budgets in the
  perception worker. Coalesce/drop obsolete pending frames instead of growing a
  queue. A delayed result retains its observation time, attachment/interaction
  identity and evidence age; release, slip or a new grasp invalidates it.
- Accumulate supported attachment evidence while the arm follows its already
  checked carry. Do not silently splice a newly estimated correction into an
  executing contact path. At the pre-contact boundary, use a fresh validated
  consensus or the explicit fallback/refusal policy; wait only for any remaining
  bounded measurement budget.
- Prepare candidate plans from immutable state snapshots in the background.
  Recheck the target, attachment, current joints and relevant scene geometry
  before execution; invalidate stale plans. Parallel candidate solving needs
  isolated solver state and bounded CPU use. It must not starve the control loop
  or create another motor-bus owner.

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
before contact without degrading control-loop latency. Replay accuracy alone
does not validate this scheduling or the physical correction.

## Regression coverage

The key-frame runner is tested against expected grasp, carry, placement, drop,
reach, occlusion and hold-correction behavior. Planner tests verify reference
poses, recorded follow samples, timing, gripper commands, joint hints and floor
references directly. They no longer require a second implementation as an oracle.
Persisted mark-to-key-frame conversion, old replay records and the HTTP routes
retain their own compatibility checks.
