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
Synchronized group motion is an additional consistency gate, not a pose override.
If a surface correction still disagrees with the group at the existing planning
tolerance, the view is withheld and the last placed pose is carried forward.
This matters for planar targets: depth can correct tilt and normal offset but
does not verify the P2P proposal's tangent motion. An arm-driven held group is
excluded from this gate because it is the attachment prediction being tested,
not independent evidence against a visible slip correction.

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
ends that the appearance match identified. Explicit user-declared appearance
symmetry now names an observed axis (`surface_normal` or `long_axis`) and a
finite order or continuous turn. The find folds only those turns after
registration. Old integer orders still mean finite turns about the visible
face normal. This declaration resolves pose ambiguity; it does not authorize a
changed grasp or relax the placement task.

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

| Mechanism                              | Implemented behavior                                                                                                                                              | Boundary or unresolved limitation                                                                                                        |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Object-relative targets                | Compose demo key frames with the current object's motion. Pre-grasp line targets update while approaching.                                                        | Follow/contact samples are recomposed each control cycle; accepted poses are required, and missing observations remain predictions.      |
| Reacquisition and surface correction   | Fresh segmentation/registration recovers identity; verified surface geometry constrains supported pose components. A remaining group conflict withholds the pose. | Visibility is not a verified full pose. Rejection can leave only prediction available.                                                   |
| Environment and attachment propagation | Groups carry established relationships through missing observations; a held group can be driven by arm kinematics.                                                | An arm-driven held group is not independent evidence that the object stayed fixed in the fingers.                                        |
| Held-object correction                 | Maintain one attachment estimate; direct matching to suitable demo held views updates the next trajectory command.                                                | With grasp prediction allowed, observe from the start of carry through contact. Strict mode still requires verification before approach. |
| Equivalent landing search              | Search the rotations permitted by the demo, choose a feasible full path, and search again if the current landing fails after grasp/hold correction.               | This is a bounded, demo-seeded search, not maximum reachability or general collision planning; details below.                            |
| Playback progress                      | Joint-lag guards slow/pause the demo clock and stop a stalled arm.                                                                                                | A changed pose inserts bounded correction samples before phase progress; there is no rewind or independent terminal task-success check.  |
| Repeated interactions                  | Rebase a reused object's reference before its next attachment and repeat the checks.                                                                              | Corrections and observations must belong to that interaction; a previous grasp cannot authorize the next.                                |

### Equivalent landings: current selection rule

The target is inferred from the current attachment and the objects referenced by
its held key frames. There are no object-name or category-specific exceptions.
The operator's saved rule determines which turns are equivalent: `exact` allows
only the demonstrated relation, `symmetry` uses the declared finite or
continuous turns, and `turn`
samples the full circle every 5 degrees (72 candidates). The latter is a sampling
resolution, not a proof that the best feasible angle is within 2.5 degrees of the
selected one.

For `turn`, the demo saves a task turn axis separately from pose-estimation
symmetry; older demos default to the surface normal. For `symmetry`, the turns
and axis both come from the target's declared visual symmetry. This keeps a
task's allowed placement rotations independent of whether the pose estimator
can distinguish an object's orientation. Each turn rotates about the selected
observed axis, carried by that object's current pose. A face-normal turn uses its measured top-surface centre;
a long-axis turn uses a centreline fitted to central RGB-D cross-sections.
This is a partial-surface estimate, not a complete shape or a guarantee that
the task permits a turn. The separate landing rule still grants that freedom.

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
on the existing interaction/key-frame constraints. Specify the allowed range
of _task_ freedom independently of an object's appearance symmetry. Solve a feasible upcoming contact relation
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

### Grasp and transit freedom at an interaction

**NOT IMPLEMENTED.** The remaining design gap is between a demonstrated tool path and a task constraint. For a
rigid pick and insertion, the required relation is the held object's pose
relative to the receiving object at contact, modulo only a declared task
symmetry. The demonstration's grip point, jaw side, wrist angle and free-space
route are preferences. They may change when another pose of the same object
makes the recorded grip awkward. A line mark before contact is a useful hover
hint, not evidence that its exact Cartesian line is required. During an isolated
rotation, orientation may matter while translation is free within a safe region;
translation becomes required again at insertion. This distinction must be
represented explicitly, not inferred from a single path or object name.

The October 15 rotated-dowel case separates these freedoms concretely. The
demo and live RGB-D surfaces each support a roughly 90 mm long round axis:
central-section median radial residuals are about 0.4 and 0.45 mm. Rotating
the exact recorded first approach by 30 degrees about the live axis changes
an out-of-range forearm pose to a 0.2 mm IK match. That alone is **not** a
valid grasp: at the second recorded frame the CAD jaw meshes would penetrate
the measured tray. An offline object-relative transform that additionally
raises the tool target about 33 mm and shifts it about 2.5 mm across the jaws
produced a 154-sample pre-grasp/grasp plan under 0.5 mm IK residual, with no
modeled self, dowel, tray or ring collision. At the contact configuration its
open fixed jaw has about 1 mm clearance and its closing jaw reaches the other
side of the observed cross-section. This is a _candidate from partial surface
and CAD geometry_, not a certified collision-free scene path or a successful
physical grasp. Unknown space and unmodeled accessories remain unresolved.

The motion core can now compose a constant object-local `adjust` through a
pre-grasp line and follow, and the later grasp attachment correction can
account for that changed gripper-to-object relation. **Automatic candidate
selection and physical validation are NOT IMPLEMENTED.** The appearance
symmetry setting alone must not silently enable these contact changes.

The editor reports a round long-axis suggestion when central cross-sections of
the designated object's visible RGB-D surface fit within the sensor-noise and
relative-radius criterion. It draws that measured line on the clicked demo
frame and includes the length and fit residual,
but does not turn on symmetry automatically. An unseen underside, markings or
off-camera attachment could break the apparent equivalence. This measurement
is useful for proposing grasp candidates; task permission and contact validity
still have to be decided separately.

Key-frame marks now declare `constraint: pose | orientation | best_effort` in the
demo editor. Old marks keep `pose`; new lifts default to `best_effort`. `pose`
keeps the recorded position and orientation. `orientation` rotates at the
position where that stage begins, including during a follow segment, and the
live controller re-anchors it once on stage entry as the reference moves. The
next `pose` stage restores positional control. `best_effort` aims at the
recorded waypoint without making exact arrival a task condition. For a
standalone line or lift, the live controller tries incremental commands until
the arm arrives, stops making encoder progress, or the next increment fails an
IK, joint-step, calibrated workspace or floor check. It holds the observed
joints and replans the next frame from there. The modeled final waypoint is
not a precondition for the attempt. A plan containing later strict stages
still uses a modeled feasible prefix; it must be split at the best-effort
stage to gain the online behavior. Neither a floor-height estimate nor a
workspace box can prove that the waypoint was reached as far as physically
possible; they only reject individual commands outside the calibration. This
is limited to the demonstrated route; **automatic collision avoidance and
alternative waypoint search are NOT IMPLEMENTED.** A separately
collision-checked guarded path uses a relaxed endpoint tolerance within its
following-error bounds. It also stops at the observed safe prefix when its
encoders stop progressing or following error reaches the bounded soft-stop
range. Large unexpected deviations still freeze. Safety stops still apply. The October 15 live
hover reached a collision-checked pose, then
froze 3.9 mm and 1.9 joint degrees from an exact target that had no contact
significance; bounded endpoint settling and tolerance support address that
failure mode while the region planner is built.

A generic planner should sample candidate gripper-to-object attachments from
the observed rigid surface and calibrated jaw geometry. It must verify that
both jaws can close on opposing interior surface patches, that the open and
closing sweeps clear support and clutter, and that a collision-checked joint
path reaches the approach. The demonstrated attachment can rank feasible
candidates. The selected attachment, followed by measured held-pose updates,
determines the next gripper target for the required object relation. The target
relation should be replanned as accepted observations change. If the sampled
set, IK seeds or search budget is exhausted, that is a search failure, not proof
that the object lies outside the arm's physical reach.

The rotated-dowel scene on October 14 exposes three separate failure modes.
The proposed full 3D object motion was about 87 degrees from the demo; this
must not be read as an 87-degree turn in the tray plane. The sequential reach
preview reported 108 mm residual at key frame 1, while an independent seed
solved the same pose to about 2 mm and 2 degrees. A flipped tool pose had a
kinematic path, but an initial offline candidate accepted as a two-jaw grasp
was a false positive: with the 3 mm proximity margin removed, one jaw missed
and the other touched the object's top near an end. The reset prototype used
the same proximity test. Its conservative RGB-D collision grid also reports a
collision for the already-parked arm in the mount region outside reliable
overhead view. The available route therefore has neither a verified pinch nor
a full-scene clearance certificate. No physical retry was sent on that basis.
The live object's simple box measured roughly 23 mm wide, versus roughly 15 mm
from the demo view; a box or CAD mesh must be validated against a known grasp
before being treated as exact contact truth. The stricter reset-prototype check
now refuses proximity-only grasps; its prior positive fixture is retained only
to test the downstream carry and placement path in isolation.

An offline comparison of the saved demo and live RGB-D object clouds gives
independent support for the large 3D motion: after applying the proposed pose,
the median demo-to-live nearest-surface distance is 0.74 mm and 95.6% of demo
points lie within 3 mm of a live point. A further 30-degree roll about the
object's long axis lowers that fraction to 76.1%; a -10-degree change raises
it to 98.2%, so some roll uncertainty remains. This is surface agreement, not
ground truth, since visibility differs and the live mask came from the tracker.
It does not justify discarding the measured rotation to make the wrist easier
to reach. A coarse offline search of rotations about the object's own long
axis found 102 near-jaw poses with a low IK residual, but none had a verified
opposing pinch; two cleared only the _observed_ obstacles after unknown mount
space was excluded. A subsequent search with finer vertical sampling found
opposing side contacts whose jaw meshes stay above the tray. Of its first 60
ranked contact candidates, 35 matched IK within 2 mm and 2 degrees. Two also
cleared observed obstacles at their final joint configurations. Neither cleared
the full unknown-space model: an upper-arm link enters a region near
(-7, -86, 235) mm outside the overhead camera's image. The already-parked arm
also intersects an unknown volume. Thus the refusal is an observation/collision
model limit, **not** evidence that the dowel is physically out of reach. Final
pose clearance alone does not certify an approach or closing sweep, so no
physical act was sent. A sampled top-down search found six geometric side
pinches, but none met the joint limits; this does not rule out other approaches.
These are exploratory samples, not a safe act plan. The
saved frame, tracker state, point-cloud plot and search scripts are under
`/home/feit/Documents/showservo-experiments/20261015/grasp-freedom/`.

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

When grasp prediction is allowed and an attachment seed exists, direct held-view
matching starts before the carry and runs beside the trajectory with one worker
request at a time. It replaces the mandatory stationary retry in this mode.
Strict mode retains its verification boundary before approach/contact; neither
mode invents an attachment when no grasp estimate exists. It uses the shared camera's frame
publication timestamp and interpolated, checked encoder history, never FK at
job completion. Missing timestamp brackets or serial gaps longer than two
motor periods reject the frame. The host publication timestamp is **not a
calibrated exposure timestamp**: sensor/processing latency remains a measurement
uncertainty. This is not multi-camera fusion.

The reference bank comes from segmented views within the current demo attachment,
including rotation during carry, and ends before opening starts. Image times
interpolate the recorded fingertip transforms. The bank retains up to seven
views with different camera directions in hand coordinates (the existing hold
work budget); views within the existing three-degree solver tolerance add no
coverage. Selection ranks these references against the current viewing direction
and predicted depth support. Ranking only proposes a reference: it cannot
approve a pose. Reference masks are refreshed with SAM3 before matching.

Failed references are skipped while the arm pose, attachment prediction and
projected depth remain unchanged within existing hold tolerances. Changed views
or occlusion allow another attempt. Accepted observations form a rolling window
of up to five views, aged out after the existing five-second hold budget. Once
consensus exists, each new accepted observation can update it; old evidence alone
cannot publish a new measurement. This bounds stale evidence but does not yet
explicitly detect slip or split the window at a sudden attachment change.

The existing segmentation, appearance-strength, surface checks and consensus
rules remain required.
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
held-view geometry is prepared once per attachment outside the act coroutine.
Prediction-enabled acts continue carrying while it prepares and matches. Strict
acts still wait at the verification boundary and plan from the verified result.
The perception worker consumes one queued job at a time,
so launching more asynchronous jobs does not create more GPU capacity and can
delay tracking. `holds._held_view` explicitly assumes a stationary arm: it reads
the fingertip immediately before fetching the frame, without a capture-time joint
interpolation contract. Moving that helper into a background task unchanged
would introduce image/arm timing error.

**Remaining concurrency work, not implemented:** introduce priority/coalescing in
the shared perception worker; calibrate exposure time against encoder reads; and
recognize attachment changes that invalidate an accumulating consensus. The
initial stationary verification remains only when prediction is disallowed or
no usable seed exists. Active inspection and general collision-aware transitions
also remain unimplemented. Historical timing measurements below predate the
continuous observer and must not be used as measurements of this implementation.

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

### Comparison of the three successful October 14 rounds

The observed position update is not yet evidence of better insertion accuracy or
speed. A read-only comparison of the three physical recordings gives:

| Act               | Held position during insertion                        | First to last arm command | Insertion sequence to release | Approximate final ring displacement |
| ----------------- | ----------------------------------------------------- | ------------------------: | ----------------------------: | ----------------------------------: |
| `20261014_075438` | Grasp prediction, earlier controller                  |                   27.83 s |                        5.29 s |                                4 mm |
| `20261014_140613` | Grasp prediction, per-frame controller                |                   36.96 s |                        8.26 s |                                9 mm |
| `20261014_163916` | Measured position accepted during per-frame execution |                   42.24 s |                        7.97 s |                                9 mm |

All three used speed 1, no injected error, and completed insertion and withdrawal.
Timing uses actual command-consumption timestamps. The insertion interval starts
with the first key-frame-6 command and ends at recorded gripper release; it does
not claim to identify first physical contact. The newest insertion interval is
0.285 s shorter than the preceding one, but this single pair cannot attribute that
difference to the correction. Its whole-arm path was also longer: approximately
691 mm versus 496 mm, using encoder FK resampled at 10 Hz. Different object starts,
grasps, selected landing angles and controller versions confound a causal speed
comparison. The stationary checkpoint still accounts for command gaps of 7.18,
5.99 and 5.80 s respectively.

Ring displacement is an independent _disturbance proxy_, not peg-to-hole error.
Offline Lucas–Kanade matching of visible ring features from the initial RGB to
the final RGB gives 3.84, 8.97 and 8.70 mm respectively, using calibrated
projection onto an approximate ring-height plane. Each match passes a forward/
backward image check and must remain on the visible ring. A separate outer-contour
ellipse measurement gives 4.52, 10.49 and 9.03 mm. Stationary-background image
motion at the ring is under 0.11 pixels in every run. Segmentation thresholds,
shadows, partial occlusion and ring tilt affect the measurements, so the rounded
values do not establish a small improvement between the last two rounds. These
evaluation scripts use the visible green ring to select pixels; they do not add
object-specific rules to the controller. Neither measurement uses P2P, group
transport or the held-pose estimate as truth. Missing rim observations during
occlusion remain gaps rather than interpolated motion evidence.

In the newest run, the attachment was accepted 2.194 s after the insertion sequence
started, with 5.779 s remaining before release. It therefore cannot explain better
initial approach alignment. The 4.987 mm command change and 1.217 mm inter-view
spread reported above remain evidence of correction application and estimator
agreement, respectively; neither is an independently measured reduction in task
error. Rotation still used prediction. The peg and bore are partly occluded at
the critical entry frames, preventing a defensible millimetre-scale alignment
comparison from these recordings alone.

A causal evaluation requires matched repeats on one controller version, with the
held observer running in both conditions and only application of its updates
changed. This keeps computation and other target adaptation comparable. Measure
approach alignment when independently visible, ring movement, contact duration
and success across repeatable grasp offsets. The current three trials establish
that the update reaches hardware; they do not establish an accuracy benefit or a
regression caused by that update.

Durable local evidence: `/home/feit/Documents/showservo-experiments/20261014/round-comparison/`
contains `metrics.json`, the reproducible measurement scripts, annotated raw
image crops, timing plots, rim-feature diagnostics and
`insertion_comparison.mp4`. The video aligns the beginning of the insertion
sequence, preserves elapsed time, and labels when the held-position update was
applied. This comparison issued no hardware commands.

### Continuous-observer rollout and separate lift failure (October 14 evening)

The reference-bank and scheduling change was deployed as `72fb49916`. After the
operator reset the scene, physical act `insertion/acts/20261014_220340` started its
held observer before carry, but stopped in key frame 4: the shoulder encoder
remained near 15.8 degrees with a 10.8-degree command, reaching the existing
five-degree transport guard. No fresh held correction was accepted or applied;
this trial does not establish an accuracy or completion benefit for the change.

A bounded probe through the gizmo target API reproduced the stall. After
releasing the peg, an empty-gripper retreat also failed and exceeded 70% shoulder
load. However, the probe helper itself imposed the same five-degree/half-second
following-error stop as playback. This is not an independent demonstration that
ordinary gizmo operation or the joint cannot move. The operator subsequently
reported that Park worked, and the API confirmed it had disconnected. Park uses
a different joint-space trajectory and a per-command twelve-degree clamp against
the measured angle, without playback's five-degree/half-second stop. Its success
does not isolate the effect of posture, path, or permitted command lead.

No accepted held correction caused this failure: none had been applied. Runtime
register diagnostics showed the expected gains and position mode, enabled
torque, no reported motor error flags and position limits well outside the
current angle. The arm was re-anchored to its observed pose to remove commanded
strain before the operator parked it. The initial request for physical inspection
was based on an overly strong interpretation of the guarded probes; a mechanical
cause remains unproven. No gains, transport tolerances or protection limits were
relaxed.

The observer logged a missing capture as “the arm went away”; this message did
not establish a disconnect. Capture diagnostics now distinguish a missing
encoder bracket, a camera lacking timestamps and intentional stopping. The
synchronization checks remain unchanged.

The operator then requested retries after Park. Four Act attempts on the same
deployed code produced two refusals before motion (pick reachability, then a
target surface-validation failure) and two physical approaches. Acts
`20261014_221606` and `20261014_221744` stopped during key frame 1, before grasp:
wrist roll stopped following near 97.36 and 97.10 degrees respectively, with
nominal commands of 127.05 and 124.85 degrees. The subsequent GUI state reported
wrist-roll overload in both cases; recovery cleared that joint's latch, and Park
completed after each. The shoulder followed during these approaches. No error
injection was enabled and neither run reached held-pose observation.

The first physical retry used 40 mm/s and 30 deg/s Cartesian limits, the second
20 mm/s and 15 deg/s. Slowing the Cartesian walk did not avoid this posture's
failure. This is not an isolated joint-speed experiment: the pose was reacquired
between runs, and Cartesian speed limits do not directly bound joint speed.
In particular, the first run's nominal wrist command accelerated to about
180 deg/s immediately before freezing. The records establish a repeatable
posture-specific approach failure, not a general inability to move the wrist or
a proven mechanical cause. Further repetitions of this approach were stopped;
the arm was parked, torque released and the original Cartesian limits restored.

Retry images, polling traces, frozen-joint states and the comparison report are
under `/home/feit/Documents/showservo-experiments/20261014/continuous-held/retry_after_park/`.

### 2026-10-15: independent pose and task axes, live insertion retry

The saved insertion demo now declares continuous pose symmetry about the
dowel's observed long axis. Its existing `turn` placement rule rotates about
the ring's measured surface normal. These are independent declarations: the
first folds equivalent dowel poses during finding; the second selects a
task-equivalent final relation with the ring. The prior object sidecar is
backed up under `/home/feit/Documents/showservo-experiments/20261015/symmetry-axis/`.

Act `20261015_104148` found the dowel (71 inliers), passed the reach check,
grasped it, and reselected a feasible carry/landing path after the first carry
plan failed. It stopped early in key frame 4, before insertion: shoulder lift
remained at 13.1 degrees while the command reached 8.0 degrees, with 61.6%
reported load. The 5-degree, 0.5-second transport guard froze the arm. The
recorded fingertip stayed near 9 mm above the base frame while the planned
carry had begun lifting it. The arm was recovered, the dowel released onto the
tray and the arm parked. This run validates the symmetry declaration's live
find and grasp, but not a successful placement; the physical cause of the
stalled lift is not established by the log alone. In particular, the 61.6%
load reading is below the reported overload threshold, temperature was normal,
and no servo fault was reported. Playback freezes after 0.5 seconds beyond a
fixed 5-degree following error; the Cartesian gizmo has a different controller
and can keep driving toward its target. This guard is a software stop, not
evidence that the arm could not lift the dowel. A comparison with the same
grasp and lift through the gizmo remains needed.

A follow-up reach check exposed a separate clearance bug. The robot's gripper
read 63.9 units while the recorded hover asked for 63.948; planning treated
that 0.048-unit difference as a contact event and refused to shorten a
best-effort hover. Clearance now uses the existing 0.3-unit gripper-still
threshold in planning and live execution. Actual opening and closing remain
strict at their contact positions.

The 2026-10-15 direct gizmo comparison reached the transformed grasp area with
the arm and closed the fingers 2.2 units short of their command. A vertical
lift then moved the fingertip from -6.7 to 40.7 mm, but camera frames showed
the dowel still touching the tray and sliding toward its left wall. A further
lift request to 90 mm stopped at 74.0 mm, 16.7 mm from target, with shoulder
lag 4.6 degrees and load 56% of full scale; there was no servo fault or freeze.
The gripper-gap check had reported “holding” throughout. It cannot establish
that the object is airborne. The arm was released and parked, leaving the
dowel against the wall. The evidence supports a grasp/contact-path failure,
not an intrinsic inability of this arm to reach the grasp area. The next trial
needs an inboard object, a checked grasp location and a lift whose progress is
verified from the object's motion before sideways carry. The existing demo
can express a vertical lift key frame, but a fixed 45 mm lift was not enough
in this off-centre case.

The original failed act's saved frames show the same pattern. Between frames
40 and 59 of `20261015_104148`, the dowel shifted and tilted while the first
carry commanded a sideways move and rotation. Aligned depth at its lower end
in frame 59 puts the visible top about 19 mm above the local tray; the dowel's
measured thickness is about 17–19 mm. Its underside was therefore at or near
the tray, not cleared for transport. This agrees with the gizmo comparison,
although the exact contact forces were not recorded. The 2.3-unit shortfall in
the gripper reading was evidence that its fingers closed on _something_, not
proof of a load-bearing grasp or an airborne object.

The reset helper also showed a stale camera frame while a second server was
recording. It searched all recording directories by modification time, so the
image could come from the wrong session. It now reads an aligned RGB-depth
pair directly from its own server. Its `held_short` return is no longer marked
`ok` when the observed fingertip has not reached its target; that distinction
matters when a loaded lift stops centimetres short without a motor fault.
