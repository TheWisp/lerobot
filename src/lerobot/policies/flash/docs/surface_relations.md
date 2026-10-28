# FLASH adaptation through surface relationships

Status: proposed

A demonstration records one successful interaction. It does not establish
which other contact locations would work, or which changes remain valid later
in the task. Object-wide symmetry labels cannot express that intent, while
independent freedoms at each keyframe can silently change an earlier grasp.
FLASH needs an editable description of permitted changes that survives the
whole interaction and lets the live robot choose a feasible path.

**Proposal.** Add two concepts to the existing objects and timeline:
**surface regions** and **relationships between regions**. An annotation
defines allowed relative placements, including where similar observed regions
may substitute for the demonstrated contact and which orientations remain
permitted. The planner jointly chooses the contact placement and robot
path. That choice becomes part of the continuing physical state. Keyframes
provide examples and validation constraints; they do not reset the choice.
The timeline and its annotated keyframes define the sequence. Labels such as
grasp, push or place are optional descriptions and select no execution logic.

This proposal covers rigid-object imitation using partial RGB-D observations.
The robot, other objects, and an environmental reference can participate in
the same relationships. Geometry supplies evidence for compatibility;
annotation supplies permission. Surface similarity alone cannot establish
friction, attachment, or successful task completion.

## Requirements

Preserving intent and causal consistency takes priority over finding a short
path. Operator convenience follows a representation that the planner can
actually enforce.

| ID                | Priority | Requirement                                                       | Target and reason                                                                                                                                                                | Check                                                                                                                           |
| ----------------- | -------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| <a id="r1"></a>R1 | P0       | Express adaptation through object regions and their relationships | The same schema covers pushing, sliding, grasping and object-object interaction; no object-name, cylinder, end-swap or symmetry-order field                                      | Author the examples below using the same schema; rename the objects without changing behavior                                   |
| <a id="r2"></a>R2 | P0       | Preserve the consequences of an adaptation                        | No independent contact reassignment at the next keyframe; actual attachment governs subsequent goals                                                                             | Shift a grasp, then verify all later hand goals account for the shift and the final object relationship remains unchanged       |
| <a id="r3"></a>R3 | P0       | Keep permission and measured evidence distinct                    | Similarity cannot create permission; missing geometry cannot count as observed agreement                                                                                         | Test an unpermitted matching patch and an unobserved contact surface; retain the demonstrated patch when adaptation is disabled |
| <a id="r4"></a>R4 | P0       | Evaluate a path from the current robot state                      | Enforce declared relationships and robot constraints across the ordered approach, interaction and dependent goals                                                                | Replay the saved joint-branch failure; test a convenient grasp that prevents the subsequent operation                           |
| <a id="r5"></a>R5 | P0       | Preserve existing timing and execution contracts                  | Async perception/planning, fixed 30 Hz motor ownership, dense interaction timing and distance-dependent transport                                                                | Use the [architecture's timing tests](architecture.md#interval-timing) with delayed replans and moved targets                   |
| <a id="r7"></a>R7 | P0       | Express separate, continuous and coupled placement freedoms       | A painted region alone cannot express discrete orientations or rotation about an offset pivot; use one relationship schema for all of them                                       | Distinguish quarter-turn alternatives from a continuous arc; verify pivot rotation moves position and orientation together      |
| <a id="r8"></a>R8 | P0       | Extract and execute a timeline without required task stages       | Keyframes select participants, relationships and interval timing; grasp state only enables a supported gripper-pose fallback, without choosing a motion mode or reference object | Extract a push with no grasp events; changing interval labels cannot change goals; closure alone cannot create an attachment    |
| <a id="r6"></a>R6 | P1       | Author and inspect through one scene viewer                       | Demo/Edit, Test and Replay share scene rendering, layers and controls; preserve provenance, units, evidence and selected correspondence in the regular dataset sidecar           | Save/reload a rule; compare the Test snapshot with its Replay; hiding layers cannot alter policy decisions or recording         |

## Observations and constraints

<a id="o1"></a>**O1. A tracked surface is partial.** The shared surface backend
stores an initial visible cloud; it does not produce a complete, verified
object model. Its [implementation reference](surface_tracking.md#2-establish-one-reference-and-coordinate-frame)
also records the failure when an asymmetric object turns over. Therefore a
region must retain its observation and alignment evidence, and the design must
work without a complete mesh.

<a id="o2"></a>**O2. Current grasp selection enumerates annotated axial rotations.**
The [selector](../grasp_symmetry.py) returns the first candidate whose prefix
and first required carry pass the planner. The [editor](../../../gui/static/flash_demo_editor.js)
exposes an object-wide axis and end interchangeability. The generic replacement
needs contact-region freedom and trajectory cost, without a required analytic
axis.

<a id="o3"></a>**O3. Surface overlap does not determine the executed joint path.**
In the [2026-10-26 experiment](../../../../../docs/proofs/flash-geometry-sequence/EVIDENCE.md#can-the-arm-guide-registration-without-a-candidate-list),
a 99.4% cloud-overlap fit selected a grasp configuration 253 degrees away in
joint-vector norm from the sequential IK result. Path feedback corrected the
choice. A wrong-object fit still passed every sampled IK goal. Therefore visual
validation and path feasibility must both constrain adaptation.

<a id="o4"></a>**O4. The legacy importer assumes a grasp-and-release sequence.**
The [sealed-demo importer](../sealed_demo_import.py) requires one held role,
one target role, and grasp/release timestamps. Its interval layout uses those
times to choose motion modes and reference roles, and its carried-object
samples are predicted from one grasp attachment. The more general
[compiler](../compiler.py) already accepts explicit interval annotations.
The new extraction path must derive its sequence from those annotations and
surface relationships; the legacy importer remains an explicit compatibility
adapter, rather than the template for every demonstration.

<a id="c1"></a>**C1. Permission describes an interaction, with object-relative scope.**
It belongs to regions and relationships that persist through time. A similar
surface elsewhere is usable only when the annotation permits it. A keyframe
can narrow the permitted set without making a fresh independent choice.
([R1](#r1), [R2](#r2), [R3](#r3).)

<a id="c2"></a>**C2. A commanded adaptation has physical consequences.** Choosing
a different contact before a rigid grasp is different from moving that contact
after the fingers close. The latter requires a permitted physical motion or
a measured update, such as slip; a planner cannot change it by relabeling the
contact. Future planning consumes the resulting relationship, including
uncertainty. A demonstrated sliding contact remains a changing relationship
instead of being treated as a rigid attachment. ([R2](#r2).)

<a id="c3"></a>**C3. Matching tolerance is not task tolerance.** Depth uncertainty,
permitted interaction variation, solver convergence, and robot limits have
separate sources and units. There is no universal surface-similarity threshold
that certifies every grasp. ([R3](#r3), [R6](#r6), [O1](#o1).)

<a id="c4"></a>**C4. A surface region does not determine allowed orientation.**
Several placements can touch the same region while only some headings are
permitted. Rotation about an offset pivot also changes position. Permissions
therefore constrain the whole relative placement, including any coupling
between translation and rotation. Equivalent final placements do not imply
that the intervening motion is valid while contact is maintained.
([R2](#r2), [R4](#r4), [R7](#r7).)

## Two concepts

### Surface region

A region selects measured parts of one participant. It can contain several
disconnected patches. It refers to source RGB-D frames, masks, and verified
alignments, with uncertainty and visibility; it need not have an analytic
shape, center, axis, or semantic landmark. A region on the robot may use its
calibrated geometry. Reuse the existing object identity and coordinate-frame
version. ([R1](#r1), [O1](#o1).)

Store source references and a bounded geometry cache. Combine observations
only where their alignment is supported; retain unlinked views separately.
Newly visible surface can add evidence without changing old region coordinates.
A failed alignment cannot silently deform or extend a region.

### Relationship

A relationship names two participants and records:

- the surface regions involved and the demonstrated arrangements to preserve;
- where similar replacement regions are permitted, initially just the
  demonstrated region;
- allowed relative orientations and any coupling between position and
  orientation, expressed in a named participant's coordinate frame;
- required geometric conditions, their tolerances and evidence sources;
- the time span and events over which the relationship applies.

The conditions use generic geometry: relative separation and orientation,
contact or clearance, and agreement of local surface shape. A fixed relative
transform is one particularly strict relationship. The existing milestone
mechanism checks whether a required relationship was achieved. The annotation
author and supporting demo frames are retained. ([R1](#r1), [R6](#r6), [C1](#c1).)

The geometric comparison covers the interacting neighborhood, including both
finger-contact areas when observable. It must preserve the required relative
arrangement and approach, not merely find two patches with similar curvature.
The selected demo neighborhood fixes the support that must be explained; the
solver cannot improve its score by shrinking it to a tiny agreeing patch.
Whole-object geometry still supplies collision and downstream constraints.
Hidden contacts retain an explicit assumption or unknown state. ([R3](#r3),
[C3](#c3).)

Relationship scope is selected on the timeline. A keyframe may introduce,
narrow or end a requirement, while each interval declares whether to preserve
the demonstrated motion and timing or generate the connecting motion. Robot
commands, including finger position, remain recorded actions. Neither an
interval name nor a gripper closing command identifies the interaction type
or establishes an attachment. ([R5](#r5), [R8](#r8).)

This does not eliminate rigid transforms from computation. They place the
measured regions and robot links in space. Their arbitrary origin is not a
semantic object center, and no named axis is required to express a permission.

### Allowed relative placements

An allowed placement specifies where a participant may be and how it may be
oriented relative to the other. Region selection constrains the location;
orientation and coupling conditions complete the permission. Several
alternative placements or continuous ranges may be allowed. All active
requirements must hold for the selected alternative. This remains information
inside a relationship, rather than a separate object-specific rule system.
([R1](#r1), [R7](#r7), [C4](#c4).)

Keep three questions separate: did the editor permit this placement, does the
observed geometry support it, and can the robot reach it while satisfying the
rest of the task? A proposal must pass all required checks. A matching surface
cannot expand the permission. Geometry that is unobserved is unknown, rather
than evidence that the region is prohibited or interchangeable.
([R3](#r3), [R4](#r4).)

Disconnected surface areas are useful when, for example, either of two handles
is permitted but the intervening body is excluded. Two end regions are another
example only if the middle is excluded; a grasp allowed along the whole shaft
uses a connected region. Separate allowed orientations at the same location
are also possible, even when the selected surface region is connected.
([R1](#r1), [R7](#r7).)

A quarter-turn permission may admit headings 0, 90, 180 and 270 degrees around
a chosen reference direction, while excluding 45 degrees. A bounded turn may
admit every heading from -30 to +30 degrees. These are illustrative annotations,
not defaults or matching tolerances. A full turn is a third choice. Any task
tolerance around an allowed heading is recorded separately from sensor
uncertainty; it must not accidentally turn discrete alternatives into free
rotation. Endpoint equivalence also does not equate different amounts of joint
or cable winding along the executed path. ([R4](#r4), [R7](#r7), [C3](#c3).)

Rotating about a point requires a direction as well as a pivot. The editor can
take that direction from a selected surface normal, an environmental reference,
or a drawn rotation handle, then show it for review. This is an annotation
reference with observation provenance, not a claim that a true symmetry axis
or geometric centre has been reconstructed. It can also be off the surface.
For surface-guided sliding, the measured contact geometry determines the
compatible change of orientation without requiring a manually drawn axis.
([R1](#r1), [R3](#r3), [R6](#r6).)

For a pivot c, a permitted rotation R changes a relative placement (p, Q) to
(c + R(p - c), RQ), with all quantities expressed in the reference participant's
coordinates. Here p is position and Q is orientation. This couples the arc
travel and turn instead of independently changing angles and XYZ. Rotating in
place keeps p fixed; moving around a pivot while keeping orientation fixed
holds Q fixed instead. The preview must distinguish these three behaviors.
Store reference-frame identity/version, the selected pivot/direction when
needed, and the permitted ranges or alternatives. Use metres and radians in
the record and readable units in the editor. ([R6](#r6), [R7](#r7).)

## Demo extraction from the timeline

Preparation consumes a regular RGB-D and robot-state episode, object
annotations, keyframes and relationship spans. It processes the recording
through the same visual pipeline as Test and the tracker debugger. Every
selected object is tracked independently of whether the gripper touches it.
Offline refinement may use later evidence, with that distinction preserved in
the result. ([R3](#r3), [R6](#r6), [R8](#r8), [O4](#o4).)

The prepared data contains:

- Source frame IDs, times, calibration and the unchanged recorded robot and
  gripper actions.
- Per-object masks, filtered observed surfaces, supported alignments between
  views, and uncertainty or gaps. Store source references and bounded caches;
  a complete mesh and a globally continuous pose are not prerequisites.
- At each keyframe, the selected participants, observed surface arrangement,
  required relationship and permitted adaptation, with their evidence.
- For each interval, its motion source and timing. A demonstrated interval
  retains its time-indexed relative surface arrangements and robot motion;
  a generated connection retains the keyframe goals, constraints and speed
  source. Store a transform sequence only where the registration supports it.
- Optional relationship predictions, separately identified from direct
  observations, so an unsupported interval is visible during review.

These quantities allow cross-recording surface matching at the selected
keyframes without treating an uncertain fitted pose as ground truth. A lost
alignment remains a visible gap until a supported bridge reconnects it;
regeneration cannot silently replace it with a smooth invented trajectory.
The editor previews these exact inputs before choosing a prepared version.
([R2](#r2), [R3](#r3), [R6](#r6), [O1](#o1).)

Grasp state belongs to the prediction layer. When direct vision is unusable,
a supported rigid gripper-object relationship can propagate the last object
estimate using measured gripper motion. Trustworthy visual evidence remains
the source for correction; a grasp label or finger command cannot override it.
A pushing or sliding contact does not justify rigid propagation. Loss of the
attachment withdraws that fallback without advancing a keyframe, switching
the target object or selecting a new motion mode. Any observed relationship
can still constrain planning; this distinction concerns prediction through
missing evidence. ([R2](#r2), [R3](#r3), [R8](#r8).)

## How adaptation works

For each active interaction, the planner searches for a placement of the hand
or moving object that preserves the demonstrated local surface relationship
within the permitted regions. Sliding and rotation are possible changes in
this relative placement. On a locally uniform shaft, many placements may
agree; on an irregular surface, the agreeing set may be small. The object
remains rigid throughout. ([R1](#r1), [C1](#c1).)

The search variables are the relative placement and a short continuous joint
trajectory. The starting point is the current measured joints and the previous
accepted plan. Joint and cable limits, intended contacts, scene collisions,
and required relationships are constraints. Actual joint travel, wrist winding,
and departure from demonstrated motion are preferences. A configurable robot
comfort range may penalize large wrist turns; a universal 90-degree cutoff
is not inferred from surface shape. ([R4](#r4), [O3](#o3).)

The solver adjusts continuous freedoms directly. It need not sample every
possible axial angle. Disconnected permitted placements, whether separated in
location or orientation, can require separate initializations. A local solver
cannot guarantee the best solution across them; record search coverage and failure instead of
claiming global unreachability. Geometric symmetry can make the fit indifferent
to a direction, but annotation must still permit exploiting that freedom.
([R3](#r3), [O2](#o2).)

Before commitment, validate the proposed choice against the ordered keyframes
and known dependent relationships. A short motion horizon must not forget a
later requirement such as keeping an interaction region accessible. Predict
the resulting relationship only under a declared model, then update it from
observations. A contact may slide, rotate or separate; rigidity is not implied
by contact. If the physical response cannot be predicted well enough for the
required outcome, report that uncertainty rather than silently using a rigid
attachment. ([R2](#r2), [R3](#r3), [R4](#r4), [R8](#r8).)

All relationships involving the same object use one consistent geometry and
predicted physical state. For example, allowing either of two contact regions
does not independently authorize choosing a future insertion region underneath
the fingers: the future contact also has to satisfy its required clearance.
This is a joint constraint check, without an object-specific end assignment.

The selected mapping and resulting relationships persist. If the chosen grasp
slides along the object, the later object-target goal stays the same while
the required hand goal changes through the updated attachment. It is incorrect
to apply one global offset to every later object-object goal, because that can
move the desired contact itself. Repeated keyframes constrain this shared
history. Replanning changes unexecuted motion while respecting existing contact
and the queue's continuity contract. ([R2](#r2), [R5](#r5), [C2](#c2).)

During occlusion, the shared estimator carries the chosen correspondence using
supported hand or environment relationships and reports the remaining
uncertainty. Permission to choose a similar region does not verify hidden slip
or repair lost identity. Reacquisition corrections to a geometry coordinate
frame must transform every linked relationship consistently; an unsupported
relink remains unknown. ([R2](#r2), [R3](#r3), [O1](#o1).)

The same relation evaluator is used by demo review, planning during tests and replay.
It consumes a timestamped scene snapshot, required relationships and the
candidate trajectory. It returns satisfied, violated or unknown conditions
with the supporting observations. It runs outside the fixed-rate motor owner.
([R3](#r3), [R5](#r5), [R6](#r6).)

## Cases the same representation must cover

These cases enumerate the distinct kinds of rigid placement permission, rather
than every possible task. They compose: several requirements can constrain the
same relationship, and several relationships can constrain the same object.
All examples retain geometry checks and robot/path feasibility. The example
angles express possible user choices, not framework constants.
([R1](#r1), [R4](#r4), [R7](#r7).)

| Case                               | What may change                                                   | What remains required                                                                                  |
| ---------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Exact interaction                  | No intentional placement freedom                                  | Demonstrated arrangement, within explicitly declared task tolerance                                    |
| Slide along a segment              | Contact position within a bounded span                            | Compatible finger contact and approach; later interaction region stays usable                          |
| Choose within an area              | Contact position within a painted patch                           | Required orientation, local contact geometry and footprint                                             |
| Choose separate areas              | Either handle, or either permitted end region                     | The intervening surface remains excluded; other constraints apply to every alternative                 |
| Quarter-turn alternatives          | Only specified headings, such as 0/90/180/270 degrees             | Permitted centre/contact arrangement; intermediate headings are not automatically valid contact states |
| Bounded turn in place              | A continuous heading interval, such as -30 to +30 degrees         | Chosen pivot/location and all other orientation constraints                                            |
| Free heading                       | A complete turn where explicitly permitted                        | Support/contact, required tilt and robot winding limits                                                |
| Move and turn about a pivot        | Position follows an arc and orientation turns with it             | Shared pivot/direction and the permitted angle interval                                                |
| Follow an arc without turning      | Position follows an arc                                           | Orientation remains fixed in the declared reference frame                                              |
| Slide over a curved surface        | Position and orientation change together as local geometry allows | Demonstrated contact arrangement; no assumed analytic cylinder or axis                                 |
| Limited tilt / upright transport   | Translation and possibly heading; tilt within an annotated bound  | An object-relative direction remains aligned with an environmental direction                           |
| Free transfer between interactions | Intermediate position/path, with distance-dependent duration      | Ordered milestones, any required orientations and collision clearance                                  |
| Changing contact during the action | A permitted relative sliding or rotating motion                   | Continued contact requirements and a real feasible path; a fixed grasp cannot be silently reassigned   |
| Several simultaneous contacts      | Only placements satisfying all contacts                           | One rigid object placement, shared consistently by every relation                                      |

An insertion or placement sequence combines these cases over its lifetime.
An upright direction is a selected or reviewed geometric reference, not an
object-class rule. A permission can also depend on a later relationship, such
as keeping an interaction region exposed. Evaluate that dependency before
committing to a grasp; do not independently pick a new correspondence at the
later keyframe. A newly incompatible later requirement calls for another
permitted physical action or a reported failure. ([R2](#r2), [C1](#c1), [C2](#c2).)

The finite pattern is: exact placement, continuous placement freedom, or
separate alternatives, with coupled constraints and a lifetime. Reflection
and rescaling are not rigid motions. A mirrored or differently sized object
needs its observed geometry and a supported correspondence; it cannot be
handled by mirroring or scaling the robot's attachment. ([R1](#r1), [R3](#r3).)

## Editing and validation

Use the existing demo image, 3D view and timeline. The editor selects the two
participants, marks their interacting regions, then expands where equivalent
local contact is permitted. It previews the resulting gripper or object poses
and the effect on later keyframes. Separate unknown regions from ruled-out
ones. An additional patch is selected using the same gesture as the first;
there is no special control for a dowel's other end. A world-relative
orientation requirement uses the same relationship inspector.
([R1](#r1), [R6](#r6).)

### One scene view for Demo/Edit, Test and Replay

Use one contextual RGB-D scene viewer in Run -> FLASH, with input adapters for
a demo frame, a test snapshot and a recorded rollout frame. All three use the
same object selection, rendering, overlay names, orbit controls and timeline
layout. Demo/Edit adds keyframe/interval authoring and relationship editing;
Test shows active progress and the choices made by the policy. Replay is
exactly the Test view driven by recorded snapshots and a playback clock,
including the recorded rules, matches, plans, decisions and outcomes. Seeking
or replaying a recording does not dispatch robot actions. These are modes of
the same view rather than separate editors. A frozen inspection or virtual
rotation preview does not command the robot. Deployment is a separate future
workflow; Test is the interactive experimentation mode. ([R6](#r6), [R5](#r5).)

In a future workflow, the user can promote a replay into a demo and annotate
it in place through Demo/Edit. This adds demonstration annotations to an
ordinary dataset episode while preserving its recorded observations, actions,
original outcome and provenance. Promotion is an explicit authoring choice;
it does not declare the original attempt successful or make all of its motion
an expert example. ([R6](#r6), [R5](#r5).)

Reconstruct the visible surroundings from the frame-aligned RGB and depth,
including the tray, table and other objects. Overlay measured object surfaces
and aligned reference surfaces in the same coordinate system. Start at the
capture viewpoint, allow orbit/pan/zoom, and provide a return-to-camera control.
Keep the user's inspection angle as data updates. Use the existing URDF-view
control conventions. Objects and rule handles remain attached to their
reference geometry when the viewing camera moves. ([R3](#r3), [R6](#r6).)

Represent the end effector with a tool point and an approach arrow; a small
crossbar can expose finger-closing direction when roll matters. A single arrow
does not specify roll, so hiding the crossbar cannot remove the underlying
orientation constraint. Derive these markers from the calibrated robot state
or the candidate placement. Authoring does not require manually aligning a
gripper mesh. Robot geometry, when available, remains usable for planning
checks independently of the displayed symbol. ([R4](#r4), [R6](#r6).)

Keep selected surfaces and editing handles sharp. Context softening and fading
of incomplete scan boundaries are display choices. Small rendering gaps can
be covered by surface patches, but missing observation is not measured
geometry. Decorative completion is excluded from contact picking and geometric
validation. Accumulating frames requires supported registration and separate
handling of moving objects; a stale object must not become part of a static
background. ([R3](#r3), [O1](#o1), [R6](#r6).)

The shared overlay menu contains:

| Layer                  | Demo/Edit                                                              | Test and Replay                                                                             |
| ---------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Scene                  | Source RGB-D and surrounding geometry                                  | Current or recorded RGB-D scene                                                             |
| Objects and matches    | Selected measured regions and their extraction evidence                | Observed object surfaces plus aligned demo/reference surfaces; missing matches are explicit |
| Rules                  | Editable rotation ranges, surface-slide regions and required relations | The same rules mapped onto the test participants, with the selected choice highlighted      |
| Keyframes and progress | Add/edit keyframes and intervals; preview their consequences           | Active/next keyframes, interval progress and milestone outcome                              |
| Motion                 | Recorded motion and any explicitly computed reference/preview          | Published future plan, sent commands and measured motion, distinguished by provenance       |

Enable relevant rules, matched surfaces and any available current plan by
default. Users can opt out through a compact collapsible in-view overlay menu;
keep the menu clear of the maximize control and the playback strip fixed below
the scene. Persist layer preferences across modes. Hovering or selecting a
layer item highlights its geometry. Layer visibility cannot change estimation,
planning or recording. Labels and line styles distinguish evidence, prediction
and alternatives without depending on many colors. ([R6](#r6), [R3](#r3).)

An allowed rotation arc or surface ribbon shows a set of possible placements,
not a motion the robot must execute. Keep four kinds of motion distinct:
recorded motion, an extracted demonstrated relative motion if available, a
candidate or published future plan, and physically observed motion. A plan
must identify what moves, its reference frame, time range and revision. When
only keyframe goals exist, display those goals without inventing a smooth path
between them. The renderer does not require a decision to store a full object
trajectory in the demo; both goal-based and trajectory-based policies can
publish the actual references and plans they use. ([R2](#r2), [R5](#r5), [R6](#r6).)

The viewer consumes a timestamped scene snapshot containing image/depth frame
references, object/reference identities and transforms, observation or
prediction provenance, skill/rule revision, active progress, and available
plans or action receipts. A delayed estimate remains associated with its
source frame; propagating it to the presentation time must be explicit and
labelled as prediction. Scene geometry and overlays must not silently mix
different frame times. The renderer consumes bounded updates outside the motor
loop. ([R3](#r3), [R5](#r5), [R6](#r6).)

Record the snapshot inputs and decision revisions through the ordinary dataset
and FLASH sidecars so Replay can reconstruct what was shown and known in Test.
Rendering caches need not be stored every frame. If older data lacks a layer,
show it as unavailable; a newly computed analysis must be labelled as such.
Comparisons across demo and rollout align by keyframe/interval progress, since
adapted transport can take longer than the demonstrated transport. This keeps
the same meaning across views without imposing the same absolute playback
time or camera pose. ([R2](#r2), [R5](#r5), [R6](#r6).)

### Placement editing and checks

Add rotation handles to the same placement preview: drag an arc to permit a
range, add a separate ghost placement to permit another orientation, or repeat
it at a chosen angular interval. A pivot handle controls whether the hand
turns in place or moves around a reference. These are editor gestures for
allowed placements, not claims that the object has an exact symmetry. The
preview distinguishes what was permitted, what current geometry supports,
and what the robot can execute. A soft preference for short wrist travel is
displayed separately from a hard permitted angular range. ([R6](#r6), [R7](#r7).)

Validation includes a shifted grasp followed by insertion, alternative contact
patches, upright transport, an irregular object, and incomplete surface views.
Check coordinate-frame invariance: changing an arbitrary region origin must
not change the physical plan. Check causal invariance: a new keyframe cannot
change an already established attachment. On simple synthetic scenes, compare
joint travel with an exhaustive reference search to measure the local solver's
misses. Record pose/contact error, outcome, rejected and unknown cases,
planning latency and motor-loop deadlines; surface-overlap percentage alone
does not establish success. ([R1](#r1), [R2](#r2), [R4](#r4), [O3](#o3).)

For placement permissions, check that quarter-turn alternatives reject an
intermediate heading, a bounded turn rejects values beyond its endpoints, and
a pivot change produces the expected coupled position change. Test combined
sliding/rotation conditions without accidentally admitting every combination
of independently valid coordinates. Moving the reference object must move the
permission with it. Equivalent final headings must not erase wrist winding or
permit a contact-preserving path through a prohibited orientation.
([R2](#r2), [R4](#r4), [R7](#r7).)

For extraction, include a push and a sliding interaction without any grasp or
release annotation. Verify selected keyframes and explicit interval settings
alone determine ordering, reference participants and timing. Rename every
interval and retain identical compiled goals. A gripper closure with a
visually stationary object must not create a rigid fallback, and trustworthy
visual motion must not be replaced by a stale attachment prediction.
([R3](#r3), [R5](#r5), [R8](#r8), [O4](#o4).)

For the shared viewer, feed the same saved snapshot through Test and Replay
adapters and compare projected geometry, rule state and plan revision. Inject
delayed estimates and missing layers; verify provenance remains visible and no
unrecorded path appears. Hide all overlays and verify estimator inputs, policy
decisions and recorded data are unchanged. Orbit and switch modes while
checking that rules remain object-relative and the timeline stays in place.
Verify replay playback and seeking never invoke the robot command interface.
([R3](#r3), [R5](#r5), [R6](#r6).)

## Alternatives and costs

Keeping analytic symmetry and per-keyframe pose tolerance is compact for known
shapes but leaves generic contact sliding and causal consequences implicit.
Explicit axis or angle tools may be useful editor shortcuts; their saved
meaning must be regions and relationships. Independent adaptation per keyframe
can produce a sequence that no physical grasp realizes. Pure nearest-surface
matching omits intent and embodiment. ([O2](#o2), [O3](#o3).)

The proposed representation has fewer object-specific concepts but requires
more work in geometry validation and trajectory optimization. Similar-looking
patches can differ in friction or strength; a successful demonstrated grasp
only supports generalizing to the extent permitted by the annotation and
available evidence. Partial observation can leave a proposal unknown. General
patch search can miss alternatives or take too long; its completeness and
latency must be measured. These limitations remain visible instead of being
hidden behind a symmetry flag. ([R3](#r3), [C3](#c3).)
