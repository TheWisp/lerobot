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
permits substituting a similar observed region while preserving a demonstrated
relationship. The planner jointly chooses the contact placement and robot
path. That choice becomes part of the continuing physical state. Keyframes
provide examples and validation constraints; they do not reset the choice.

This proposal covers rigid-object imitation using partial RGB-D observations.
The robot, other objects, and an environmental reference can participate in
the same relationships. Geometry supplies evidence for compatibility;
annotation supplies permission. Surface similarity alone cannot establish
friction, attachment, or successful task completion.

## Requirements

Preserving intent and causal consistency takes priority over finding a short
path. Operator convenience follows a representation that the planner can
actually enforce.

| ID                | Priority | Requirement                                                       | Target and reason                                                                                                                             | Check                                                                                                                           |
| ----------------- | -------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| <a id="r1"></a>R1 | P0       | Express adaptation through object regions and their relationships | The same schema covers grasping, placement, insertion and orientation constraints; no object-name, cylinder, end-swap or symmetry-order field | Author the examples below using the same schema; rename the objects without changing behavior                                   |
| <a id="r2"></a>R2 | P0       | Preserve the consequences of an adaptation                        | No independent contact reassignment at the next keyframe; actual attachment governs subsequent goals                                          | Shift a grasp, then verify all later hand goals account for the shift and the final object relationship remains unchanged       |
| <a id="r3"></a>R3 | P0       | Keep permission and measured evidence distinct                    | Similarity cannot create permission; missing geometry cannot count as observed agreement                                                      | Test an unpermitted matching patch and an unobserved contact surface; retain the demonstrated patch when adaptation is disabled |
| <a id="r4"></a>R4 | P0       | Evaluate a path from the current robot state                      | Enforce declared relationships and robot constraints across the ordered approach, interaction and dependent goals                             | Replay the saved joint-branch failure; test a convenient grasp that prevents the subsequent operation                           |
| <a id="r5"></a>R5 | P0       | Preserve existing timing and execution contracts                  | Async perception/planning, fixed 30 Hz motor ownership, dense interaction timing and distance-dependent transport                             | Use the [architecture's timing tests](architecture.md#interval-timing) with delayed replans and moved targets                   |
| <a id="r6"></a>R6 | P1       | Make permissions and decisions reviewable                         | Edit in the demo viewer; save provenance, units, evidence and selected geometry correspondence in the regular dataset sidecar                 | Save/reload a rule and inspect the chosen contact and rejection reason in replay                                                |

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

This does not eliminate rigid transforms from computation. They place the
measured regions and robot links in space. Their arbitrary origin is not a
semantic object center, and no named axis is required to express a permission.

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
possible axial angle. Disconnected permitted regions can require separate
initializations from a generic patch search. A local solver cannot guarantee
the best solution across them; record search coverage and failure instead of
claiming global unreachability. Geometric symmetry can make the fit indifferent
to a direction, but annotation must still permit exploiting that freedom.
([R3](#r3), [O2](#o2).)

Before commitment, validate the proposed choice against the ordered approach,
interaction and known dependent goals. A short motion horizon must not forget
a later requirement such as keeping an interaction region accessible. Predict
the resulting attachment for planning; after contact, use its measured or
explicitly assumed value and uncertainty. ([R2](#r2), [R4](#r4).)

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

The same relation evaluator is used by demo review, live planning and replay.
It consumes a timestamped scene snapshot, required relationships and the
candidate trajectory. It returns satisfied, violated or unknown conditions
with the supporting observations. It runs outside the fixed-rate motor owner.
([R3](#r3), [R5](#r5), [R6](#r6).)

## The same representation in different tasks

| Example                      | Regions and relationships                                                                                             | Where the freedom comes from                                                          |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Grasp farther along a shaft  | Finger surfaces and a permitted object region; preserve the demonstrated local grasp geometry                         | Similar contact geometry at several locations                                         |
| Use either end               | One permitted region with two disconnected patches; retain the later object-target requirement                        | Both patches satisfy the same annotated relation; no end-swap type                    |
| Place at a different heading | Moving-object and support regions; preserve support/contact and any required footprint constraints                    | Several headings satisfy the same surface relationship                                |
| Carry upright                | An object-relative reference direction and an environmental direction, represented by a required orientation relation | Translation remains free while that relation holds; no cup type                       |
| Insert an object             | Two interaction regions with the demonstrated sequence of relative arrangements                                       | Only replacements satisfying the whole interaction remain allowed; no shaft/hole type |

Directions in the upright example are geometric references, selected or inferred
and reviewed. They are not assumptions about the object's class. This is an
example of intent that local surface resemblance alone cannot establish.
([R1](#r1), [C1](#c1).)

## Editing and validation

Use the existing demo image, 3D view and timeline. The editor selects the two
participants, marks their interacting regions, then expands where equivalent
local contact is permitted. It previews the resulting gripper or object poses
and the effect on later keyframes. Separate unknown regions from ruled-out
ones. An additional patch is selected using the same gesture as the first;
there is no special control for a dowel's other end. A world-relative
orientation requirement uses the same relationship inspector.
([R1](#r1), [R6](#r6).)

Validation includes a shifted grasp followed by insertion, alternative contact
patches, upright transport, an irregular object, and incomplete surface views.
Check coordinate-frame invariance: changing an arbitrary region origin must
not change the physical plan. Check causal invariance: a new keyframe cannot
change an already established attachment. On simple synthetic scenes, compare
joint travel with an exhaustive reference search to measure the local solver's
misses. Record pose/contact error, outcome, rejected and unknown cases,
planning latency and motor-loop deadlines; surface-overlap percentage alone
does not establish success. ([R1](#r1), [R2](#r2), [R4](#r4), [O3](#o3).)

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
