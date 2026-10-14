# Pose verification and measured surface constraints

Accepted Point2Pose updates can contain orientation drift even when every track
is visible and passes the existing depth check. In the October 12 stacking
recordings, clear views of the stationary target cube produced this failure.
The depth check tests the tracked points' plausible depth and the distribution
of the observed fit points; it does not establish that the returned orientation
matches the visible surface.

Since October 13, `surface_constraint.py` arbitrates the proposal before control
acceptance, including initial finds. Raw comparisons remain in
`pose_verification.py`. The policy corrects a supported observed plane's tilt and
normal offset, retains non-planar proposals only when the general surface check
agrees, and rejects unverified observations. Group motion still carries an
accepted pose between observations; it is not an absolute orientation authority.
The native P2P estimator and its model updates are unchanged by this correction.

## Measured constraint

The policy uses reference RGB-D from designation and current masked RGB-D. It
does not fit against P2P's growing map or impose an upright, table or object-category
prior. Five-pixel erosion excludes mask rims. A bounded robust plane fit uses
alternating samples for fitting and validation, requires majority support in
both, and rejects patches with too little physical span. The existing 3 mm
correspondence allowance is split between reference and current planes. Normal
and offset disagreement between the split fits must fit the existing 3°/3 mm
placement tolerances. A quadratic validation check rejects predictable curvature
beyond the per-surface noise budget and measured model residual. A small sphere's
visible cap is a tested counterexample to relying only on a plane residual.

For a supported corresponding patch, apply the shortest rotation aligning its
normal, about its predicted patch centre. Then move only along the observed
normal onto that plane. This constrains three pose components; tangential motion
and twist remain the proposal's. Both old and corrected patches must have nearby
majority overlap, using the existing 6 mm local association radius. Corrected
point support must remain compatible with the noise allowance.

If a reference plane is supported but the current patch cannot be verified,
the proposal is rejected even if the older point-distance ICP check says
consistent. An unavailable or failed check never supplies a hold observation.
Non-planar references may retain an unchanged proposal when the general surface
check agrees; this is not a six-dimensional accuracy guarantee.

The policy is on by default for the P2P control path. Geometry runs before the
existing visibility/depth/centre-support acceptance gates and before following
can set an arm target. Replacing teach/track while geometry is running discards
that result. Both the primary object and secondary targets use corrected poses.
Held-object correction consumes accepted observations rather than raw P2P fits;
recorded demo holds and fresh reference finds undergo the same surface policy.
When registration back to the original designation cannot establish a carried
hold, the runner still measures at its first held observation line. It matches
fresh live RGB-D against still held views from this demonstration attachment.
SAM resegments each reference from its recorded mask's interior prompt; matching
and surface verification use that same refreshed mask. These alternative
reference transforms never enter the designation-based live tracker.

For a reference fingertip pose `D`, current fingertip pose `L`, camera-to-base
transform `B`, and reference-to-live visual registration `M`, the two attachments
are `inverse(D) @ B` and `inverse(L) @ B @ M`. Their ratio gives the gripper
correction without first matching the carried view back to the table view.
Multiple fresh observations must agree within the existing hold tolerances.
Rotation also requires at least two reference views agreeing within the planner's
angular tolerance. If it does not, explicitly allowed grasp prediction supplies
rotation while translation aligns the same matched material patch centre.
This is logged and rendered as position measured, rotation predicted; it cannot
satisfy a request that forbids predicted orientation. Inconclusive checks retain
their individual find IDs and reasons. A shared hold wait budget bounds retries.

The selected demo planning references are also measured before the key-frame runner
can dispatch. Planning inverts these checked reference poses; inverting a raw,
tilted demo pose would otherwise reintroduce its error into a corrected live pose.
Rejection or a changed reference selection prevents using that reference.
`act.json` preserves both versions in `demo_reference_constraints` without
rewriting the original recording.
Original worker poses remain available in the model/recording for diagnosis.

This is a measured planar-patch constraint, not complete multi-surface fusion or
an active-inspection controller. Symmetry, hidden slip and unobservable pose
components remain unresolved. No statistical ground-truth accuracy claim follows
from agreement with the same depth sensor.

## Comparisons

- **Groups:** save the first synchronized P2P/group pair for each tracking
  session and group designation. Predict the current pose as
  `G_now @ inverse(G_reference) @ P_reference`. Keep this reference across
  accepted P2P updates so that small successive errors cannot erase accumulated
  drift. This compares relative motion, not independent absolute registration;
  an error already present at the reference remains undetected by this check.
  Use the existing group interpolation and 0.2 s synchronization allowance;
  record the nearest frame and its time difference. Stale or unsupported group
  evidence is unavailable. Group ownership/carry context is recorded too.
- **RGB-D refit:** project reference geometry into the current object mask and
  fit its overlapping surface to that frame's depth. Exclude invalid depth,
  mask rims and nearer foreground occlusion. Use one local ICP start, without
  an upright, table, cube or object-specific constraint. The alternative needs
  majority support within the existing 3 mm point-noise allowance, capped RMS
  within that allowance, and an observable centre within the planner's reach
  tolerance. A different pose must also lower the surface-fit cost to count as
  a disagreement. Insufficient overlap or support is inconclusive.

Both comparisons flag rotation above the planner's existing 3 degree reach
tolerance or displacement of the **same object centre** above its 3 mm reach
tolerance. Transform translation columns are not centre positions. These
tolerances are actionable diagnostic bounds, not calibrated statistical
confidence intervals. Actual matched motion is allowed, including tilted
objects. Missing evidence is never labelled agreement.

Surface work is bounded to 400 points per surface and 12 iterations per ICP
phase. The mask erosion is the same five pixels as the initial find; the fit
radius and foreground exclusion use twice the existing point-noise allowance.
The four recorded fixture checks took roughly 11–12 ms each after first import
on the development host. Each live check records its own elapsed time. Work
runs on the render executor before acceptance and adds tracking-loop latency.
Exceptions are logged and recorded as unavailable; the affected proposal cannot
reach control or held-pose history. Plane work uses at most 1600 points and 64
deterministic RANSAC triples per surface, with bounded refinement.

## Reading a result

The live picture and replay show per-object comparisons in a panel below the
camera image. A conflict shows its rotation/centre difference and whether that
P2P candidate was used, corrected before use, or rejected. A third row reports
the surface decision, correction angle and three observed components. Replay's
**Next pose disagreement** button jumps
between affected frames. Legacy acts say verification was not recorded; replay
does not manufacture historical checks.

`act.json` stores each frame's `pose_verification` and an act-level
`pose_disagreements` event list. These retain frame time, raw P2P pose,
acceptance, group prediction/reference time, surface alternative, residuals,
point counts, thresholds and unavailable/inconclusive reasons. The `constraint`
record includes the policy version, control proposal, support, validation errors,
curvature evidence and rejection reason. `raw_p2p_used` distinguishes an unchanged
proposal from a corrected observation. The arrays now
also preserve other objects' raw deltas, models and fit points. The server log
reports entry into each object/source conflict, including outside an act;
repeated frames remain in the recording without repeating the server warning.

Neither comparison is ground truth. Symmetry, limited overlap, a poor mask,
shared depth error or sample timing can produce ambiguity. A local fit may
remain in the wrong minimum; a consistent result does not certify pose
accuracy. Group carry can depend on the assumed supporting object or gripper.
Review the recorded evidence before deciding which estimate to trust.

## Recorded regression fixture

`tests/showservo/fixtures/pose_disagreement_20261012.npz` contains the cube's
original designation surface (400 sampled points), recorded target transforms,
cropped masked metric depth, masks and crop-adjusted intrinsics. It includes no
RGB images. The source demo is `pick_place`, cube designation frame 57.

| Fixture  | Source act / frame    | P2P-to-refit rotation | Result       |
| -------- | --------------------- | --------------------: | ------------ |
| baseline | 20261012_205715 / 0   |          2.64 degrees | consistent   |
| latest   | 20261012_205715 / 124 |         10.56 degrees | disagreement |
| earlier  | 20261012_192802 / 10  |          7.95 degrees | disagreement |
| clear    | 20261012_194250 / 30  |          8.72 degrees | disagreement |

These are alternative-fit differences, not independently measured absolute
orientation errors. The stored transforms reproduce the historical target
geometry; a live check instead uses that frame's raw P2P candidate, so its
number can differ from a stale accepted target's historical error. Synthetic
tests additionally cover real rigid motion, a genuinely tilted surface,
occlusion and cumulative P2P drift against stationary groups. Integration tests
cover separate object anchors, fail-closed diagnostic failure, raw/control pose
separation, acquisition, historical hold validation and replay visibility.

## October 13 constraint replay

The nine synchronized static snapshots from `tracker-readiness/143413` improve
against the earlier independent implementation of the visible-normal check:
before fresh initialization, median disagreement is 22.63° raw versus 0.85°
corrected; after initialization it is 5.78° versus 0.88°. The corrected maximum
is below 0.98°. A cached pre-initialization frame remains excluded. These checks
share sensor data and are not independent physical ground truth.

Every saved production output from both complete local acts was processed:
189 + 266 frames, 910 object observations, without changing native P2P state.
For the target cube with at least 97% of tracks visible and native tracking
available, the first act keeps 91/104 proposals, reducing paired median normal
disagreement from 6.01° to 0.21° (corrected maximum 1.67°); the second keeps
107/107, reducing 2.62° to 0.49° (maximum 0.84°). Rejected frames remain explicit.
These are surface-policy results in the native initialization frame, not a
complete GUI/control replay or proof of placement success. The separate old
visibility and centre-support gates may reject more observations.

The actual demo planning references also pass this policy: gamepad frame 98
receives a 0.80° correction and cube frame 294 a 4.54° correction. Held-out normal
residuals are 0.30° and 0.23°, respectively. This identifies an additional source
of erroneous placement tilt in the recorded reference, separate from live drift.

Availability decreases deliberately. In particular, the recorded `baseline`
fixture's current patch fails the new plane check despite its old ICP-consistent
label. The held gamepad has no frames with 97% track visibility in these two
native replays; the historical raw-hold bypass must not be used to claim that
in-hand ambiguity is fixed. Full-pose ground-truth regression across the nine
reference videos has not been rerun for this frontend policy.

Scripts, input coverage/hashes, all decisions and timing are preserved in the
`20261013/surface-constraints` experiment archive. This change uses no physical
robot motion during offline validation.

## Live perception validation of the constraint

Build `b368ad0e3` was deployed with the arm disconnected, following disabled and
no act running. After fresh acquisition, 120 distinct stationary tracker frames
were sampled over 23.3 seconds. All 120 cube proposals passed after correction;
115/120 gamepad proposals passed, with five rejected for unsupported planarity.
No accepted proposal lacked a control pose. Both demo reference corrections were
present in the live state. The observed median tracking rate was 5.19 FPS; the
combined per-object diagnostic and constraint costs were 24.3 ms for the cube and
18.9 ms for the gamepad. These are observed timings, not a frame-rate guarantee.

Held-out normal disagreement for accepted cube observations was 0.19° median and
0.54° maximum. A separate check using the previous normal-fitting implementation,
a fixed acquisition mask and 13 exactly synchronized depth/pose snapshots measured
0.52° median and 0.84° maximum for the used cube pose. Both checks share the sensor;
neither is independent full-pose ground truth. The old full-pose ICP diagnostic
still flagged 97 cube proposals, which were constrained before acceptance; this
does not imply that all components of those full-pose disagreements were resolved.

This verifies live visible-face correction and explicit rejection, not in-hand
accuracy, active inspection or physical placement. Raw drift can continue inside
P2P; the frontend constraint does not change its map. Live states, geometry,
rendered panels, acquisition evidence and the normal audit are archived alongside
the offline results in `20261013/surface-constraints/live`.
