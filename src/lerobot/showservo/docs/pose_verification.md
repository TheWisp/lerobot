# Pose disagreement diagnostics

Accepted Point2Pose updates can contain orientation drift even when every track
is visible and passes the existing depth check. In the October 12 stacking
recordings, clear views of the stationary target cube produced this failure.
The depth check tests the tracked points' plausible depth and the distribution
of the observed fit points; it does not establish that the returned orientation
matches the visible surface.

There is no continuous alignment rule between P2P, point groups and RGB-D. The
initial find uses surface fitting. During tracking, an accepted P2P update
replaces the stored pose, and group motion carries that pose between updates.
The diagnostics in `pose_verification.py` report conflicts without changing
these acceptance or control rules. They are not a fix for the underlying P2P
error and do not stop an act.

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
runs on the render executor after acceptance, but adds tracking-loop latency.
Diagnostic exceptions are logged and recorded as unavailable, without stopping
the existing tracking/control path.

## Reading a result

The live picture and replay show per-object comparisons in a panel below the
camera image. A conflict shows its rotation/centre difference and whether that
P2P candidate was accepted. Replay's **Next pose disagreement** button jumps
between affected frames. Legacy acts say verification was not recorded; replay
does not manufacture historical checks.

`act.json` stores each frame's `pose_verification` and an act-level
`pose_disagreements` event list. These retain frame time, raw P2P pose,
acceptance, group prediction/reference time, surface alternative, residuals,
point counts, thresholds and unavailable/inconclusive reasons. The arrays now
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
cover separate object anchors, diagnostic failure, acceptance preservation,
recording and replay visibility.
