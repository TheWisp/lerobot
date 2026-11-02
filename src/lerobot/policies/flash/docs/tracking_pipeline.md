# FLASH tracking pipeline

Status: proposed

**Estimate a rigid object's pose from its visible surface.** Use aligned RGB-D
and SAM masks, image matches, and optional group/gripper predictions. Grow the
surface as new sides are observed. This is perception, not robot control.

## Core loop

**Priority: usable mask + measured depth first; 2D tracking second.** When
they disagree about position or orientation that the visible geometry constrains,
the geometry wins. More image matches or higher matching confidence cannot
override that evidence. This applies to UFM, LK and future 2D trackers alike.

![Geometry, image evidence and optional motion predictions feed one pose estimate](assets/tracking_pipeline/pipeline.png)

- Fit the stored surface to this frame's visible mask and measured depth. Image
  matches supply starting guesses and distinguish poses compatible with that
  geometry; they do not replace the geometric checks.
- A partial view may leave some position or rotation uncertain. Image matches
  and group/gripper predictions help estimate those unresolved parts. When no
  usable object observation remains, a valid group or gripper relationship can
  still supply a **predicted** pose; otherwise report lost.
- Apply the same mask/depth quality checks to every hypothesis. Bad or missing
  depth is unknown, not evidence against an object behind it. Mask and depth
  are not infallible; deciding their reliability remains part of the
  [uncertainty problem](#uncertainty-and-correction).
- Add **measured** new surface only after checking its alignment. Hidden frames
  can update pose through prediction, but provide no new object surface.

## Initialization and frame sequence

![Sequence of initialization, per-frame data exchange and recovery](assets/tracking_pipeline/sequence.png)

The sequence expands the same loop above. It is the target behavior, not a claim
that every branch is implemented yet. Inputs are synchronized RGB, depth, camera
intrinsics and a visible-object mask. SAM maintains its own causal segmentation
state; missing segmentation and missing usable depth are recorded separately.

**Image source:** usually the previous processed frame. After a tracking failure,
try the last image successfully aligned to the stored surface (a **connected
observation**) and, within a configured budget, older stored views.
**Geometry source:** the accumulated surface before processing this frame.
**Prediction source:** a timestamped pose inferred from a known co-movement
relationship; it works even without object pixels. These sources record their
frame IDs. A new mask does not by itself recover the old pose.

## Shared image matching

**Implemented foundation (2026-11-02):** `image_matching.py` gives LK and UFM
one prepared-pair/query interface, explicit backend settings, and separate native
diagnostics. Existing callers retain their original image coordinates, frame
selection and filtering. The refactor is checked against the previous code.

**Experimental implementation (2026-11-02):** `shared_image_tracking.py` runs
one LK or UFM pass on consecutive full-scene images. The benchmark can opt in
through `--shared-matching-config`; its existing paths remain comparison controls.
This changes sampling and group evidence, so adapter equivalence does **not**
establish equal tracking quality for this next step.
The first full-clip comparison found regressions, including a sideways dowel
estimate during rotation. Subsequent geometry fixes improved aggregate accuracy
across 21 labelled scenes, with individual regressions and higher processing cost.
It remains opt-in; see the
[full-suite measurements](../../../../../docs/proofs/flash-tracker-restart/RESEARCH.md#2026-11-02--full-suite-validation-after-the-geometry-fixes).

Select one configurable 2D matching backend, initially LK or UFM, for an experiment.
Object tracking and surrounding-group tracking consume its same results. Group
tracking maintains identities and co-movement relationships; it does not choose
or invoke a second image tracker. Two consumers of the same matches are one source
of evidence, not two independent votes.

```text
Consecutive RGB frames + requested points / image region
                          ↓
             Matching adapter: LK OR UFM
                          ↓
         Shared point identities + mask/depth checks
                 ↙                         ↘
        Object measurements          Surrounding groups
                 ↘                         ↙
                  Surface fit + pose estimate
```

The common result contains source/target frame IDs, source/target pixels in original
image coordinates, valid matches, image coverage, processing time and backend
reliability information. Dense UFM output can answer point queries by sampling its
flow field; sparse LK can match the requested points directly. Neither adapter
needs to invent a dense field, a 3D pose, or hidden-object observations. Native
visibility, forward/backward error and other scores retain their meaning; a UFM
visibility value is not interchangeable with an LK error or a probability of pose
accuracy. Acceptance settings are explicit per backend.

SAM identifies visible object pixels. Shared depth-cleaning and boundary-margin
settings govern which pixels supply object geometry. Ownership remains separate
from depth quality: a discarded boundary pixel does not become background. A point
center inside SAM does not imply all image context used for matching is inside.
Mask-aware matching is an experiment, including possible loss of useful context
and artificial boundaries created by zeroing masked pixels.

Matching continues over surrounding RGB when the object's mask is empty. Crop
coverage and image scale are explicit; points outside coverage or without a valid
match cannot be reported as observed. Matching the same pair/region is shared or
batched across consumers. Older-frame recovery is explicitly separate from
consecutive-frame identity tracking. Backend changes preserve point IDs only
when correspondence continuity is retained; otherwise reseeding starts new IDs.

Configuration selects the backend, image scale/resolution, point budget and its
acceptance settings. The same configuration is recorded by the benchmark and used
by the consumer. Compare LK and UFM on identical recordings and masks, including
visible-point drift, mistaken retention through occlusion, pose quality and cost.
The shared experiment does not run a second backend or silently retry image
inference. Older observations can reconnect through surviving point IDs in a
bounded history (default 64 frames); beyond it, no image correspondence is
claimed. Other surface/group evidence may still provide a pose. New seeds have
new IDs. Backend, resolution, point budgets and history length are saved with
the result and preserved by GUI reruns and cache keys.

**Current experiment, 2026-11-02:** both endpoints of object matches must pass
the configured mask erosion (2 px by default) and depth checks. A match can
still drift inside a valid surface. With `GrowingSurfaceConfig.surface_priority`,
surface fitting and group-disagreement checks use the same geometric objective;
image/group matches choose between equally fitting solutions, rather than
outweighing a better measured-surface fit. The weighted objective is retained as
an ablation. This is not a calibrated confidence model.

Only the visible part of the saved surface must match the current measured
surface. Outside the visible mask, nearer or unknown depth can hide a point;
farther depth can contradict it. A partial saved surface is **not** required to
cover the entire current outline. `ShapeConfig.require_full_mask_coverage` is
off by default; enabling it assumes a complete reference, which our growing
partial surfaces do not provide. No scene name or grasp timing selects these
rules. The recorded gamepad regression checks approach occlusion and subsequent
pickup; it does not establish accuracy throughout manipulation.

## What persists between frames

1. **Surface patches:** retain sampled camera measurements, source frame and the
   transform placing each patch in fixed object coordinates. The combined surface
   is a rebuildable cache, with explicit point/patch budgets.
2. **Pose and evidence:** estimated rotation/translation, source time, model
   version, observed/predicted/lost status, fit diagnostics and source frame IDs.
   Adding a side never changes the coordinate origin or the A/B/C landmarks.
3. **Connected observations:** the last connected image and bounded older views
   for recovery. A prediction alone is not a newly connected image.

Check a new patch against observations that were not used to create it **before
letting that patch influence fitting**. Otherwise, a badly placed patch could
make the next fit look good simply because it matches its own wrong placement.
This check reduces one source of circular evidence; it does not prove correctness.

## Uncertainty and correction

**We have not solved confidence estimation.** Initially record mask/depth quality,
geometric and image disagreement, visible coverage, pose sensitivity in all six
directions, and time without observation. Test whether those quantities predict
actual error on labelled clips. Do not call a fit score an accuracy probability.

Position and orientation are one 6D pose. A 6 × 6 covariance could represent local
uncertainty and correlations in all six directions; separate pictures below are
only illustrations. A Kalman filter is a possible later experiment, not a remedy
for a wrong surface or ambiguous identity. [Explanation and prior art](#e3-surface-and-pose-uncertainty).

To undo an addition, exclude its patch and rebuild the surface. To reposition it,
change that patch's transform and rebuild. Retained raw patches make this possible;
a permanently averaged cloud would not. **Automatically deciding which patch is
wrong is still unresolved.** Rebuilding also does not repair later pose estimates:
reprocess dependent frames offline, or refit the current live observation.

![Separate patches can be corrected; later dependent poses require reprocessing](assets/tracking_pipeline/surface_revision.png)

## Scope and first experiment

Rigid objects without a supplied mesh. With a completely new side and no overlap,
recognizable features or valid motion connection, the old pose cannot be recovered
reliably. Keep the new observation separate; do not silently reset the landmarks.
[Example](#e2-a-new-side-after-occlusion). Deformation, transparent objects and
unreliable segmentation/depth remain explicit limitations.

Keep the implementation in the experiment harness first. Compare fixed and growing
surfaces on the same recordings, masks and image matches. Test removal/repositioning
with deliberate bad additions before trusting automatic repair. Keep the live
defaults unchanged until benchmark and visual evidence justify moving them.

**Run all benchmarks** processes listed examples in a background worker and
preserves a code/config snapshot, prior results and a downloadable report. Evaluate
pose error separately from availability; unlabelled videos have no proven accuracy.
Supplied-mask tests do not measure SAM accuracy. The same causal estimator is intended
for demos, benchmarks and the debugger; the independent robot loop consumes its
outputs at its own fixed rate.

## Examples and references

### E1 Partial observation

![Existing insertion and stacking examples: cyan is the input mask; orange is the stored surface moved by each estimated pose](assets/tracking_pipeline/partial_observation.png)

**Actual baseline examples, 2026-10-31:** ring at 9.67 s above; gamepad at 7.50 s
below. Cyan is the supplied visible mask. Orange is the initial measured surface
moved by each estimate, **not raw image matches**. Both methods can drift. These
pictures motivate the design; they do not demonstrate the proposed fix.

Reliable visible ends of a dowel can determine tilt, while a smooth side may leave
spin unresolved. An occluder's depth must not become the object's surface. SAM
and depth can themselves be wrong.

### E2 A new side after occlusion

![Illustration: front observed, hidden while turning, then a previously unseen back appears; connecting evidence determines whether it can join the old surface](assets/tracking_pipeline/disjoint_views.png)

Overlap with an older view, identifiable features, or valid motion can connect
the new side. A motion-only connection remains predicted; new depth does not
independently confirm that connection.

Without a connection, recognizing the same object does **not generally determine
the transform** between old and new surfaces. Keep the old model; retain the new
observation separately within the memory budget; report its placement in the old
coordinates as unknown. Resume the old landmark trajectory only after connecting
evidence appears. A new local track may continue, but it is not yet the old
A/B/C track. Never force the back onto the front or silently reset A/B/C.

### E3 Surface and pose uncertainty

![Surface noise versus whole-object position and orientation uncertainty; illustrative spreads, not benchmark measurements](assets/tracking_pipeline/uncertainty.png)

Surface uncertainty includes noisy depth and incorrectly joined patches. **Position
and orientation together are one 6D pose.** The two panels below separate them
only to make the illustration readable; the estimator must keep their correlations.
Uncertainty can differ along any translation or rotation direction. For example, position
may be uncertain by 10 mm sideways but 2 mm vertically, with 8 degrees of tilt
uncertainty. These are illustrative standard deviations, not measured accuracy.

A **6 × 6 pose covariance** stores variances and correlations of three small
translations and three small rotations. Specify coordinates, rotation convention,
pivot, metres and radians. An ellipse shows only the position part. One local
Gaussian cannot represent two separate end assignments or completely unknown
spin. Report such ambiguity instead of pretending it is precise.

**Prior art:** [ROFT](https://arxiv.org/abs/2111.03821) uses Kalman filtering for
RGB-D object pose/velocity tracking with optical flow and delayed visual
estimates. [Brossard et al.](https://arxiv.org/abs/1909.05722) study covariance of
point-cloud alignment, including poor initialization, weak geometric constraints
and sensor bias. [PoseRBPF](https://arxiv.org/abs/1905.09304) represents multiple
rotational possibilities. These are useful precedents, not validation of our
model-free growing surface.

## Terms

- **Object coordinates:** fixed coordinates attached to the object, without needing
  its centre of mass. A pose places these coordinates in the camera or scene.
- **Surface patch:** sampled depth measurements from one visible part of an object.
- **Alignment:** rotation and translation putting two observations into the same coordinates.
- **Image match:** two image locations proposed to represent the same physical spot.
  Persistent point tracks remember such matches across more frames, not a new sensor.

This is the current design. Dated experiments stay in the
[research record](../../../../../docs/proofs/flash-tracker-restart/RESEARCH.md).
The [sealed debugger](surface_tracking.md) and
[earlier growth experiments](../../../../../docs/proofs/flash-surface-multiview/README.md)
remain historical evidence, not competing specifications.
