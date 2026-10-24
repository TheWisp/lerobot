# RGB-D registration of the held dowel

**Updated result: adding a fixed earlier surface and whole-mask/depth correction
removes the obvious tilt once the dowel is clearly visible.** The corrected
reference axis agrees with the measured visible surface direction to 2.3°
median over frames 360–399, versus 15.5° for UFM transporting the same initialized
reference without correction. This is same-sensor geometric consistency, not
independent pose ground truth or verified hidden-point accuracy.
[Watch the full 120-frame before/after](shape_correction.mp4); see the
[whole-shape experiment](#whole-shape-correction) for its limits.

**Earlier result: none of the tested pairwise methods kept a sufficiently stable object
frame.** RoMa v2 `fast` was the best of that controlled comparison, but the
three landmarks chosen on the dowel at frame 280 differed from a rigid-grasp
reference by 19.3 mm median and its long-axis direction differed by 17.6°
median over frames 281–399. The latter is visible in the camera image. A
10-frame recent-view re-anchor improved the numbers slightly; a single distant
anchor stopped matching. This is evidence for maintaining several object views,
not a validated live tracker or a robot test.

[Watch the synchronized 1× video](registration_comparison.mp4). It shows all
120 frames, including the missed color-ICP matches. Cyan is the _same_ SAM3
mask on every panel. Yellow A/B/C are three fixed 3D points transported from
frame 280; their connecting lines show direction. White hollow circles are
where a rigid attachment to the recorded gripper would have moved those same
points. The renderer draws a landmark even when it is hidden. The gripper
reference is approximate, not pose ground truth.

In particular, A begins near the grasped end. UFM projects A outside the
_visible_ dowel mask in 103/120 frames; even the independent rigid-gripper
prediction does so in 107/120. In the final frame, depth at UFM's projected A
is about 24 mm **nearer** than the point, consistent with a foreground
occluder. Thus A outside the outline does not alone show a tracking error.
The outside-mask projection is inconclusive, but UFM's **22.9° final
long-axis difference** from the rigid-grasp reference is a serious failure for
insertion, not a harmless rotation around the dowel. A–B spans 58.6 mm and is
within 4.1° of the measured surface's principal axis at frame 280. This
benchmark does not establish which portion of the difference comes from
tracking versus actual grasp slip or calibration/timing error; it does establish
that the estimated pose is not validated for insertion control.

![All methods at frame 399](frame_399.jpg)

## Controlled comparison

The input is the recorded, aligned RealSense RGB and depth for the insertion
demo at `~/.cache/huggingface/lerobot/demos/insertion/recording`, with the same
cached causal SAM3 dowel masks as the [earlier mask/depth experiment](../flash-mask-depth-tracker/README.md).
Every frame uses the earlier depth cleanup: reject missing depth, an inner
2-pixel mask boundary, a local depth span above 10 mm, and disconnected
surface fragments; sample the retained surface at 3 mm. RGB match endpoints
must land on this measured surface in **both** frames. No gripper pose, later
frame, CAD model, dowel shape or task landmark enters registration.

For the learned matchers, the union of the two current masks defines one 320px
RGB crop. UFM uses its published `UFM-Base-DINOv2L-init` checkpoint and
source-to-target flow/covisibility. RoMa v2 uses its published v2.0.1 checkpoint
in `turbo` (320px) or `fast` (512px) mode. We keep endpoint matches with
covisibility/overlap at least 0.5, lift both pixels using their **own** aligned
depth, reject 3D outliers with a 6 mm RANSAC threshold, and fit one rigid
transform to the surviving 3D pairs. Colored ICP instead directly aligns the
same retained colored 3D samples and normals; three geometric/color weight
settings were tried. These are declared study settings, not calibrated safety
thresholds, and were not tuned against ground-truth object poses.

We compared each result _after estimation_ with the recorded gripper's relative
motion from frame 280. This assumes a rigid, non-slipping grasp, reasonably
correct camera calibration and frame synchronization. Violations can contribute
to these errors. Video/mask agreement is a second visual check, but cannot tell
us the exact hidden surface pose. The axis error uses the directed vector from
landmark A to B; no symmetry minimization was applied.

| Adjacent-frame method              | Valid pairs / 119 | Median one-step landmark difference | Median accumulated landmark difference | Median axis difference | Median registration time |
| ---------------------------------- | ----------------: | ----------------------------------: | -------------------------------------: | ---------------------: | -----------------------: |
| Colored ICP, geometric weight .968 |               109 |                              3.2 mm |                                48.3 mm |                  46.0° |               1.0 ms CPU |
| RoMa v2 turbo                      |               119 |                              1.6 mm |                                22.5 mm |                  27.8° |              32.4 ms GPU |
| RoMa v2 fast                       |               119 |                              1.7 mm |                            **19.3 mm** |              **17.6°** |              51.9 ms GPU |
| UFM Base                           |               119 |                              1.7 mm |                                20.6 mm |                  21.4° |              32.1 ms GPU |

The best color ICP weight missed frames 306–311, 313, 315–316 and 318.
Weights .7 and .3 had only 53/119 and 41/119 valid pairs, respectively; they
did not improve coverage. The learned matchers returned an accepted transform
for every adjacent pair, but that acceptance does **not** mean the object frame
stayed correct. In particular, a small one-step error can accumulate or rotate
the same chosen landmarks away from the dowel.

### Failure pattern and likely mechanism

The axis difference grows gradually while the gripper occludes the rotating
dowel. For UFM it first exceeds 10° at frame 309 and 20° at frame 331. The
clean, connected visible surface shrinks from 224 sampled points at frame 290
to 96 at frame 310. Yet UFM accepts 192/197 candidate pairs at frame 310 with
a 0.8 mm median fit residual; RoMa `fast` accepts 780/780 with the same 0.8 mm
residual. These local fit scores do not test whether the matches preserved the
_same material points_ or the global object direction. On a smooth, partially
hidden dowel, different local correspondences can fit the same visible patch.
Each estimated frame-to-frame transform is then multiplied into the previous
pose, so a directional bias accumulates.

The surface becomes much more visible again (270 clean samples at frame 399),
but UFM still differs by 22.9° from the rigid-grasp axis. This pipeline only
matches adjacent frames; it has no persistent object map or old-view
relocalization to correct a drifted pose. At frame 399, the measured surface's
principal 3D axis differs from UFM's transported axis by about 29° and from
the gripper prediction by about 16°. That independent surface check supports
rejecting the UFM pose, but the visible patch is incomplete and is not exact
pose ground truth. The data does not isolate how much of the gripper-reference
difference came from correspondence bias, true grasp slip, or timing/calibration.

The time column covers crop, matching, outlier rejection and rigid fitting. It
excludes loading the model and the input preparation, which separately took
about 36–40 ms median per frame for disk reads and full-image depth cleanup.
SAM3 inference was already cached and is also excluded. These numbers are
single-process offline measurements on an RTX 5090, not a measured live frame
rate or a guarantee that all work can be overlapped with the 30 Hz robot loop.

## Re-anchoring ablation

Using RoMa `fast`, each frame was instead matched directly to a recent
keyframe. The keyframe is replaced after a fixed number of frames;
the new global transform comes only from earlier estimated transforms, never
from the gripper reference. This is a controlled interval test, not a proposed
fixed live reseed rule.

| Maximum keyframe age | Valid / 119 | Median landmark difference | Median axis difference | Last landmark difference |
| -------------------- | ----------: | -------------------------: | ---------------------: | -----------------------: |
| 10 frames            |         119 |                    18.1 mm |                  13.5° |                  19.7 mm |
| 20 frames            |         119 |                    18.2 mm |                  14.1° |                  18.1 mm |
| 40 frames            |         106 |                    21.2 mm |                  13.7° |                  23.2 mm |
| Single anchor at 280 |          31 |                    22.3 mm |                  23.3° |                  33.5 mm |

The lower 40-frame axis number is **not** a success: 13 transforms were
withheld. A more distant image can lack enough correct shared visible surface
even when it is the same object. The improvement at 10–20 frames is too small
to solve drift; it suggests testing a multi-view object map that preserves
coordinates, identifies genuinely new surface, and relocalizes when overlap
returns. The one-step matcher should provide candidate correspondences to
that map, rather than define the object frame by chaining alone. This proposed
map was **not implemented** here.

### Four-history ablation

We also tested matching each new frame to frames N−1, N−2, N−4 and the frame
with the most clean surface samples between N−5 and N−20. All four source
poses came from the estimator's own past; gripper poses were used only for
evaluation. One variant selected the candidate whose transported A/B/C points
best agreed with the other candidates. A second jointly fitted balanced 3D
matches from the closest agreeing candidates. The latter subsampled at most
256 matches per source to prevent one image from dominating the fit. These are
simple causal study rules, not a validated runtime fusion policy.

| Method                                 | Valid / 119 | Median landmark difference vs gripper | Median long-axis difference vs gripper | Median matcher time |
| -------------------------------------- | ----------: | ------------------------------------: | -------------------------------------: | ------------------: |
| Adjacent RoMa `fast`                   |         119 |                               19.3 mm |                                  17.6° |               52 ms |
| Four-history candidate medoid          |         119 |                               19.7 mm |                                  16.9° |              205 ms |
| Four-history joint fit                 |         119 |                               19.8 mm |                                  17.1° |              205 ms |
| One recent anchor, up to 10 frames old |         119 |                               18.1 mm |                                  13.5° |         about 52 ms |

The four-history variants did not meaningfully fix drift. Over the clearer
frames 360–399, the median difference from the visible depth surface's
principal axis was 34.0° for adjacent matching, 32.0° for four-history
candidate selection, 31.1° for its joint fit and 22.7° for the recent anchor.
This surface axis is an incomplete-view diagnostic, not ground truth. In the
candidate-selection run, the chosen source was N−1 or N−2 on 88/119 frames;
the older clear frame was rarely chosen. Four estimated deltas are correlated:
their source poses already share earlier drift, and an old clear image may no
longer overlap the rotated current surface. Adding deltas by vote or average
does not make an absolute object anchor. A proper multi-view map or windowed
pose graph would need persistent object coordinates and direct consistency
checks against the current mask and depth; this experiment does not rule out
that design. See `multi_history_ablation.py` and
[`results/keyframe/multi_history_20.json`](results/keyframe/multi_history_20.json)
for the exact study settings and per-frame traces.

## Whole-shape correction

`shape_grounding.py` adds a correction after each matched frame-to-frame delta.
The next proposal starts from the previously **corrected** pose. The reference
is the fixed measured surface from an earlier clear frame (100 in the main
test, 0 in the repeat), not a cylinder, CAD mesh, gripper pose or future image.
This matters: the earlier frame-280 reference contained only about 58 mm of
visible shaft, whereas frame 100 contained about 106 mm. A surface that lacks
an end cannot enforce that end's position.

The correction minimizes distances between the whole reference and current
cleaned surface in both directions. Reference points hidden behind nearer
measured foreground are not pulled onto that foreground. Missing depth provides
no surface evidence. Visibility is interpolated between rays without blending
their depth values across holes or object boundaries. The mask additionally
penalizes exposed reference points outside the object and observed outer mask
extent outside the projected reference's convex envelope. That envelope is a
conservative containment constraint, not a reconstructed mesh or an exact model
of concavities.

For the current frame, this uses **all** cleaned surface components, with bounded
influence from distant 3D samples. Keeping only the largest component had thrown
away the lower end when the gripper separated the visible dowel into two pieces.
The earlier reference still uses its largest connected clean surface. A generic
PCA initialization can escape an incorrect local fit; it is refined only if its
whole-shape score already beats the local result, and the orientation branch
closest to the proposed motion is chosen. PCA itself is not the published pose.

| Matcher / reference   | Before: median axis difference from current depth, frames 360–399 | After | Median correction optimization time |
| --------------------- | ----------------------------------------------------------------: | ----: | ----------------------------------: |
| UFM / frame 100       |                                                             15.5° |  2.3° |                               29 ms |
| UFM / frame 0         |                                                             15.0° |  2.4° |                               31 ms |
| RoMa fast / frame 100 |                                                             29.8° |  2.3° |                               29 ms |

Each before/after pair transports the **same** reference principal axis and uses
the same initial surface alignment, so these values differ from the earlier
A–B/gripper metrics. The correction's whole-interval median current-surface
axis difference is 5.1° for UFM/frame 100; at the partly occluded frame 320 it
is still 20.6°. Current visible-patch PCA also becomes a poor axis reference
during occlusion. The gripper-based checks remain discrepant; they cannot
establish ground truth here. The data establishes better clear-view alignment,
not fully solved six-dimensional pose or material-point transport through
occlusion. Axial rotation and end identity are not independently validated.

![Before and after at frame 399](shape_frame_399.jpg)

The new video shows the same fixed reference surface in yellow and the same
three reference landmarks in white in both panels. Cyan is the current SAM3
mask. Points hidden by foreground are not drawn, while landmarks remain visible
through occlusion. These landmarks are selected from the new complete-length
reference; they are not the earlier frame-280 A/B/C. Every frame from 280 to
399 is included at recorded speed, including partial-view failures.

All knobs are in `ShapeConfig`: 5 mm depth tolerance and 2 px mask margin reuse
the earlier observation settings; 30 mm caps distant-sample influence, inherited
from the earlier coarse association radius. The 3 mm sample spacing controls
compute; 50 evaluations and 1e−5 convergence tolerance bound the optimizer.
They are study settings, not calibrated robot safety limits. The reported time
measures `correct()` only; it excludes fitter setup, the cached matcher, SAM3,
input loading and depth preprocessing.

**This correction is implemented in the offline experimental path only.** It
does not change FLASH's live tracker, dataset poses or robot commands. A single
earlier view still lacks unseen geometry; an incomplete or contaminated
reference can bias a valid new view. Automatic reference selection, a complete
multi-view surface, and calibrated acceptance/uncertainty remain unimplemented.

`run_shape_grounding.py` reproduces the causal sequence from cached deltas;
`render_shape_grounding.py` renders the comparison. Exact configurations, pose
traces and per-frame measurements are under `results/shape/`. Six focused tests
pass, including recovery from a known 23° synthetic tilt, missing depth,
foreground occlusion and a pose that tries to hide behind observed depth.
Disabling `WholeSurfaceFit.correct` makes the known-tilt test fail. The exported
H.264/yuv420p video decodes all 120 frames successfully.

## Reproduce

The local benchmark source is `compare.py`; `keyframe_ablation.py` and
`render.py` create the ablation and video. CSV, summaries and complete pose
traces are in `results/`. Inputs and pretrained checkpoints are local cache
files; they are not committed. Reuse the earlier experiment's mask-generation
instructions to regenerate SAM3 masks. The UFM and RoMa source revisions used
here were `ff78d6c3db807c91578acccb58ee67d4394d5e5f` and
`95c9968145c8906b7b59383258e9f73b02853d89`, respectively. The isolated
benchmark environment used PyTorch 2.8.0+cu128, OpenCV 4.12.0, and an RTX 5090. Colored ICP used the repository Open3D 0.20 environment.

UFM's constructor normally downloads a separate DINOv2 backbone before
replacing it with its full checkpoint; the benchmark skips that redundant
pretrained initialization. It does **not** omit the UFM checkpoint's encoder
weights. The UFM convenience API normalizes RGB to float32 on the input
device, so the benchmark passes GPU tensors and casts the complete checkpoint
to float32 before prediction.

Validation: two focused tests verify known 3D rigid motion with outliers and
require measured, valid depth on both match endpoints; both passed. All 119 real
transitions were scored, and the encoded
H.264/yuv420p video has 120 frames at recorded speed with fast-start metadata.
Frames 340 and 399 were inspected at full resolution. No FLASH runtime,
dataset, tracker default, GUI, or robot setting was changed.
