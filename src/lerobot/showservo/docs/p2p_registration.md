# P2P returned a different pose from the one its fit statistics described

**Replay coverage correction:** historical local comparisons below use frames
2–130 of a 191-frame act, not its entirety. See the corrected scope and extended
local validation in [regression evidence](p2p_regression.md).

The October 12 investigation found a concrete upstream registration defect:
`SVDClusterRANSACRegister.register` computes a surface-refined pose and scores
its residuals, but returns the earlier sparse-fit seed. This is a control-flow
bug reproducible on the CPU, without neural inference or low-precision GPU
arithmetic. It affects both objects in the session.

In the inspected upstream implementation:

1. `selected_T` receives the chosen RANSAC candidate.
2. `_maybe_refine_with_sdf` returns `refined_T`, fitted to the reconstructed
   depth surface. The selected cluster's `T` is replaced with this value.
3. Residuals, inlier flags and the final support check describe `refined_T`.
4. The function returns `selected_T` and saves it in `_last_selected_T`.

Thus the front end can adopt the unrefined pose using quality statistics from
a different pose. If refinement fails the final support gate, upstream
deliberately falls back to the seed and recomputes its statistics; that path
was already internally consistent.

The experimental correction lives directly in the maintained P2P fork, on
[`fix/sdf-pose-return`](https://github.com/TheWisp/point-to-pose/tree/fix/sdf-pose-return).
After the final support gate passes, `selected_T` becomes `refined_T`, so the
return and selection history describe the scored pose. Skipped/disabled
refinement, missing candidates and the existing support fallback are preserved.
No geometric constraint or tolerance is added. This intentionally changes
results; it is distinct from the bridge's historical performance patches.

**Rolled back from the live selection on October 13.** Complete replay and
reference-sequence tests found tracking regressions. The active dependency pin
is again the pre-correction `c3d2068`; `4abb818f` and the fix branch remain
available for offline experiments. The code defect described here is therefore
present in the restored baseline. Contract correctness alone did not justify
keeping the corrected candidate active.

The earlier source-hash-guarded bridge correction has been removed. LeRobot
now requires the exact fork commit in `benchmarks/p2p.lock.json` and rejects a
different or modified checkout before loading models. See
[P2P dependency workflow](p2p_dependency.md) for setup and controlled updates.

## Evidence

A six-point controlled test calls the real upstream `register` function while
fixing the candidate and refinement outputs to isolate return routing. With a
20 mm seed error and a correct refined pose, the old return claims six inliers
but its returned pose actually has zero. The repaired return has six inliers
and exactly matches the reported residuals. The same test checks the real
upstream final-support fallback, skipped refinement and disabled refinement.

A separate CPU probe uses saved replay candidates from act
`20261012_205715`, frames 2–16, and P2P's own SDF builder and refinement. The
SDF is rebuilt once from the recorded initialization depth/mask and held fixed;
the recorded sparse candidate is fixed too, isolating refinement and return
routing. No plane or upright constraint enters the estimator. An independently
fitted dominant visible top normal is used only to score its result.

| Visible-surface normal disagreement | Unrefined candidate | Refined candidate |
| ----------------------------------- | ------------------: | ----------------: |
| Median, 15 frames                   |        5.67 degrees |      2.10 degrees |
| Maximum                             |        7.31 degrees |      4.00 degrees |

All 15 frames improve. Their original returned poses disagree with the
reported residual arrays; the corrected return matches those arrays in every
case. All refined candidates retain at least 21 sparse inliers. This makes the
bug relevant to the observed tilt, not just a synthetic mismatch.

The existing SDF-off ablations do **not** establish that refinement was
ineffective: accepted refined poses were being discarded. Their historical
measurements remain valid for the defective implementation. Refinement can
still affect its flags and later state even when its pose is discarded.

The saved early point model is unchanged, so model growth is not required for
the initial drift. A fixed-correspondence, unweighted fit of those recorded
inputs still rotates a median 2.84 degrees. Holding track pixels fixed while
changing depth gives 0.83 degrees; holding depth fixed while changing pixels
gives 1.26 degrees (15 frames). These separate input ablations are not additive
and do not reproduce P2P's uncertainty weights or hypothesis selection. They
show residual sensitivity to the measurements, not a second isolated software
defect or proof that TAPIR alone caused the error.

## Limits and verification

The initial counterfactual above uses a fixed SDF and recorded seeds. It does
not establish six-dimensional ground truth, complete placement accuracy,
occluded-view robustness or moving-object benchmark accuracy.

**Subsequent replay found a tracking regression.** On 129 recorded
frames (2–130) of act `20261012_205715`, the unmodified baseline and corrected
pipeline were each run from the same recorded initialization, with Python,
NumPy and Torch seeds set to 17. The correction uses fork commit `4abb818f` and
LeRobot `ed84954cc`; the baseline uses P2P `c3d2068` and the bridge from
`c3c2cf633`, before the runtime return-value correction. The configuration and
recorded RGB-D inputs are unchanged. These runs include tracking, masks, model
growth and reconstruction feedback.

| Recorded quantity                                  | Baseline | Corrected |
| -------------------------------------------------- | -------: | --------: |
| Median reported cube rotation from initialization  |    6.99° |    10.95° |
| Maximum reported cube rotation from initialization |   15.42° |    41.97° |
| Cube frames marked lost                            |       20 |        10 |
| Gamepad frames marked lost                         |       34 |        23 |

The 41.97° estimate at frame 40 is accepted, while the arm and cable partly
occlude the cube. An additional complete corrected replay produces identical
cube matrices at every frame. The fix improves some frames (frame 16: 7.03° to
1.66°; frame 124: 6.01° to 2.84°) but cannot be treated as an overall accuracy
improvement. These rotation magnitudes are reported estimates, not measured
ground-truth orientation errors. Lower lost counts do not establish accuracy.
Physical stacking validation was withheld after this replay result; the arm
was parked and disconnected. Returning consistent statistics fixes the code
defect, but refinement under partial occlusion remains an unresolved estimator
problem. The cause of its larger rotations has not yet been isolated between
the current surface fit and subsequent model/reconstruction feedback.

**Tracing the bad frame changes the immediate diagnosis.** An instrumented
replay reproduces the corrected poses exactly. At frame 40, the sparse RANSAC
seed is already at 41.97°; SDF optimization fails to find an accepted step and
returns that seed unchanged. The jump check accepts 13 of 16 correspondences,
all marked confirmed, with a 3.50 mm mean inlier residual under the configured
5 mm correspondence threshold. The retained image points occupy a small visible
patch. Removing one point from a weighted rigid refit changes its orientation
by up to 12.27°, versus 2.10° and 2.80° in the earlier frames 2 and 16. This is a
measured sensitivity, not ground-truth angular error or a calibrated confidence
interval. Point count and residual alone do not establish orientation precision.

Freezing SDF integration after initialization changes frame 40 to 16.35°, but
frame 41 then reaches 36.17°. This isolates a contribution from reconstruction
feedback while showing that freezing the surface alone does not solve the
failure. These experiments are process-local and do not change the pinned
implementation. The broader [regression protocol](p2p_regression.md) is required
before promoting any proposed estimator change.

Two further 44-frame ablations (frames 2–45) separate more of that feedback:
disabling the global keyframe graph still reaches 35.63°. Disabling new
keyframes and point sampling entirely reduces the maximum to 10.93°, but lost
cube frames increase from 10 to 21. The latter also changes point-tracker
queries, point promotion, graph updates and SDF updates; it does not isolate
one of them. It is evidence that the model-update path amplifies the failure,
not a validated replacement tracker. Permanently freezing the model would lose
the ability to learn newly visible surfaces. None of these ablations were
deployed, and no object-specific motion prior or threshold was introduced.

The fork's `test/resgister/test_svd_cluster_ransac_register.py` exercises the
real registration function with controlled candidate/refinement outputs.
Accepted translation and rotation fail before the fix; all six cases pass
after it, together with 15 existing pose-jump-guard tests. LeRobot tests verify
commit/cleanliness enforcement, non-destructive setup, rejection before model
loading, and propagation of source/configuration identity into recordings.

A subsequent native CPU check removes the controlled candidate and refinement
outputs: real RANSAC, selection, SDF refinement and support gating run on three
recorded frames, under both rig and published YCBInEOAT registration settings.
All six calls disagree with their returned pose before the fix and agree after
it (largest residual mismatch becomes zero). Inlier counts happen to be equal
before/after in these examples. This check still fixes the initial SDF and uses
uniform uncertainty weights; it is not a complete tracking replay.

Reproduction scripts, JSON results, logs and the inspected upstream source are
preserved under the experiment archive's `p2p-rootcause` directory. No camera,
arm connection or motion was used by the offline probes/replays. The existing
GUI connection was used to park the idle arm before stopping its worker for
the full replay. The old P2P checkout was preserved; the fork was installed
separately.

## Clear-view depth-edge investigation, October 13

The initial false tilt does not require occlusion or model growth. A new
instrumented baseline replay reproduces both objects' matrices bit for bit on
the 129-frame window while recording the actual RANSAC refit inputs. During the
unobstructed cube window, frames 2–16, one tracked depth edge is the most
influential point in every leave-one-out fit of all correspondences.

That point starts at pixel (355, 311), sampling 413 mm depth. At frame 2 its
tracked location is (354.62, 309.59); P2P truncates the coordinates and samples
405 mm at (354, 309). Later samples differ from initialization by 8–10 mm.
The saved 5×5 depth neighbourhoods show the abrupt surface-depth transition.
This is an unstable depth correspondence on a visible object, rather than
evidence that the object actually tilted. All correspondence weights are equal
in these 15 frames because the uncertainty floor dominates; unequal weights
do not explain this example.

In 12 of the 15 frames, this point enters RANSAC's final least-squares fit and
then fails the residual threshold. RANSAC recomputes the inlier flags but does
not refit the pose after this change, so the rejected point still influences
the returned pose. This is a demonstrated limitation of its one-pass refit,
separate from the SDF return/statistics defect above. At frame 4 the point stays
within the threshold even after tilting the fit, so repeated refitting alone
cannot remove every bad correspondence.

The group tracker was replayed from the same initial RGB-D and masks over the
same complete recording, with its normal sampling, depth lookup and reseeding.
It received no P2P poses or robot/grasp information. Its existing depth lookup
uses a 5×5 median and rejects neighbourhoods whose 10th–90th percentile depth
spread exceeds 12 mm. Applied to the suspect P2P track, this rejects its initial
depth and all early current depths except frame 4. This explains a concrete
input-quality advantage; it does not establish that groups always track better.

The following comparison uses frames 6–16, when group context is available,
and scores every method against the same observed top-surface normal:

| Method                                                   | Median disagreement | Maximum disagreement |
| -------------------------------------------------------- | ------------------: | -------------------: |
| P2P baseline                                             |               5.25° |                7.13° |
| Group tracker, including surrounding context             |               1.77° |                3.73° |
| Diagnostic fit using only the group's own tracked points |               1.76° |                3.16° |
| Direct depth-surface fit from initialization             |               1.60° |                3.26° |
| Experimental P2P consensus refit                         |               2.10° |                3.65° |

The own-point row is an independent diagnostic fit, not the production group's
`own_ok` decision. The production group can use stationary surrounding context;
frames 2–5 retain its initial pose and are excluded from this common-window
table. Surface normals check two orientation directions, not full 6D pose.
Depth fitting and scoring share the measured depth, so this is not independent
ground truth. No upright or cube-specific prior enters any estimator. Changing
the scoring mask erosion from 5 pixels to 3, 7 or 11 changes the measured normal
by at most 1.23° in this window. These scores use a new common initial normal
and should not be mixed with the earlier frozen-SDF probe's 5.67° median.

Replacing just the fitter on P2P's fixed correspondences with the group's
robust fitter at the same 5 mm threshold reduces median disagreement to 2.10°.
Using 10 mm instead leaves 5.49° on the common window. The observed advantage
therefore involves both depth association and robust fitting, not a universally
better pose algorithm. A frozen-input rounded-depth probe also improves the
score, but it has not passed complete-sequence validation and is not deployed.

### What the two trackers actually share

The camera reader takes one SDK frameset, applies `rs.align(rs.stream.color)`,
and publishes the color/depth pair together under a lock. The GUI disables
decimation and hole filling, but leaves spatial and temporal depth filtering
enabled. It uses color-stream intrinsics for aligned depth. Alignment therefore
establishes a common pixel grid; it does not guarantee error-free surface
association at an edge. SDK frame pairing also does not establish identical
exposure times. Ordinary cached frames carry host publication time. The diagnostic capture
added below separately retains SDK timestamps; historical act recordings do
not have that evidence. See the
RealSense documentation on [frame synchronization](https://dev.realsenseai.com/docs/frame-management/)
and [alignment and occlusion](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/).

In the live application both trackers read that camera source, but independent
loops can consume different frames. This offline comparison instead supplies
identical saved image pairs and initial masks explicitly. They do not share a
precomputed sparse point cloud: each selects and tracks image points, then
performs its own depth lookup. P2P truncates pixel coordinates and samples one
depth value; groups use rounded neighbourhoods and the median/spread check
described above. These operations produce different fitting correspondences
even from the same aligned images.

SAM operates on RGB in that same color-image coordinate system. P2P propagates
masks over subsequent frames; groups segment at designation and then track
points. Sharing coordinates does not imply identical later masks or point
sets. The frozen-correspondence fitter ablation separately holds the actual
P2P 3D inputs fixed, isolating the refit from these input differences.

### Consensus-refit candidate: rejected for live use

Fork commit `583bf9d93fdfe76483e1f16980e97c176a7fa89f` on
[`experiment/consensus-refit`](https://github.com/TheWisp/point-to-pose/tree/experiment/consensus-refit)
repeats the sparse fit when its consensus changes, using the existing residual
threshold, support minimum and iteration budget. An unchanged consensus still
takes one fit. The bounded loop does not guarantee convergence if membership
cycles. It starts from baseline `c3d2068` and does not include the SDF return fix.

Four recorded-correspondence regression cases cover weighted/unweighted fits
and a nonidentity prior. A known-motion clean case and 15 existing jump-guard
tests also pass: 20 tests total. Restoring the original `_RANSAC` in memory
makes all four regression cases fail while the clean case passes. These tests
establish the narrow refit behavior, not overall tracking accuracy.

The 129-frame robot replay window improves reported cube rotation from
initialization: median 6.99° → 4.76°, maximum 15.42° → 11.35°. Cube lost frames
change 20 → 19, and gamepad lost frames 34 → 33. Rotation magnitude is not
ground-truth angle error. All nine reference sequences were then replayed,
and the candidate **fails the broader regression comparison**; see
[complete results](p2p_regression.md#consensus-refit-experiment-october-13).
Neither estimator experiment is selected for live use. The unchanged baseline
remains pinned, and no physical stacking trial followed these offline runs.

Evidence, scripts, raw traces, manifests and the inspected figure are preserved
in `/home/feit/Documents/showservo-experiments/20261013/p2p-clearview`. The
remaining research question is how to improve depth association and geometric
support without degrading tracking and model updates on moving objects. The
clear-view cause is now concrete; a generally reliable replacement is not yet
established, nor does this isolate every cause of the physical misplacement.

## Complete depth-lookup experiments, October 13

Two independent forks start from baseline `c3d2068`, retaining its registration
solver and SDF behavior. `9120a453` rounds pixel coordinates to the nearest
integer consistently at initialization, tracking and model growth. `d44f1e90`
uses the group's existing 5×5 median/spread rule for those three sparse lookup
sites, leaving dense reconstruction unchanged. The 12 mm spread limit and
three-neighbour minimum are comparison settings inherited from groups, not
newly calibrated sensor constants or a proposed universal policy.

Both improve the clear-view cube window (frames 2–16): median top-surface-normal
disagreement is 5.25° for baseline, 2.05° for nearest pixel and 2.29° for the
neighbourhood rule; maxima are 7.13°, 6.45° and 4.47°. These are the same
observed-normal diagnostic, not independent 6D ground truth.

The 129-frame recording window reverses the apparent benefit of the neighbourhood
rule: reported cube rotation reaches 76.40°, versus baseline 15.42°. Nearest
pixel reaches 12.92°, but fails the nine-video regression suite. Both full
7,449-frame reference suites completed; neither candidate is selected for live
use. The [regression tables](p2p_regression.md) retain worsening sequences as
well as improvements. Five nearest-pixel tests plus 15 existing jump tests pass;
restoring the old lookup in memory fails four of the five new tests. Four
neighbourhood tests plus the jump tests pass; restoring the old lookup fails
three of the four. Narrow test success does not override full replay failure.

### A delayed keyframe can replace the current pose with an older pose

A stage trace of `bleach_hard_00_03_chaitanya` reproduces the original baseline
and consensus-refit matrices exactly over the first 60 frames. The first bitwise
refit difference occurs at frame 5. At frame 25 it changes a jump-guard decision;
subsequent model updates change later inputs. The first refit-minus-baseline ADD
increase exceeding 5 mm occurs at frame 54. This 5 mm cutoff locates a trace point;
it is not an acceptance threshold added to the estimator.

At frame 54, the consensus-refit frontend emits an 11.05 mm ADD pose, then the
pipeline emits 34.49 mm ADD. Local optimization is disabled. The manager promotes
a keyframe sampled at **frame 51**; `KeyFrameGraph.update` optimizes that historical
pose, and `ModularPipeline.step` assigns it directly to the current object's
pose. Pending-keyframe promotion preserves the sampling frame and its RGB-D
observations. The assignment therefore drops the motion from frame 51 to 54,
even if graph optimization itself makes no correction. This is P2P's internal
keyframe graph, not LeRobot's group tracker.
The inspected upstream base `ffcbf24b` contains the same promotion and assignment
code; blame traces both to upstream commit `e56d45d9`. Our local baseline did not
change those lines. This attribution concerns that inspected version, not a
claim about any future upstream revision.

Skipping only the graph update for keyframe 51 preserves exactly the same
registration inputs/output at frame 54 and changes final ADD from 34.49 to
11.05 mm. All 54 earlier outputs are bit-identical. The first attempted
intervention targeted keyframe 54 and changed nothing because no such keyframe
was promoted; both runs are retained. This establishes the source of this
specific jump, not the cause of every later lost frame. Baseline also takes this
path: at frame 55, its sparse/final ADD changes from 8.53 to 39.61 mm.

Experimental fork `32297ecd` preserves current-to-keyframe motion before graph
optimization mutates keyframe poses, then applies it to the optimized pose. If
`C` is the current pose and `K` the original keyframe pose, the new current pose
is `C @ inverse(K) @ K_optimized`. The historical keyframe retains
`K_optimized` for its own SDF observations. The most recent promoted keyframe
supplies one correction per object, avoiding repeated corrections for a batch.
This preserves the estimator's relative motion; it is not a claim that either
input pose is ground truth. No threshold changes or object-specific rules enter
this candidate. Six tests
exercise the real pipeline step with known motion, noncommuting rotations,
current and delayed frames, batch ordering, multiple objects and jump rejection.
All six plus 15 existing jump tests pass; the original step restored in memory
fails five new cases and passes the current-frame control. Full replay results
are recorded in the regression document; this is an experimental dependency.

### Measuring camera pairing and filtering

`POST /api/showservo/camera/diagnostics` with `{"frames":60}` captures from the
GUI-owned reader without opening a second camera. The endpoint accepts 1–120
frames and returns the saved path. Each NPZ includes the paired RGB/final depth,
aligned depth before filtering, and depth immediately before temporal filtering.
The manifest records both original sensor frames' SDK timestamps, domains,
frame numbers, supported sensor metadata, depth units, intrinsics and filter
settings. Raw/intermediate arrays use the unrotated color grid; published RGB
and final depth follow the configured rotation. The GUI currently uses no
rotation. The extra arrays are copied only during a requested diagnostic batch.

A separate stateless disparity-to-depth conversion observes the intermediate
stage without advancing or resetting the live temporal filter. Reader/API tests
cover unchanged filtered output, paired arrays, separate sensor metadata,
request lifetime and GUI ownership: ten tests pass. Making both sensor timestamp
reports equal in memory makes the pairing regression test fail.

A 60-frame stationary capture on this rig found median RGB-minus-depth SDK frame
timestamp difference 0.018 ms, maximum 33.376 ms. Pair 55 reused depth frame 2220
with a new RGB frame, while the preceding pair also used depth frame 2220.
Reported `sensor_timestamp` differences were 9.307 ms median and 42.667 ms maximum.
These metadata represent different events: the SDK describes sensor timestamp
as exposure midpoint and frame timestamp as readout/transmission start. This is
reported timing, not an independently calibrated measurement of physical
exposure alignment. See the SDK's [metadata definitions](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_frame.h).

Across this static capture, the median per-frame 95th percentile of absolute
pre-temporal to final depth change was 1 mm on valid pixels. This does not bound
moving-edge artifacts. The historical recording lacks raw stages and sensor
metadata, so neither timing nor filtering has been established as the cause of
its pose error. The new capture disproves the stronger assumption that every
published pair necessarily contains simultaneous, fresh sensor observations.

### Where contradictions are visible

The existing act replay shows a pose-verification panel below the image and a
count of disagreement frames in its title. Its disagreement control jumps to
the next flagged frame. Per-object rows distinguish P2P/group disagreement from
RGB-D surface disagreement and show whether P2P nevertheless accepted the pose.
Unavailable or inconclusive evidence is explicit; old acts say verification was
not recorded. The diagnostic does not silently replace the control pose.
Twenty recording/rendering/geometry tests pass. A current live capture also
shows the panel flagging an accepted cube pose roughly 6° from its surface refit;
that screenshot is saved as `p2p-depth-lookup/live_pose_verification.jpg`. Stage-by-stage graph traces and
the camera timing plot in this experiment are offline evidence, not additional
live overlays.
