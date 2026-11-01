# FLASH restart: postmortem and research

2026-10-31. Code audit and literature review, not a replacement implementation.
The two preserved experiments are `proto/show-and-servo` at `80d90b783` and
`docs/flash-redesign` at `c72e1e1c3` (tag `flash-redesign-sealed-20261031`).
The latter's [handover](../../../src/lerobot/policies/flash/docs/SEALED_20261031.md)
records what was saved and verified. This research continues separately on
`research/flash-tracker-restart`.

Current design: [FLASH tracking pipeline](../../../src/lerobot/policies/flash/docs/tracking_pipeline.md).
That is the maintained proposal; this file preserves dated research and results.

## Did the redesign become worse?

There is no matched end-to-end comparison that establishes a global ranking.
There are confirmed regressions, and the user's worse experience must not be
dismissed because the newer system records more diagnostics. The main failure
of the workstream was replacing a demonstrably useful path without first
preserving its behavior as a comparison baseline.

Show-and-servo adapted recorded gripper motion using the target object's motion
and a held-object/gripper correction. Point2Pose linked the demo to the scene;
point groups updated motion within the recording. This could execute an action
without freshly verifying the held object's whole pose every frame. Its
[sealed report](https://github.com/TheWisp/lerobot/blob/80d90b783/docs/showservo/sealed/LESSONS.md)
records five stacks among five clean completed acts on October 12, but also one
insertion and two misses in the later October 18–19 examples. Aborted starts are
excluded from that stacking count. These are useful outcomes, not a controlled
success-rate estimate. The same report found serious held/contact pose errors.

| What changed or went wrong                                        | Evidence                                                                                                                                                                                                                                                                                                                                                                                  | Consequence                                                                                                                                                                                                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| We changed the estimator while rebuilding the product.            | The new surface path uses UFM motion followed by fitting a fixed partial reference; Point2Pose is not part of that path. [Implementation](../../../src/lerobot/policies/flash/docs/surface_tracking.md).                                                                                                                                                                                  | A successful old rollout cannot establish the new path works; changing several layers obscures the cause of a regression.                                                                                    |
| Surface correction can make an initially better pose worse.       | On the labelled cracker turn, final source-point displacement was 13.8 mm for the raw UFM chain, 31.2 mm after motion-constrained fixed-reference fitting, and 25.2 mm with the experimental growing map. [Recorded ablation](../flash-surface-multiview/README.md).                                                                                                                      | This is a concrete fitting failure on that clip, not just stricter rejection or display latency. It is not a whole-system comparison.                                                                        |
| The map and fitting objective were mismatched.                    | The model normally contains the first observed side, but the objective also asks new visible points to match it. Removing that reverse term alone worsened some labelled cases. [Ablations](../flash-surface-multiview/README.md).                                                                                                                                                        | New sides can pull the fit incorrectly; simply loosening the fit or adding points is not an established fix.                                                                                                 |
| Useful evidence does not stay together throughout estimation.     | UFM correspondences initialize fitting but are absent from its final objective. Group/object-point disagreements are handled later in `CausalPoseService`.                                                                                                                                                                                                                                | Geometry can discard a useful motion answer before a later gate rejects it. The benefit of one joint fit remains to be measured.                                                                             |
| Shared processing and event timing arrived incrementally.         | Shared demo/live scene estimation: [5d04bf37a](https://github.com/TheWisp/lerobot/commit/5d04bf37a). Shared vision processing: [8a394de5a](https://github.com/TheWisp/lerobot/commit/8a394de5a). Recorded RGB-D timing: [8916e1eda](https://github.com/TheWisp/lerobot/commit/8916e1eda). Motor-time attachment events: [3c58e4fb2](https://github.com/TheWisp/lerobot/commit/3c58e4fb2). | A good debugger result did not initially imply the same demo or Test result. Timing defects also affected physical pose interpretation. These links identify fixes; they do not say those exact bugs remain. |
| Acceptance and UI work outpaced an independent accuracy baseline. | Local demos have no independent pose labels; the frozen external baseline supplies masks and some saved traces use older configurations. [Benchmark limits](../../../benchmarks/FLASH_TRACKING.md).                                                                                                                                                                                       | Better residuals, fewer rejected poses, passing tests and smoother overlays can each coexist with worse physical tracking. The editor exposed implementation details without settling the tracker.           |

Some failures became more visible because rejection increased; some examples
also became harder as testing expanded to external objects. Those explain why
aggregate impressions are hard to compare. They do not excuse the confirmed
fitting and integration problems. The available evidence supports preserving
the old end-to-end behavior for comparison, not declaring Point2Pose universally
best: the [pose benchmark](../flash-full-pose-benchmark/README.md) contains both
wins and failures for it.

## How motion and surface evidence currently cooperate

```text
Current mask + sensor depth → measured partial surface ──────┐
Two images + measured depth → point matches → pose seed ─────┤
Fixed reference surface ────────────────────────────────────┤
                                                          ↓
                                                   surface-only fit
                                                          ↓
Object/group motion → separate comparison / prediction → selected pose
```

Previously tracked or predicted points do not fill the current depth image.
Points measured on the object constrain its motion under a rigidity assumption.
Points on a tray or another support constrain the object's motion only if their
relationship holds. Robot encoders measure the gripper; whether an object stays
fixed to it is likewise a hypothesis. These three statements have different
evidence even if each can produce a proposed rigid motion.

### Did groups include the object's own points?

**Yes, in both sealed implementations.** The current UFM comparison does not
run those persistent point trackers. Audit of the exact sealed commits:

| Implementation                 | Points on the object                                                                                                                                                                        | Points outside the object                                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Show-and-servo, `80d90b783`    | TAPIR tracks points selected inside the object mask. `TrackedObject.model` stores their positions in a fixed object frame; `_place_objects` fits that frame to surviving visible point IDs. | The same grouping system can carry the object using other members of its group, provided their relative motion remains consistent.                                               |
| Redesign, `c72e1e1c3`          | `CausalRigidPointTracker` tracks image corners seeded inside the mask, uses measured depth, and fits their initial 3D positions to their current positions.                                 | `scene_groups.py` deliberately selects surrounding points outside object masks; `CausalPoseService` combines their relationship predictions with the separate object-point path. |
| Current offline UFM comparison | Each image pair's matches are already fitted to **one rigid motion**, but no persistent point-ID map is maintained across the clip.                                                         | No surrounding-group or gripper input in these ablations.                                                                                                                        |

The redesign also contains a growing-point experiment:
[`VisibleRigidTracker`](../../../src/lerobot/policies/flash/visible_rigid_tracking.py).
It stores each accepted point in a fixed object frame, estimates motion from
visible known points, and adds newly visible points only after estimating that
motion. Hidden landmarks move with the same frame. This is used by the
TrackEverything pose/visible-tracking experiments, not the current UFM baseline
or the live `CausalRigidPointTracker`. We should reuse it as a comparison rather
than rediscovering the mechanism.

There were integration limits: the redesign's fast service calls
`tracker.update(frame)` without a fresh mask; slow visual estimates are checked
later. Its live object-point tracker loses points and is reseeded by accepted
visual estimates, rather than continuously growing its own point map. Surface
fitting and those tracks are reconciled later, not fitted jointly. In the older
system, the held-object mode could bypass the own-point update in favor of an
externally supplied gripper pose. These are code findings, not proof that any
one caused every observed failure.

**Expected benefit to test:** with only part of a rigid object visible, correctly
identified, sufficiently spread points can determine its motion and carry its
hidden points with it. This could preserve identity and reduce frame-to-frame
drift as visibility changes. Outside groups or a firmly attached gripper can
also help when no object points remain visible. They require an additional
relationship assumption that the object's own rigidity does not.

**Limit:** many matches can slide together across a smooth surface while still
passing a rigid fit and remaining inside the mask. A small or nearly straight
set of points may also leave some rotations uncertain. Rigidity cannot supply
missing point identity or unseen independent motion. Own-point tracking is
visual evidence, not an independent vote merely because it is called a group.

Next comparison: retain the current baseline, add persistent own-object tracks
with the same masks/depth, then add surrounding co-motion separately. Measure
partial occlusion, reappearance, smooth visible surfaces and broken support
relationships separately. Test joint surface/point fitting as another change;
do not silently replace the baseline with several changes at once. Predictions
must never be inserted as fresh sensor measurements.

Audit sources: [old grouping](https://github.com/TheWisp/lerobot/blob/80d90b783/src/lerobot/showservo/groups.py),
[old tracker frontend](https://github.com/TheWisp/lerobot/blob/80d90b783/benchmarks/group_live.py),
[redesign own points](https://github.com/TheWisp/lerobot/blob/c72e1e1c3/src/lerobot/policies/flash/causal_points.py),
[redesign surrounding groups](https://github.com/TheWisp/lerobot/blob/c72e1e1c3/src/lerobot/policies/flash/scene_groups.py).

## Initial research direction, 2026-10-31

The updated design is maintained in [Tracking pipeline](../../../src/lerobot/policies/flash/docs/tracking_pipeline.md).
The initial proposal below is retained as research history. The updated design
makes automatic surface growth part of the core and keeps source contributions
so corrected registrations can revise the bounded map. This bookkeeping does
not imply automatic detection or rollback of every past error.

The agreed scope is rigid-object tracking from partial observations. Surface
measurements and optional motion constraints contribute to one estimate:

```text
SAM mask + RGB-D → cleanup → measured visible surface ───┐
Previously observed object surface ─────────────────────┤
RGB-D point correspondences → measured motion ──────────┤
Optional relationships → predicted motion ──────────────┤
                                                       ↓
                                      surface and motion estimation
                                                       ↓
                                     pose + uncertainty + source time
                                                       ↓
                                  update surface with supported observations
```

The object frame stays attached to the accumulated geometry; it is not
recentered on each visible fragment. Newly exposed surfaces extend the model.
How to accept those additions without letting wrong alignment corrupt it is an
experimental question, not a solved property of this diagram.

Each contributor identifies what it constrains, the frames/times it relates,
its uncertainty, and the condition under which it applies. Object point tracks
assume rigidity; surrounding groups assume a continuing relationship to the
object; gripper motion assumes a firm grasp. These can constrain the same fit
without pretending to be new depth measurements. The estimator must also work
when any optional source is absent. Deformable objects need a different motion
model later; rigid fitting must not silently label their deformation as motion.

Mask and depth cleanup are replaceable helpers whose effects are measured
separately. Our [depth-model pilot](../flash-contact-observation/temporal-depth-pilot/README.md)
found mixed results: LingBot helped the dish's withheld-depth test, while simple
nearest filling beat both tested models on several other examples. This tested
agreement with withheld sensor values, not true depth in holes or transparent
objects. Model-completed depth must remain labelled as inferred; it cannot
independently confirm a pose fitted using that same completion.

Later experiment: [Amodal SAM](https://arxiv.org/html/2604.20748v1) predicts
the complete 2D object outline, including hidden regions. The paper adapts
original SAM for images and SAM2 for video; it is not based on SAM3. Treat
its completed outline as an optional hypothesis input, kept separate from
the visible mask and measured depth. Depth at an occluded pixel belongs to
the covering surface and must not become an object point. Compare with and
without completion on controlled occlusions and labelled pose clips, measuring
hidden-outline accuracy, pose error, recovery and latency before adopting it.

**Deferred, not implemented:** reuse SAM3's visible masks, image features and
recent frames in a separately trained outline-completion component. There is
precedent in [HyperDAM](https://arxiv.org/abs/2609.34396), which uses frozen SAM3
features to predict a full bounding box, not a detailed hidden outline, and was
evaluated on hyperspectral video. **Outline completion is not pose estimation:**
it does not establish 3D depth or which physical surface points correspond
across frames. It could supply an additional image constraint; its value must
be measured by improved pose accuracy and recovery, not outline appearance
alone. Keep this separate from the immediate motion-and-surface fitting work.

## Research relevant to the restart

Primary sources were read on 2026-10-31. The lessons below are recommendations
for experiments, not claims that their reported accuracy transfers to our rig.

| Source                                                                                                                                     | Relevant mechanism                                                                                                                                                                            | What it does not establish for us                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Co-Fusion](https://github.com/martinruenz/co-fusion), [MaskFusion](https://github.com/martinruenz/maskfusion)                             | Segment independently moving objects, track them against evolving surface models, and fuse their labelled RGB-D observations. Model growth is part of tracking.                               | Their older segmentation and motion assumptions are not a validated solution to our smooth toys, grip occlusion or SAM errors.                                                                                                                              |
| [BundleTrack](https://arxiv.org/abs/2108.00516), [BundleSDF](https://bundlesdf.github.io/)                                                 | Use temporal feature matching and automatically maintained memory frames; jointly refine related poses. BundleSDF additionally learns object geometry in a separate thread.                   | A memory frame is an internal computational choice, not a manually chosen canonical view. Neural reconstruction and historical pose revision add cost; neither is required before comparing simpler fits.                                                   |
| [NVIDIA 3DObjectReconstruction](https://github.com/NVIDIA/3DObjectReconstruction#configuration)                                            | Its BundleTrack optimization exposes both 3D correspondence and point-to-plane terms, with robust loss. Useful implementation reference for retaining correspondence evidence during fitting. | It is a reconstruction workflow, not a demonstrated drop-in real-time tracker; its documentation also reports optimizer speed and memory problems.                                                                                                          |
| [ROFT](https://arxiv.org/abs/2111.03821)                                                                                                   | Combines fast optical-flow motion with slower segmentation/pose outputs through filtering and explicit delay handling.                                                                        | It does not solve unknown-object model construction; smoothing cannot recover an unobserved motion or fix a systematically wrong shape.                                                                                                                     |
| [Temporally Consistent Object 6D Pose Estimation](https://data.ciirc.cvut.cz/public/projects/2024TemporalPose/)                            | Combines pose observations, motion models and estimated uncertainty in an online factor graph.                                                                                                | This is evidence for a fusion approach, not proof that our confidence estimates or partial surfaces are sufficient.                                                                                                                                         |
| [Visuo-tactile estimation, 2025](https://arxiv.org/html/2503.19893v1), [VICON, October 2026 preprint](https://arxiv.org/html/2610.05180v1) | Combine visual and hand/contact evidence as separate constraints. VICON explicitly tests rendered geometry against depth and changes the motion relation with contact evidence.               | The former uses object meshes and tactile sensors; VICON uses reconstructed meshes, an instrumented glove and offline bidirectional refinement with many gates. Their results do not validate a causal RGB-D-only core or justify copying their thresholds. |
| [Switchable constraints](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf)                                 | Allows an optimizer to reduce or reject inconsistent relationship constraints.                                                                                                                | Robust optimization cannot guarantee detection of a wrong relationship when all independent visual evidence is absent. It need not be our first implementation.                                                                                             |

The useful distinction is between **a measured surface**, **measured point
correspondences**, and **an assumed motion relationship**. Keep those inputs
separate even if one estimator uses all of them. Do not average poses solely
because their producers output a confidence number, or count the same RGB-D
pixels twice as independent evidence.

An external-information interface can initially describe a proposed motion or
relationship, the objects and timestamps it concerns, its error/unsupported
directions, and the assumption that permits using it. A gripper integration
would supply such a relationship; external RGB-D recordings simply omit it.
Whether to implement this with a filter, a per-frame joint fit, or a short
window of poses remains an experimental choice. No rollback system or new
editor is implied.

## What certainty should mean

A single score is insufficient as the internal representation: position may be
well constrained while rotation is not. The proposed output contains the best
pose, its timestamp, whether it is observed or predicted, and uncertainty about
translation and rotation. A local covariance can describe one plausible answer;
separate plausible answers require multiple hypotheses or a distribution.
[PoseRBPF](https://arxiv.org/abs/1905.09304) and
[SpyroPose](https://arxiv.org/abs/2303.05308) demonstrate pose distributions,
including rotational ambiguity. They are research references, not validated
drop-in components for our unknown-object tracker.

For a simple UI score, define a checkable meaning: probability that pose error
is within a stated tolerance. Calibrate it on held-out labelled sequences;
do not convert a fitting residual into a percentage. Research on
[conformal pose uncertainty](https://github.com/NVlabs/ConformalKeypoint)
supports calibrating uncertainty sets, but its keypoint setting and statistical
assumptions do not automatically transfer here. Prediction can keep a pose
available during occlusion while uncertainty grows. No information source can
guarantee accuracy when the object moves unseen and every motion assumption fails.

## Minimal experiment infrastructure

The existing suite contains two rig demos, nine labelled YCBInEOAT clips and ten
Caryx clips. All 21 source directories were present in the local inventory on
2026-10-31. At the sealed revision, the runner could freshly process the external
cases, but the rig rows only scored saved preparations. A common fresh rerun of
both was the first missing boundary; scoring old files after an algorithm edit
does not test that edit. The local experiment below addresses this boundary.

The proposed harness needs only a sequence list, a causal frame reader, a
replaceable tracker call, and a result writer/scorer. Reuse existing dataset
readers, evidence formats and scoring where they fit. Identical recorded frames,
calibration, initialization, configuration and optional source inputs allow
controlled comparisons. Evaluation labels remain outside the tracker. Publish
one comparison report and frame-aligned videos with the same persistent pose
axes; save per-frame timings separately from startup and file encoding.

Expand with whole sequences and different cameras, objects and motions, keeping
splits by recording rather than adjacent frames. The two local demos remain
diagnostic because they lack independent pose labels. External supplied masks
isolate fitting; separate SAM runs test the full perception path. Candidate
datasets include [DexYCB](https://arxiv.org/abs/2104.04631) and
[HO-3D](https://arxiv.org/abs/1907.01481) for labelled RGB-D hand/object motion.
Their annotations are evaluation references, not permission to give the tracker
an object mesh or contact state unavailable in normal use. Existing cases are
development data; genuinely unseen sequences are reserved before tuning.

Algorithm changes belong first in this offline loop. An editor, policy rewrite
or hardware rollout is unnecessary to answer whether tracking improved.

## Comparisons that would settle the choices

Use identical RGB-D, masks, frame schedule and initialization. Compare raw
point-motion propagation, the archived surface-correction path, and a joint
fit retaining point correspondences plus depth/outline evidence. Separately
toggle fixed versus accumulated geometry and optional support-motion input;
otherwise several simultaneous changes would again prevent diagnosis.

Include visible turns, newly exposed sides, partial/full occlusion, a moved
support, an object moving independently of that support, and deliberately
incorrect motion assumptions. Test boundary filtering separately with supplied
masks and SAM output. Keep raw observations separate from predicted geometry.
An automatic map-growth test must catch wrong accepted alignments, not merely
show that it added points. No manual canonical-view selection is required.

Score pose error against labels (translation, rotation and carried-point error),
error distributions alongside published/lost coverage, confidently wrong poses,
recovery delay, map corruption, and processing latency/resource use. Fix the
evaluation definition before tuning; the current 21 cases are development data,
so a new holdout is necessary. Replay can compare perception on fixed input;
it cannot prove how a changed controller would perform physically.

The first inspection tool needs object naming/clicking, playback and optional
evidence overlays. Task keyframes and adaptation editing belong to the later
policy workflow.

## Local experiment: UFM motion versus surface correction

User direction after review: continue UFM, park Point2Pose, and iterate locally
without per-step commits or product work. The new runner is
[`benchmarks/flash_tracker_experiment.py`](../../../benchmarks/flash_tracker_experiment.py).
It uses the archived core's actual UFM preparation and surface fit. Input
adapters read current RGB-D and saved masks; they do not consume saved object
poses. External masks are supplied annotations and the rig masks are cached SAM
outputs, so this first experiment isolates motion and fitting, not segmentation.

Both candidates share the same UFM matches. One chains only the measured motion;
the other applies the archived surface correction. Neither includes group or
gripper fallback, nor the product's acceptance gates. A returned pose is a
candidate, not a claim of correctness. The comparison preserves failed frames.
The surface reference is the first usable observation, held fixed for this
baseline; this does not implement the proposed gradual model refinement.

All 21 existing source sequences passed image-size, count and timestamp checks.
The first fresh pass processed both full rig demos (713 and 628 frames), the
labelled cracker turn and stationary sugar box (181 frames each), and moving-camera
Caryx-032 (174 frames). On the 115 post-initialization cracker frames where both
methods returned a pose, median transported-point error was **1.88 mm with UFM**
and **8.62 mm after surface correction**. UFM then lost its connection; the
surface fitter kept returning candidates. On the stationary sugar box, all 180
post-initialization frames had poses: medians were **0.59 and 0.91 mm** respectively.
These are development cases scored against supplied pose annotations, not a
general ranking or proof of accurate held-object tracking.

The follow-on pass in `~/.cache/flash-tracker-restart/labelled-expansion-001/`
completed the other seven labelled clips with unchanged settings. There are now
fresh results for 12 sequences: both rig demos, all nine labelled clips, and
Caryx-032. The other nine Caryx sources have passed input checks only.

| Labelled clip        | Paired frames / 180 | UFM median error, mm | After surface correction, mm |
| -------------------- | ------------------: | -------------------: | ---------------------------: |
| Cracker turn         |                 115 |                 1.88 |                         8.62 |
| Stationary sugar box |                 180 |                 0.59 |                         0.91 |
| Bleach 0             |                 180 |                13.87 |                        28.34 |
| Bleach hard          |                 180 |                 8.53 |                        20.87 |
| Cracker, Yale hand   |                   0 |              No pose |         No paired comparison |
| Mustard 0            |                 180 |                 2.59 |                         9.21 |
| Mustard easy         |                 180 |                10.76 |                         5.90 |
| Sugar box, Yale hand |                 180 |                 4.40 |                        12.28 |
| Soup can, Yale hand  |                 180 |                 7.16 |                         8.68 |

Surface correction worsened median error on seven of eight comparable clips,
but improved one: removing it universally is not established. On the remaining
cracker clip UFM failed on the first pair; surface fitting nevertheless returned
all 180 poses, with a 533.48 mm median error. The results retain that failure.
The retry experiment recovered no additional frames on these labelled clips.
Per-frame traces and p90 errors are in the result files; this table alone is
not an uncertainty or availability metric.

Local results: `~/.cache/flash-tracker-restart/baseline-001/`. Each run saves its
configuration, exact input digest, per-frame outputs/timings and a copy of the
uncommitted core/runner source. Startup and inference timings are separate;
video rendering is a separate pass. Serial comparison time is not live-loop
frame rate. Inference runs do not modify source recordings, prepared demo skills
or robot control state. The suite is at
`~/.cache/flash-tracker-restart/suite-20261031.json`.

```bash
PYTHONPATH=src:. uv run --no-project --active python -m benchmarks.flash_tracker_experiment \
  --suite ~/.cache/flash-tracker-restart/suite-20261031.json \
  --case ycb-cracker_box_reorient --output /path/to/new-result-directory
PYTHONPATH=src:. uv run --no-project --active python -m benchmarks.flash_experiment_video \
  /path/to/new-result-directory/ycb-cracker_box_reorient
```

Use the existing FLASH inference environment (`~/.cache/flash-live-tracking/.venv`).

The local GUI at `http://127.0.0.1:9102/#/run/flash` now opens a small experiment
browser: recording list, video, pose-calculation selector, and independent
mask, reference-surface and 2D-match controls.
The mask is the shared input, not a different segmentation produced by either
pose method. Missing estimates remain missing. Export completed runs with
`python -m benchmarks.flash_experiment_browser /path/to/result/case`; the default
output is `~/.cache/flash-tracker-restart/browser/`. Export verifies the original
RGB-D/mask digest and preserves camera timestamps in video and overlays.
Playback performs no inference and does not modify prepared skills or source
datasets. The first browser export mixed five original two-method runs with seven
later runs that also contained recovery retry. This was an incomplete experiment
rollout, not a decision to enable recovery only for particular object types.
The five original entries now use their completed retry runs; the input digests
and every original UFM/surface-fit pose were verified identical before replacing
those playback exports. The older exports remain archived locally.

| Viewer option                    | How UFM affects the displayed pose                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Pose from UFM matches            | Match RGB pixels, read depth at both endpoints, fit one 3D motion, accumulate it.                                        |
| Pose after surface correction    | Apply that same motion to the previous corrected pose, then fit the initial reference surface to current depth and mask. |
| Pose with earlier-frame recovery | Like the first option; after a broken link, retry matching from the last successfully linked image.                      |

The white cloud and coloured axes depict the **initial measured surface moved
by the selected estimated pose**, including predicted hidden parts. They are
neither current depth measurements nor raw UFM matches. This uses the old live
surface debugger's white-cloud convention, projected into the camera image.
`--record-correspondences` separately saves sampled UFM pixel pairs before pose
estimation, their filtering stages, and their exact source/target frame IDs.
The coloured 2D layer can show raw predictions or those agreeing with the fitted
rigid motion; agreement is not ground-truth correctness. Display sampling does
not affect inference. Normally the source is the previous usable image; missing
mask/depth can skip images, and recovery can compare much older images. The
viewer keeps that frame pair and time gap in the trail control’s hover text. The local
`match-display-001` rerun records this evidence using unchanged tracker settings;
older runs without correspondence evidence cannot display raw matches.
**UFM point trails** use TrackEverything-style coloured heads, constant colour
per point, thin fading paths and a dimmed RGB background. Mask and pose layers
remain opt-in. The previous overlay faded unrelated pairwise segments; it did
not follow the same points. `benchmarks/flash_flow_paths.py` now composes saved
UFM flow into display particles. New recordings sample the full network output
at each particle before saving its path. Older recordings without those paths
fall back to interpolation of the saved grid. Unsupported paths end; uncovered
areas get new IDs. These are composed UFM predictions, **not TrackEverything
inference or verified material-point identities**. Raw paths preserve predictions
outside the current mask. Hollow heads denote low UFM co-visibility; filtering
labels conservatively use the contributing saved samples. Display changes do
not change pose estimation. The four recordings named in the drift audit below
have been rebuilt using the full output, with their historical poses preserved.
Colours and trails follow the [TrackEverything visualizer](https://github.com/ayushjain1144/trackeverything/blob/main/utils/vis_utils.py).
Unlike its model-based static/dynamic classification, ours only dims subpixel
image motion for display; camera motion is not removed. Trail length is Off or
5/10/20/30 frames (10 by default). Rebuilding trails from recorded IDs on each
seek/method selection prevents stale ghosts and invented links across missing
frames. Display density is bounded to 96 particles per object per frame.
Surface correction can shift/rotate the saved cloud away from UFM's proposed
motion: its second optimization scores geometry, not the original RGB pixel
pairings. Earlier-frame recovery instead retries the first method from an older
successfully linked image; it does not combine the two pose methods.
All 32 current exports now contain it. Across their 7,567 frames, all three pose
streams and their availability were exactly unchanged from the earlier runs.
The exporter checks every match's frame ownership; the GUI regression checks
that rejected raw endpoints remain distinct from the fitted pose. Nineteen tests
pass, and deliberately replacing recorded target pixels with source pixels
makes the correspondence test fail. The actual viewer passed 576 seek/layer/method
checks across the 32 recordings with no JavaScript errors. Evidence and the
actual-dowel pipeline illustration are under the local cache's
`pipeline-explanation-20261031/` directory.
The inline overlay also checks that recovery draws its recorded older-image
pair and that switching methods clears it without seeking the player. Gamepad UI frame 207's source mask appears to include
the gripper circuit-board edge; its 32 sampled predictions into frame 210 all
fall outside the current object mask. This demonstrates that wrong source
segmentation must be distinguished from wrong correspondence predictions.

The follow-up `external-expansion-001` freshly processes the remaining nine Caryx
clips with all three candidates. `earlier-examples-001` adds the three earlier
lid/jar, bottle/plate and device/dish recordings (Caryx episodes 31, 50 and 110).
The expanded local suite is `suite-extended-20261031.json`, under the same cache
root, with 24 recordings. These runs use unchanged tracker settings and input
masks; this fills the missing comparisons rather than improving the tracker.
Caryx-000 and Caryx-015 remain deformable negative controls, outside the rigid
tracking accuracy claim. The Caryx clips lack independent pose annotations;
their returned-pose counts must not be called tracking accuracy.

The first recovery experiment (`--retain-motion-reference`) retries from the
last successfully connected image after a failed pair. In
`~/.cache/flash-tracker-restart/retained-reference-001/`, it restored ring
candidates from 290 to 695 of 712 frames, but did not recover the dowel or the
labelled cracker turn. The original two candidates and input hashes were
identical to the baseline in all five cases. Each extra UFM attempt cost about
35 ms; repeatedly retrying a poor saved view is therefore not a sufficient
recovery strategy. The ring has no independent pose labels, so its increased
availability is not evidence of increased accuracy. This is not a new default.

Uncertainty remains an experiment, not a calibrated percentage. We save match
counts, disagreement with measured depth, visible extent, and which directions
the surface constrains. On cracker frame 130, the final surface residual was
only 2.59 mm while the annotated transported-point error was 22.75 mm: a small
surface residual alone cannot certify the pose. Here, point error measures the
distance between the same initial surface points moved by the estimated pose
and by the annotated pose; it is not error at an annotated interaction tip.

The initial proposed next experiment was to retain UFM correspondences as
constraints during surface fitting. The occlusion review below changes that
priority: neither source is a reliable reference under weak visibility. A joint
fit remains an experiment, but forcing agreement with UFM cannot be the remedy
when UFM is also wrong.

### Playback and occlusion audit, 2026-10-31

The bleach-hard source has adjacent stored RGB filenames separated by 5.0203,
2.8927 and 2.3595 seconds. The exported video preserves those exact gaps; playback
holds the preceding image. These are absent source images, not inference stalls.
The files alone do not establish whether capture or dataset packaging omitted
them. Playback now shows the recorded interval to the next image.

On the gamepad at 7.50 s (frame index 170), the saved mask covers a narrow rim.
All 181 accepted UFM point pairs pass the rigid fit, with a 1.81 mm median
residual, yet that fit proposes a 15.16-degree turn in one frame. On the ring
at 9.67 s (index 290), 20/20 pairs agree to 1.26 mm and propose 16.55 degrees.
These are proposed changes, not independently measured pose errors. The narrow
visible support and the user's observed drift show why point count, inlier
fraction and residual alone are insufficient. The gamepad then has no saved
mask at indices 173–196. The raw UFM chain cannot reconnect after a failed link;
surface fitting continues returning candidates from the initial partial model.
Neither comparison includes group or gripper motion. Crops and trace values are
saved locally in `~/.cache/flash-tracker-restart/occlusion-audit-20261031/`.

**Proposed experiments, not fixes already implemented:** first separate bad or
incomplete masks, wrong point correspondences, unreliable depth and insufficient
visible geometry. Use labelled occlusion clips and manually checked rig windows;
compare corrected visible masks against the unchanged saved masks. A useful
estimator should update the position/rotation directions the visible evidence
supports, retaining uncertainty in the others. Then test optional group and
gripper motion separately, including deliberately broken co-motion assumptions,
and test reconnection to earlier clear observations after reappearance. A hidden
object's prediction must remain distinguishable from a fresh visual measurement.
Success means lower labelled error and fewer confidently wrong poses, alongside
availability and recovery time; simply producing more poses is insufficient.

### Evidence priority and coverage review, 2026-10-31

The viewer loads all saved methods before playback. Switching methods selects
another result for the same decoded video frame; it neither runs inference nor
initializes that method from the previously displayed pose. Twelve switches
during active Caryx-053 playback preserved time/frame and caused no video reload
or seek. Different methods can still visibly disagree at the instant of switching.

**Working design, not an implemented fusion rule:** reliable measured depth and
the visible mask constrain the pose. UFM matches and group/gripper motion are
fallible additional evidence. There is no established universal ranking between
them: a firm grasp can outlast textureless visual matching, while slip invalidates
the grasp assumption. Groups derived from the same images are not independent
votes. A low surface-fitting cost is not the same thing as reliable measurement.
For example, a visible patch of a flat unmarked plate constrains depth and tilt,
but without edges or markings it cannot establish sliding along the plane or
rotation within it. Distinctive corners can constrain much more, even in a small
patch. Do not force hidden model points into the visible mask. Research on
[ICP uncertainty](https://arxiv.org/abs/1909.05722) separates sensor noise,
insufficient geometric constraints and convergence to the wrong solution; our
current diagnostic strengths are not calibrated probabilities.

Caryx-053 is a camera placed onto a white plate, with camera viewpoint change.
The experiment input is only 192×256 pixels. At frame 60 the plate has 447
accepted point pairs and a 1.65 mm median rigid-fit residual; those numbers do
not certify material-point identity. At frame 138 the accumulated UFM surface
extends beyond the plate's outer rim; surface fitting brings it visibly closer.
This confirms drift, not its full 3D magnitude or a single root cause. The mask
excludes the finger in the inspected frame 120; an initial reading of the small
preview as finger contamination was incorrect. Points drawn over the black
camera can represent the plate hidden underneath and are not by themselves
evidence of error. Keep this case for testing textureless tracking and partial
occlusion separately. Figures and browser checks: local cache
`flash-tracker-restart/review-20261031/`.

The Laval clock was omitted from the imported subset, not deleted. It has now
been processed with the same unchanged three candidates in `clock-restored-002`
and exported to the browser (25 recordings). `suite-with-clock-20261031.json`
adds it to the suite. Its RGB-D and saved SAM masks are inputs, not its older
estimated poses. Playback is explicitly nominal 30 fps; source timestamps are
absent. Laval has pose labels, but this fresh export has not yet been scored by
the harness's YCB-only scorer. At that stage, earlier DexYCB, synthetic-plane and
additional local-object experiments remained outside this pose viewer. In
particular, the cached DexYCB pilot has only an initial mask and intentionally
empty later masks; importing it unchanged would test missing masks, not the
tracking question. The viewer must not be described as all historical benchmarks.

Prioritize a small diverse panel before expanding the corpus:

| Source                                                                                 | Question it tests                                                            | Important distinction                                                                                                                          |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Own demonstrations, Caryx plate/device, Laval clock                                    | Robot occlusion; smooth surface; front/back rotation and reappearance        | Rig/Caryx have no independent pose labels; Laval does                                                                                          |
| [DexYCB](https://dex-ycb.github.io/) / [HO-3D](https://github.com/shreyashampali/ho3d) | Grasping and sustained hand occlusion with RGB-D and object-pose annotations | Use complete temporal sequences; many objects overlap the YCB family                                                                           |
| [WildRGB-D](https://wildrgbd.github.io/)                                               | Unfamiliar objects, camera motion, newly exposed surfaces                    | Primarily camera orbit around objects, including static-hand captures; camera poses are not independently measured dynamic-object trajectories |
| [T-LESS](https://arxiv.org/abs/1701.05498)                                             | Textureless geometry, similar shapes and symmetry                            | Sampled viewpoints; useful for registration, not automatically a continuous motion benchmark                                                   |
| [ClearPose](https://github.com/opipari/ClearPose)                                      | Transparent objects, bad/missing sensor depth                                | Raw sensor depth is input; rendered completed depth and pose labels are evaluation references                                                  |

ScanNet/ScanNet++ and ARKitScenes are useful camera-motion/depth controls, not the
first choice for hand-manipulated object motion. CA-1M, FunGraph3D and RoboSpatial
primarily address detection, functional structure and spatial relationships.
[HOT3D](https://facebookresearch.github.io/hot3d/) lacks a native aligned sensor
depth stream; [Aria Digital Twin](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)
offers rendered ground-truth depth, which must not be mixed with real sensor
depth when measuring sensor failure robustness. Keep synthetic cases separate.

Next controlled comparisons: visible-surface fitting alone, plus UFM, plus group
motion, plus both; add gripper motion separately on the rig with a stated grasp
assumption. Include clear/partly occluded/reappeared windows, corrected-mask
controls and known-broken co-motion. Keep some objects/recordings held out and
report per-case pose error, missing estimates, recovery time and compute cost.
Do not weight a family more heavily merely because it contributes more frames.

SAM supplies useful evidence beyond a binary mask: per-pixel mask scores,
predicted mask quality, and a video-tracker object-visibility score. Visibility
means whether the object appears, not what fraction is hidden or where its
hidden surface lies. SAM3's text-detection presence score is a different signal.
Neither is depth or pose confidence. Our existing overlay adapter already uses
`object_score_logits` for lost-track recovery, but the current experiment reads
binary masks only. Preserve these signals separately in a future SAM pass and
measure their relation to actual errors before using them to weight fitting.
Sources: [SAM2 paper](https://arxiv.org/html/2408.00714v2),
[SAM3 tracker outputs](https://github.com/facebookresearch/sam3/blob/main/sam3/model/video_tracking_multiplex.py),
[SAM3 image outputs](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py).

### Additional samples, 2026-10-31

The viewer now has **32 entries**, including seven additions processed with the
same three methods and unchanged tracker settings:

| Added entry                                                |  Frames | What it adds / mask input                                                                                |
| ---------------------------------------------------------- | ------: | -------------------------------------------------------------------------------------------------------- |
| DexYCB blue mug                                            |      72 | Hand occlusion; fresh causal SAM3 masks from one initial click, replacing the pilot's empty later masks. |
| Rig yellow block                                           |      75 | An object pushed during the robot run; existing recorded masks.                                          |
| WildRGB-D car, scene 000                                   |     361 | Complete camera orbit clip, different sensor and background; dataset masks.                              |
| WildRGB-D car, scene 002                                   |     300 | Complete camera orbit clip with surrounding objects; dataset masks.                                      |
| Synthetic plane, smooth tilt / textured tilt / translation | 31 each | Controlled inputs, explicitly labelled synthetic; existing masks.                                        |

These are four real clips plus three controls, not seven new independent real
scenes. The two WildRGB-D clips show the same toy-car type. Their calibration
follows the [official reader](https://github.com/wildrgbd/wildrgbd): transpose the
stored intrinsics and convert millimetres to metres. All native frames are
retained. Their time comes from the metadata frame rate; the mug's playback is
nominal 30 fps. Neither has original per-image capture timestamps in this import.
Only the selected WildRGB-D scenes were downloaded, not the full archive.

Local suite: `~/.cache/flash-tracker-restart/suite-diversity-20261031.json`.
Results: `diversity-001` and `diversity-wild-001` under that cache root.
Preparation scripts, source metadata, mask previews and the sealed-code audit
are in `expansion-20261031/`. No source recordings were overwritten.

These new entries are visual diagnostics, not newly scored pose benchmarks.
The imported mug and synthetic controls are not connected to pose scorers;
WildRGB-D camera poses are not independent labels for dynamic object motion.
Both car clips return poses on every frame, which establishes availability,
not accuracy. Their mask provenance is displayed correctly as dataset annotation.
Persistent object-point tracking and surrounding-group inputs have **not** been
added to these runs; that remains the controlled comparison described above.
Verification: decoded every added video and checked its frame count/timestamps
against the saved overlays; exercised all three methods at the beginning,
middle and end of all seven entries (63 browser checks, no JavaScript errors).
The relevant 11 harness/API tests and exporter lint/format checks passed.

### UFM drift audit (2026-10-31)

Two integration issues were separated from model error:

- The crop uses `warpAffine`, but the inverse used a resize-style half-pixel
  adjustment. Fixed the inverse; an image encoding its own x/y coordinates
  reproduces the failure before the fix. Historical pose results remain as
  originally calculated; the correction applies to subsequent experiments.
- Composing flow from a grid sampled every four network pixels can exaggerate
  drift. New diagnostic recordings save particle paths sampled directly from
  the full network output. Rebuilt both FLASH demos, DexYCB blue mug and
  Caryx-007 using that full output, without changing their poses or masks.

For 50-pair windows, the distance between sparse-grid and full-output paths
(median / 90th percentile, original image pixels) was 0.057 / 0.153 for dowel,
0.094 / 0.222 for gamepad, 0.007 / 0.486 for mug, and 0.050 / 2.272 for Caryx-007.
The largest Caryx discrepancy was 18.94 pixels. **This measures display-induced
error, not accuracy against physical point labels.**

Identical-image controls have known zero motion. Repeatedly applying the full
UFM output 60 times still moved points by median 0.13, 0.20, 0.25 and 0.97 pixels
respectively. Thus model bias can accumulate even without scene motion or
occlusion. Synthetic image translations were recovered to median 0.03–0.08 px.
The RGB order, uint8 normalization, flow direction/units and checkpoint API
agree with the upstream implementation. No upstream model-code bug established.
These controls do not establish the accuracy of tracks through real occlusion,
nor prove that crop resolution and segmentation inputs are optimal.

UI: neutral trail label, no duplicate video badge, detailed pair statistics on
hover. Evidence/scripts: local `drift-audit-20261031/` cache. Regression coverage
includes crop/image round-trip, full-output motion missed by the sparse grid,
recording/export preservation, and seeking/switching in the browser.

### Rebuilding examples from the viewer (2026-10-31)

“Rebuild all examples” reruns the listed cases with current code and each case’s
recorded configuration, using the saved RGB-D and masks. A separate process
reports frame progress, prevents concurrent rebuilds and survives a GUI restart.
Each complete playback replaces its list entry; old files remain available and
failed cases keep their previous playback. The same selection is retained when
its replacement is published. Recording, tracking, fitting and export use the
existing experiment pipeline, including full-output UFM trails.

Verified through the actual button with two 12-frame labelled YCB clips and real
UFM inference. Automated checks cover partial failure, immutable old results,
process-lock ownership, page reload and form replacement during requests.

For objective UFM assessment, measure raw pixel correspondence error against
independent labels, accumulated error for the same point over time, and errors
among accepted points alongside retained coverage. Mask/depth agreement and
forward/backward consistency are diagnostics, not material-point ground truth.
The current catalogue has nine YCB clips with pose labels; the FLASH demos have
no independent point-track labels. [TAP-Vid](https://tapvid.github.io/) supplies
labelled real/synthetic tracks for direct evaluation. Keep final 3D pose error
separate so fitting errors are not attributed to UFM.

### Benchmark reporting and design review (2026-11-01)

The same action is now labelled **Run all benchmarks**. Each job writes an HTML
results table and JSON report, reachable from the experiment sidebar. Reports
retain failed/not-completed cases, pose availability, labelled pose errors and
shared preparation/fitting times. Unlabelled accuracy is explicitly unmeasured.
The job reuses saved masks and case configurations; it does not rerun SAM or
implement the proposed growing-surface tracker.

Verification: 32 experiment/browser/sampling tests passed, including real
loading/fitting/scoring/export with substituted neural outputs, report HTTP
delivery and browser button/progress/result links. Another 91 JavaScript and
asset-version checks passed. The latter exposed an editor fingerprint mismatch
already present at HEAD; its cache version was corrected without editing that
editor. Lint and the GUI async check passed. No new full-catalogue inference
result is claimed by these reporting checks.

The maintained design now distinguishes session state from frame updates and
describes revisable surface additions. Automatic identification of a wrong
addition and calibrated pose uncertainty remain explicitly unresolved. These
are design proposals, not functionality exercised by the current benchmarks.
