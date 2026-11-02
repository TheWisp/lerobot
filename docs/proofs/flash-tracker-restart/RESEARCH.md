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

### Reversible surface growth experiment (2026-11-01)

This experiment follows pushed checkpoint `3b42021a1`.
The maintained design was shortened, with a sequence diagram replacing the
second flowchart. It describes pose estimation, not robot commands.

The first causal implementation is
[`experimental_surface_session.py`](../../../src/lerobot/policies/flash/experimental_surface_session.py).
It reuses the baseline's mask/depth cleaning and surface objective, adds image
correspondences to the fitting objective, and compares fixed versus growing
surfaces. Patches retain sampled camera measurements and placement separately.
A later aligned observation must support a pending patch before it affects
fitting. Exclusion/repositioning rebuilds the surface; the coordinate origin is
unchanged. This is repeatability evidence, not calibrated confidence.

An initial age-based memory limit discarded the broad first view after 32 newer
patches, even when they did not cover that surface. The corrected memory policy
evicts the patch losing the fewest uniquely covered voxels. A regression test
preserves an old broad view against redundant recent additions. This policy can
still retain wrongly aligned geometry; it is not an accuracy test.

Artifacts and exact inference source/config snapshots:
`~/.cache/flash-tracker-restart/growing-surface-003/`. The two earlier interrupted
runs and the smoke run are retained locally as implementation-debugging evidence.
All thresholds and memory budgets are in
[`flash_surface_experiment.json`](../../../benchmarks/flash_surface_experiment.json).

On the eight labelled clips with comparable returned poses, growth improved
median point displacement in four and worsened it in four. Comparisons use only
frames where both variants return a pose. For example, mustard-easy improved
from 6.1 to 2.9 mm; tomato-soup worsened from 9.0 to 16.9 mm. These are errors of
the same initial surface points against annotated object motion, not nearest
surface residuals. Rotation scoring does not remove symmetry-equivalent rotations.
The remaining labelled clip has no post-initialization pose from either variant.

![Same source images; fixed and growing surfaces with independently scored pose errors](growing-surface-comparison.png)

The figure deliberately includes improvement and regression. At bleach clip
frame 72, 99.4% of current measured points were within the configured 5 mm of the
growing surface, yet pose error was 32.5 mm. Local surface agreement therefore
does not establish correct coordinates for the accumulated model. Detecting and
repairing such errors automatically remains unresolved.

The session accepts an external timestamped pose prediction. Synthetic tests
cover prediction with an empty mask, no hidden-shape growth, and return to visual
tracking without resetting the reference. **These real-video comparisons do not
include group or gripper predictions.** Direction-specific evidence fusion,
calibrated covariance and automatic patch repair are not implemented; covariance
is explicitly absent, with conditional surface sensitivity recorded separately.
The local demos have no independent pose labels, so availability is not accuracy.

Browser playback displays each frame's actual model version, while its axes keep
the original reference. If recovery used another source image whose raw matches
were not recorded, it does not substitute unrelated previous-frame trails.

The completed sweep contains 32 examples and 7,567 frames. All were published
into the existing FLASH list; its **View results** control opens the comparison,
with a separate JSON download. The largest combined cloud contained 8,188 points, within the configured
8,192 limit; retained patches stayed within the 32 active + 8 pending budget.
Original sampled patches and model versions remain in the local run archive.

Verification: 136 focused tests passed, including baseline equivalence, bounded
storage, patch correction, prediction/recovery and playback provenance. The
report test also passed after correcting its per-method timing labels. Repository
checks passed. Headless playback against the actual port 9102 server opened both
local demos at 10 seconds with the new growing model, found all 32 examples and
the completed report, and reported no JavaScript errors. This verifies the
experiment and display plumbing; it does not establish accurate tracking under
occlusion. Live defaults remain unchanged.

### Benchmark review UI (2026-11-01)

FLASH now shows whole-clip scores beside the selected method's playback and
per-scene comparison bars for accuracy, median/p90 error, availability and
additional processing time. Accuracy percentages use the saved 10 mm / 10° or
20 mm / 20° counts; missing poses fail, initialization is excluded, and unlabelled
clips remain unscored. Clicking a bar opens that method's playback. Existing
exports load only their own matching score file; reruns publish replacements as
each example completes. Cancellation retains completed exports and earlier
results. The user's active benchmark survived the port 9102 server update.

The sidebar's **View results** opens this overview, including during reruns;
selecting a recording returns to playback. All 32 scenes now have five-method
results. Initially only nine had reference poses loaded for accuracy measurement.
The remaining scenes explicitly show that limitation instead of empty accuracy bars;
availability and timing remain accessible. This does not mean the original
datasets necessarily lack pose annotations—only that this benchmark has not
loaded them.

113 GUI, JavaScript and asset checks passed, including real scoring through the
browser and cancellation of a verified test worker while protecting an unrelated
process. Headless review of the actual server found all 32 scenes, displayed the
new panels/charts and reported no JavaScript errors. Inference is unchanged:
“UFM-assisted” describes the current combined fit, not a guaranteed priority of
depth over UFM.

The scorer now also reads the Laval clock's supplied object poses and the three
synthetic clips' exact 3D point trajectories: **13 of 32 scenes have accuracy
scores**. All 203 imported clock images match resized source frames exactly;
its label coordinates are converted from OpenGL to the RGB-D camera convention.
Synthetic label motion agrees with the renderer's known transforms. Labels are
evaluation-only and never reach the tracker. The nine YCB scores are unchanged.

Saved predictions were rescored using `benchmarks.flash_rescore_experiment`;
video, overlays and prediction files were verified unchanged. Previous summaries
and scoring code are archived locally. The GUI and downloadable report now show
the new scores, and future reruns retain the label configuration. The clock is
still difficult: UFM returns only 3 of 202 post-initialization poses; surface
correction meets 10 mm / 10° on 21 frames, and the growing fit on 10. The synthetic
cases are controls, not evidence of robustness to real occlusion.

130 targeted tests passed, covering camera conventions, delayed initialization,
missing poses, role-specific labels, invalid frame/time mappings, saved-result
rescoring and browser results.

### Shared 3D inspection (2026-11-02)

Benchmark playback and the demo/rollout scene now share calibrated RGB-D
projection and orbit/pan/zoom controls. PR #266's GPU pixel-grid approach keeps
RGB and depth aligned without rebuilding the background cloud on the CPU.
Every displayed packet contains one frame's RGB, depth and saved poses.

Orange shows the tracker estimate; green shows the independently labelled pose.
Without a supplied object model, green uses the initial measured partial surface.
Only Growing surface offers a drawing choice: **Starting surface** or **Expanded
surface**. This changes the orange geometry, never the estimated pose or score.
The same initial points are used by scoring throughout, so added geometry cannot
hide their motion error. Optional dataset meshes remain evaluation-only.

All 32 examples decoded at a middle frame with network access forbidden.
Tests cover RGB/depth projection, missing depth, label coordinate conversion,
frame/method switching during a request, growing geometry and local-only dataset
loading. 107 dataset/GUI/asset checks passed, plus eight existing scene/editor
checks. Tracking and scoring algorithms are unchanged by this viewer work.

DexYCB's blue object is a **pitcher**, not a mug. The preview is now matched to
BOP scene 006011 / original camera 841412060263: all 72 depth frames agree pixel
for pixel; RGB differs slightly in encoding. BOP and raw DexYCB labels use a
constant different model origin, but their relative motion agrees within
1.5e-7. Local provenance records these checks. The supplied BOP model is available
as the green 3D reference. There are now **14/32 scenes with accuracy labels**;
rescoring changed no predictions, masks or videos. For this clip, growing surface
meets 10 mm / 10° on 33/71 frames; surface correction on 41/71. UFM meets it on
23/71 and returns no pose for the remainder, so its low median error alone is
not evidence of better whole-clip tracking.

### More labelled clips and repeatable reruns (2026-11-02)

Added seven DexYCB hand/object clips (tomato can, tuna can, banana, bowl, mug,
drill, marker) and three BOP YCB-Video clips (clamp, mustard, bleach). They retain
consecutive 640×480 RGB-D frames, source calibration and original depth units.
Pose labels are evaluation-only. These ten clips use **dataset masks**, so they
test tracking given those masks, not SAM's segmentation quality. Their timing is
nominal 30 fps; source capture timestamps were not supplied. H2O remains pending
approved access to its official download. Local input provenance and source
hashes are under `~/.cache/flash-tracker-restart/labelled-expansion-20261102/`.

The catalogue now contains 42 scenes, with pose labels for 24. The expanded
five-method rerun is recorded in
`browser/.rebuild/rebuild-20261102-002523-c7e32c1a/` under the same cache root.
Its progress and completed results remain available in Run → FLASH.

**Rerun all benchmarks** now reuses a completed clip only when decoded RGB,
depth, masks, calibration, timing, labels, settings, tracker source, UFM/DINO
source, checkpoint contents and execution environment match. Changed or corrupt
results rerun. **Ignore cache** forces fresh computation, including fresh timing
measurements; cached timing numbers describe the original run. Reuse is at clip
level, not individual frames. A real unchanged-clip probe completed in 4.04 s
without loading UFM or rerunning tracking/export. This first expanded run still
recomputes older results because they lack a verified cache record.

### Do the labels themselves match the observation? (2026-11-02)

DexYCB supplies a previously scanned YCB object model. Its per-frame poses are
fitted to real multi-camera depth and human-marked image points; they are not
exact simulator states. See the [DexYCB annotation method, §2.3](https://dex-ycb.github.io/assets/chao_cvpr2021.pdf)
and [YCB scanning description](https://registry.opendata.aws/ycb-benchmarks/).
The paper's reported fingertip annotation errors concern hands, not object poses.

![Pitcher label compared with RGB and measured depth](label-quality-pitcher.png)

For the existing blue pitcher clip, the green viewer geometry agrees with the
source model/pose to 0.00006 mm: the visible discrepancy is not introduced by
that display transform. A separate triangle rendering check compares the
model's nearest surface with the depth at the **same image pixel**. It uses the
benchmark's visible-object mask, excludes missing depth, and does not refit the
label or discard points based on their disagreement.

Across all 72 frames, with a two-pixel mask edge excluded, the median absolute
depth gap is **8.38 mm**, p90 **26.17 mm**, and **56.66%** of compared pixels are
within 10 mm. Without excluding the edge: 8.69 mm, 29.31 mm and 55.22%.
First-frame median: 6.01 mm; last-frame median: 17.05 mm. The dataset's own mask
sometimes covers an unmodelled sleeve in front of the pitcher; using that mask
would wrongly score the sleeve's depth as the object's surface.

These numbers measure **depth agreement, not the percentage of correct poses**.
They combine model-shape error, pose-label error, calibration, mask error and
sensor error; this audit does not separate them. Our usual 10 mm / 10° score
measures agreement with labelled _motion_, and labels compared with themselves
necessarily score 100%. Constant model/pose offsets can cancel in relative
motion, so this audit neither invalidates every tracking score nor proves the
tracker is right when it disagrees. It establishes disagreement between the
supplied model/poses and recorded measurements, not which one is wrong.

`benchmarks.flash_label_depth_audit` reproduces the depth check from a case file,
mesh and explicit metres-per-mesh-unit scale. Local reports and comparison
rendering are in `~/.cache/flash-tracker-restart/label-quality-pitcher-20261102/`.
Tests check an analytically known tilted plane, front/back surface selection,
missing depth, and a deliberately incorrect pose through the real input readers.

The three new BOP clips provide a useful comparison. Over **all projected model
pixels with valid depth**, without using the supplied masks, clamp/mustard have
median gaps of 1.36/1.48 mm and 96.30%/95.84% within 10 mm. Bleach has 74.39%
within 10 mm; most remaining pixels have a nearer observed surface, consistent
with occlusion, not automatically a wrong label. Its supplied visible-mask
interior has a 2.29 mm median gap. These masks are not independent validation:
[BOP generates them from the model, labelled pose and observed depth](https://github.com/thodan/bop_toolkit/blob/master/scripts/calc_gt_masks.py).
Local reports, original meshes and download provenance are in
`~/.cache/flash-tracker-restart/label-quality-bop-20261102/`.

**Import audit, same pitcher clip.** All 72 imported RGB frames match their
same-index original more closely than any other original frame. All unsaturated
depth pixels are unchanged. The PNG adapter clears 53 pixels at the uint16
maximum (65.535 m) to missing depth across three frames, all outside the object
mask; these do not enter the reported comparisons. The 640×480 color intrinsics
and millimetre depth units agree with the official loader. Already-aligned depth
receives no second alignment.

![Original DexYCB rendering versus our imported model](label-import-audit.png)

An independent check downloads the original `textured_simple.obj` and executes
the object-rendering portion of [DexYCB's example](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/examples/visualize_pose.py)
with raw `pose_y`, bypassing FLASH and BOP conversion. It reproduces the visible
offset: last-frame median depth gap **16.28 mm**, versus **17.05 mm** in our
earlier rendering. The original/BOP surfaces, rendered with the same rasterizer,
differ by a median **0.18 mm** in depth. The two rasterizers differ by about
1.17 mm median on the same mesh; the last decimal is not meaningful accuracy.

Testing pose offsets from −5 to +5 frames on identical measurement pixels finds
that −2 reduces the moving-frame depth gap (frames 45–66) from 11.38 to 5.60 mm.
However, the unshifted pose covers more of the visible blue object in RGB; this
does **not** establish a timestamp correction. The source recording guide uses
approximate synchronization, but hardware timestamps are unavailable here.
No frames, poses or benchmark scores were shifted. **No import/convention bug
was found for this clip; pose annotation, calibration, model shape, depth and
capture timing remain possible contributors. Calling the labels themselves
wrong was too strong.** Scripts, raw model provenance and per-frame comparisons
are in `~/.cache/flash-tracker-restart/dexycb-alignment-audit/`.

### Point groups: two controlled comparisons (2026-11-02)

**Historical first implementation.** The measurements below describe the
initial integration, before stationary support, dedicated object points and
proposal comparison were added. See the completion entry below for the
current behavior; these results must not be attributed to the updated code.

Run → FLASH adds **Depth + mask fit, UFM + groups**, with fixed and growing
surfaces. Both reuse the existing scene point tracker: optical flow follows
pixel identities, depth supplies their 3D positions, and a rigid fit estimates
their shared motion. Points inside the object support its motion directly.
Surrounding points are associated only after they have moved with the visibly
tracked object. Proximity alone does not establish that relationship.

The group pose initializes the same surface fit used by the two corresponding
baselines. If the surface fit cannot produce a pose, an available group pose
is returned as **predicted**; it cannot extend the surface. These experiments
do **not** yet compare conflicting surface/group proposals before accepting a
visible fit. Group association thresholds are recorded in `GroupConfig`, using
the earlier tracker's defaults. There are no object-specific rules or gripper inputs.

Purple points in both 2D and 3D are the actual measured points used by the group
prediction. The short 2D trails join persistent point IDs. They are separate
from the fitted object surface and from UFM matches. Missing support is shown
as zero group points; historical results without this evidence say to rerun.

The 42-scene, seven-method run is
`browser/.rebuild/rebuild-20261102-011645-c96e7bb4/` under the cache root.
It preserves source/config snapshots, raw group proposals, supporting points,
final poses, and processing times. Side-by-side full videos of both FLASH demos
and analysis scripts are in `~/.cache/flash-tracker-restart/group-experiment-20261102/`.

**Completed: 42/42 scenes, seven methods, no execution errors.** Of these,
21 real clips have pose labels; three additional labelled clips are synthetic
controls. On the 21 real clips, the mean per-clip fraction within **20 mm and
20° of the labels** is:

| Surface | Without groups | With groups |
| ------- | -------------: | ----------: |
| Fixed   |         68.87% |      69.16% |
| Growing |         59.37% |      59.34% |

Missing poses count as failures; clips receive equal weight. These are label
agreement scores, with the label-quality limitations described above. On
common returned frames, the median point-error change is below 1 mm in 19/20
fixed-surface comparisons and all 20 growing-surface comparisons. The remaining
fixed-surface comparison improves by 1.12 mm. One labelled cracker-box clip has
no returned poses in either variant and is included as zero in the table.

The fixed-surface variant returns 191 group-only predictions over the real
recordings. Only 34 have pose labels, and **12/34** meet the 20 mm / 20° limits.
The growing variant returns two group-only predictions, both unlabelled.
Groups can fill gaps, but these results do not justify treating their output
as reliable or promoting either variant as an accuracy improvement.

Scene point tracking takes a median **1.71 ms/frame** across real-clip medians
(range 1.10–2.78 ms). Per-object fit-stage medians increase by 1.85 ms for fixed
surfaces and 2.34 ms for growing surfaces, including association work and any
change in fitting iterations. The scene tracker is shared across objects.
This is offline batch timing, not a robot-loop frame-rate measurement.

Both FLASH demos still drift visibly. With the growing surface, missing poses
change from 28 to 26 for the dowel (712 evaluated frames), remain 31 for the
gamepad and 117 for the cube (627 each). Neither demo produces a group-only
fallback pose: support is absent in the missing frames. At gamepad frames
265–266 (11.71–11.77 s), the group-enabled estimate rotates 106° while the image
changes little; no group prediction exists there. The saved cube mask also
ends up on the tape roll, an unchanged segmentation error in both versions.

**Why this is not the old debugger's full behavior:** that service additionally
seeds points specifically on each object and permits hypotheses based on
stationary nearby support. Its point/group-versus-surface comparison code also
requires a reference cloud; the debugger's constructor does not supply that
cloud, so those two comparisons are inactive there. Its separate depth
contradiction check does use an automatically saved first surface. This small comparison uses a subset of the
global scene points and requires observed co-motion for surrounding support.
The old debugger's saved logs contain group predictions, but they are often
from different camera frames than its slow recorded RGB-D samples; they are
not an aligned accuracy benchmark for the new methods.

At dowel frame 263 (8.77 s), 49 surrounding points provide a prediction, but
the surface fit moves the reference surface a median 24.8 mm away from it.
Only 25 cleaned surface measurements and no UFM image pairs enter that fit.
Applying the old debugger's comparison rule to these two recorded proposals
returns **ambiguous**, rather than favoring the surface fit. Frame 350 returns
**insufficient evidence**. This is a diagnostic on recorded proposals, not a
closed-loop rerun of the old service, and does not establish that the group
pose was correct. Both its anchor and the accumulated surface may already be
wrong. The next comparison should test when a partial observation can justify
replacing the point-motion prediction, together with seeding points on each
object's mask; simply adding the scene tracker has not reproduced debugger stability.

Another inherited limitation is the association rule's centre-displacement
test. A controlled 30° rotation about the object's centre moved its surface
points a median 24.2 mm, with surrounding points following the same rigid
motion, but established no surrounding relationship. Existing associations
can predict rotation; learning one from rotation alone is the missing case.
This rule was kept unchanged during the benchmark, rather than tuning a
threshold mid-run. Testing a measure of whole-surface movement is a separate
follow-up; this probe does not establish the cause of any particular video failure.

### Why the dowel still disappears: debugger comparison (2026-11-02)

This audit concerns the initial group integration above. It did not reproduce either sealed system. This is an
integration gap, not evidence that grouping cannot bridge this occlusion.

- At frame 150 (5.00 s), the current scene-wide pool contains **11** points
  inside the dowel mask, below its configured 12-point minimum. Running the
  debugger's object-specific seeding on the same image/mask yields **52**
  points. This is a seeding comparison, not 52 proven persistent tracks.
- At frame 180 (6.00 s), the saved dowel mask is empty. No own visible points
  can help then. The new adapter has no surrounding relationship because its
  association requires object motion first. The debugger permits stationary
  support and retains a surrounding-point prediction.
- Replaying all **713 frames**, feeding the same saved `conditioned_growing`
  proposals to the unchanged debugger service, supplies a current-frame pose
  in **all 26 frames** missing from `group_growing`. Five use a verified group;
  21 use a proximity-based, unconfirmed support hypothesis. No gripper input
  or stationary-hold timer was enabled. These counts measure availability,
  not correctness.
- Passing the reference cloud to that same service enables its existing
  group/point disagreement checks. This diagnostic also fills all 26 gaps,
  now using a verified group. Neither version fixes the whole trajectory:
  at frame 350 (11.67 s), both the benchmark and debugger-call comparison
  accept a badly oriented saved surface fit. The dowel is upright; the
  reference surface is drawn sideways. Reusing a group service alone is
  therefore insufficient to repair erroneous direct observations.

The replay isolates group integration: identical RGB-D, masks and saved surface
proposals, processed in source order. It does **not** rerun the debugger's
different vision frontend or reproduce asynchronous scheduling. The debugger's
four tracking modules are unchanged from sealed `c72e1e1c3`.

The first sealed system (`80d90b783`) uses causal BootsTAPIR rather than LK
optical flow, seeds object and surrounding points explicitly, and maintains
object-to-group membership through occlusion. Its own report also documents
incorrect attachment to stationary background during slow insertion; retaining
a pose is not proof of accurate motion. Source: `showservo/groups.py`,
`benchmarks/group_live.py`, and sealed lesson 7 at that revision.

Evidence, reproducible scripts, full-length comparison video, source-point
counts and screenshots: local directory
`~/.cache/flash-tracker-restart/group-experiment-20261102/debugger-audit-dowel/`.
The governing verification rule is **Prove the refactor equivalent** in
`.agents/skills/verifying-changes/SKILL.md`: borrowing the tracker class did not
establish parity with its surrounding association and pose-selection logic.

### Completed group integration (2026-11-02)

**This run predates the sustained-agreement rule below.**

The same two group options now use these components from the sealed debugger:

- Dedicated optical-flow points inside each object's mask, plus shared scene
  points outside it. Another tracked object can supply surrounding points.
- Surrounding points become support after agreeing with two observed object
  poses, including when both are stationary. That agreement is evidence for
  shared motion, not proof of attachment. A later accepted observation that
  separates from the group removes the association.
- Before changing the pose or growing the surface, compare the proposed
  surface fit with the point-motion prediction against the current mask and
  depth. Keep the prediction when the observation cannot distinguish them;
  allow a better-supported observation to correct it. Predictions never add
  measured surface points or establish a new surrounding association.

The comparison had an inherited bug: valid depth contradicting a proposed pose
could be counted as missing evidence. It now distinguishes a contradictory
measurement inside the object mask from unavailable depth or an occluder.
This permits a clearly supported observation to correct a wrong prediction.

Purple points in 2D and 3D show the measured own-object and surrounding support,
even when only one source currently supplies the prediction. Persistent IDs
join the trails; reseeding starts new IDs. Recorded evidence distinguishes
the selected source, both point sets and the surface-versus-group comparison.

The five existing no-group options are unchanged. Group parameters are saved
in `GroupConfig`; there are no object names, task phases or gripper inputs in
this integration. This completes the experiment's missing components, not
tracking accuracy or exact equivalence with every heuristic in the old service.

Regression coverage includes stationary occlusion, moving support, separation,
support on another tracked object, real optical-flow tracking without any
global scene points, rejection before surface mutation, and correction from
contradictory depth. Reintroducing each of five defects fails its named test.
The closed-loop rerun of both FLASH demos and one labelled bleach clip is
`browser/.rebuild/rebuild-20261102-080126-66fb2b03/`; outputs use the same recorded
RGB-D and masks as the first comparison. Availability and accuracy must be
reported separately.

That three-clip run completed without execution errors. All five no-group
methods reproduced their previous saved poses exactly. For the growing
variant, adding groups reduced missing poses from 28 to 5 for the dowel,
31 to 24 for the gamepad and 117 to 2 for the cube. The remaining five dowel
gaps are frames 265–269: neither object points nor surrounding support supply
a prediction, and the proposed surface fit does not overlap the observation.
Both object poses and their group associations still have errors.

The bleach clip demonstrates why a whole-clip score is insufficient: at
10 mm / 10°, both group variants pass 63/180 frames, all in the first 12 s,
versus 1/180 for their no-group counterparts. The fixed-surface group variant
still loses its pose at 26.42–27.20 s and ends 117.8° from the label. The growing
variant's median labelled point error worsens from 19.76 to 21.31 mm. Improved
early scores do not establish success during the later rotation.

### Sustained agreement before joining; prompt separation (2026-11-02)

The user identified a real weakness in the two-observation association rule.
At dowel frame 350, 17 of its 399 surrounding support points projected inside
the ring mask. The combined purple overlay also showed the ring's own points,
but the problematic association was not merely a display misunderstanding.

The revised experiment stores each nearby tracked point's first measured
position relative to the observed object. It must remain within the configured
distance tolerance for `support_confirmation_s` before becoming support
(initial experiment setting: 1 s). A contradiction removes it immediately;
the distance tolerance remains configurable through `fit_threshold_m` (5 mm).
Comparing with the first relative position prevents many small deviations
from accumulating unnoticed. New point identities start their own histories.

An occlusion interrupts confirmation of pending points; it does not count as
contradiction of established support. Current object-point motion can break
an existing surrounding relationship before a surface fit is accepted. The
viewer can isolate one object's group support while retaining both poses.

The duration and tolerance are experiment settings, not calibrated confidence.
An erroneous accepted object pose can still corrupt the association; this is
not a solution to all pose errors. The updated closed-loop comparison is
`browser/.rebuild/rebuild-20261102-082912-f5cd72ff/`. Earlier scores above belong
to the earlier implementation, not this revised rule.

The three-clip rerun finished without execution errors; the five no-group
methods again reproduced their saved poses exactly. The stricter association
rule is **not an accuracy improvement established by this test**. Compared
with the preceding two-observation implementation, missing poses increased:

| Growing-surface method | Previous groups | Sustained-agreement groups | No groups |
| ---------------------- | --------------: | -------------------------: | --------: |
| Dowel                  |               5 |                         32 |        28 |
| Gamepad                |              24 |                         31 |        31 |
| Cube                   |               2 |                        113 |       117 |

The bleach growing result is unchanged: its higher early score still does not
solve the later rotation. The fixed variant has 16 missing poses, previously 14.

A concrete failure of the stricter rule is recorded at dowel frames 100–150.
Of 175 previously admitted surrounding points still tracked, 90% moved less
than 2.02 mm in the fixed camera frame. The accepted object pose rotated
14.95°, making those points appear to move by a median 33.30 mm relative to
that estimated object frame. Thus pose instability can remove useful support;
this is not evidence that the whole group physically separated. Comparing
raw own-object motion with surrounding motion, independently of changes to
the fitted pose anchor, needs investigation before treating the rule as a fix.
The 5 mm tolerance is configurable, but increasing it would not resolve this
source-of-evidence problem. Joining and separation also currently share this
parameter with point fitting and pose comparison; they are not independently
tuned confidence estimates.

Measurements and the reproducible probe are in
`~/.cache/flash-tracker-restart/group-experiment-20261102/`:
`support-confirmed-drift-probe.json`,
`support-confirmed-drift-measurements.npz`, and
`support-confirmed-drift-probe.py`. The probe selects previously admitted IDs,
not arbitrary points elsewhere in the scene. Tolerance, source-time joining,
slow separation, and occlusion behavior have regression tests; restoring each
of five defects makes its targeted test fail. These tests establish the rule's
implementation, not the quality of its input pose estimates.

Playback now uses the original show-and-servo triangle/quad wireframe renderer
from `80d90b783`, copied unchanged into `gui/group_wireframe.py`. The adapter
uses recorded measured support and stable track IDs. These display links do
not add tracking constraints. Group visibility has an independent checkbox
in 2D and 3D; the object selector limits only the support overlay. Existing
42 exports received display-only updates with the recorded poses preserved.
“Rerun this scene” sends only the selected published recording to the existing
cancellable worker. The actual UI rerun `rebuild-20261102-091650-50b8e378`
recomputed the 181-frame bleach clip, exactly reproduced all seven pose traces,
and left the other 41 recording entries unchanged. The dowel's support loss
reported above remains unresolved; this UI work makes it easier to inspect.

### Why the accepted pose corrupted its own group reference (2026-11-02)

The 14.95° change at dowel frames 100–150 accumulated over 50 updates; it was
not one jump. Almost all of it was roll: the long direction changed only 0.36°.
The surrounding points independently imply 0.086° rotation (166 of 175
previously admitted, still-tracked points agree within the existing 5 mm fit
tolerance). These are measured-motion diagnostics, not independent pose labels.

The implementation does not jointly constrain the fit with those points:

1. `GroupHints.predict` selects object-point motion whenever available, even
   when surrounding support is also available. The selected pose initializes
   the depth/mask/UFM fit; group measurements are absent from its objective.
2. All 50 updates pass `compare_pose_proposals` as `agree`: each correction is
   small relative to that frame's prediction. Twelve pass even though both
   mask coverage and depth p90 worsen under that check's own measurements.
3. `GroupHints.observe` overwrites both object and surrounding pose anchors
   with each accepted fit. The next prediction consequently includes the
   preceding fitting error. This is not independent confirmation of its pose.
4. Association history still uses the earlier object-relative coordinates.
   Accumulating roll changes those coordinates for distant tray points, and
   their admitted support falls from 187 points at frame 100 to zero at 144.

Removing the anchor corrections algebraically from the saved object-point
motion gives 3.31° across this interval, versus the accepted 14.95°. This
isolates the feedback contribution; it is not a rerun with a corrected fusion
algorithm. Surface information is recorded after acceptance but does not
control which pose corrections are trusted. Thus a small accepted update
currently means neither high confidence nor an improvement over all evidence.

Next experiment: preserve an independent measured group-motion history and
test how it constrains the surface fit, including real separation and later
visual corrections. Simply freezing the object or enlarging the separation
tolerance would not test that requirement. No tracker behavior was changed
for this diagnosis.

Reproducible probes, per-update measurements and the two-frame/curve figure:
`~/.cache/flash-tracker-restart/group-experiment-20261102/pose-feedback-*`.

Sealed-branch code comparison: in `80d90b783`, `showservo/groups.py` fits scene
groups from visible TAPIR tracks lifted through depth, before placing objects.
Hidden members retain predicted positions but do not vote as visible measured
points. An object's own fit changes its group-relative placement only after
the visible-support check and a significant disagreement; small fit changes
leave that relationship unchanged. The code explicitly cites noisy small-object
rotations as the reason. The downstream `SeenThrough` depth test does not refit
the upstream group's points.

In `c72e1e1c`, `causal_service.py` checks direct fits against object-point and
surrounding predictions, then rebases their object-pose anchors on accepted
vision. `_observe_environment` likewise refreshes the relationship. Thus the
reference-update feedback pattern predates this experiment; its previous
presence is not evidence of correctness. That branch's live surface debugger
used this service only with optional `group_fallback` enabled (default off).
In the current experiment the fitted pose changes the point-to-object
relationship and support membership, **not** the raw optical-flow pixels or
their measured depth. This distinction is essential to interpreting the drift.

### Joint fit from independent point evidence (2026-11-02; local experiment)

The two group variants now fit measured depth/mask, UFM correspondences and
independent group point pairs together. A source supplies fixed object
coordinates and current camera coordinates, marked **measured** (tracked object
pixels plus depth) or **predicted** (object points carried by surrounding
measured motion). Each source has a configurable, point-count-normalized Huber
residual. The default weight of 1 and the existing 5 mm residual scale are
experimental settings, not calibrated uncertainty. Correlated sources can
still agree on a wrong answer.

The final fitted pose no longer rebases ongoing point tracks or determines
whether their surrounding associations survive. Group learning uses measured
object-point motion even if the downstream surface fit is rejected. Conversely,
a surrounding prediction alone cannot establish its own co-movement relationship.
Initialization, or recovery after all independent object motion is lost, still
needs a coordinate reference from an accepted surface observation; these events
are recorded as `point_reference_reset`. This is not a guarantee that a newly
established reference is correct.

Predicted points enter the motion objective only. They never become measured
surface patches. A clear new depth observation can correct a wrong prediction;
the existing post-fit observation check remains in place. The five methods
without groups are unchanged. No gripper source is injected in this experiment.

Verification covers actual LK tracking, synthetic measured co-movement and
separation, missing masks, the real surface optimizer, normalized source weights,
and scalar/batched residual parity. Restoring pose feedback, removing the joint
point residual, or making group learning wait for final pose acceptance each
fails its targeted regression test. The first partial rerun was cancelled when
the acceptance dependency was found; the complete rerun is
`rebuild-20261102-122643-joint-points`.

Complete-demo result (same recordings and masks; no independent pose labels):

| Object              | Missing frames before | Missing frames after |
| ------------------- | --------------------: | -------------------: |
| Dowel, 713 frames   |                    32 |                   30 |
| Ring, 713 frames    |                     0 |                    0 |
| Gamepad, 628 frames |                    31 |                   31 |
| Cube, 628 frames    |                   113 |                    0 |

These counts measure availability, **not accuracy**. The new cube trace has
275 predicted frames; inspection shows that it stays under the gamepad instead
of following the late erroneous mask to the tape roll. Dowel rotation during
the stationary frame-100–150 interval falls from 14.95° to 3.47°, with 69
surrounding points still available at frame 150 (previously zero).

This is not a general quality improvement yet: held-dowel estimates and the
growing surface are visibly worse in part of the carry. At frame 350 there are
15 own-object tracked points and no surrounding support. The fit falls back to
that own-point prediction after an ambiguous comparison; both proposed poses
already disagree strongly with depth (about 128 mm p90 in that comparison).
Keeping a point source independent does not repair its initially wrong reference
or later drift. The observation check and reference recovery still need work;
this experiment does not justify treating returned poses as correct.

All five non-group methods exactly reproduce their previous pose matrices
across both demos. The 74 focused tests pass. Both exports were published back
into the existing FLASH list; the server and source recordings were unchanged.
Uncommitted code was copied into the run's provenance archive. Local comparison,
rendering scripts, screenshots and mutation-test logs are under
`~/.cache/flash-tracker-restart/group-experiment-20261102/joint-points/`.
The [52-second before/after video](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_independent_groups_20261102.mp4)
contains both full demos at source speed, including the failures. Only this
video was uploaded; experimental source changes remain uncommitted.

### Persistent predictions and safer surface additions (2026-11-02; local experiment)

A tracked background group already measures the effect of camera motion. If an
object has stayed fixed relative to those points, the group can carry its hidden
points without a separately supplied camera pose. The relationship must end on
observed separation, not merely because the object disappears. A paired test now
checks the same hidden-object prediction with and without known camera motion.

Two extrapolation errors discarded surrounding support prematurely. We compared
a noisy pose fitted from a small visible fragment over the entire hidden object,
and used that pose to re-express distant surrounding points. Separation now
compares the group's prediction at the **measured object points** with their
actual positions. An established, agreeing group supplies the coordinate
reference for maintaining its surrounding membership. The final surface pose
still does not update those point tracks. Real measured separation still breaks
the relationship using the configured tolerance. Raw point drift can nevertheless
produce a false contradiction; this change does not solve that uncertainty.

Both group methods now retain an output after initialization when all measured
support is lost. Such frames are labelled `last_world_pose`, distinct from a
measured group prediction, and record the time since support was available.
Known camera poses transform that retained state into the current camera view.
If neither camera motion nor background motion can be measured, retaining it is
only a stationary assumption, not recovered tracking. There is no invented pose
before initialization and no calibrated confidence estimate yet.

The sprawling surface had a separate cause: real depth points were inserted at
wrong object coordinates after a bad fit. A similar fit in the next frame then
confirmed them. New additions are now projected into retained earlier views:
points in observed empty space, or outside the earlier mask without a nearer
occluder, are rejected. Points behind earlier measured depth remain possible
hidden surfaces. The existing mask/depth tolerances apply; no dowel-specific rule
is used. Retained views are bounded by the surface patch budget.

Point-by-point rejection alone was insufficient in the first trial: a mostly
contradicted extension left a small unexplained tail in previously hidden space.
We now defer that whole extension unless the existing configurable
`patch_overlap` fraction is consistent with retained views (default 0.5). This
is an experimental consistency threshold, not a probability. It can reject valid
growth when an earlier mask, depth or alignment was wrong; it does not implement
automatic repair of already accepted bad patches. Tests include an actual newly
visible back surface that must remain admissible.

The first trial, `rebuild-20261102-132342-joint-points`, completed the dowel but was
cancelled during gamepad processing after the unexplained-tail failure. The final
full rerun is `rebuild-20261102-133839-joint-points`, using the same recordings and
masks. It changes both growing methods as well as the group methods. No gripper
motion is supplied. These demos have no independent pose labels, so availability
and visual improvement must not be reported as measured pose accuracy.

The 82 focused tests pass. Restoring either group extrapolation error, removing
historical surface checks, bypassing the whole-extension check, or removing
persistent output makes its regression test fail. Evidence, code snapshots,
comparisons and render scripts remain under
`~/.cache/flash-tracker-restart/group-experiment-20261102/persistent-pose/` and the
rerun directory. Experimental code remains uncommitted.

Both full recordings completed and were verified in the running FLASH browser
without page errors. Both group methods output a pose on every frame after
initialization. For the growing variant:

| Object  | Frames | Missing before → after | New prediction sources when direct fitting is rejected |
| ------- | -----: | ---------------------: | ------------------------------------------------------ |
| Dowel   |    713 |                 30 → 0 | 20 own-point, 32 last-known-pose                       |
| Ring    |    713 |                  0 → 0 | None                                                   |
| Gamepad |    628 |                 31 → 0 | 13 own-point, 31 last-known-pose                       |
| Cube    |    628 |                  0 → 0 | 276 surrounding-group                                  |

The unsupported 32 dowel and 31 gamepad frames are **not successful group
tracking**. The cube is the clearest example of surrounding points maintaining a
hidden object's prediction. Qualitatively, the dowel's carry/release surface is
much less distorted; the gamepad still has visible alignment errors during
carry. Final stored point counts fall from 1171 to 435 for the dowel, 817 to 520
for the ring, 909 to 457 for the gamepad and 280 to 235 for the cube. No final map
points violate the initial retained view's ray check; that is internal
consistency with one observation, not independent shape or pose accuracy.

The [full before/after video](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_persistent_pose_and_surface_20261102.mp4)
shows previous above and current below, at source speed. It includes both full
recordings and labels last-known-pose intervals separately from measured group
predictions. The gamepad section displays the cube's surrounding support.

### Gamepad jump before lifting: rejected observation-check experiment (2026-11-02)

Inspection of the previous full rerun locates the first major break at frames
168–170 (7.41–7.50 s). The dedicated object point tracker loses its fit; a scene
point fit supplies a shifted reference for reseeding. The next own-point
prediction differs from the surrounding group by 7.92 mm, which removes 186
surrounding tracks. The reported orientation then reaches 21.2° from its starting
orientation at frame 170, while the gripper is covering the object.

This was not an unavoidable lack of evidence. At frame 169, the surrounding
prediction explains the initial measured surface better than the own-point
prediction: mask/depth support 65.1% versus 58.2%, and depth residual p90
2.32 mm versus 4.51 mm. These are internal observation checks, not labelled pose
accuracy. The existing comparison returns `prior_favored`, but was not consulted
before destroying the relationship. The point tracks' mutual agreement alone
had been treated as proof of real separation.

The tested change ran the same depth/mask comparison at that upstream decision when measured
point disagreement exceeds the configured tolerance. Only when it favored the
surrounding proposal did it reject the own-point proposal and retain support;
other verdicts retained the previous separation behavior. Rejected points did not
enter the downstream joint fit or define its reseeded reference. This comparison
used independently predicted transforms and the initial measured surface, never
the downstream fitted pose. It adds no threshold. It does **not** solve ambiguous
comparisons, sparse observations that admit many rotations, or reliable
attachment acquisition during the lift. No gripper encoder prediction is supplied.

Paired regression cases give the same coherent point displacement with either
stationary or genuinely moved RGB-D geometry. Only the first keeps the group;
the second must separate. Removing this check makes the stationary case fail
while the genuine-movement case still passes. The 84 focused tests pass.
The full rerun is `rebuild-20261102-141801-group-observation-check`; evidence and
scripts are under `~/.cache/flash-tracker-restart/group-experiment-20261102/gamepad-diagnosis/`.

The prior dowel display contains 296 sampled surface points initially and 435
at the end, joined only by short nearest-neighbor lines. It is a partial measured
surface, not a watertight mesh. Its incomplete display is not proof of a wrong
pose, but neither do the retained-view checks establish a correct complete shape.

**Full-sequence result: rejected and reverted.** Both demos completed. Dowel and
ring counts and final surfaces are unchanged. For the gamepad, the check reduces
the early orientation change at frame 170 from 21.2° to 4.7°, and preserves the
tray support through the covering interval. However, after the gamepad lifts,
the estimate stays on the tray: the retained surrounding prediction wins while
object points are unavailable, and the correct new observation fails to recover
its connection to the original surface. Availability is not correctness. The
change and its two synthetic tests are preserved in the local evidence directory
(`rejected-guard.patch`, `groups-rejected-guard.py`, `tests-with-rejected-guard.py`),
but removed from the active algorithm. Complete output remains under the job;
its browser exports are archived in `rejected_exports` so they do not replace
the previous examples. The synthetic pair did not cover loss followed by genuine
separation and reappearance; that temporal case is necessary before another
acceptance-rule change. No new threshold or gripper-specific rule was added.

### LK versus UFM near the gamepad occluder (2026-11-02)

`compare_lk_ufm.py` seeds the same image corners at frame 150 and advances both
methods through every frame to 172, without pose fitting affecting their image
positions. UFM uses the deployed DINOv2-L checkpoint and existing 320-pixel crop;
LK uses the existing 21-pixel patch and forward/backward check. Both endpoints
are checked against the same raw SAM mask and depth-support test for this
comparison; production UFM pose fitting additionally uses cleaned/trimmed depth.
No robot action, geometry fitting, or predicted hidden point is an image match.

At frame 168, 11 common accepted points also have their original position inside
the current mask with valid depth. Their displacement from frame 150 has median
4.25 px / p90 8.07 px for LK, versus 1.26 px / p90 1.70 px for UFM. This is a
stationary-object diagnostic, **not labelled flow accuracy**. The common set
shrinks to two at frame 169, so later aggregate comparisons are not meaningful.
UFM is better on this visible subset, but still accepts covered features: the
illustrated point B moves 24.1 px onto another visible location, with UFM's
current pair co-visibility output 0.993. That output is not the probability that
its accumulated point identity is correct. Both methods can keep the endpoint
inside SAM while switching which physical spot they follow.

The matching-patch experiment tried 7, 11, 15 and 21 pixels with identical initial
seeds, depth checks and pyramid settings. Smaller windows reduce aggregate drift
but discard different points; on identical surviving IDs the improvement is
mixed. Therefore changing the window alone is not a validated fix. The raw mask
captures also rule out the suspected circuit-board segmentation contamination
at frames 150 and 168: white markers over the board in the comparison are earlier
point positions that have since become covered, not new points seeded there.

**Current implementation, audited rather than inferred:**

| Stage                               | Matcher/input                                                  | Where SAM/depth filtering happens                                                                                     |
| ----------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Surface + UFM pose proposal         | UFM on unmasked RGB crops                                      | Accept matching pixel pairs only when both endpoints pass the cleaned object mask/depth; default mask margin is 2 px. |
| Dedicated object points for groups  | LK on full-resolution grayscale, 21×21 window, pyramid level 3 | Seed/accept point centers in the raw mask; depth support and forward/backward test. No 2-pixel mask erosion here.     |
| Surrounding scene points for groups | Separate LK on half-resolution grayscale                       | Track scene corners; depth support; current raw object mask determines ownership.                                     |

Even applying the same 2-pixel margin to LK would not contain a 21×21 matching
patch within the object; at half resolution it spans roughly 42×42 sensor pixels.
Coarser pyramid levels also use wider context. The linked picture shows a center
9.6 px inside SAM while 6% of its finest-level matching patch is outside. This
establishes the implementation's exposure to mixed motion, not the sole cause
of every slipped point. UFM also sees surrounding RGB; segmentation is currently
an endpoint/geometry check, not an input attention constraint.

Sources: [OpenCV's LK description](https://docs.opencv.org/4.12.0/d4/dee/tutorial_optical_flow.html),
[UFM model and usage](https://github.com/UniFlowMatch/UFM). The consolidation
proposal is maintained in the pipeline design; it is not implemented by this
comparison. Local evidence includes raw RGB/masks, per-point coordinates and
visibility, the window ablation, complete before/after recordings, and scripts.

Phone-accessible evidence:
[LK patch crossing the mask](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_LK_mask_patch_20261102.jpg),
[same-point LK/UFM comparison](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_LK_UFM_occlusion_20261102.jpg),
[short point-tracking video](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_LK_UFM_occlusion_20261102.mp4),
[full gamepad before/after showing the rejected guard's later failure](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_rejected_group_guard_20261102.mp4).
The short video repeats the same source interval three times and holds its final
frame briefly; the full recording uses original timestamps. After reverting,
the existing 26 group tests pass. No physical trial or robot control was run.

The user's abstraction clarification makes the consolidation backend-neutral:
LK and UFM are selectable providers of 2D matches behind the same contract.
Persistent point identities, SAM/depth filtering, grouping and surface fitting
belong downstream. This is a design revision, not a claim that the active
experimental code has already replaced its two separate tracking paths.

### 2026-11-02 — Matching adapters, before sharing inference

Extracted the duplicated LK forward/backward calls and the UFM field query into
`image_matching.py`. Both return query-ordered source/target image pixels,
acceptance and backend-native diagnostics. Explicit settings select LK or UFM;
UFM inference is reusable across point queries. SAM ownership, depth cleaning,
point IDs, group association and pose fitting remain downstream.

This step deliberately preserves the current full-resolution object LK,
half-resolution scene LK and cropped UFM paths. It does **not** yet make them
consume one shared inference result, change the mask margins, or fix slipping.
The implementation status is marked in the existing pipeline design.

**Equivalence:** compared frozen pre-refactor source with the new implementation
on every frame of both FLASH demos and the external bleach-bottle clip:

| Recording                       | Frames | Depth-lifted pairs | Outcome       |
| ------------------------------- | -----: | -----------------: | ------------- |
| Gamepad / cube                  |    628 |            320,288 | Exactly equal |
| Dowel / ring                    |    713 |            440,289 | Exactly equal |
| YCB bleach hard 00 03 chaitanya |    181 |             29,876 | Exactly equal |

The comparison checks raw correspondences and rejection stages, 3D motion,
scene point identities, group membership/state, reseeding and predicted poses.
It includes 113 empty object masks and 61 failed UFM rigid fits in the two demos.
Both implementations receive the same real UFM network output after verifying
that their requested input images are identical. Group feedback uses the same
recorded final-fit results. The unchanged final surface optimizer was not rerun;
its geometric and group inputs were compared instead. This establishes refactor
equivalence on these recordings, not that their tracking is accurate.

150 targeted tests pass. Two recorded-frame fixtures retain the original
implementation's outputs for permanent, hardware-free regression tests. Injecting
a one-pixel endpoint error into either backend makes the real-consumer fixture
test fail. A float32 threshold-comparison test also guards a subtle conversion
regression. Existing benchmark source hashing automatically includes the new file.

No extra model inference was introduced. With captured UFM output, CPU
postprocessing rises from about 0.55 to 0.69 ms per image pair (+0.13 ms); this
small allocation/contract cost is recorded rather than claimed as zero overhead.
Paired LK measurements show no slowdown; their second-call cache advantage is
not evidence of a speedup. No hardware run, server restart or commit was made.

[Machine-readable evidence](image-matching-refactor-20261102.json) includes source
hashes, counts, timing distributions and environment versions. The full original
source snapshot, runnable `audit.py`, logs and measurements are retained locally
under `~/.cache/flash-tracker-restart/image-matcher-refactor-20261102/`.

### 2026-11-02 — One matching pass for objects and surrounding groups

Implemented an opt-in shared LK or UFM pass. Both consumers use the same persistent
point IDs and mask/depth checks; object pose cannot move those points. Shared
object matches contribute once to fitting, not once as image evidence and again
as an independent group measurement. Matching continues with empty object masks.
Point budgets, image resolution and history length are configurable and recorded.
GUI reruns preserve these settings and cache them separately.

Ran both choices through the complete gamepad (628 frames), dowel (713) and
labelled bleach-bottle (181) clips: exactly one matching call per consecutive
frame pair, regardless of object count. This is a new algorithm configuration;
the adapter-equivalence result above does not guarantee equal quality here.

**Not ready to replace the baseline.** For the bottle, using surface growth and
groups on the same 180 evaluated frames:

| Matching                          | Median point error | p90 point error | Median rotation error |
| --------------------------------- | -----------------: | --------------: | --------------------: |
| Existing UFM + separate LK groups |             4.0 mm |          7.8 mm |                  2.7° |
| Shared UFM                        |             7.3 mm |         22.8 mm |                  2.0° |
| Shared LK                         |             4.6 mm |          9.8 mm |                  3.6° |

These compare identical reference points placed by estimated versus labelled
poses. The control was rerun with current group logic: the earlier 21.3 mm result
used older group code and is **not** the valid baseline for this comparison.

Both demos retain poses throughout, but neither has independent pose labels.
Shared UFM visibly fails during dowel rotation: at frame 350 the real dowel is
upright while its estimate is sideways. Projecting the unchanged initial surface
confirms a pose error, not merely bad surface growth. Shared LK is better in that
frame, but also drifts; availability is not correctness.

Shared matching, seeding and depth processing take median 35–37 ms with UFM and
12–13 ms with LK across these clips. This excludes subsequent surface fitting;
it is not end-to-end tracker FPS. The experiment changes image crop/scale,
sampling and group inputs together. The next useful experiment isolates these
changes on the failing rotation interval, rather than changing acceptance limits.

203 targeted tests and an additional browser test pass; four deliberate defects
(pixel scaling, second tracker, exhausted point budget and double-counting) each
fail their regression test. Rerun/cache and browser-label tests also pass after
export changes. The existing default stays unchanged; new recordings are named
“shared LK” and “shared UFM” in FLASH. Changes remain local and uncommitted.

[Full demo comparison video](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_shared_UFM_comparison_20261102.mp4)
shows the existing path above shared UFM, at recorded timestamps.
[Machine-readable measurements](shared-matching-comparison-20261102.json) identify
the inputs, settings and preserved source snapshots. Complete runs and drivers
are under `~/.cache/flash-tracker-restart/shared-matching-20261102/`.

**Follow-up: the gamepad also fails before pickup.** At frame 100 (4.37 s),
44 accepted points surviving from initialization have median accumulated UFM
displacement 3.89 px. A visible 30×30 button patch independently matches its
original position with zero-pixel displacement (normalized correlation 0.985),
and the mask bounds stay within one pixel. Thus the image tracks already drift
while the gamepad is stationary. At frame 144 (6.34 s), the object points disagree
with the environment prediction by 5.09 mm; the 5 mm separation rule removes the
surrounding group. At frame 160 (7.04 s), the final estimate has rotated 23.3° from
initialization while the object remains on the tray. The surface fit still accepts
it. This is not limited to complete occlusion or to the dowel rotation. The origin
of the image drift (sampling/scale versus model behavior) remains unisolated;
increasing a group tolerance is not a demonstrated fix. The local
`gamepad_pregrasp_audit.py` and `gamepad-pregrasp-160.jpg` preserve the diagnostic.

### 2026-11-02 — Recorded stationary-gamepad regression

Added a real RGB-D → pinned UFM → shared groups → growing-surface integration
test. The visually annotated stationary interval is frames 0–164 (0–7.24 s),
including gripper occlusion. Frames 165–170 show pickup beginning. Inference runs
through frame 280; the annotation is used only afterward. Inputs are hashed.

| Maximum false motion before pickup |  Before |   After | Test limit |
| ---------------------------------- | ------: | ------: | ---------: |
| Surface-center displacement        | 9.22 mm | 1.15 mm |       5 mm |
| Rotation from initialization       |  24.49° |   3.98° |         5° |

The limits were fixed before changing the algorithm. The estimate subsequently
moves 59 mm by frame 280; this prevents freezing from passing, but is not proof
of accurate held-object tracking. There are no independent gamepad pose labels.

The fixes are geometric, not timed or object-specific:

- Keep the existing 2 px boundary exclusion at **both** match endpoints. Tests
  verify it, but interior matches can still slide to wrong material points.
- Use the measured-surface objective consistently when fitting and deciding
  whether object matches contradict a surrounding group. A drifting image
  hypothesis cannot win merely by outweighing better geometric agreement.
- Do not pull hidden reference points toward the visible fragment. Nearer or
  unknown depth outside the visible mask is not evidence of an empty ray.
- Do not force an incomplete reference to cover the whole current outline.
  That assumption initially regressed the bottle: an excluded bottom surface
  pulled the correctly aligned reference down. Removing the assumption fixed
  the regression; the complete-reference option remains explicit.

Cross-check on the full 181-frame labelled bleach-bottle clip: median point
error 7.34 → 4.42 mm, p90 22.81 → 8.33 mm, and frames within 10 mm / 10°
136/180 → 170/180. **Tradeoff:** median rotation error 2.00° → 3.90°; not every
metric improved. On the full 713-frame dowel demo, the previous sideways estimate
at frame 350 becomes upright. That is visual evidence, not full-clip accuracy.

169 checks pass, including the two recorded tests and generic stationary/moving/
rotating shapes at three scales. Deliberately reinstating wrong image weighting,
unknown-depth penalties, complete-outline assumptions, or removing boundary
filtering makes the relevant tests fail. Freezing every output pose passes the
stationary check but fails the pickup check. Changes remain local and uncommitted;
the running server and historical benchmark playbacks were not replaced.

[Before/after video, including pickup](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_gamepad_stationary_regression_20261102.mp4).
[Inputs, settings, measurements and source hashes](gamepad-stationary-regression-20261102.json).
The test and replay command are documented in
[`tests/policies/flash/fixtures/README.md`](../../../tests/policies/flash/fixtures/README.md).

### 2026-11-02 — Full-suite validation after the geometry fixes

Ran all seven methods on **42 clips / 8,611 source frames**: 21 real labelled
scenes, three synthetic controls, and 18 unlabelled scenes (including both FLASH
demos and two deformable negative controls). All completed. RGB, depth, masks,
intrinsics, timestamps, labels and initial surface samples match the saved
baselines. Tracker sources stayed unchanged throughout. SAM masks were reused;
this tests downstream tracking, not segmentation changes.

For depth/mask fitting with shared UFM, groups and growing surface:

| Real labelled frames     |              Before |               After |
| ------------------------ | ------------------: | ------------------: |
| Within 10 mm **and** 10° | 1,229/2,927 (42.0%) | 1,622/2,927 (55.4%) |
| Within 20 mm **and** 20° | 1,720/2,927 (58.8%) | 2,154/2,927 (73.6%) |
| Returned a pose          |         2,702/2,927 |         2,927/2,927 |

**Better overall, but not regression-free.** At 10 mm/10°, 12 scenes improve,
five worsen, four tie. The regressions are DexYCB marker, banana, drill, tomato
can and bowl. At 20 mm/20°, eight improve, eight worsen, five tie. On the same
2,702 frames with poses in both versions, median point error improves
10.42 → 4.78 mm and median angle 9.46 → 4.66°. These are annotation-relative
motion errors; neither absolute initialization nor reconstructed-shape accuracy
is measured. Labels remain imperfect and scoring is not symmetry-minimized.

All seven methods improve in aggregate. Groups versus the current otherwise
identical growing-surface method add 70 available poses but produce one fewer
passing frame at 10 mm/10°; availability alone is not an accuracy improvement.
Historical versus current selected-path timing is about **83 → 195 ms/frame**
(median across the 21 real labelled scenes), excluding SAM/display. Those cases
ran sequentially; later concurrent unlabelled runs are excluded from timing.

The full runner exactly reproduces the earlier post-fix integration outputs:
gamepad frames 0–280 and all 713 dowel frames. Gamepad false pre-pickup motion
remains 1.15 mm/3.98°; motion by frame 280 is 59.19 mm. Both complete demos retain
poses throughout, but have no independent full-motion accuracy labels.

This compares accumulated changes, including separate LK/UFM → shared UFM,
against saved baselines. It does not isolate the last fix. The three prior
shared-UFM comparisons are also preserved separately. No tracker tuning or
product changes were made during this sweep. The code and evidence are preserved
together as a research milestone, with the limitations above.

Visual review distinguishes two failures: the tomato can clearly gets worse;
the selected marker frame shows two inaccurate estimates, so it does not show
a previously successful frame becoming a failure. The marker's aggregate score
still decreases. Neither example identifies the cause yet.

![Tomato can: the same initial surface projected by the label (green) and the estimate (orange), before and after](full-suite-tomato-can-20261102.jpg)

![Marker: both estimates are inaccurate in this selected frame; green is the label and orange the estimate](full-suite-marker-20261102.jpg)

The benchmark uses **shared UFM**, not LK. FLASH also retains explicitly named
"shared LK" comparison recordings; their method labels follow the recording's
actual backend. The main playback catalogue still contains historical results;
the full-suite report is a separate snapshot. A historical playback is not the
new result merely because the server runs the new code.

![Per-scene comparison](full-suite-validation-20261102.png)

[Inputs, settings, hashes, per-scene scores and integration checks](full-suite-validation-20261102.json).
Full traces, baseline summaries, paired-frame checks and the HTML report are in
`~/.cache/flash-tracker-restart/full-validation-20261102/`.
