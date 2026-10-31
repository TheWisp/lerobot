# FLASH restart: postmortem and research

2026-10-31. Code audit and literature review, not a replacement implementation.
The two preserved experiments are `proto/show-and-servo` at `80d90b783` and
`docs/flash-redesign` at `c72e1e1c3` (tag `flash-redesign-sealed-20261031`).
The latter's [handover](../../../src/lerobot/policies/flash/docs/SEALED_20261031.md)
records what was saved and verified. This research continues separately on
`research/flash-tracker-restart`.

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

## Core to investigate

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
2026-10-31. The runner can freshly process the external cases, but the rig rows
only score saved preparations. A common fresh rerun of both is the first missing
boundary; scoring old files after an algorithm edit does not test that edit.

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
policy workflow. No tracker, editor, motor behavior or active demo was changed
in this research pass.
