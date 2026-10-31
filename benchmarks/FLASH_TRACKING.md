# FLASH RGB-D tracking benchmark

This is the current, versioned entry point for measuring FLASH tracking changes.
The [suite](flash_tracking_suite.json) freezes **nine YCBInEOAT clips**,
**ten Caryx clips**, and **our two rig demonstrations** (dowel insertion and
gamepad stacking). The rig cases are a diagnostic tier without independent
visibility, mask, or pose labels. Raw datasets and
per-frame evidence remain in the local cache; the [baseline report](flash_tracking_baseline.md)
and [machine-readable baseline](flash_tracking_baseline.json) are saved here.
All existing cases are development cases: they were examined during research.
Add a genuinely unseen holdout before claiming generalization.
The dated [failure ledger](../docs/proofs/flash-contact-observation/FAILURE_LEDGER.md)
explains why some cases fail; it is evidence, not the current pipeline specification.

Each case uses the normal `SceneEvidenceWriter` output. The scorer checks that
all frames are present, then reports, per visible object and per clip:

- **Observed / predicted / lost** pose counts. Prediction means a fallback pose
  exists; it does not mean that pose is correct.
- Agreement of the tracked mask with the supplied reference mask, longest loss
  streak, accepted depth residual, and recorded rejection reasons.
- For YCB only, pose error against supplied annotations: displacement of the
  same 256-or-fewer first-frame interior measured points in millimetres and full
  rotation in degrees. The point sample and random seed are fixed by the scorer.
  It is first-frame-relative; initialization is excluded. Error is reported only
  on frames with a published pose, beside the fraction of frames covered.
- For the rig demonstrations, counts of observed, predicted, and lost poses,
  and frames with a saved object mask. These count **all** 713 insertion or 628
  stacking frames because object visibility was not independently annotated.
  A saved mask or accepted depth residual is not proof that the mask or pose is
  correct. These cases can detect coverage regressions, but cannot establish
  pose accuracy or physical task success.

The frozen external baseline **feeds supplied masks into perception**, so its 1.00
mask agreement is expected and says nothing about SAM3. Candidate reruns can use
SAM3 prompts instead; the scorer still uses supplied masks only as evaluation
labels. The rig demo traces have no
independent mask agreement score. YCB annotations are evaluation
labels, not input to tracking or independent sensor ground truth. Caryx has no
object-pose labels; pose accuracy there is unknown. The two domains have different
cameras and conditions, so the report does not merge them into one success rate.
The rotation metric does not factor out shape symmetry. The earlier Caryx camera
convention audit remains a caveat for pose interpretation on that domain.
Offline wall time from new reruns includes model initialization and is not live
robot FPS. Frame-time and memory profiling still need separate instrumentation.

SAM3 candidate reruns test the path from text prompt through mask and pose on
labeled clips, but the suite does **not** test task execution. That requires
independently checked contact or task outcomes for rig runs. The saved
demonstration masks cannot serve as both predictions and truth.

## What transfers to FLASH

The benchmark, demo preparation, and live run call the same surface tracker and
scene/group pose estimator. A change to those shared algorithms is available to
all three after their processes restart. Their **inputs and settings differ**:

| Path                               | Masks and object reference                                                 | Scene/group fallback | Gripper attachment                                    |
| ---------------------------------- | -------------------------------------------------------------------------- | -------------------- | ----------------------------------------------------- |
| YCB / Caryx benchmark rerun        | Supplied masks or SAM3 prompts; first observed view is the local reference | Yes                  | No                                                    |
| Saved rig-demo rows in this report | Saved SAM3 masks and prepared reference from the older extraction          | Yes, as saved        | Demo attachment interval, without live motor feedback |
| Fresh demo preparation             | SAM3; prepared appearance and surface reference                            | Yes                  | Demo attachment interval, without live motor feedback |
| Live tracking debugger             | SAM3; local reference made from the clicked/named frame                    | Optional             | No                                                    |
| Live FLASH run                     | SAM3; prepared appearance and surface reference                            | Yes                  | Yes, updated from motor and vision evidence           |

The two frozen rig-demo benchmark traces were prepared with pose-trace selection version 6;
the current extractor is version 9. **Scoring a saved trace does not rerun the
tracker.** The frozen YCB configuration also differs from the rig demo in SAM3
resolution (504 versus 448), group image scale (1.0 versus 0.5), and static
prediction age (unlimited versus 4.5 s). A benchmark gain is therefore a reason
to test a candidate on the rig, not evidence that the rig behavior already
improved.

To carry a change through: rerun the affected external benchmark cases with a
candidate configuration; prepare a **new version** of the rig demo using that
change and review its pose/mask replay; test the raw surface behavior in the live
debugger with matching vision settings; then test a live FLASH attempt using the
new prepared version. The live launcher rejects vision or scene settings that
disagree with its prepared version. The debugger can exercise the same group
estimator with **Use scene/group pose fallback** enabled before opening the
camera: it forwards every camera frame and slower vision proposals. It has no
motor-backed gripper attachment or prepared multi-view reference. The group
pose is shown in a separate view on its own source RGB-D frame; the direct fit
keeps its own source frame. The diagnostics report both frame IDs and provenance.
A fresh rig-demo
benchmark entry must point to the newly prepared evidence; changing code while
scoring the old directory cannot change its results.

The scorer reproduces the earlier YCB state counts and conditional point errors.
As an end-to-end check, the current runner processed all 181 frames of the moving
cracker-box case and produced the same 28 observed, 87 predicted, 66 lost states
and 6.5 mm conditional median point error. Its offline wall time was 22.9 s
including model initialization; this is not a live frame-rate measurement.

An end-to-end SAM3 rerun of that same 181-frame cracker-box clip used the prompt
`cracker box`, with no supplied mask passed into inference. Its median mask overlap
with the reference masks was 0.817. It published a pose on 113 frames (115 for the
supplied-mask baseline), directly observed 93 frames (28 baseline), and was lost
for 66 consecutive visible frames late in the clip. On the 112 frames where both
runs published a pose, median measured-point error against the pose annotations
was 3.74 mm with SAM3 and 6.45 mm with supplied masks. This is one development
clip, not evidence of a general pose-accuracy improvement; the late loss remains
an unsolved failure. Full local evidence is in
`/tmp/flash-sam3-cracker-20261031-b/` and the score is
`/tmp/flash-sam3-cracker-score-v2.json`.

Two more 181-frame checks show why mask initialization must be measured too.
`mustard bottle` gave 0.795 median mask overlap and 99/181 frames with a pose,
versus 64/181 with supplied masks. `sugar box` produced **no mask at all** from
text alone. One click inside the visible box at the initial frame gave 0.905
median mask overlap and a pose on all 181 frames; its median pose error was
1.11 mm versus 0.90 mm with supplied masks. These are development clips and the
click result is not a claim that SAM3 is universally accurate. Local scores:
`/tmp/flash-sam3-extra-score-20261031-b.json` and
`/tmp/flash-sam3-sugar-click-score-20261031.json`.

The cracker-box loss is not just a missing mask. At frame 150, its SAM3 mask
overlaps the label by 0.864, yet the first saved surface is 28.9 mm from the
current clean depth points at the annotated pose. Adding views from four earlier
**accepted** frames (0, 30, 60, 90) changes that to only 28.2 mm: they largely
show the same side. Registering frame 150 with its evaluation pose label reduces
the same-frame distance to 3.9 mm, but that is an **oracle experiment**, not a
tracking method; the label never enters inference. It also does not solve frame
180, where the current SAM3 surface remains 36.4 mm from the label-registered
views. The next estimator change needs a way to register a genuinely new side
after occlusion and reject incomplete or contradictory surface support. Merely
loosening the existing 5 mm acceptance limit would publish an unverified pose.

A review-only check on the insertion demo illustrates why new reference views
need more than a one-way fit. The first score asks how closely measured points
fit the saved surface. The reverse check projects the saved surface into the
recorded depth image. A surface in front of measured depth is more concerning;
a surface behind it may simply be hidden. With 1,427 clean points from the
prepared mask at dowel frame 100 as
the first view, adding frame 150 reduced the median first gap across 272
observed frames from 1.30 to 1.07 mm, and the second from 2.32 to 1.86 mm.
Adding frame 500 or 550 also reduced the first gap (to 1.28 or 1.14 mm), but
increased the absolute reverse gap to 5.79 or 9.14 mm. For frame 500, splitting
the gap by direction shows a much smaller change in surface closer than measured
depth (median 0.87 to 1.15 mm) than in surface farther away (2.13 to 4.44 mm).
That increase alone does **not** show a contradiction: at frame 224, frame
500's added surface projects mostly behind the observed dowel. This direction
is not automatically safe either: in the recorded gamepad frame 430, a known
bad pose has a 124 mm absolute reverse gap almost entirely behind the measured
surface, despite a 2.35 mm measured-point-to-reference gap. The live acceptance
gate therefore remains conservative until visibility or independent motion
evidence can distinguish a hidden patch from a misplaced one. These are **internal consistency
scores**, not accuracy: the same prepared pose trace supplied both
registration and evaluation. The reusable `reference_view_candidates` review
function reports both checks without changing a prepared reference. The
**Run → FLASH → Edit demo → Objects → Clear views** control now runs this
check on the selected frame and separates surface in front of measured depth
from surface behind it. A later view needs independent support and explicit review
before demo preparation or a live trial can use it.
The same control also ran on the gamepad–cube demo: gamepad frame 136 reduced
the measured-point gap from 0.92 to 0.78 mm across 211 other observed frames,
while the projected-surface gap stayed about 1.04 mm. This shows the check is
not tied to the dowel, but it still lacks independent pose labels.

An independent registration probe used only the dowel's frame-100 measured
surface, each candidate frame's mask and depth, and the existing whole-surface
fit. Four geometry-based initializations were refined and ranked by fit cost;
the prepared pose trace was used **only afterward** for comparison. At frame
150 the fitted and recorded surfaces had centers 2.6 mm apart, but the
independent fit was _worse_ against measured depth (4.13 versus 2.77 mm p90).
At frame 500 their centers were 1.4 mm apart and the independent fit improved
the visible-point distance (3.89 versus 7.93 mm p90), but it differed by
165.5° in orientation (possibly an interchangeable-end assignment), and the
winning solver branch did not converge. The
large transform-translation differences from this probe are not physical
object displacements: this saved object's frame origin is near the original
camera, so rotation about the object moves the arbitrary origin. Geometry
alone has not yet established the stable point identity needed to add that
later view. These images show the measured mask boundary in cyan and the
saved frame-100 surface projected in red under the two placements:
[frame 150](../docs/proofs/flash-reference-view-registration/dowel-frame-150.jpg),
[frame 500](../docs/proofs/flash-reference-view-registration/dowel-frame-500.jpg).
The generic `reference_view_registration` candidate generator now makes the
four geometry fits available beside the existing trace-based check in the demo
editor. It also ran on the gamepad view at frame 136; its lowest-cost proposal
had 0.94 mm measured-surface and 1.08 mm projected-surface-to-depth p90 gaps, but its
optimizer did not converge. The editor previews any candidate against the
same recorded camera frame without changing the saved reference.

The editor also compares a gripper-motion prediction during the declared carry
interval. It starts with the selected pre-grasp object pose and moves it by the
recorded gripper motion; later visual corrections are excluded so the selected
frame cannot confirm itself. On dowel frame 500, this raw prediction had 17.05 mm
measured-surface and 36.38 mm projected-depth p90 gaps. Refining it against that
frame improved those gaps to 6.45 and 25.24 mm. The best geometry-only proposal
had 3.97 and 11.67 mm gaps, although its solver did not converge. Gripper motion
is useful as a labeled alternative, but these numbers do not justify accepting
it or the geometry fit as the object pose without further evidence.

From the repository root, in an environment with FLASH's inference dependencies:

```bash
PYTHONPATH=src:. python benchmarks/flash_tracking_benchmark.py \
  --require-all --output /tmp/flash-tracking-baseline.json
```

To compare a candidate, rerun one or more frozen cases into a **new** directory.
The runner uses the shared FLASH perception path and writes a resolved suite that
points at its evidence; it never overwrites the baseline.

```bash
PYTHONPATH=src:. python benchmarks/flash_tracking_run.py \
  --case ycb-cracker_box_reorient --case caryx-032 \
  --run-root ~/.cache/flash-tracking-candidate-001
PYTHONPATH=src:. python benchmarks/flash_tracking_benchmark.py \
  --suite ~/.cache/flash-tracking-candidate-001/resolved_suite.json \
  --baseline benchmarks/flash_tracking_baseline.json \
  --require-all --output /tmp/flash-tracking-candidate-001.json
```

`--config-bundle PATH` on the runner permits a controlled parameter comparison.
For an end-to-end segmentation rerun, copy the suite to a candidate JSON and add
`"segmentation": {"source": "sam3", "prompts_by_role": {"object": "cracker box"}}`
to the selected case. Use a descriptive prompt for each role in a multi-object
case. If text does not initialize the intended object, add
`"seed_points_by_role": {"object": [x, y]}` with a point in the first RGB frame.
The default `"source": "supplied"` retains the frozen baseline behavior.
The comparison checks source hashes and matching frame visibility. It reports
pose availability separately from **paired** pose error on frames where both
runs published a pose, so dropping hard frames cannot appear as an accuracy gain.
The candidate's own conditional error and mask agreement remain in its report.

To add a case, first preserve its RGB-D source, timestamps, calibration, object
selection and reference masks in a regular LeRobot dataset (or a labeled YCB
sequence). Add a unique case to the suite with its source type, artifact path,
frame count, domain and diagnostic tags. Run the existing observation extractor
or this runner to make a complete scene-evidence journal. Independent pose labels
are optional and belong only in scoring. Keep new cases separate by domain and
mark genuinely unseen cases before using them for tuning. The case schema and
source hash make changes to the benchmark explicit.
