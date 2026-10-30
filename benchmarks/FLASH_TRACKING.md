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

The external runs **feed supplied masks into perception**, so their 1.00 mask
agreement is expected and says nothing about SAM3. The rig demo traces have no
independent mask agreement score. YCB annotations are evaluation
labels, not input to tracking or independent sensor ground truth. Caryx has no
object-pose labels; pose accuracy there is unknown. The two domains have different
cameras and conditions, so the report does not merge them into one success rate.
The rotation metric does not factor out shape symmetry. The earlier Caryx camera
convention audit remains a caveat for pose interpretation on that domain.
Offline wall time from new reruns includes model initialization and is not live
robot FPS. Frame-time and memory profiling still need separate instrumentation.

This suite does **not** yet test the complete path from selecting/segmenting an
object through tracking and task execution. A full-path evaluation needs fresh
SAM3 output scored against separately annotated masks and object identity, then
pose coverage and error on the same frames, plus independently checked contact
or task outcomes for rig runs. The saved demonstration masks alone cannot serve
as both predictions and truth.

## What transfers to FLASH

The benchmark, demo preparation, and live run call the same surface tracker and
scene/group pose estimator. A change to those shared algorithms is available to
all three after their processes restart. Their **inputs and settings differ**:

| Path                               | Masks and object reference                                        | Scene/group fallback | Gripper attachment                                    |
| ---------------------------------- | ----------------------------------------------------------------- | -------------------- | ----------------------------------------------------- |
| YCB / Caryx benchmark rerun        | Supplied masks; first observed view is the local reference        | Yes                  | No                                                    |
| Saved rig-demo rows in this report | Saved SAM3 masks and prepared reference from the older extraction | Yes, as saved        | Demo attachment interval, without live motor feedback |
| Fresh demo preparation             | SAM3; prepared appearance and surface reference                   | Yes                  | Demo attachment interval, without live motor feedback |
| Live tracking debugger             | SAM3; local reference made from the clicked/named frame           | Optional             | No                                                    |
| Live FLASH run                     | SAM3; prepared appearance and surface reference                   | Yes                  | Yes, updated from motor and vision evidence           |

The two saved rig-demo traces were prepared with pose-trace selection version 6;
the current extractor is version 8. **Scoring a saved trace does not rerun the
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
