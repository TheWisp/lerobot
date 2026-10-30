# FLASH RGB-D tracking benchmark

This is the current, versioned entry point for measuring FLASH tracking changes.
The [suite](flash_tracking_suite.json) freezes **nine YCBInEOAT clips** and
**ten Caryx clips** selected in the earlier failure audit. Raw datasets and
per-frame evidence remain in the local cache; the [baseline report](flash_tracking_baseline.md)
and [machine-readable baseline](flash_tracking_baseline.json) are saved here.
All existing cases are development cases: they were examined during research.
Add a genuinely unseen holdout before claiming generalization.

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

The current runs **feed supplied masks into perception**, so their 1.00 mask
agreement is expected and says nothing about SAM3. YCB annotations are evaluation
labels, not input to tracking or independent sensor ground truth. Caryx has no
object-pose labels; pose accuracy there is unknown. The two domains have different
cameras and conditions, so the report does not merge them into one success rate.
The rotation metric does not factor out shape symmetry. The earlier Caryx camera
convention audit remains a caveat for pose interpretation on that domain.
Offline wall time from new reruns includes model initialization and is not live
robot FPS. Frame-time and memory profiling still need separate instrumentation.

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
