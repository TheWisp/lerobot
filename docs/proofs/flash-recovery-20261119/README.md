# Recovery regression and first experiments

Use the [tracking glossary](../../../src/lerobot/policies/flash/docs/glossary.html)
for exact definitions and code citations. Earlier “image identity” means
[point track ID](../../../src/lerobot/policies/flash/docs/glossary.html#point-track-id);
“local 3D alignment” means
[local surface fitting](../../../src/lerobot/policies/flash/docs/glossary.html#surface-fitting);
“object anchor” means
[object point anchor](../../../src/lerobot/policies/flash/docs/glossary.html#object-anchor).
“Smoothing” referred to the
[Kalman pose update](../../../src/lerobot/policies/flash/docs/glossary.html#kalman-update).

The compact regression replays eleven real RGB-D observations through
`LiveSurfaceSession`: real image matching, scene groups, surface preparation,
registration, Kalman updates, accepted-image changes and map growth. SAM and
learned depth are frozen at their recorded outputs. The test archive is 6.4 MiB;
it requires neither the original video nor the camera.

Open [the visual explanation](index.html) for five annotated camera screenshots.
All overlays come from actual replay coordinates. Alternative poses are labeled
as diagnostic alternatives. The ring's rejection involves depth in a retained
view and cannot be established from RGB appearance alone.

The screenshots use the original full history. The short regression starts
fresh and reproduces the corresponding surface-fitting and Kalman-update failures;
it does not restore or fake the original accumulated map/group state.

## Implemented translation initialization

The production `WholeSurfaceFit.fit_once` now compares the motion proposal with
a translation aligning the means of the stored and current 3D object surfaces.
It preserves the proposed rotation and runs one bounded solve from the lower
geometry-cost start. The comparison matters: a newly visible side can move a
surface's measured centre even when the object centre has not moved. The
geometry checks, Kalman settings, and map-update rules are unchanged.

The sparse cube and pointer regressions now pass with ordinary recorded settings,
including reacquisition after all image tracks are lost. The wrong-object control
also passes on the production path. Restoring the original `fit_once` in a
subprocess makes all four recovery tests fail again, at frames 2741, 3000, 2747,
and 3008. No video or expected pose is an estimator input.

The separate ring Kalman-fusion regression remains a strict expected failure.
The live server was not changed or restarted. The screenshots and original
experiment tables below describe the pre-fix implementation at `9ff25cf80`.

The new full-history replay publishes 315 observed updates, 84 lost updates and
one prediction, compared with 216/182/2 before the fix. These are availability
counts, **not pose-accuracy scores**. The compact regression additionally checks
visible location and depth; full-history exact 6D accuracy remains unlabelled.
See [implementation verification](translation-initialization-results.json).

Verification: the recorded suite, including actual cached UFM inference, passes
20 tests with the one ring expected failure. A broader selection of fitting,
publication, live tracking, shared matching and Kalman tests passes 264 tests,
with 4 optional skips, 3 expected failures and 11 ordinary failures. All 11 also
reproduce with the original `fit_once`: one image-assisted partial-view fit,
six wrong-image/rotation cases, and four gamepad placements. They remain visible
failures rather than being reclassified or relaxed for this fix.

## Original experiments

### What each matching step receives

The shared LK/UFM pass receives the previous and current full RGB images resized
to 640 × 362 in this recording. Its caller uses masks to allocate queries and
filter object matches, but it does not create separate object crops for the two
images. Depth is used after image matching to form 3D point pairs.

The small_gicp surface fit receives stored object points and current points lifted
from usable depth inside the SAM mask. Projecting its saved target points back
into the recorded mask verifies 915/915 inside for frame 2741 and 207/207 inside
for frame 5381; neither target cloud includes points outside that mask. This checks
mask inclusion, not whether SAM labelled every pixel correctly. Full scene depth
remains available separately for rejecting incompatible placements.

The relevant code paths are `SharedFrameTracker.update`/`pairs`,
`SurfaceTrackingSession.prepare`, `surface_observation.observation`, and
`LibraryMatcher.prepare`. Selecting only object pixels is distinct from aligning
their centres: their 3D coordinates still retain the object's camera-space
position. The original missing translation initialization below occurred despite the object
surface already being selected.

### Missing initialization from the current surface centre

**Corrected above; historical diagnosis.** The path at `9ff25cf80` chose one starting pose from image-tracked motion, a group
prediction, or the previous estimate. It did not first align the stored surface
centre with the current measured surface centre. The centring in
`LibraryMatcher.prepare` changes coordinates for numerical computation;
`LibraryMatcher.refine` converts the initial transform into those coordinates,
preserving its original displacement. This is not the centre-alignment step.

A separate controlled replay of cube frame 2741 kept the original starting
rotation, previous pose, images, masks, depth and fit checks unchanged. Only the
initial translation changed:

| Initial translation   | Starting centre gap | Starting rotation change | Surface fit |
| --------------------- | ------------------: | -----------------------: | ----------- |
| Original              |             43.4 mm |                       0° | Rejected    |
| Align surface means   |              0.0 mm |                       0° | Accepted    |
| Align surface medians |              1.1 mm |                       0° | Accepted    |

This identifies missing initialization from current geometry as a fixable
implementation defect in this case. The old object's displacement is not an
inherent barrier to recovery. Partial visibility can bias the visible surface
centre, so a centred starting estimate still needs the normal fitting and checks.
The controlled trials measure surface-fit acceptance before downstream filtering;
they do not supply independent pose ground truth or establish all-case recovery.

[Recorded trial inputs, outputs and exact translation formulas](center-initialization-ablation.json)
retain this evidence separately from the earlier comparison of two fitted poses.

### Recovery policies

These policy interventions exist only in the benchmark runner. This original
experiment did not change production source or live-server configuration. The
subsequent production translation fix is documented above; running `baseline`
now uses that updated fitter with the recorded settings.

- `baseline`: recorded settings.
- `recover`: if the ordinary local surface fit fails, try the predicted pose and then a
  translation aligning surface medians. All existing geometry checks remain.
  This is a diagnostic recovery proposal, not evidence of object identity.
- `baseline_reset`: enable the existing Kalman reset-on-geometry-conflict option.
- `recover_reset`: combine those two interventions.
- `rematch`: when fewer than the configured minimum 3D pairs remain, perform
  fresh matching directly from the accepted image to the current image. Sample
  depth-supported source pixels without requiring surviving point track IDs;
  retain mask, endpoint-depth, robust-fit and geometry checks. This experiment
  does not resurrect lost scene-group IDs.
- `rematch_reset`: combine fresh matching and the existing filter-reset option.

The ordinary UFM path recovers the sparse cube and pointer sequences, but both
still fail after all old point tracks are lost. Fresh reference-to-current UFM matching
recovers them with 145 and 108 validated 3D pairs respectively. The ring still
needs the separate filter change. Actual UFM inference uses cached checkpoint
`980825b59331413bb51e5773f9272bfce525bce2`; it is not a fake flow field.

The LK surface-recovery experiment also recovers the cube and pointer without
temporal pairs; the combined filter intervention recovers the ring. Subsequent
frames run from the changed accepted image/map, rather than branching each
frame from the failing baseline state.

The complete-loss tests keep the map/reference unchanged during missing
measurements. A wrong-object control supplies the green ring after initializing
the cube and checks rejection. Visible-location assertions use reviewed image
regions. These checks do not provide exact 6D ground truth or a general
lookalike-object guarantee.

## Original longer-history check

All 404 retained results from the four selections were replayed from their
actual initialization frames. The baseline reproduces all 404 recorded statuses.
There are 400 subsequent updates:

| Policy                       | Observed updates | Lost | Predicted |
| ---------------------------- | ---------------: | ---: | --------: |
| Existing LK                  |              216 |  182 |         2 |
| LK surface-recovery + reset  |              400 |    0 |         0 |
| UFM fresh rematching + reset |              399 |    0 |         1 |

These are availability counts, **not accuracy scores**. The remaining UFM
prediction is pointer frame 3491: the fused candidate is rejected and the
compatible prior is retained. This shows that reset-on-prior-contradiction does
not resolve every fusion failure. Full-history exact orientation and map
correctness remain unlabelled. Do not promote either experiment solely because
more poses are published.

## Original experiment verification

The baseline defects are strict expected failures, not hidden passing tests.
The opt-in experiments must satisfy the same recovery assertions. Reintroducing
the defects produced the intended failures:

- Removing alternative starting poses fails cube 2741 and pointer 3000.
- Disabling the filter reset fails ring 4759 with the original fusion rejection.
- Removing fresh matching fails UFM reacquisition after complete loss.

See [the fixture documentation](../../../tests/policies/flash/fixtures/motion_recovery_20261117/README.md)
for scope, provenance and test commands. `experiment-results.json` records the
experiment counts; `experiment-trace.jsonl` contains individual outcomes and
compact full-history pose traces.

Reproduce a compact experiment:

```sh
PYTHONPATH=src:. python -m benchmarks.flash_motion_recovery --output /tmp/recovery-lk.json
HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m benchmarks.flash_motion_recovery \
  --backend ufm --policies baseline rematch rematch_reset --output /tmp/recovery-ufm.json
```

For the longer check, add `--full-recording /path/to/saved-recording`. To rebuild
the explanatory images from the original full history:

```sh
PYTHONPATH=src:. python -m benchmarks.flash_recovery_screenshots \
  /path/to/saved-recording /tmp/recovery-figure-data
PYTHONPATH=src:. python -m benchmarks.flash_recovery_figures \
  /path/to/saved-recording /tmp/recovery-figure-data docs/proofs/flash-recovery-20261119
```
