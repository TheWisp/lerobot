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

## Experiments

These policies exist only in the benchmark runner. No production source or
live-server configuration was changed.

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

## Longer-history check

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

## Verification

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
