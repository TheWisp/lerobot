# Kalman innovation-gate comparison — 2026-11-11

Parent: `554a7e633177118605d4e11c1bd42041c717458e`. The sole tracking-policy
change between comparison arms is the historical innovation veto versus ordinary
Kalman correction.
`reset_on_geometry_conflict=false` in both; all noise, fitting, geometry, image,
group and growth settings are otherwise identical. After the comparison and
user visual review, the veto and its experimental switch were removed from
production. The benchmark retains the historical veto solely as a comparison oracle.

## Recorded checkpoint

`tests/policies/flash/fixtures/green_toy_kalman_recovery_20261111.npz` preserves
full-frame RGB, selected Fast-FoundationStereo depth, masks, map, filter state
and reconstructed LK pairs for frame 928 of
`saved-14a68e75c4bc47388628c786427cea78`. The accepted source is frame 904.
The original seed was not logged: surviving IDs and pixels rounded to 0.001 px
reconstruct the same 31 retained pairs. The test's gated control reproduces the
recorded fitted pose to matrix tolerance 1e-9 and innovation score to 1e-7.
It does not supply a candidate pose to the fitter.

With the gate, score 45.30853568 exceeds 12.59158724 and publication is lost.
Without it, the actual fit → Kalman → final geometry path publishes immediately:
100% mutually visible overlap, 1.918 mm median surface residual, no coherent
empty-space protrusion. An overconfident wrong-prior negative control still
rejects the resulting invalid blend and restores the filter and accepted map.

## Paired continuous replays

Both arms start at the first retained observation, not the pruned original live
initialization. Recorded RGB, selected depth and masks are reused; shared LK,
scene grouping, object association, small_gicp, Kalman and growth run again.
The cube's saved matcher setting is overridden to small_gicp in both arms.

| Clip                                               | Frames | Observed including initialization, gate on → off | Predicted |     Lost |
| -------------------------------------------------- | -----: | -----------------------------------------------: | --------: | -------: |
| Green toy `saved-14a68e75c4bc47388628c786427cea78` |    194 |                                         52 → 172 |     0 → 0 | 142 → 22 |
| Cube `saved-3b9b9f3d70a74fbda3d65772f3f32b9c`      |    223 |                                        160 → 192 |    8 → 14 |  55 → 17 |

Every measured publication was independently re-evaluated using that update's
pre-growth geometry: 51/171 for the toy and 159/191 for the cube, all passing
(initialization excluded). Median per-frame surface residuals were
1.705→1.484 mm and 2.522→2.428 mm respectively. No ground-truth object pose is
available in these live clips: availability and fit metrics are not accuracy.

The toy gains 120 available frames with none lost. In its 928–1157 interval,
availability changes from 0/37 to 32/37 under the paired fresh initialization.
The cube gains 45 available frames but loses seven that the gated arm publishes:
2521, 3257, 3264, 3269, 3276, 3283, 3289. Four are fitter/geometry failures;
three are rejected fused poses. The final six occur in a clear view near the
clip's end. These are availability differences, not established correctness
regressions.

User visual review on 2026-11-11 identified the shown gate-on cube poses as
wrong. Withholding those poses is preferable to publishing them and therefore
counts in favor of the ungated arm. The comparison crops show frames 2521,
3257 and 3289. This records the visual assessment separately from the numerical
fit metrics; neither availability nor a passing geometry check establishes
correct object pose.

The cube has 7→18 post-fusion geometry rejections, and the toy has 0→3. Removing
the temporal veto therefore does not disable the final validity check. Candidate
and map histories diverge after different acceptances, affecting later solves.

## Controls and tests

Six rendered controls have identical output poses and availability with/without
the gate: visible motion, hidden motion/reappearance, hidden stationary object,
a 12-pixel visible strip, group plus camera motion, and corrupted depth followed
by clear frames. Unobserved motion remains unknowable during full occlusion;
the hidden-motion accuracy assertion applies after reappearance. Corrupted
depth remains rejected and does not update the map. The recorded wrong mug
checkpoint remains withheld by geometry in both arms.

64 focused tests passed across `test_surface_kalman`, `test_surface_uncertainty`,
`test_green_toy_kalman_recovery`, `test_ungated_kalman_controls`,
`test_controller_motion_recovery`, `test_controller_visibility_recovery` and
`test_recorded_cube_publication_geometry`. Restoring the gate makes the new
frame-928 recovery test fail with `lost`. Ruff and diff whitespace checks pass.

A broader check also found the existing
`test_partial_surface_fitting::test_image_assisted_candidate_selection_keeps_both_measured_views[visible_icp]`
failure (3.038° versus a 1° limit). It reproduces with the parent commit's
Kalman and uncertainty modules loaded in an isolated interpreter. No runtime
files were reverted for this check. The new experiment uses small_gicp.
An additional historical `test_combined_recovery` assertion expects Kalman
diagnostics on `gamepad_271`, but both HEAD and current code reject it earlier
with `no_observation_compatible_candidate`; neither publishes or changes the map.

No physical camera or new SAM/depth inference was exercised. The production
path now uses the tested ungated correction. After removal of the switch, 68
focused tests passed, including the earlier controller recovery fixtures, the
missing-motion tests and the shared live/video entry point. Tests that previously
required innovation rejection now assert weighted correction and final geometry.
Restoring the historical veto makes the frame-928 recovery assertion fail.
The final pre-commit run passed 71 tests, including three worker-loop checks for
reselecting a processed image and keeping tracking active after an expired click.
These worker checks fake hardware/model producers; they do not constitute a
physical camera validation.

## Reproduce

Interpreter: `~/.cache/flash-tracker-restart/coarse-open3d-env/bin/python`.
Environment: `PYTHONPATH=src:.:tests/policies/flash OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2`.

```bash
python -m benchmarks.flash_kalman_gate_ablation \
  ~/.cache/flash-tracker-restart/live-history/saved-14a68e75c4bc47388628c786427cea78 \
  --out /tmp/toy-gate-comparison
python -m benchmarks.flash_kalman_gate_ablation \
  ~/.cache/flash-tracker-restart/live-history/saved-3b9b9f3d70a74fbda3d65772f3f32b9c \
  --set 'config.shape.registration.backend="small_gicp"' --out /tmp/cube-gate-comparison
```

Full per-frame JSON, pre-growth clouds, contact sheets and H.264 comparison
videos are retained locally under
`~/.cache/flash-tracker-restart/green-toy-followup-20261111/` in `gate-ablation/`
and `cube-gate-ablation/`. Videos use source capture timestamps and hold each
processed image until the next; they do not simulate live availability latency.
