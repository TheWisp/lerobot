# Pointer frame 3997: local constraints versus the actual depth check

Captured 2026-11-20 from chronological replay of
`saved-747dd29b27a745418524e2744ac29cb5` on production base `215b0a36e`.
The 399 KiB NPZ and JSON contain the two measured depth/mask views, prepared
3D surfaces, camera matrices, geometry settings, the valid surface fit, and
the unconstrained Kalman pose/covariance at this boundary. They contain no
video, RGB, ground-truth pose or future observation.

This is a correction-boundary regression, not a complete tracking replay.
The existing eleven-frame recovery fixture separately covers the live pipeline.
The loader reconstructs real `WholeSurfaceFit`/`LatestObservationFit` validators;
the exporter checks their fit and Kalman verdicts against the running replay.

The one-step linear constrained optimization succeeds, but its proposed position
and rotation fail the actual nonlinear depth check. The test must return no
proposal. Removing that final verdict causes the test to fail.

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/policies/flash/test_constrained_kalman_experiment.py::test_linear_constraints_can_pass_while_actual_depth_rejects
```

Regenerate with `benchmarks.flash_constrained_kalman --full-recording <recording>
--capture-frame 3997 --capture-path tests/policies/flash/fixtures/constrained_correction_20261120/pointer3997
--output /tmp/constrained.json`. The exporter lives in
[flash_constraint_checkpoint.py](../../../../../benchmarks/flash_constraint_checkpoint.py).
