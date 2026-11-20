# Geometry-constrained Kalman correction: feasibility and cost

Measured 2026-11-20 on `research/flash-recovery-regression`, production base
`215b0a36e`. This is a **benchmark-only shadow experiment**: proposals are evaluated
but never written back to the tracker, its uncertainty, or its surface map.
The retired prior-in-fit implementation is not used.

The [Kalman pose update](../../../../src/lerobot/policies/flash/docs/glossary.md#kalman-pose-update)
combines a predicted position and rotation with the current surface fit.
The experiment asks whether a contradictory result can be moved to a position
and rotation that pass the **unchanged geometry checks**. Passing those checks
does not establish that the position and rotation are correct.

## Findings

One bounded local projection produced compatible proposals for **28 of 30**
contradictory Kalman results in the full recording. Allowing three linearizations
also produced **28 of 30**, with a worse latency tail. The prototype therefore
does not universally find a compatible constrained solution within its budget.

There was an already-valid surface fit in every one of those 30 cases. Returning
that fit is a feasible fallback, but would discard the filtering protection that
motivated this work. The failure is in the bounded approximation, not evidence
that no compatible pose exists.

A rendered moving-strip counterexample demonstrates a separate limitation:

| Estimate on the first moved frame                | Rotation error | Median material-point error |
| ------------------------------------------------ | -------------: | --------------------------: |
| Surface fit before Kalman                        |        0.0014° |                     6.95 mm |
| Ordinary Kalman result, rejected by geometry     |         10.65° |                    26.54 mm |
| One constrained correction, accepted by geometry |      **9.27°** |                **13.85 mm** |

The object turns 25° between observations separated by 250 ms, and a nearer
occluder leaves a 20-pixel strip visible. The surface fit's **rotation** is nearly
correct; its position already has error. The projection reduces Kalman's error
but still returns a substantially incorrect rotation. This is a counterexample
to using geometry compatibility alone as an accuracy criterion.

An untextured symmetric box turned 180° produces identical rendered depth and
mask images. Its unchanged placement passes geometry despite 180° material
rotation error. No geometry-only correction can resolve that ambiguity from
these two measurements. The renderer's truth is used only for scoring.

## Performance

Ryzen 9 9950X CPU, OpenCV one thread, `OPENBLAS_NUM_THREADS=1`, NumPy fitting
kernels because Numba is unavailable. The existing registration uses small_gicp.
No GPU inference is included. Exact versions, source hashes and raw measurements
are in [summary.json](summary.json), [full-history.json.gz](full-history.json.gz)
and [controls.json.gz](controls.json.gz).

Full history: 400 updates after four initializations; 350 reach Kalman correction.
Of these, 320 already pass geometry and 30 fail. Each method was evaluated five
times on each identical incoming state, alternating method order. That gives
150 repair timings per method. No competing benchmark ran during this timing run.

The following costs are **additional to the ordinary final geometry check**,
which the tracker already performs. They include constructing the approximation,
solving, and checking proposed results. They exclude replay instrumentation,
decoding, SAM, learned depth, fitting, and a future covariance update.

| Method                                                   | Compatible proposals / 30 states | Added median on a failing state | Added p95 | Added maximum |
| -------------------------------------------------------- | -------------------------------: | ------------------------------: | --------: | ------------: |
| One local constrained calculation                        |                               28 |                        10.32 ms |  13.71 ms |      14.24 ms |
| Up to three local calculations                           |                               28 |                        10.83 ms |  27.44 ms |      40.61 ms |
| Six checks along the fit-to-Kalman path, comparison only |                               30 |                        17.39 ms |  21.93 ms |      22.40 ms |

For already-compatible results, the prototype returns the identical pose without
building constraints; routing after the shared check has a measured median of
about 0.0012 ms. Averaged over all 400 baseline updates, the one-calculation
repair cost is about 0.81 ms/update. That average does not describe a repair
frame's latency or predict the event rate in other recordings.

In the one-calculation method, sampled geometry evaluations take 4.52 ms median
and the small quadratic solve 0.35 ms median. Remaining time includes sampling,
coordinate calculations and exact validation. The finite-difference derivative
requires six extra sampled geometry evaluations. Analytic derivatives and reuse
of the already-validated fit could reduce cost, but that optimized implementation
has not been measured. Neither an iteration cap nor the observed maximum is a
hard real-time guarantee.

## Method and its approximations

The reference method is covariance-weighted estimate projection from
[Gupta and Hauser, 2007, sections 3–5](https://arxiv.org/pdf/0709.2791).
The implementation and adaptation are in
[flash_constrained_kalman.py](../../../../benchmarks/flash_constrained_kalman.py).

The local objective favors closeness to the **current unconstrained Kalman
result**, using its 6×6 covariance. It does not run another registration or change
the earlier motion prediction. Rotation increments are camera-axis radians;
position increments are metres of the same fixed point in object coordinates.

For each measured view, at most 256 spatially sampled 3D points supply bounds
on visible surface distance, outline/free-space error and scene-depth error.
The whole map separately supplies current-point overlap distances. Existing
errors at the valid fit may remain but must not grow; ordinary distances receive
the existing depth tolerance. Current-to-retained checks preserve the existing
exception for points agreeing with the saved surface. Nearer occluders and missing
depth retain the normal production interpretation.

These per-sample bounds are an approximation to a validator that uses outlier
fractions, pixel visibility and connected contradictory regions. Sampling can
miss a relevant point, nearest correspondences can change, and an occlusion edge
can change which constraint applies. The approximation has no global implication
that the real geometry checks will pass.

Each local calculation uses one-sided finite differences and a six-variable
quadratic optimization solved by SLSQP, capped at 20 iterations. The search is
limited to ±45° and ±50 mm per component around the current fit. One calculation
uses eight sampled geometry evaluations, including initialization. Three use at
most 22. A proposal is returned only if optimization succeeds, its linear
constraints pass, and the full nonlinear production geometry checks pass.
There is no proof of a globally nearest feasible pose.

The path comparison searches only between the valid fit and the rejected
Kalman result. It always retains the fit as a fallback and uses six bisection
checks. It does not solve the six-variable projection problem; nonconvexity also
means bisection need not find the closest valid interval. Its higher compatible
count does not establish more accurate estimates.

An initial pilot counted one extra compatible pose from a nonconverged solver.
The final experiment requires convergence and excludes it. All headline counts
and timings use that stricter rule.

## Coverage and reproducibility

The full replay uses the original cube, pointer, ring and second cube histories.
Baseline history remains unchanged by every proposal. The 28 alternative poses
are **not** a measured increase in recovery for a new chronological tracker;
accepting them would change subsequent maps and filter states.

[flash_constrained_controls.py](../../../../benchmarks/flash_constrained_controls.py)
runs 13 rendered conditions: missing observations with and without motion,
visible translation, stationary partial visibility, camera/group motion,
corrupted depth, sudden 25° and 60° rotations, a 50 mm translation, 8- and
20-pixel moving strips, noisy depth, and a symmetric half-turn. It also evaluates
six existing real checkpoints from other objects. All six stop before Kalman
because fitting finds no compatible candidate; they provide no evidence of
projection accuracy. Missing measurements likewise cannot invoke this correction.

The 8-pixel moving strip fails before Kalman. The 20-pixel strip reaches the
correction and exposes the wrong-rotation result above. Other controls also
include geometry-compatible wrong fits that remain unchanged. These limitations
are retained rather than excluded from the scores.

A new **399 KiB** geometry-only checkpoint saves the pointer frame 3997 failure:
[fixture description](../../../../tests/policies/flash/fixtures/constrained_correction_20261120/README.md).
It contains no video. The linear optimization succeeds on this input while the
actual depth check rejects its result. Removing the final geometry verdict makes
the named regression fail. Synthetic symmetry and narrow-strip counterexamples
are generated at test time.

Verification: **39 tests passed**, four optional UFM tests skipped, and the
existing production ring failure remains a strict expected failure. A separate
in-memory mutation that ignores the final geometry verdict fails the pointer
checkpoint test. The source implementation is unchanged after that mutation.

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m benchmarks.flash_constrained_kalman \
  --output /tmp/constrained-compact.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m benchmarks.flash_constrained_kalman \
  --full-recording /path/to/saved-747dd29b27a745418524e2744ac29cb5 \
  --repeats 5 --output /tmp/constrained-full.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m benchmarks.flash_constrained_controls \
  --output /tmp/constrained-controls.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/policies/flash/test_constrained_kalman_experiment.py \
  tests/policies/flash/test_recorded_motion_recovery.py \
  tests/policies/flash/test_surface_kalman.py
```

## What the measurements establish

The method can repair many contradictions at a bounded amount of solver work,
with measured CPU cost in the table. The unchanged final validator remains
necessary. More local iterations did not improve the final compatible count.

They do not establish accurate pose recovery under partial visibility, a universal
feasible-solution guarantee, or a calibrated uncertainty update. Covariance is
used only as a local weight and is not modified. Publishing these proposals in
a recurrent estimator still requires an uncertainty policy that represents weak
or ambiguous observations, plus chronological accuracy testing. Production
tracking and defaults remain unchanged.
