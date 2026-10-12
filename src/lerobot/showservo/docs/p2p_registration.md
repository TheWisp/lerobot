# P2P returned a different pose from the one its fit statistics described

The October 12 investigation found a concrete upstream registration defect:
`SVDClusterRANSACRegister.register` computes a surface-refined pose and scores
its residuals, but returns the earlier sparse-fit seed. This is a control-flow
bug reproducible on the CPU, without neural inference or low-precision GPU
arithmetic. It affects both objects in the session.

In the inspected upstream implementation:

1. `selected_T` receives the chosen RANSAC candidate.
2. `_maybe_refine_with_sdf` returns `refined_T`, fitted to the reconstructed
   depth surface. The selected cluster's `T` is replaced with this value.
3. Residuals, inlier flags and the final support check describe `refined_T`.
4. The function returns `selected_T` and saves it in `_last_selected_T`.

Thus the front end can adopt the unrefined pose using quality statistics from
a different pose. If refinement fails the final support gate, upstream
deliberately falls back to the seed and recomputes its statistics; that path
was already internally consistent.

The experimental correction lives directly in the maintained P2P fork, on
[`fix/sdf-pose-return`](https://github.com/TheWisp/point-to-pose/tree/fix/sdf-pose-return).
After the final support gate passes, `selected_T` becomes `refined_T`, so the
return and selection history describe the scored pose. Skipped/disabled
refinement, missing candidates and the existing support fallback are preserved.
No geometric constraint or tolerance is added. This intentionally changes
results; it is distinct from the bridge's historical performance patches.

**Rolled back from the live selection on October 13.** Complete replay and
reference-sequence tests found tracking regressions. The active dependency pin
is again the pre-correction `c3d2068`; `4abb818f` and the fix branch remain
available for offline experiments. The code defect described here is therefore
present in the restored baseline. Contract correctness alone did not justify
keeping the corrected candidate active.

The earlier source-hash-guarded bridge correction has been removed. LeRobot
now requires the exact fork commit in `benchmarks/p2p.lock.json` and rejects a
different or modified checkout before loading models. See
[P2P dependency workflow](p2p_dependency.md) for setup and controlled updates.

## Evidence

A six-point controlled test calls the real upstream `register` function while
fixing the candidate and refinement outputs to isolate return routing. With a
20 mm seed error and a correct refined pose, the old return claims six inliers
but its returned pose actually has zero. The repaired return has six inliers
and exactly matches the reported residuals. The same test checks the real
upstream final-support fallback, skipped refinement and disabled refinement.

A separate CPU probe uses saved replay candidates from act
`20261012_205715`, frames 2–16, and P2P's own SDF builder and refinement. The
SDF is rebuilt once from the recorded initialization depth/mask and held fixed;
the recorded sparse candidate is fixed too, isolating refinement and return
routing. No plane or upright constraint enters the estimator. An independently
fitted dominant visible top normal is used only to score its result.

| Visible-surface normal disagreement | Unrefined candidate | Refined candidate |
| ----------------------------------- | ------------------: | ----------------: |
| Median, 15 frames                   |        5.67 degrees |      2.10 degrees |
| Maximum                             |        7.31 degrees |      4.00 degrees |

All 15 frames improve. Their original returned poses disagree with the
reported residual arrays; the corrected return matches those arrays in every
case. All refined candidates retain at least 21 sparse inliers. This makes the
bug relevant to the observed tilt, not just a synthetic mismatch.

The existing SDF-off ablations do **not** establish that refinement was
ineffective: accepted refined poses were being discarded. Their historical
measurements remain valid for the defective implementation. Refinement can
still affect its flags and later state even when its pose is discarded.

The saved early point model is unchanged, so model growth is not required for
the initial drift. A fixed-correspondence, unweighted fit of those recorded
inputs still rotates a median 2.84 degrees. Holding track pixels fixed while
changing depth gives 0.83 degrees; holding depth fixed while changing pixels
gives 1.26 degrees (15 frames). These separate input ablations are not additive
and do not reproduce P2P's uncertainty weights or hypothesis selection. They
show residual sensitivity to the measurements, not a second isolated software
defect or proof that TAPIR alone caused the error.

## Limits and verification

The initial counterfactual above uses a fixed SDF and recorded seeds. It does
not establish six-dimensional ground truth, complete placement accuracy,
occluded-view robustness or moving-object benchmark accuracy.

**Subsequent full replay found a tracking regression.** On all 129 recorded
frames (2–130) of act `20261012_205715`, the unmodified baseline and corrected
pipeline were each run from the same recorded initialization, with Python,
NumPy and Torch seeds set to 17. The correction uses fork commit `4abb818f` and
LeRobot `ed84954cc`; the baseline uses P2P `c3d2068` and the bridge from
`c3c2cf633`, before the runtime return-value correction. The configuration and
recorded RGB-D inputs are unchanged. These runs include tracking, masks, model
growth and reconstruction feedback.

| Recorded quantity                                  | Baseline | Corrected |
| -------------------------------------------------- | -------: | --------: |
| Median reported cube rotation from initialization  |    6.99° |    10.95° |
| Maximum reported cube rotation from initialization |   15.42° |    41.97° |
| Cube frames marked lost                            |       20 |        10 |
| Gamepad frames marked lost                         |       34 |        23 |

The 41.97° estimate at frame 40 is accepted, while the arm and cable partly
occlude the cube. An additional complete corrected replay produces identical
cube matrices at every frame. The fix improves some frames (frame 16: 7.03° to
1.66°; frame 124: 6.01° to 2.84°) but cannot be treated as an overall accuracy
improvement. These rotation magnitudes are reported estimates, not measured
ground-truth orientation errors. Lower lost counts do not establish accuracy.
Physical stacking validation was withheld after this replay result; the arm
was parked and disconnected. Returning consistent statistics fixes the code
defect, but refinement under partial occlusion remains an unresolved estimator
problem. The cause of its larger rotations has not yet been isolated between
the current surface fit and subsequent model/reconstruction feedback.

**Tracing the bad frame changes the immediate diagnosis.** An instrumented
replay reproduces the corrected poses exactly. At frame 40, the sparse RANSAC
seed is already at 41.97°; SDF optimization fails to find an accepted step and
returns that seed unchanged. The jump check accepts 13 of 16 correspondences,
all marked confirmed, with a 3.50 mm mean inlier residual under the configured
5 mm correspondence threshold. The retained image points occupy a small visible
patch. Removing one point from a weighted rigid refit changes its orientation
by up to 12.27°, versus 2.10° and 2.80° in the earlier frames 2 and 16. This is a
measured sensitivity, not ground-truth angular error or a calibrated confidence
interval. Point count and residual alone do not establish orientation precision.

Freezing SDF integration after initialization changes frame 40 to 16.35°, but
frame 41 then reaches 36.17°. This isolates a contribution from reconstruction
feedback while showing that freezing the surface alone does not solve the
failure. These experiments are process-local and do not change the pinned
implementation. The broader [regression protocol](p2p_regression.md) is required
before promoting any proposed estimator change.

Two further 44-frame ablations (frames 2–45) separate more of that feedback:
disabling the global keyframe graph still reaches 35.63°. Disabling new
keyframes and point sampling entirely reduces the maximum to 10.93°, but lost
cube frames increase from 10 to 21. The latter also changes point-tracker
queries, point promotion, graph updates and SDF updates; it does not isolate
one of them. It is evidence that the model-update path amplifies the failure,
not a validated replacement tracker. Permanently freezing the model would lose
the ability to learn newly visible surfaces. None of these ablations were
deployed, and no object-specific motion prior or threshold was introduced.

The fork's `test/resgister/test_svd_cluster_ransac_register.py` exercises the
real registration function with controlled candidate/refinement outputs.
Accepted translation and rotation fail before the fix; all six cases pass
after it, together with 15 existing pose-jump-guard tests. LeRobot tests verify
commit/cleanliness enforcement, non-destructive setup, rejection before model
loading, and propagation of source/configuration identity into recordings.

A subsequent native CPU check removes the controlled candidate and refinement
outputs: real RANSAC, selection, SDF refinement and support gating run on three
recorded frames, under both rig and published YCBInEOAT registration settings.
All six calls disagree with their returned pose before the fix and agree after
it (largest residual mismatch becomes zero). Inlier counts happen to be equal
before/after in these examples. This check still fixes the initial SDF and uses
uniform uncertainty weights; it is not a complete tracking replay.

Reproduction scripts, JSON results, logs and the inspected upstream source are
preserved under the experiment archive's `p2p-rootcause` directory. No camera,
arm connection or motion was used by the offline probes/replays. The existing
GUI connection was used to park the idle arm before stopping its worker for
the full replay. The old P2P checkout was preserved; the fork was installed
separately.
