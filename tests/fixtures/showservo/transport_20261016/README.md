# Recorded pose-fusion regression inputs

Small extracts of insertion acts `20261016_110227` and `20261016_110310`.
The JSON files identify source frame numbers and SHA-256 hashes of the original
`act.json`. No camera, network service, model weights or local demo directory is
needed to run these regressions.

- **11:02, frames 39–40:** an accepted RGB-D pose explicitly overrides group
  disagreement, followed by group catch-up. Applying that motion again adds
  more than 15 mm. The corrected transport must preserve the visual pose.
- **11:03, frames 50–52:** the accepted visual anchor agrees with group motion.
  Subsequent group movement must remain usable. This is a non-regression control,
  not another instance of the first run's double-counting error.

Group poses are the recorded comparison poses at each frame's verification
time. These sparse samples reproduce the fusion decision, not every intervening
live group update. The recorded control matrix is retained for audit; it is not
pose ground truth. Tests rename the object so the historical scene label cannot
select special behavior.

Each NPZ contains the saved aligned depth, segmentation mask, raw pose, sparse
reference cloud and dense reference surface for that frame. `expected_pose`
was generated with the pre-batching plane fit from commit `73812a7d4`, through
the same surface verification and constraint functions. It checks numerical
non-regression, not independently measured accuracy.

The JSON also retains frame capture timestamps and the timing breakdown for
both full runs. These are profiling baselines. Unit tests intentionally do not
assert machine-dependent FPS or wall-clock thresholds; performance comparison
must rerun numerical work under controlled conditions and separately measure
active live control. See `docs/showservo/run_review_20261016_1102.md` for the
original timing interpretation and limitations.
