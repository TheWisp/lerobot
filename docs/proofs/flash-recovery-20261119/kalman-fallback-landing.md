# Geometry-checked Kalman fallback — landing verification

2026-11-21. The tracker can recover a pose when the surface fit passes the
geometry checks but both the fused Kalman pose and its prediction fail them.
The checked fit becomes both the returned pose and the next internal pose.
The pre-correction covariance, previous matching image and its pose, and stored
surface remain unchanged on that fallback frame. Ordinary accepted corrections
can update the matching image and surface again.

This is the retained-image fallback from the broader comparison. It does **not**
keep the rejected prediction internally, inflate covariance by pose disagreement,
or advance the matching image during fallback. See the canonical definitions of
[Kalman pose update](../../../src/lerobot/policies/flash/docs/glossary.md#kalman-pose-update)
and [accepted observation / reference image](../../../src/lerobot/policies/flash/docs/glossary.md#accepted-observation--reference-image).

## Production equivalence

Fresh replay with the experimental policy wrappers bypassed exactly reproduced
the good experiment on all nine recordings: 2,183 sampled RGB-D frames, containing
2,221 object updates across 16 object selections. Returned and internal poses,
statuses, accepted-image IDs, map versions, and fallback decisions all match.
These are replayed frames, not seconds. Saved tracking poses never initialize
these replays; recorded RGB, masks and selected depth are the inputs.

All 54 known-pose controls also match: 14 general cases, 36 combinations of
rotation, visible-strip width and depth noise, and four nearer/farther movements
of 50 or 100 mm. Returned poses and scored errors match exactly. Registration
uses one thread for repeatability. The camera recordings have no independent
true 6D pose, so less time lost does not establish greater pose accuracy.

| Example                                | No fallback: seconds lost | Landed fallback: seconds lost |
| -------------------------------------- | ------------------------: | ----------------------------: |
| Latest three-object run                |                    15.592 |                         1.768 |
| Older evening helicopter               |                    24.745 |                         5.971 |
| Original five-selection regression run |                    24.851 |                         7.009 |

Durations run from a lost output to the next returned pose, capped at the final
recorded timestamp. Selections never bridge gaps. Full per-selection scores,
source hashes and equivalence results are in
[the landing measurements](kalman-fallback-landing.json).

## Regression protection and limits

- The recorded controller test exercises the default production behavior and
  the next update. It requires the checked pose internally, unchanged covariance,
  unchanged matching image/pose, and unchanged map. Reintroducing each of the
  three discarded state changes separately makes this test fail.
- The compact ring recording checks recovery followed by a normal correction
  that resumes matching-image and map updates. The wrong-object test still
  rejects the ring as the cube. Missing observations and group vetoes retain
  their rejection behavior.
- Recorded green-toy geometry rejects an invalid fused pose in both the
  historical baseline and the fallback path. The fallback must return its
  independently fitted, geometry-compatible pose instead.
- The known accuracy gap remains an explicit strict expected failure: a clean
  24-pixel visible strip after a 25-degree turn can pass geometry with 8.009 mm
  median 3D point error for eight returned poses over 1.75 seconds. The baseline
  rejects those poses. Preserving uncertainty and the map does not remove this
  ambiguity. Two-thread live registration also showed variation in the prior
  broader experiments; the exact equivalence claim here is for one thread.

No additional numerical fit or retry loop is introduced. This verification makes
no new performance claim. SAM and depth inference are not rerun. The live server
was not restarted. The architecture diagram's JavaScript and internal anchors
were checked; visual browser review was unavailable because the browser blocked
the local file URL.

## Reproduction

With the replay environment, run the committed compact fixture and controls:

```sh
PYTHONHASHSEED=0 PYTHONPATH=src:.:tests/policies/flash OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest tests/policies/flash/test_controller_visibility_recovery.py \
  tests/policies/flash/test_recorded_motion_recovery.py \
  tests/policies/flash/test_surface_kalman.py \
  tests/policies/flash/test_green_toy_kalman_recovery.py \
  tests/policies/flash/test_kalman_recovery_control.py -q

PYTHONHASHSEED=0 PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m benchmarks.flash_motion_recovery --policies baseline checked_fit \
  --output /tmp/flash-fallback-compact-replay.json
```

The baseline flag is explicit in the replay helper so changing the runtime
default cannot silently change the comparison. Known-pose controls use the
committed November 20 settings fixture. Accuracy scores use the returned pose
when it differs from the internal prediction. Full local replay receipts are
under `/tmp/flash-fallback-landed-live` and `/tmp/flash-fallback-landed-controls`.
