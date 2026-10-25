# Exact CPU batching and bounded frame pipelining

Measured on 2026-10-25 on the local Ryzen 9 9950X / RTX 5090. The deployed
camera interpreter uses SciPy 1.17.1. OpenMP/OpenBLAS were limited to one
thread; the explicit exact nearest-neighbour searches use up to eight workers.

## What changed

The CPU solver's six finite-difference candidates share two larger exact tree
queries. There is one implementation of the mask/depth/envelope residual for
both scalar and batched paths. No neighbours, projected hull, solver steps,
tolerances or geometry samples are approximated or reused between candidates.
SciPy installations without the public `least_squares(workers=...)` callback
retain scalar differentiation.

The shared runner overlaps the next frame's SAM/depth/UFM preparation with the
current frame's CPU fit. Image motion is prepared independently of fitted pose;
composition and fit completion remain ordered. At most one frame is prepared
ahead. Camera callers acquire the latest input when preparation starts; offline
callers supply all input frames. Freeze/add/reset drain pending work first.
Reference registration stays serial until all roles are registered. Appearance
backends without staged preparation retain their ordinary update path.

The worker, offline extractor and live debugger use this runner. Symmetry
canonicalization remains in the ordered completion stage. The motor loop and
its scheduling are unchanged. All acceleration switches are saved in the shared
vision configuration; see the [pipeline reference](../../../src/lerobot/policies/flash/docs/surface_tracking.md).

## Paired recorded observations

`benchmark.py` preloads aligned RGB-D and recorded SAM masks, runs actual UFM,
then compares every candidate pose matrix, acceptance decision, mask and source
frame. Timing excludes SAM inference, image decoding, UI publication and robot
control. First intervals that include model warmup are excluded from steady
rates. Each result file contains two repetitions; there are no concurrent
camera inference workers during these measurements.

| Recorded workload                                 |      Baseline | Exact batching | Batching + pipelining |
| ------------------------------------------------- | ------------: | -------------: | --------------------: |
| Red hand, 35 frames                               | 8.7 updates/s |           10.0 |             16.8–18.0 |
| Three objects, 30 frames, staged serial scheduler |     3.05–3.11 |      3.56–3.67 |             5.29–5.30 |

The **pre-change** multi-object scheduler already overlapped one object's fit
with another's matching. It is a stronger baseline than staged serial
execution: reconstructing that method from commit `8c32aee5b` gives **3.76–3.79
updates/s**, versus **5.10–5.22** for the new default on identical observations
(about 37% higher throughput). `original_vs_pipeline.json` records this fairer
comparison. The same scalar geometry code runs in that reconstructed schedule.

All compared pose matrices were exactly equal, including the three-object
comparison with the original scheduler. Every sampled candidate passed the
same existing acceptance checks (35/35 red-hand frames; 30/30 frames per role
in the three-object sequence). **This proves preservation of these results,
not their absolute pose accuracy.** The three-object scene was stationary;
the red-hand recording includes motion. The existing backside/occlusion
limitations are not fixed by scheduling.

`object_workers.json` compares two and three simultaneous object jobs. Two
workers achieved 5.17–5.30 updates/s, three 4.99–5.16. More workers increased
per-object fit time through contention; the default remains two. This is a
measured local default, not a universal optimum for every CPU.

Raw measurements:

- [`red_hand.json`](red_hand.json)
- [`three_objects.json`](three_objects.json)
- [`original_vs_pipeline.json`](original_vs_pipeline.json)
- [`object_workers.json`](object_workers.json)

The recordings remain local under
`~/.cache/huggingface/lerobot/flash_debug_traces/`; each JSON names its trace and
exact frame IDs. The three-object trace is
`trace-20261025-192743-253063024` (green ring, mini gamepad, yellow block).

## Live camera measurement

`live_three_objects.json` samples 47 distinct publications with SAM and UI
packet publication included and diagnostic trace recording disabled. It measured
3.87 updates/s and median source-image age 464 ms. All three roles passed the
existing checks on every sampled frame. Camera capture remained independent.
This is a different acquisition from the paired replay, not an additional
pose-equivalence claim or a demonstration of successful robot control.

A fresh serial-vs-default camera comparison used the same stationary scene,
with each session selecting the three objects afresh. With batching/pipelining
disabled, 31 publications gave 2.47 updates/s and median image age 384 ms;
with both enabled, 49 publications gave **4.06 updates/s and 457 ms**.
See `live_three_objects_serial.json` and `live_three_objects_final.json`.
This is about 65% higher throughput, with a 73 ms increase in median source
latency. It is a comparison with the staged serial configuration, not the old
per-object scheduler; the paired replay above is the old-scheduler comparison.
All roles passed the same checks in these samples. Fresh segmentation/seeding
and camera noise differ between sessions, so exact equivalence comes from the
paired recorded test, not these live numbers.

Pipeline throughput is not source latency. A prepared image can wait for the
preceding CPU fit; the debugger reports publication intervals and source age
separately. There is no unbounded pending-frame queue. CPU scaling alone still
does not deliver ten updates/s for this three-object live workload.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  /home/feit/.cache/flash-live-tracking/.venv/bin/python \
  docs/proofs/flash-cpu-pipeline/benchmark.py \
  --trace ~/.cache/huggingface/lerobot/flash_debug_traces/trace-20261025-192743-253063024 \
  --frames 30 --modes original batch_pipeline --output /tmp/flash-pipeline.json
```

Regression checks in `tests/policies/flash/test_surface_acceleration.py` compare
scalar/batched residuals and complete fits on recorded ring and held-dowel
surfaces, enforce source-frame ownership and one-frame lookahead, force actual
overlap, cover missing masks and recovery, pause/drain/resume, and inference
failure while the input queue has no next frame. Existing cross-consumer tests
exercise the same backend through extraction, worker and debugger.

Two deliberate regressions were also exercised and restored: publishing the
next frame's masks made the source-ownership test fail, and reusing one
candidate's nearest distances made the recorded-fit equivalence test fail.
The camera was also reopened after cleanup changes; the first-selection path
and replacing a previous selection now have explicit regression coverage.

The final targeted suite passed 45 tests; six browser tests were skipped in the
project test environment because Playwright was absent there. Separately, the
installed GUI interpreter's Playwright drove the actual `:9102` Run → FLASH →
Live tracking view, rendered the real three-object packet at source frame 556,
and reported no browser JavaScript errors. The camera worker exited with code
zero after each completed live sample. No robot motion was commanded.
