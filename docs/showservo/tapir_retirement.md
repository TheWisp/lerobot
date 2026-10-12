# TAPIR retirement: memory and numerical behavior

Measured on the rig on 2026-10-12, for the groups-view retirement change
`b609ade03`. Retired queries leave TAPIR's query points, feature rows and causal
state; group track IDs remain stable. Re-designating an object retires its
superseded own tracks. This bounds retained GPU query state for the measured
replay, but does not make total memory independent of the number of tracks ever
created: stable-index CPU arrays still grow.

## Recorded replay

The original session replayed `camera_20261012_042615`, every second frame, with
object clicks `gamepad=166,335` and `cube=286,336`. Its handoff reported:

| Measurement    | Original (beforeA) | Retirement enabled |
| -------------- | -----------------: | -----------------: |
| Active queries |               1152 |  503 of 1136 added |
| Causal state   |           1080 MiB |            472 MiB |
| GPU reserved   |             5.9 GB |             3.9 GB |

Two original-code replays agreed within 0.001 mm. Retirement matched the original
until the first drop; afterwards the gamepad differed by median 0.36 mm, maximum
0.63 mm and maximum turn 0.57 degrees. The cube differed by median 0.06 mm,
maximum 0.39 mm. These are differences between implementations, not ground-truth
pose errors. GPU reserved memory also depends on allocator history.

## Isolated check at the first retirement

The follow-up replay captured the exact tracker state at step 150 (source frame
300), before dropping ten of its 546 query rows. Each comparison restored that
state and predicted the same next RGB frame, source frame 302. The two branches
used all rows or the same state sliced to the 536 surviving rows. Neither branch
updated groups or selected new points during this comparison.

Environment: RTX 5090, PyTorch 2.8.0+cu128, four TAPIR refinement iterations,
256-pixel tracking input, query chunk size 4096. Pixel distances below are in the
original 848-by-480 image.

| Arithmetic          | Median shift | 99th percentile | Maximum shift | Changed surviving points | Visibility changes |
| ------------------- | -----------: | --------------: | ------------: | -----------------------: | -----------------: |
| bf16 autocast       |  0.057310 px |     0.196247 px |   0.933077 px |               535 of 536 |                  0 |
| fp32, TF32 disabled |         0 px |            0 px |          0 px |                 0 of 536 |                  0 |

Repeating the full-query branch from the identical state was exactly equal in
both modes. The fp32 comparison promoted the saved features and causal state to
fp32 and explicitly disabled autocast: Point2Pose's SAM2 module enables bf16
autocast globally at import, so merely omitting a local autocast context does not
produce a full-precision control.

This isolates query-count-dependent low-precision arithmetic at the first
divergence. It supports the explanation for the replay's later pose differences;
it is not a full fp32 replay or a bound for other scenes. Production retains bf16
and its existing tracking settings.

## Regression checks

The affected files' tests passed: 125 tests across `test_group_live_tapir.py`,
`test_groups.py` and `test_stream_objects.py`. Independently disabling TAPIR
retirement, old-own-track retirement, or latest-model capture made the respective
regression test fail. Each source file was restored and touched afterwards; the
checkout matched the pushed commit before the GUI restart.

## Live trial and diagnostic transport

Build `0a958b3c6` ran act `20261012_130212`, with the normal key-frame runner,
the demo's lift and hold correction enabled, from separated objects. The act
completed without a groups-view crash and the arm parked, but did not stack:
camera recording `camera_20261012_130145` shows the gamepad beside the cube after
release. The measured median height at the cube remained 30.3 mm. Completion of
the motion is not a successful physical outcome, and this run does not establish
why the placement missed.

The `tracker_memory_20261012.jsonl` sampler collected 57 usable observations over
282.7 seconds, including warm-up, the act and the still scene afterwards. The
followed object's reported track count rose from 25 to 100 to 200. The bridge's
GPU memory peaked and ended at 6624 MiB. Sampled groups logs reached 1279 active
queries and later fell to 962; its GPU process memory peaked and stayed at
7138 MiB. These are `nvidia-smi` process measurements, including allocator caches,
not live tensor bytes. One trial cannot establish a long-run memory bound or
prove the bridge's track table caused all its memory growth.

The live model endpoint exposed a separate integration bug: the worker sent
`model_xyz`, but `/worker/result` omitted it from the accepted array names. The
first endpoint test called the apply helper directly and missed this boundary.
The regression now posts the worker's NPZ through HTTP before reading the model
endpoint, for both trusted and rejected views, and checks JSON state remains
serializable. The raw worker fit is available separately from the trusted motion
for inspection; it does not become an accepted pose merely by being exported.
Both HTTP regression cases failed before the transport fix. Afterwards the full
stream-objects test file passed all 104 tests, including both cases.

Before that transport fix, the yellow cloud always fell back to taught points.
Untrusted views still intentionally use taught points carried from the last
trusted pose by groups. The post-trial image showed that cloud near the original
pick position while the gamepad lay by the cube. This is evidence of a stale
displayed belief, not proof of adopted model contamination. A held transform
alone is not necessarily the final rendered transform because groups can carry it.
