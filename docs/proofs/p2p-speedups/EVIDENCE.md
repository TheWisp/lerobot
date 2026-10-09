<!-- Captured evidence for one change: what was observed, not what was designed.
     Design documents live beside the code they describe. -->

# Evidence: where Point2Pose's step went, and the same answers in less time

Captured 2026-10-09/10 on `proto/show-and-servo` at `aa59a1fca`, RTX 5090, with the GUI's worker and its bridge
loaded but idle (`nvidia-smi pmon` showed no compute from them) and no point groups view running. The live tracker's
last status on the rig read 327 ms a frame with two objects in the session; offline, without the groups view beside
it, the same two objects took 140-154 ms.

Two recorded camera streams (848x480, RealSense `152222071508`), one init mask per object from SAM 2.1 and a click:

| Stream     | Recording                                     | What happens                                                      |
| ---------- | --------------------------------------------- | ----------------------------------------------------------------- |
| static     | `camera_20261009_230432`, 392 frames          | nothing moves; each object keeps its 25 tracks                    |
| moving act | `camera_20261009_210943` frames 1830-2389 / 3 | the arm grasps the gamepad and sets it on the cube; tracks to 182 |

Times are median / p90 in ms over every step after the init and 10 warm-up steps, through the bridge's pipe as
`pregrasp_worker.P2PBridge` drives it, unless marked in-process.

## The step, stage by stage (static, two objects, in-process, CUDA-synchronised timers)

| Stage                       | Before        | After       |
| --------------------------- | ------------- | ----------- |
| SAM2 propagation            | 35.6 / 36.6   | 32.5 / 33.2 |
| · image prep (CPU)          | 2.9 / 3.2     | 0.6 / 0.7   |
| · image encoder             | 19.4 / 19.7   | 19.4 / 19.6 |
| · memory attention          | 9.1 / 9.5     | 9.1 / 9.5   |
| TAPIR (one chunk of 64)     | 34.0 / 36.9   | 30.3 / 33.6 |
| pose fit                    | 79.3 / 82.3   | 19.1 / 20.1 |
| · SDF refinement            | 64.1 / 66.5   | 16.3 / 17.2 |
| · cluster RANSAC            | 12.6 / 13.2   | 1.9 / 2.0   |
| · dense-cloud depth lookups | 4.2 / 4.6     | 0.7 / 0.9   |
| serialisation and pipe      | 5.1           | 5.4         |
| whole step                  | 151.8 / 158.6 | 84.1 / 88.4 |

"After" here leaves out the two changes that synchronised timers would distort (TAPIR on its own stream, the SDF cost
memo). The SDF refinement built its Jacobian in a Python loop over up to 1500 points, 8 times per object per frame.
TAPIR blocked on about 540 host-to-device copies a step; its kernels were 11.9 ms of its 34 (`torch.profiler`). Each
further 64 tracks cost TAPIR about 25 ms in the moving act (28, 53, 78 ms at 1, 2, 3 chunks).

## Whole step through the pipe

| Stream, objects | Before        | Exact speedups | + one TAPIR chunk   |
| --------------- | ------------- | -------------- | ------------------- |
| static, 1       | 110.7 / 115.6 | 52.6 / 57.8    | (one chunk already) |
| static, 2       | 152.8 / 157.4 | 64.8 / 68.1    | (one chunk already) |
| moving act, 1   | 93.1 / 110.1  | 66.4 / 71.9    | 44.3 / 51.9         |
| moving act, 2   | 140.3 / 161.8 | 88.6 / 97.5    | 59.4 / 65.4         |

Re-run on the bridge as committed with this change (`benchmarks/p2p_bridge.py`), moving act, two objects: 141.2 /
163.3 before, 88.2 / 96.1 with the exact speedups, and 139.3 / 161.0 before against 58.3 / 66.7 with one chunk.
Bridge start-up (2.1-2.2 s) and init (3.2-3.5 s) did not change.

## The answers

**Exact speedups.** Every frame's per-object motion (`delta_i`) compared bit for bit with the unpatched bridge: no
frame differs, in-process and through the pipe, one and two objects, static (392 frames) and moving (187); the
re-run above, 0 of 187 frames for either object. Two unpatched runs are identical to each other. Each rewritten
function was also compared with Point2Pose's own: RANSAC over 561 calls (candidate, remaining pool and the random
generator's state), the SDF gradient and refinement over 80 trials, the depth lift over 12 cases including holes and
the uncertainty path, SAM2's input preparation.

**One TAPIR chunk** (`query_chunk_size: 4096` in `benchmarks/p2p_rig.yaml`), moving act, two objects, against the
unpatched bridge:

| Object  | Frames differing | At its centre: median / max mm | Turn: median / max deg |
| ------- | ---------------- | ------------------------------ | ---------------------- |
| gamepad | 1 of 187         | 0.00 / 0.30                    | 0.00 / 0.75            |
| cube    | 160 of 187       | 0.19 / 0.77                    | 0.78 / 3.88            |

The cube's 3.88 degrees hold from about frame 2136, after Point2Pose has lost it and each run keeps a different last
pose. For scale: the RANSAC seed alone (0 against 1) moved the cube's centre at most 0.47 mm and its turn at most
1.75 degrees on single frames, and on the static stream the cube's turn wanders a median 3.6 degrees (max 6.9) while
its centre stays within 1.3 mm. The tracks themselves moved a median 0.059 px (p90 0.29, max 6.5).

## Not changed

TSDF fusion runs on the CPU, since PyCUDA is not installed in Point2Pose's environment ("Failed to import PyCUDA.
Running fusion in CPU mode"): 0.9-1.0 s of each init and 25-138 ms on keyframe steps. After these changes SAM2's
image encoder (19.4 ms) and memory attention (4.9 ms an object) are the largest costs.
