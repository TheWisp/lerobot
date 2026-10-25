# CPU batching and thread scaling for the FLASH surface fitter

The current CPU `WholeSurfaceFit` evaluates a six-variable pose with SciPy.
Numerical derivatives request six independent nearby poses at each optimizer
step. Each residual searches the same fixed and observed `cKDTree`s. This
experiment combines the six candidates' nearest-neighbour queries into two
larger CPU calls, then computes every candidate's mask, depth and projected
outline residual exactly as before. It leaves the optimizer and branch choice
unchanged. The candidate is in [`benchmark.py`](benchmark.py), **not in the live
tracker**; raw runs are in [`results.json`](results.json).

On 2026-10-25, the benchmark ran on a Ryzen 9 9950X (16 cores, 32 threads),
SciPy 1.18.1, with OpenBLAS and OpenMP limited to one thread. It used three
recorded RGB-D fit cases: a moving ring, a held dowel and a stationary dowel.
Each case was run three times after building its clouds and search trees. For
all cases, residual arrays at seven sampled candidate poses and the final fit
poses matched the production scalar path exactly in this run; the benchmark
aborts if they do not.

| Recorded fit     | Current CPU | Batch, 4 tree workers | Batch, 8 tree workers | Thread six derivatives, 2 workers |
| ---------------- | ----------: | --------------------: | --------------------: | --------------------------------: |
| Moving ring      |      450 ms |                362 ms |                304 ms |                            358 ms |
| Held dowel       |      207 ms |                155 ms |                166 ms |                            176 ms |
| Stationary dowel |       54 ms |                 38 ms |                 38 ms |                             47 ms |

To check scaling, the same three **independent recorded fits** were scheduled
as if they belonged to one frame. This measures fitter makespan, not live
perception FPS or three-object tracking quality; the objects came from
different scenes. The existing runtime already schedules object fits in
parallel. More object workers did not improve this CPU workload after two.

| Three-fit makespan          | One object worker | Two object workers | Three object workers |
| --------------------------- | ----------------: | -----------------: | -------------------: |
| Current fitter              |            707 ms |             566 ms |               588 ms |
| Batched, 4 tree workers     |            555 ms |             438 ms |               447 ms |
| Batched, 8 tree workers     |            514 ms |             429 ms |               442 ms |
| Threaded derivatives, 2     |            586 ms |             567 ms |               561 ms |
| Threaded derivatives, 4     |            616 ms |             584 ms |               592 ms |
| Single-threaded tree search |           1063 ms |             857 ms |               860 ms |

The main gain is **batching the tree queries**, not simply spawning more
threads. For this mix, batching with two object workers reduced fitter makespan
by about 24% versus the current two-worker path (566 to 429 ms). The optimal
tree-worker count differs between the ring and dowel, so the values above are
not universal defaults. Using SciPy's separate finite-difference worker pool
can help one large fit, but gives almost no benefit when the independent object
fits already occupy CPU threads. Disabling the existing parallel tree queries
slows the mixed workload substantially.

This does not make the current three-object _vision pipeline_ 10 FPS. In the
[stationary live-debug timing](../flash-rgbd-registration/multi_object_optimized_timing.json),
SAM masks took about 34 ms and the three serialized UFM matches took about
36–38 ms each after subtracting lock wait. That is roughly 146 ms of serial
mask/match work per processed frame before any fit, so even a free fitter would
be limited to about 7 updates/s in that measured scene. The yellow block's pose
was rejected in that sample, and these recorded-fit timings were measured
without concurrent SAM/UFM load. Live speed and pose quality still need a
paired test before changing the default.

The prototype uses SciPy 1.18's `least_squares(workers=...)` map callback to
batch derivative candidates. LeRobot permits SciPy versions back to 1.14, so
that API's compatibility across supported installs must be checked or handled
before integration. The candidate also duplicates the residual formula for
the experiment; production integration should share one implementation with
the scalar path to keep future geometry fixes aligned.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  docs/proofs/flash-cpu-scaling/benchmark.py \
  --output /tmp/flash-cpu-scaling.json
```
