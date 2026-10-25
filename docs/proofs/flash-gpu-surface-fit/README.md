# FLASH CUDA surface-fit experiments

The live tracker still uses the CPU `WholeSurfaceFit`. The opt-in
[`experimental_gpu_fit.py`](../../../src/lerobot/policies/flash/experimental_gpu_fit.py)
copies one prepared RGB-D frame and surface reference to CUDA, then runs the
nearest-surface queries, mask/depth residuals, finite-difference Jacobian and
six-variable pose solver there. SAM3 and UFM already use the GPU. Frame
preparation (depth-to-points, distance image and convex hull) and coarse pose
initialization remain on the CPU. The sidecar is not imported by the live
tracking path.

**Correction:** the first CUDA nearest-point implementation checks all point
pairs, while production searches a spatial index. Both return the same nearest
distance, but this was not an algorithm-identical GPU port; its speed does not
establish that GPU spatial search is inherently slower. The separate
[`experimental_gpu_kdtree.py`](../../../src/lerobot/policies/flash/experimental_gpu_kdtree.py)
now transfers the production `cKDTree` splits, leaves, point order and
coordinates to CUDA for exact tree traversal. Both GPU implementations remain
experimental and outside the live tracker.

Run the matched-input comparison from the repository root:

```bash
PYTHONPATH=src python docs/proofs/flash-gpu-surface-fit/compare.py \
  --repeats 5 --output docs/proofs/flash-gpu-surface-fit/comparison_live_gpu.json
```

Both paths receive the same four recorded frames, reference points, RGB-D
observation, camera calibration, initial pose and CPU `ShapeConfig`. The
measurement includes a warm GPU, five repeated fits, the one-time per-frame
GPU upload and incremental peak VRAM. The saved run had the live three-object
camera-debug worker active on the same GPU. The full output, including pose
matrices and raw times, is in
[`comparison_live_gpu.json`](comparison_live_gpu.json).

| Recorded case              | Shared preparation | CPU fit median | GPU fit median | GPU upload | Extra peak VRAM | GPU pose vs CPU |
| -------------------------- | -----------------: | -------------: | -------------: | ---------: | --------------: | --------------: |
| Ring moving                |              17 ms |         502 ms |         521 ms |      88 ms |         134 MiB |   16.5 mm, 8.6° |
| Held dowel                 |               7 ms |         273 ms |         384 ms |       2 ms |           6 MiB |    1.9 mm, 5.1° |
| Stationary dowel, frame 0  |               7 ms |         100 ms |         176 ms |       1 ms |          18 MiB |   15.9 mm, 2.4° |
| Stationary dowel, frame 13 |               7 ms |          64 ms |         246 ms |      10 ms |          18 MiB |   10.6 mm, 1.4° |

At the initial pose, CPU and CUDA residuals agree to within `3e-8` on every
case. CUDA pose search is a separate damped solver and currently freezes the
projected hull topology within each refinement; the CPU solver recomputes it.
Consequently the full-fit timing and final-pose differences above are **not a
controlled CPU-versus-GPU implementation comparison**. They compare two
optimizers, as well as two devices. CUDA reached a lower CPU-measured objective
for the ring, a higher one for the held dowel, and similar values for the two
stationary frames. These frames have no independent ground-truth pose, so a
lower score does not establish a more accurate physical pose.

For a controlled measurement of the _same calculation_,
[`compare_same_poses.py`](compare_same_poses.py) records candidate poses from
the CPU solver, then evaluates each identical pose through both residual
implementations. Both compute exact nearest-point distances: the CPU uses a
`cKDTree` index; CUDA computes all point pairs in bounded tiles. Across 80
poses per case, the residual arrays agree within `3e-8`:

| Recorded case       | CPU residuals | GPU residuals |
| ------------------- | ------------: | ------------: |
| Ring moving         |        106 ms |        146 ms |
| Held dowel          |         39 ms |        176 ms |
| Stationary dowel 0  |         45 ms |        170 ms |
| Stationary dowel 13 |         45 ms |        111 ms |

These are sequential, synchronous evaluations of the same candidate poses;
GPU batching could change throughput for a solver that can issue candidates
together. The raw three-repeat data is in
[`same_pose_residuals.json`](same_pose_residuals.json). The different search
algorithms are mathematically equivalent here; the **different optimizer is a
comparability flaw** in the earlier full-fit benchmark.

The same-pose test was repeated with the CUDA tree traversal. All 80 residual
arrays per case still agree with the CPU within `3e-8`. Median CPU/GPU times
were **105/176 ms** for the ring, **40/118 ms** for the held dowel,
**45/95 ms** for stationary frame 0 and **42/84 ms** for frame 13. The
[raw tree comparison](same_pose_tree_residuals.json) measures this specific
kernel, not a theoretical GPU limit. It avoids exhaustive pairs but still
traverses divergent branches and transfers each result back to the CPU solver.

[`compare_same_solver.py`](compare_same_solver.py) goes further: it runs the
production `WholeSurfaceFit` SciPy optimizer and branch selection unchanged,
replacing only `residual()` with CUDA evaluation. Three-repeat median fit times
under the active camera worker were:

| Recorded case       | CPU solver + CPU residual | Same solver + GPU residual | Final pose difference |
| ------------------- | ------------------------: | -------------------------: | --------------------: |
| Ring moving         |                    495 ms |                     752 ms |          3.4 mm, 1.8° |
| Held dowel          |                    277 ms |                     925 ms |          1.6 mm, 3.6° |
| Stationary dowel 0  |                    100 ms |                     300 ms |         16.0 mm, 2.4° |
| Stationary dowel 13 |                     67 ms |                     344 ms |          5.4 mm, 0.6° |

The [raw same-solver result](same_solver_comparison.json) still has different
final poses. The production solver estimates derivatives by perturbing its six
pose variables at very small steps. At a `1e-8` metre translation step, the
CPU and CUDA residual difference is amplified into a roughly 5% difference in
one derivative on the two stationary frames; at `1e-5` metre it is below
0.005%. Thus _the optimizer code is now identical_, but its numerical path is
sensitive to tiny residual changes. The controlled same-pose residual test
above remains the clean speed comparison for the objective alone. Changing
the production derivative rule requires its own accuracy and regression study.
The [same-solver tree comparison](same_solver_tree_comparison.json) is also
slower and still produces numerically divergent final poses; exact nearest
distances alone do not repair derivative sensitivity.

### Where the CUDA time goes

The earlier conclusion that "the GPU is slower" needs the workload attached.
[`profile_residual_path.py`](profile_residual_path.py) times 80 _identical CPU
solver candidate poses_ with the same exact `cKDTree` traversal on CPU and
CUDA. The [saved profile](residual_path_profile.json) was taken on an RTX 5090
with PyTorch 2.11, after warmup, without the live camera worker. The
"resident" column has pre-uploaded poses and precomputed hulls; it therefore
isolates residual execution from those per-pose CPU costs.

| 80 candidate poses | CPU full residual | CUDA resident residual | CUDA with pose upload, CPU hull and result download | CPU nearest only | CUDA nearest only |
| ------------------ | ----------------: | ---------------------: | --------------------------------------------------: | ---------------: | ----------------: |
| Moving ring        |           94.9 ms |                92.0 ms |                                            106.9 ms |          78.5 ms |           65.8 ms |
| Held dowel         |           33.9 ms |                63.6 ms |                                             77.1 ms |          12.6 ms |           14.0 ms |

The profiler counted **113 CUDA kernel launches and two stream
synchronizations per candidate pose** in the result-downloading path. Each
candidate fits at most 2048 query points per direction; the held-dowel case has
only 780 reference and 100 target queries. The CPU's compiled `cKDTree` handles
these small searches cheaply, while the CUDA residual invokes many separate
tensor kernels. On the larger ring, CUDA nearest search wins in isolation, but
the other residual work plus per-pose upload, CPU hull construction and result
download consume that gain. On the small dowel, the CUDA path is slower even
when poses and hulls are already resident. In the same-solver experiment,
SciPy needs each result before it can select its next pose, so those transfers
and synchronizations cannot be amortized by queuing many candidates.

The older full-fit comparison ran while SAM3/UFM work was active on the same
GPU, whereas this breakdown did not. GPU contention is a plausible additional
factor, but these measurements do not isolate its contribution. A native GPU
solver could batch its finite-difference poses; the current native prototype
does that but changes convergence and hull handling, so it has not established
an equivalent faster replacement. This profile identifies the costs in the
**current implementation**, not a hardware limit on GPU surface fitting.

Reproduce the breakdown from the repository root:

```bash
PYTHONPATH=src .venv/bin/python docs/proofs/flash-gpu-surface-fit/profile_residual_path.py \
  --output /tmp/flash-residual-profile.json
```

**Decision:** keep the CPU fitter as the default. The measured CUDA full-fit
paths have not been both faster and pose-equivalent; solver-level pose
equivalence is not yet established. Moving the remaining
7–17 ms of frame preparation to CUDA cannot
recover the observed fit-time gap by itself. Before considering a switch,
benchmark a GPU implementation with equivalent convergence and hull updates
on multi-object live frames, then compare both measured pose quality and total
fresh-pose rate while SAM3 and UFM share the GPU. CUDA batching, fewer kernel
launches or a compiled solver may help, but are unmeasured here.
