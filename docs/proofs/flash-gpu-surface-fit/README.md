# FLASH CUDA surface-fit sidecar

The live tracker still uses the CPU `WholeSurfaceFit`. The opt-in
[`experimental_gpu_fit.py`](../../../src/lerobot/policies/flash/experimental_gpu_fit.py)
copies one prepared RGB-D frame and surface reference to CUDA, then runs the
nearest-surface queries, mask/depth residuals, finite-difference Jacobian and
six-variable pose solver there. SAM3 and UFM already use the GPU. Frame
preparation (depth-to-points, distance image and convex hull) and coarse pose
initialization remain on the CPU. The sidecar is not imported by the live
tracking path.

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

**Decision:** keep the CPU fitter as the default. The same-pose residual path
is slower on CUDA in these cases, and solver-level pose equivalence is not yet
established. Moving the remaining 7–17 ms of frame preparation to CUDA cannot
recover the observed fit-time gap by itself. Before considering a switch,
benchmark a GPU implementation with equivalent convergence and hull updates
on multi-object live frames, then compare both measured pose quality and total
fresh-pose rate while SAM3 and UFM share the GPU. CUDA batching, fewer kernel
launches or a compiled solver may help, but are unmeasured here.
