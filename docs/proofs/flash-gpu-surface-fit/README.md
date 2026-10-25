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
Consequently the final poses can differ even when both are scored by the same
CPU objective. CUDA reached a lower CPU-measured objective for the ring, a
higher one for the held dowel, and similar values for the two stationary
frames. These frames have no independent ground-truth pose, so a lower score
does not establish a more accurate physical pose.

**Decision:** keep the CPU fitter as the default. The current GPU sidecar is
slower on all four fits once upload is included, and pose equivalence is not
established. Moving the remaining 7–17 ms of frame preparation to CUDA cannot
recover the observed fit-time gap by itself. Before considering a switch,
benchmark a GPU implementation with equivalent convergence and hull updates
on multi-object live frames, then compare both measured pose quality and total
fresh-pose rate while SAM3 and UFM share the GPU. CUDA batching, fewer kernel
launches or a compiled solver may help, but are unmeasured here.
