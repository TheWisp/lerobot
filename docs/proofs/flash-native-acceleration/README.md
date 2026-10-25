# CPU fitting and UFM batching experiments — 2026-10-25

**CPU paths deployed after these experiments; UFM batching remains research only.** These candidates preserve the current surface-fitting objective instead of replacing it with another registration algorithm. The independent benchmark baseline is production code at `3aa668b99`.

## Deployment

The shared demo/run/debug backend now enables `compiled_surface_fit` and `reuse_surface_search` by default. Both are explicit boolean fields in `VisionPipelineConfig`, also visible in the debugger's Camera & tracking parameters JSON. Setting either to `false` restores that part of the prior implementation. UFM still processes each object separately.

The backend owns one bounded search pool, shared by its object sessions and closed after pending fitting workers finish. The compiler warms the float32/float64 kernels before processing frames and caches generated machine code. Numba is optional: without it, a logged fallback retains the original NumPy calculation. This rig's existing worker environment has Numba 0.68.0 installed with its existing NumPy 2.2.6; other environments can install those versions to reproduce this configuration. Per-frame diagnostics report whether compiled fitting is active and the reused worker count.

Deployment verification passed 68 tests, including the real production fitter, shared pipeline, configuration round-trip, missing-compiler fallback and search-pool teardown. Reversing the production depth visibility direction makes the recorded-fit equivalence test fail. A fresh paired 30-frame/three-object run through the production backend reproduced every pose and decision exactly and measured 5.17 → 7.10 updates/s with saved masks; see [deployment measurements](deployed_pipeline.json). These are not SAM-inclusive live FPS.

The compiled CPU calculations are the strongest result: they reduced fitting time on all four recorded fit cases and reproduced the baseline poses and acceptance decisions across three sequential recordings. Reusing search threads improved total throughput in those recordings, but slowed the isolated moving-ring fit. UFM batching reduced neural inference time, but changed some poses; it is not an exact replacement.

## What was changed

`candidates.py` is imported only by these experiments and their tests.

- **Compiled:** Numba compiles the four-ray depth visibility calculation and projected-outline distances into native CPU loops. It avoids temporary arrays and Python work. Double precision, operation order and `fastmath=False` are retained; these kernels release the GIL.
- **Pool:** persistent threads split the same exact nearest-neighbour queries into contiguous pieces. Each piece runs `cKDTree.query(workers=1)`; results retain their original order. The production SciPy implementation creates and joins threads per parallel query.
- **Concurrent candidates:** a separate experiment evaluates independent numerical-derivative candidates on four threads, with internal search threading disabled to avoid nesting.
- **UFM batch:** independent object crops share one neural call instead of one call per object. The same checkpoint, resolution, precision and thresholds are retained.

The CPU candidates retain the production SciPy solver, samples, reference cache, weights, pose parameterization, convergence criteria, coarse alternatives, mask/depth checks, UFM model and causal pipeline. No tolerance was relaxed and no object-specific rule was added. The original experiments installed Numba in an isolated environment; deployment retains it as an optional dependency.

## Whole-pipeline measurements

These use actual recorded RGB-D, saved SAM masks, real UFM and the shared `PerceptionPipeline`/`PerceptionRunner`, including the existing bounded preparation pipeline. **SAM inference, camera capture, display and robot control are excluded.** These numbers are processing throughput, not live camera or motor-loop FPS. The insertion test tracks the dowel through the vision backend; it does not include gripper/group fusion.

Steady updates per second, excluding the first three completion intervals:

| Recording                                     |    Baseline |    Compiled | Compiled + 8 search workers |
| --------------------------------------------- | ----------: | ----------: | --------------------------: |
| Three objects, 30 recorded frames, two passes |   5.19–5.22 |   6.49–6.84 |                   7.13–7.34 |
| Moving red hand, 200 recorded frames          |       16.12 |       16.94 |                       18.00 |
| Full insertion demo, 713 frames, two passes   | 15.96–15.99 | 20.01–20.12 |                 20.64–20.82 |

Both CPU variants reproduced every compared pose matrix exactly, with unchanged masks, source timestamps, statuses and acceptance decisions. The three-object sequence accepted all 90 object observations, the hand all 200, and the insertion sequence accepted 533 and rejected 180 in every mode. The repeated three-object baseline also reproduced itself exactly.

The insertion sequence has 686 available pose matrices and 27 frames without a pose (172–198). A second full pass explicitly asserted identical pose availability, so absent estimates cannot silently pass the numerical comparison.

These are **equivalence results**, not ground-truth accuracy measurements. Preserving an accepted pose does not establish that it is physically correct. The recordings cover different motions and rejection/recovery, but cannot prove equivalence for all future scenes or library versions. Cold Numba compilation and thread startup are not included in steady throughput; integration would need explicit warmup. The hand measurement has one pass per mode and the other recordings two, so small timing differences should not be overinterpreted.

Raw results: [three objects](pipeline_three.json), [moving hand](pipeline_hand.json), [full demo](pipeline_demo.json).

## Isolated fitting

Median wall time in milliseconds, three repeats after warmup. Each case also compares 20 candidate residual vectors and the complete solve, including final pose, cost, evaluation count, convergence and chosen coarse branch. All those comparisons were exactly equal in these measured cases.

| Recorded fit               | Baseline | Pool 8 | Compiled | Compiled + pool 8 | Compiled + 4 concurrent candidates |
| -------------------------- | -------: | -----: | -------: | ----------------: | ---------------------------------: |
| Moving ring                |    304.8 |  345.8 |    238.9 |             254.7 |                              386.6 |
| Held dowel, frame 315      |    202.9 |  191.9 |    142.3 |             137.6 |                              113.3 |
| Stationary dowel, frame 0  |     62.1 |   64.5 |     46.6 |              43.6 |                               43.6 |
| Stationary dowel, frame 13 |     40.9 |   40.3 |     31.3 |              29.4 |                               30.0 |

More threads are not uniformly better: query size, outline complexity, work imbalance and scheduling overhead matter. Compiling the numerical loops consistently helped these cases; parallelizing whole candidates did not. [Raw fit measurements](fits.json) also include pools of two and four threads and process CPU time.

## UFM batching correctness

Twenty-seven actual image pairs cover the three-object scene, moving hand and insertion demo before grasp, during rotation/holding and around release. The checkpoint is `infinity1096/UFM-Base-DINOv2L-init`, using its default BF16 neural inference.

| Neural execution        | Median time for 27 pairs, three repeats |
| ----------------------- | --------------------------------------: |
| Separate calls          |                                802.4 ms |
| Batches of two          |                                671.1 ms |
| Batches of three        |                                657.5 ms |
| Separate calls repeated |                                801.0 ms |

Batching took about 18% less neural inference time. This isolated measurement also batches different times as a numerical stress test; it is not a claim that a causal single-object tracker can batch future frames.

Repeating scalar inference was exact. Batching changed fitted poses in four of 27 pairs, despite retaining the algorithm and precision. The largest difference was insertion frames 500→501:

- Median flow difference: 0.000106 pixels; maximum: 0.0174 pixels.
- A depth-sampling boundary changed the retained metric correspondences from 130 to 129.
- Final orientation differed by 0.587 degrees.
- Displacement of the actual reference surface was 0.121 mm median and 0.223 mm maximum.

The transform's translation column differed by 4.04 mm, but that is displacement of its arbitrary origin, **not object displacement**: the origin is far from the observed surface. Use the surface displacement above to interpret this result. No ground truth establishes which result is better. Both variants' surface-distance p90 exceeded the existing 5 mm gate on this difficult pair.

`check_ufm_trajectory.py` then captured all 87 real model calls in the 30-frame, three-object pipeline and replayed it with same-frame batched outputs. Replaying captured scalar outputs first reproduced the original exactly, checking the harness itself. With batched outputs, all 90 object observations remained accepted, but the maximum orientation difference reached 0.265 degrees and maximum surface displacement 0.199 mm. This short sequence does not bound longer-term accumulation or behavior near an acceptance threshold.

The batched network therefore remains experimental. A committed small fixture pins the real depth-sampling difference through `measured_pairs`, `robust_fit` and `WholeSurfaceFit`, without requiring network access or a checkpoint in pytest. The benchmark's independent scalar path was also checked against the actual production `ufm_pairs` consumer, with exact metric point pairs.

Raw results: [27 pairs](ufm_batches.json), [consumer cross-check](ufm_consumer_check.json), [causal sequence](ufm_trajectory.json).

## Correctness checks

The regression suite compares against the unchanged production implementation, not constants generated by the candidate. Coverage includes:

- Exact query distances and indices, duplicates, empty and noncontiguous batches, ordering, simultaneous callers and closed-pool behavior.
- Both recorded ring and held-dowel fits; original, missing, nearer, deeper and mixed depth; moved/rotated poses and image boundaries.
- Complete fits and derivative batches with compiled calculations, pooled searches and concurrent candidates; solver decisions and pose matrices must agree.
- The shared pipeline through motion, loss and recovery, including timestamps, evidence and masks. A positive control verifies that the object actually moved.
- UFM object ownership/order, empty batches, mismatched input/output cardinality, and the recorded non-equivalence case through real depth sampling and fitting.

The compiled tests explicitly require Numba and skip if it is absent. The recorded verification used the isolated environment with Numba installed, not the base project environment. See `verification.json` for the test result and deliberate mutation checks.

## Reproduction

Measured on Ryzen 9950X (16 physical cores / 32 logical CPUs) and RTX 5090. NumPy 2.2.6, SciPy 1.17.1, Torch 2.10.0+cu128, Numba 0.68.0, llvmlite 0.50.0. BLAS/OpenMP and OpenCV were limited to one thread; candidate search/candidate pools are explicit. Tests and other benchmark jobs were not run concurrently with timing measurements.

The experiment environment `/tmp/flash-native-benchmark-venv` inherits the existing live UFM environment and adds Numba with the **same NumPy version**. For a fresh environment, install the live tracker's dependencies first, then `numba==0.68.0 numpy==2.2.6`; do not silently change NumPy while comparing implementations. `PYTHONPATH=src`, `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` were set on every run.

From the repository root, using that Python:

```bash
python docs/proofs/flash-native-acceleration/benchmark_fits.py --repeats 3 --output /tmp/fits.json
python docs/proofs/flash-native-acceleration/benchmark_pipeline.py --trace ~/.cache/huggingface/lerobot/flash_debug_traces/trace-20261025-192743-253063024 --frames 30 --repeats 2 --output /tmp/three.json
python docs/proofs/flash-native-acceleration/benchmark_pipeline.py --trace ~/.cache/huggingface/lerobot/flash_debug_traces/trace-20261025-181728-660815692 --start 0 --frames 200 --repeats 1 --modes baseline compiled compiled_pool --output /tmp/hand.json
python docs/proofs/flash-native-acceleration/benchmark_pipeline.py --frames 713 --repeats 1 --modes baseline compiled compiled_pool --output /tmp/demo.json
python docs/proofs/flash-native-acceleration/benchmark_ufm.py --repeats 3 --output /tmp/ufm.json
python docs/proofs/flash-native-acceleration/check_ufm_trajectory.py --trace ~/.cache/huggingface/lerobot/flash_debug_traces/trace-20261025-192743-253063024 --frames 30 --output /tmp/trajectory.json
python -m pytest -q tests/policies/flash/test_native_acceleration_candidates.py tests/policies/flash/test_surface_acceleration.py tests/policies/flash/test_vision_pipeline.py tests/policies/flash/test_surface_sampling.py tests/policies/flash/test_surface_debug.py tests/policies/flash/test_flash_perception_worker.py tests/policies/flash/test_flash_offline_perception.py tests/policies/flash/test_flash_perception_service.py
```

The full recordings and cached checkpoint remain local. The committed numerical fixtures support hardware-free regression tests, but do not contain the RGB needed to reproduce the neural benchmark on another machine. `benchmark_pipeline.py` with no `--trace` reads the insertion demo and saved full-demo SAM masks documented in its loader. No experiment modifies a recording, GUI configuration or live runtime default.

Upstream references: [SciPy worker lifecycle](https://github.com/scipy/scipy/blob/v1.17.1/scipy/spatial/_ckdtree.pyx), [Numba performance guidance](https://numba.readthedocs.io/en/stable/user/performance-tips.html), [UFM batched API](https://github.com/UniFlowMatch/UFM/blob/main/uniflowmatch/models/base.py). This experiment does not evaluate Ceres or small_gicp replacements: changing the objective or optimizer would need a separate accuracy comparison.
