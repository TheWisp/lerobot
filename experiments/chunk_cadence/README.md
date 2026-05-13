# Chunk Cadence Backtest

Measures **action-to-state latency** and **command-stream jitter** of the SO-107 predictive-lookahead controller when driven by chunked actions, simulating how a real-time-chunked policy (RTC / pi_0 / SmolVLA-style) would behave.

The experiment was built to answer one question:

> **Is chunk-aware predictive lookahead worth the implementation cost?** I.e. does reading L ms ahead into a known action chunk actually deliver L ms of latency reduction, and what does it cost in jitter?

## Scripts

| File                      | Purpose                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `backtest.py`             | Runs one cell: connects to a real predictive follower, sends action chunks at a configured cadence and bias, logs per-tick command + state. |
| `analyze.py`              | Per-trace metrics: lag (state vs motor_cmd, per-segment Pearson xcorr), jitter (Savitzky-Golay residual std), chunk-boundary jumps.         |
| `summarize.py`            | Cross-cadence table for a single replicate.                                                                                                 |
| `summarize_replicates.py` | Aggregates mean ± std across multiple replicate runs. Writes optional JSON snapshot.                                                        |
| `run_sweep.sh`            | Driver: runs every `(source, update_every, lookahead) ∈ {N×L}` combination for one replicate. Idempotent (skips already-finished cells).    |

## Methodology — read first

**Sources.** Two trajectory inputs, both 14-DOF bi_so107:

- A safe-trajectory JSON: smooth pre-recorded motion, ~13 s. Used as the calibration trajectory because it's deterministic and reproducible.
- A dataset episode (`thewisp/cylinder_ring_assembly` ep. 287, ~20 s): realistic task motion with more dynamics.

**Chunked emission.** At each emission `i = 0, 1, 2, …` we slice `chunk_i = gt[k_i : k_i + 50]` where `k_i = i × update_every` (in source frames). Emissions happen every `update_every / chunk_fps` seconds of wall time. Default chunk size is 50, default `update_every ∈ {2, 4, 8, 16}`.

**Bias-on-overlap model.** Each chunk gets a small per-emission bias added to its overlap region — frames `[update_every, 50)` — but the first `update_every` frames are clean ground truth. This represents how a real RTC policy would emit slightly different chunks for the same wall-time range (consecutive chunks disagree on the overlap; the leading edge of each chunk is "what this chunk uniquely owns until the next one arrives"). Bias evolves as a random walk capped at `bias_threshold_deg` (default 0.5).

**Two configs per source × cadence.** Each `(source, update_every)` runs twice — once with `--lookahead-ms 0` (no compensation: motor reads chunk[0]) and once with `--lookahead-ms 80` (motor reads chunk[+80 ms]). Both use the same `--bias-seed 42`, so the chunk streams are bit-identical between the two runs; the only thing that differs is what the motor was commanded.

**Action-to-state latency.** Per-joint cross-correlation lag between the motor command and the observed state. The L=0 run measures **τ_motor** directly (state trails motor by its natural response time). The L=80 run measures the **residual lag** — how much state still trails the trajectory despite the lookahead. The difference `τ_motor − residual` is the effective compensation delivered by the lookahead.

**Jitter.** Per-config, deviation from each command stream's own Savitzky-Golay (window=15, polyorder=3) smoothed version. This isolates HF noise around the smooth trend from the (intentional) time-shift the lookahead introduces.

## Reproducibility recipe

These all matter:

1. **Adaptive xcorr off.** `backtest.py` forces `adaptive=False` and pins `max_lookahead_ms` to the configured `lookahead_ms` so the controller's internal L stays fixed at exactly what we ask for. With adaptive on, L drifts during the run and post-hoc-computed motor_cmd doesn't match what actually drove the motor.
2. **Rest-to-rest framing.** Each run moves the robot to rest position before AND after the chunk loop. Back-to-back runs are initial-condition-identical (no thermal / friction history from a previous run's end pose biasing the comparison).
3. **Same bias seed across paired runs.** `--bias-seed 42` is the default. Two runs with the same seed produce bit-identical chunk streams; the only difference is the motor's response.
4. **Multiple replicates.** Run `run_sweep.sh 1`, `run_sweep.sh 2`, `run_sweep.sh 3` for cross-run reproducibility. Replicates of the safe trajectory at this seed agree on τ_motor / residual to < 0.01 ms — the system is fully deterministic once adaptive is off and the bias seed is fixed.

## Non-obvious gotchas (real bugs we hit)

The analyzer's `_xcorr_lag` had a subtle bug for two iterations before getting right:

1. **Naive `scipy.signal.correlate / N` normalization.** Returns raw dot products; at non-zero lags, only `N − |k|` samples contribute but the normalization is still by `N`, so non-zero-lag correlations look artificially smaller → peak biased toward lag=0.
2. **Global-mean-subtraction with trending signals.** Even after subtracting the global mean, a monotonically descending trajectory still dominates the inner product; the small time-shift barely budges the peak. With this method, a 100 ms motor τ rounded to 0 ms.

**Fix in place:** per-segment Pearson correlation at each candidate lag. Use each lag's overlap region as its own sample (local mean, local std, local inner product). The temporal alignment of the SHAPE wins over the trend amplitude. See `analyze._xcorr_lag` docstring.

Other settings that bite:

- **Motion-std filter for lag aggregation.** Below ~20 deg state std per joint, the cross-correlation peak is noise-dominated. We filter to high-motion joints (>20 deg) before taking the median.
- **xcorr resolution is 33 ms (1 source frame).** Sub-frame lags round. For finer resolution, upsample state to 100 Hz before xcorr — see "Future work" below.

## Findings (snapshot: 2026-05-13)

Three replicate runs across `update_every ∈ {2, 4, 8, 16}` × `lookahead ∈ {0, 80}` for both sources. Full numbers in [`results_2026-05-13.json`](results_2026-05-13.json).

| Source                         | τ_motor (L=0)       | Residual lag (L=80) | Compensation delivered    |
| ------------------------------ | ------------------- | ------------------- | ------------------------- |
| Safe trajectory                | **100.2 ± 0.01 ms** | 33.4 ± 0.00 ms      | **66.8 ms** (84% of L=80) |
| Cylinder ring assembly ep. 287 | **116.9 ± 0.01 ms** | 50.1 ± 0.00 ms      | **66.8 ms** (84% of L=80) |

- **τ_motor is independent of cadence** (same motor, same dynamics). The two trajectories produce slightly different τ values because different load regimes (joint poses, gravity).
- **Lookahead delivers a fixed ~67 ms absolute lead, regardless of trajectory** and regardless of cadence in N ∈ {2..16}. The "missing" 13 ms (= 80 − 67) is consumed by something outside the lookahead itself — likely fixed bus / encoder / controller-tick stack latency.
- **Jitter cost varies with cadence (peaks at N=4-8)**, but ranges from 0.01 to 0.09 deg, well under 1 deg in absolute terms.
- **A no-bias control run** (`bias_threshold_deg=0`) shows zero jitter cost from the lookahead path — the jitter is entirely a property of the bias model representing inter-chunk policy drift, not inherent to lookahead reading into known chunks.

The proof of concept is clean: **chunk-aware predictive lookahead is essentially free in jitter when chunks are exact**, and delivers a substantial latency reduction (~67 ms of 100 ms eliminated) under realistic chunked-policy emission cadences.

## How to reproduce

```bash
# Assumes:
#   - Predictive follower hardware at ~/.config/lerobot/robots/white_pred.json
#   - Safe trajectory recorded at ~/.config/lerobot/robots/white.trajectory.json
#   - Local copy of cylinder_ring_assembly dataset (or change CYLINDER_SOURCE)
#
# Override these via env vars if your setup differs:
#   ROBOT_PROFILE / SAFE_SOURCE / CYLINDER_SOURCE / VENV

./run_sweep.sh 1
./run_sweep.sh 2
./run_sweep.sh 3

# Aggregate across replicates
./summarize_replicates.py outputs/chunk_cadence \
  --output-json results_$(date +%F).json
```

Each replicate runs the full sweep (16 cells, ~7-8 minutes on bi_so107). The script skips any cell whose `.npz` already exists, so partial runs can be resumed.

## Future work hooks

When iterating on RTC (or any chunk-emitting policy), this infrastructure picks up cleanly:

- **Swap the bias model.** Edit `make_chunk()` in `backtest.py` to model a different drift pattern. The "first-N-clean, rest-biased" pattern is one possibility; uniform per-emission noise, growing-with-frame-index noise, or noise from an actual policy's per-chunk samples are all candidates.
- **Test new lookahead values.** Run with `--lookahead-ms` ∈ {40, 60, 100, 120} to find the sweet spot for a given motor.
- **Different chunk sizes.** `--chunk-size` defaults to 50; pi_0 uses 50, ACT uses up to 100.
- **Different trajectories.** Pass a dataset episode via `<repo_id>@<ep_idx>` or a safe-trajectory JSON path.
- **Sub-frame lag resolution.** Upsample state to 100 Hz before xcorr to resolve sub-frame motor τ.
- **Side-by-side bias models on the same trajectory.** Add a `--bias-model {overlap_only|uniform_per_chunk|growing}` flag and refactor `make_chunk` into named implementations.

## Files in `outputs/`

Raw `.npz` traces are written to `outputs/chunk_cadence/<source>_run<N>/trace_*.npz` plus `meta_*.json` sidecars. These are NOT committed (large, regenerable). A summary JSON IS committed (`results_<date>.json`, ~10 KB per run set).

Raw schema (per trace):

- `t` — wall-time of each tick relative to chunk-loop start
- `gt` — ground-truth trajectory sampled at this tick's wall time
- `motor_cmd_lookahead` — what the controller would output at the configured L (= actually what was sent, in lookahead runs)
- `motor_cmd_no_lookahead` — post-hoc: what the controller would output at L=0
- `state` — observed motor position (`robot.get_observation()` per tick)
- `bias` — current per-chunk bias values (informational)
- `emission_idx` — which chunk was active at this tick
- `joint_names` — joint name order
