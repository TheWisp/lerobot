# Online flash-DAgger (v0 prototype)

**Online LoRA correction adapter for HVLA S1.** Captures operator interventions
during a teleop session and runs a synchronous LoRA fit at episode end. The
fitted LoRA is hot-swapped into the live policy. The frozen base means a bad
fit can be peeled off without losing the original policy.

This is the on-robot counterpart to the offline phase A–F experiments at
[research/flash_dagger/SUMMARY.md](../research/flash_dagger/SUMMARY.md).

## Status: v0 prototype

Run-ready end-to-end, but with sharp edges:

- HVLA-only (S1 flow-matching). ACT/PI0 generalization is a future PR.
- Decoder-only LoRA — encoder is never fine-tuned (see "Cached context"
  below). Means novel visual features can't be learned online.
- Single continuous LoRA per session (Phase D recipe). No per-correction
  stack / dispatch.
- Synchronous fit at episode end. Async / chunk-boundary swap is v1.
- GUI: an "Online Training" section in the run-policy panel
  (`src/lerobot/gui/api/run.py` + `src/lerobot/gui/static/run.js`). Metrics
  go to JSONL + CSV under `output_dir` — no live dashboard yet.

## Quick start

### Smoke test (offline, no robot)

Replays existing eval episodes through the system as if they were
interventions. Validates the integration without hardware:

```bash
python -m lerobot.policies.hvla.flash_dagger.smoke \
    --checkpoint outputs/flow_s1_no_s2_merged_raw/checkpoints/checkpoint-50000 \
    --eval-repo-id eval/eval_cylinder_ring_assembly_apr_24 \
    --train-repo-id thewisp/cylinder_ring_assembly_merged_raw \
    --episodes 247 76 \
    --output-dir outputs/flash_dagger_smoke \
    --steps 30
```

Outputs in `outputs/flash_dagger_smoke/`:

- `summary.jsonl` — one row per fit cycle (Layer A losses, swap decision)
- `curves/cycle_NNNN.csv` — per-step training loss
- `layer_diag/cycle_NNNN.csv` — per-LoRA-layer ‖BA‖ + effective rank
- `lora/cycle_NNNN.pt` — saved LoRA + metadata (base hash bound)
- `lora/latest.pt` — pointer to most recent

### On-robot session

**GUI (recommended):** open the Run tab, configure the HVLA policy as usual,
then expand the "Online Training" section, check Enable, fill in
Train Repo ID (the dataset feeding the "old" replay slot), and launch. Other
fields default to the validated phase-D recipe (rank=16, 100 steps, 10/25/65
mix).

**CLI equivalent** — append these flags to whatever `s1_process` invocation
the GUI builds:

```bash
... existing s1_process args ... \
    --flash-dagger-mode \
    --flash-dagger-train-repo-id thewisp/cylinder_ring_assembly_merged_raw \
    --flash-dagger-output-dir outputs/flash_dagger_online \
    --flash-dagger-rank 16 \
    --flash-dagger-steps 100
```

Operator UX (no new keys for v0): SPACE-key intervention works as today.
After each intervention episode ends, the system pauses ~30–60 s for the
fit, swaps LoRA, and the next rollout uses the updated policy.

## Module layout

| File             | Purpose                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| `config.py`      | `FlashDaggerConfig` — rank, mix ratios, steps, etc.                     |
| `lora.py`        | LoRA module + base-hash binding + merge / peel / diagnostics            |
| `buffer.py`      | `InterventionFrameBuffer` (per-episode), `FlashedEpisodePool` (session) |
| `mix.py`         | `ThreeWayMixDataset` — generalized 10/25/65 sampler                     |
| `fitter.py`      | `fit_step_loop`, `evaluate_loss`, `InterventionChunkDataset`            |
| `metrics.py`     | Layer A + Layer B logging (JSONL + CSV)                                 |
| `persistence.py` | LoRA save/load with hash verification                                   |
| `system.py`      | `FlashDaggerSystem` — orchestrator with lifecycle hooks                 |
| `smoke.py`       | End-to-end offline smoke test                                           |

## Lifecycle hooks (called by `s1_process.py`)

| Hook                          | When               | What it does                                    |
| ----------------------------- | ------------------ | ----------------------------------------------- |
| `on_intervention_start()`     | SPACE pressed      | Begin frame capture                             |
| `on_tick(obs, action)`        | Per main-loop tick | Append to buffer iff intervention active        |
| `on_intervention_end()`       | SPACE released     | Stop capture; keep frames buffered              |
| `on_episode_end(ep, success)` | Episode boundary   | Trigger fit if buffer non-empty + success       |
| `shutdown()`                  | Session end        | Final log line; per-cycle saves already on disk |

## Safety mechanisms

- **Frozen base** — only LoRA params have `requires_grad`; original policy weights never overwritten.
- **Forget tripwire** — at end of fit, `loss_old_val` (held-out training-set sample) drift % is checked against `forget_drift_abort_pct` (default 50%). On exceed, the cycle's LoRA is reverted via `load_lora_state_dict` and the swap is rejected. Logged to `summary.jsonl` with `swap_accepted=false`.
- **Base-hash binding** — every saved LoRA carries SHA-256 of the non-LoRA params. `load_lora` refuses to load against a different base. Prevents silently composing a LoRA with the wrong checkpoint.
- **Min frames** — `min_intervention_frames` (default 50) gates very short interventions; below this the fit is skipped.

## Multi-intervention episodes

Multiple interventions in a single episode each open their own buffer
_segment_. At episode end the chunked dataset builds sliding-window chunks
**within** each segment — never across — so a chunk can't pair an obs from
intervention A with actions from intervention B (which would be a different
time/scene and corrupt the training signal).

Train/val split is per-segment temporal: the last `val_pct` of each segment
is held out as val (contiguous tail), the rest is train. Chunks are valid
only if they fit within their segment (start + chunk_size ≤ segment length).

## Architecture: cached context (decoder-only LoRA)

Phases B–F validated LoRA on the S1 _decoder_ only — DINOv2 + obs encoder
stay frozen. We exploit that frozen-encoder property to skip wasted
re-encoding every fit step:

- **Pre-encode at startup**: the replay-pool (5000 frames) and forget-val
  (500 frames) are run through `encode_observations` once at session start
  and stored as `(context, action_chunk, action_is_pad)` samples on CPU.
  Cost: ~30 s. Memory: low GBs of context tensors.
- **Encode live captures at cycle start**: each intervention tick's raw
  obs is encoded once via `obs_to_s1_batch` → `prepare_batch_for_encode_observations`
  → `encode_observations`. Cost scales with intervention length only.
- **Fit cycle**: every step is decoder-only — `compute_per_sample_loss_from_context`
  takes pre-encoded context and runs `denoise_step + flow MSE`. No DINOv2
  forward in the inner loop. Roughly 5–10× faster per step than re-encoding.

**Limitation:** this design forecloses fine-tuning the encoder. If a future
correction requires new visual features (e.g. a novel object the encoder has
never seen), the cached-context path can't represent it — the LoRA only
modifies the decoder's mapping from context to actions. Reverting to
re-encode-each-step would re-enable encoder fine-tuning at a heavy compute
cost. Phases B–F validated decoder-only adaptation as sufficient for the
correction regimes flash-DAgger targets.

## Known v0 caveats / TODOs

1. **No live visualization** — operator just sees stdout logs during the fit. A GUI panel is left to a follow-up.
2. **Single continuous LoRA** — no rollback granularity; if a bad correction lands and isn't caught by the tripwire, the operator must restart the session or peel via offline tools.
3. **`compute_per_sample_loss_from_context` import is HVLA-specific** — generalizing to ACT/PI0 means swapping that out for a per-policy `loss_fn` factory.
4. **`peft` library not used** — we have our own LoRA in `scripts/flash_dagger_lora.py` (validated through phases A–F). Migrating to `peft` is a future cleanup.
5. **Tail-of-segment frames waste capacity** — a segment of length L contributes only `L - chunk_size + 1` valid sliding-window chunks; the last `chunk_size - 1` frames can't be a chunk start (their actions would extend past segment end). Capturing for ~`chunk_size` more ticks _after_ intervention ends would let every in-intervention starting frame produce a valid chunk. Open question: those post-intervention actions are policy or held-leader, not active demos — may or may not be acceptable training targets.
6. **Decoder-only adaptation** (see "Architecture: cached context" above) — the encoder is frozen and pre-encoded contexts are reused across fit steps. Reverting to re-encode-each-step would re-enable encoder LoRA at significant compute cost.

## See also

- [research/flash_dagger/SUMMARY.md](../research/flash_dagger/SUMMARY.md) — offline phases A–F (recipe, data, plots)
- [scripts/flash_dagger_phase_d_rehearsal.py](../scripts/flash_dagger_phase_d_rehearsal.py) — original phase-D driver (the recipe online flash-DAgger ports)
- [scripts/flash_dagger_lora.py](../scripts/flash_dagger_lora.py) — LoRA module (lifted as-is into this package)
