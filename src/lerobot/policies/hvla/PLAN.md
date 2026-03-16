# Plan: Hierarchical VLA — Migrate into LeRobot

## Context

We have a working dual-system VLA split across two codebases (LeRobot + OpenPI) with WebSocket IPC. Goal: consolidate into a standalone LeRobot module with shared-memory IPC, stripping S2 down to VLM-only (no action expert — S1 is the action policy).

---

## Module Structure

```
src/lerobot/policies/hvla/
├── __init__.py
├── s2/
│   ├── __init__.py
│   ├── config.py                 # S2 VLM config (PaliGemma dims, no action expert)
│   ├── model.py                  # VLM-only: embed_prefix, extract_latent, AR subtask decode
│   ├── paligemma.py              # PaliGemma forward, deembed, KV cache (no gemma_expert)
│   ├── preprocessing.py          # Image resize/normalize, prompt tokenization
│   ├── fast_tokenizer.py         # FAST token wrapper (training loss only)
│   └── train.py                  # S2 training: subtask + FAST cross-entropy losses
├── s1/
│   ├── __init__.py
│   └── (re-export ACTWithVLMPolicy or future replacement)
├── ipc.py                        # SharedLatentCache: CPU shared memory + metadata
├── s2_process.py                 # S2 process entry: load VLM, extraction loop → shared mem
├── s1_process.py                 # S1 process entry: load S1, read shared mem, robot loop
└── launch.py                     # Spawn S1 + S2 processes via torch.multiprocessing
```

---

## Design Decisions

### Latent dimension: 2048 (configurable)

| System | Latent dim | Projection |
|--------|-----------|------------|
| OpenHelix | 512 | Linear from 4096 |
| Dual Process VLA | 4096 | MLP |
| RoboDual | 256 + 8 tokens | Perceiver |
| **Ours** | **2048** | MLP: 2048→1024→512 in S1 |

Keep 2048 (PaliGemma hidden dim). Add configurable `shared_latent_dim` with optional learned projection in S2. S1's MLP projector handles the final mapping to its model dim.

### Shared memory: CPU (`/dev/shm`)

- `torch.Tensor.share_memory_()` — zero-copy between processes via `/dev/shm`
- [2048] float32 + age scalar = 8KB. GPU→CPU copy in S2 + CPU→GPU copy in S1 < 0.1ms total
- Each process manages its own CUDA context independently — no GPU contention

```python
class SharedLatentCache:
    def __init__(self, latent_dim=2048):
        self.latent = torch.zeros(latent_dim).share_memory_()
        self.timestamp = torch.zeros(1, dtype=torch.float64).share_memory_()  # wall-clock seconds
        self.update_count = torch.zeros(1, dtype=torch.long).share_memory_()
        self.ready = multiprocessing.Event()  # S1 waits for first latent

    def write(self, latent: torch.Tensor):
        self.latent.copy_(latent.cpu())
        self.timestamp[0] = time.time()
        self.update_count += 1
        self.ready.set()

    def read(self) -> tuple[torch.Tensor, float]:
        return self.latent.clone(), self.timestamp[0].item()

    @property
    def age_ms(self) -> float:
        return (time.time() - self.timestamp[0].item()) * 1000
```

### Latent age as S1 input

Age is encoded via a separate path and **added** (not concatenated) to avoid awkward dims:
- `age_embedding = MLP(1 → 64 → 512)` projects age scalar to S1's model dim
- S1 projector stays `2048 → 1024 → 512` (no dim change, no alignment issues)
- `s2_token = s2_projector(latent) + age_embedding(age_seconds)` — additive fusion
- No 2049 dimension; tensor alignment preserved

**Same scalar in training and inference:**
- Training (delay augmentation): `age_seconds = k / dataset_fps` where k is the simulated frame delay
- Inference: `age_seconds = SharedLatentCache.age_ms / 1000.0`
- Both feed the identical `age_embedding` MLP path

### S1 training: pre-extract latents + delay augmentation

**Pre-extraction** remains the right approach:
- S2 inference (50ms/frame) on the same GPU as S1 training causes contention
- DataLoader workers can't efficiently share a GPU model
- 186k frames × 50ms = ~2.5 hours once, produces 1.5GB .npy file
- Extraction script now imports from `hvla.s2` locally (no server, no OpenPI)

**Delay augmentation** (new, critical for inference robustness):
- During S1 training, replace `latent[frame_idx]` with `latent[frame_idx - k]`
- `k ~ Uniform(0, MAX_DELAY_FRAMES)`, where `MAX_DELAY_FRAMES = ceil(0.5s × dataset_fps)`
- At 30fps: k ∈ [0, 15] — simulates 0-500ms S2 staleness
- Clip at episode boundaries
- Pure data augmentation — zero infrastructure cost

**Trade-offs:**
- k=0 only (current): overfits to perfect sync, degrades when S2 is stale
- k ∈ [0, 15] (500ms): robust to typical S2 latency, slight training noise
- k ∈ [0, 45] (1.5s): too noisy, S1 may learn to ignore latent

### S2 checkpoint compatibility

**Yes** — VLM-only model loads from existing Pi0.5 safetensors checkpoint. Filter out `gemma_expert.*` keys at load time. PaliGemma (SigLIP + Gemma 2B) weights are fully compatible. Saves ~40% GPU memory by not loading action expert.

### LoRA for S2 training

Apply LoRA (via `peft` or manual) to both PaliGemma Gemma 2B AND SigLIP:
- **Gemma 2B targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (72 adapters across 18 layers)
- **SigLIP targets**: attention QKV + output projections (27 encoder layers)
- **Rank: 32 (recommended)**. Rank 16 (~8M params) may underfit; rank 64 (~32M) has diminishing returns and 2× memory. Rank 32 with both VLM + vision ≈ ~24M trainable params, ~90MB.
- **Vision tower IS finetuned** (based on prior training results showing it's necessary). LoRA on SigLIP keeps memory manageable vs full finetuning.
- Dominant memory cost is the frozen base model (~5GB). Gradient checkpointing recommended for activation memory.
- Only subtask + FAST losses backprop through LoRA adapters

### Action chunking and timing

**Current S1**: chunk_size=100, 30Hz target, actual ~22Hz (45-50ms/step)

**For ≥30Hz with ~40ms inference:**
- `n_action_steps=2`: query at 15Hz, execute at 30Hz. 66ms budget - 40ms inference = 26ms slack.
- Temporal ensembling (coeff=0.1-0.15): query every step, blend overlapping chunks. Needs <33ms inference.
- **Action time-shift**: For open-loop chunk execution (`n_action_steps > 1`), compensate for prep+inference delay by skipping into the chunk. Timer starts **after** `get_observation()` (not before), because the model was trained on data with the same camera latency baked in — it already predicts actions calibrated to that. Only the additional prep+inference time needs compensation:
  ```python
  obs = robot.get_observation()
  t_after_obs = time.perf_counter()  # start HERE, not before get_observation
  actions = policy.predict_action_chunk(prep(obs))
  dt = time.perf_counter() - t_after_obs  # prep + inference only
  skip = min(int(dt * fps), len(actions) - 1)
  action = actions[skip]  # corresponds to "now"
  ```
  Not needed with temporal ensembling (re-queries every step, ensembler handles staleness).
- **Horizon**: 50 actions (1.67s at 30Hz) is sufficient. With ensembling, tail actions have near-zero weight.

### Robot control loop

**Yes, exists in `dual_system_infer.py`:**
- 30 FPS target with `time.sleep(max(0, 1/fps - elapsed))`
- Per-step: obs capture (2ms) → S1 prep (10-24ms) → S1 infer (25ms) → send action (0.3ms)
- S2 runs async in background, updates shared cache

**Prep bottleneck (10-24ms) — root cause analysis:**

Main cost: `obs_to_s1_batch()` transfers 42MB/step (4× 720×1280×3 float32) CPU→GPU, then resizes on GPU.

Per image (lines 234-238 of dual_system_infer.py):
1. `.float() / 255.0` → allocates 10.5MB float32 on CPU
2. `.to(device)` → 10.5MB CPU→GPU PCIe transfer
3. `F.interpolate()` → GPU resize 720×1280 → 224×224

4 cameras × 10.5MB = 42MB. At PCIe 3.0 x16 (~12GB/s): ~3.5ms just for transfer.

Variance (10-24ms) from GPU scheduling contention when S2 runs on same GPU.

**Fix (implement before migration):**
1. Resize on CPU first: `torchvision.transforms.functional.resize()` before `.to(device)`. 224×224×3 float32 = 0.6MB per image → 2.4MB total (96% less transfer).
2. Or: transfer uint8 to GPU (2.6MB total), convert + resize on GPU.
3. Move normalizer stats to GPU at init (one-time), not per-call.
4. Skip `DeviceProcessorStep` (tensors already on device from obs_to_s1_batch).
5. Pre-allocate GPU batch tensors and reuse across steps (avoid per-step allocation).

Expected improvement: prep from 10-24ms → 3-5ms. Measured: steady-state ~6ms, spikes to ~15ms from CPU cache contention when S2 thread is active.

**Shared image preprocessing pipeline:**
S1 and S2 both resize the same camera images (720×1280 → 224×224). Currently each does it independently — S1 in `obs_to_s1_batch()`, S2 on the server side. With both in the same codebase, the resize should happen **once** in a shared preprocessing step, and both S1 and S2 read from the resized result. This eliminates redundant CPU work (~5ms × 2 = 10ms saved per frame) and reduces memory pressure from duplicate buffers.

**GPU priority for S1 (latency-critical):**

When S1 and S2 share a GPU, S2's 230ms prefix forward can cause S1 inference spikes. Options:
1. **CUDA MPS** (recommended): `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70` for S1 process gives it 70% of SMs. S2 gets 30%. Standard multi-process GPU sharing solution.
2. **CUDA stream priorities**: Only works within same process (not applicable with separate processes).
3. **Separate GPUs**: Simplest but requires 2 GPUs.

Start with CUDA MPS. The launch script should start `nvidia-cuda-mps-control` and set thread percentages per process.

**Robot hardware (SO107 bimanual, Feetech STS3215):**
- Serial at 1MHz baud. `sync_write("Goal_Position")` is fire-and-forget, non-blocking (microseconds)
- Each motor has internal PID (P=16, I=0, D=32). Host sends goal positions; motors interpolate internally.
- `Return_Delay_Time=0`, `Acceleration=254` (maximum). No host-side interpolation or queueing.
- **30Hz is a software choice, not hardware limit.** Motors can accept commands much faster. Bottleneck is camera fps + inference time.

---

## S2 Training (VLM-only)

**No action expert.** S2 training = VLM finetuning with two losses:

1. **Subtask cross-entropy** (weight=10.0): AR-decoded subtask vs ground truth
2. **FAST token cross-entropy** (weight=1.0): discrete action tokens vs FAST-encoded GT actions

The FAST loss teaches the VLM to understand action semantics without generating continuous actions. The subtask loss teaches scene understanding and task decomposition.

Dataset needs: per-frame `high_level_task` + `low_level_subtask` annotations.

---

## S2 Weight Loading (VLM-only from Pi0.5)

```python
def load_vlm_only(checkpoint_path, config):
    state_dict = safetensors.torch.load_file(checkpoint_path)
    vlm_keys = {k: v for k, v in state_dict.items()
                if not any(k.startswith(p) for p in [
                    "gemma_expert", "action_in_proj", "action_out_proj",
                    "time_mlp", "state_proj"
                ])}
    model = S2VLMModel(config)
    model.load_state_dict(vlm_keys, strict=False)
    return model
```

---

## Port from OpenPI (VLM parts only)

| OpenPI source | What to port | Skip |
|---------------|-------------|------|
| `pi0_pytorch.py` | embed_prefix, extract_prefix_latent, sample_low_level_task | embed_suffix, denoise_step, sample_actions, forward (flow matching) |
| `gemma_pytorch.py` | PaliGemma forward, deembed, embed_image/language, KV cache | gemma_expert, dual-branch forward |
| `preprocessing_pytorch.py` | Image preprocess, tokenization | Action normalization |
| `gemma_config_lite.py` | PaliGemma config | Action expert config |
| `tokenizer.py` | FAST encode/decode | — |

---

## Implementation Sequence

0. **Prep optimization** — fix image transfer bottleneck in dual_system_infer.py (benefits all policies now)
1. **S2 config + VLM model** — PaliGemma-only model, embed_prefix, extract_latent, AR decode
2. **S2 preprocessing + FAST tokenizer** — image pipeline, prompt format
3. **IPC** — SharedLatentCache with CPU shared memory
4. **S2 process** — load VLM, extraction loop, write to shared memory
5. **S1 process** — read shared memory, robot control loop (adapt from dual_system_infer.py)
6. **Launch script** — spawn both processes
7. **S2 training** — subtask + FAST loss, LoRA on SigLIP + Gemma 2B (rank 32)
8. **S1 training** — adapt extract_s2_latents.py to use hvla.s2, add delay augmentation + age embedding
9. **Latent extraction script** — local, no server
10. **Rename** `docs/plans/dual_system_vla_plan.md` → `_legacy.md`, copy final plan to `docs/plans/hvla_migration_plan.md`

Step 0 first (immediate benefit). Steps 1-2 parallel. Step 3 independent. Steps 4-6 depend on 1+3. Step 7 depends on 1-2. Steps 8-9 depend on 1.

## Future: Training-Time RTC + Flow Matching S1

**Training-time RTC** ([arxiv 2512.05964](https://arxiv.org/abs/2512.05964)): Condition chunk generation on the unexecuted tail of the previous chunk as a prefix. During training, simulate inference delay and provide ground-truth prefix. During inference, feed previous chunk's tail — model generates smooth continuation. Zero blending overhead.

**Flow matching for S1**: Consider replacing ACT (CVAE, single-pass) with a small flow-matching policy for S1. Benefits:
- Native RTC prefix guidance during denoising (no architectural hacks)
- Training-time RTC is straightforward with tokenwise flow matching conditioning
- Better multi-modal action distributions (CVAE mode-averages)
- Consistent with S2 architecture (both PaliGemma-family)
- LeRobot's `pi05` policy already has flow matching infrastructure to build on

The HVLA S1 slot is swappable by design (`hvla/s1/__init__.py`). The S2 latent + age embedding interface stays the same regardless of S1 architecture.

---

## Verification

- [ ] S2 loads VLM-only from Pi0.5 checkpoint (~40% less GPU memory)
- [ ] `extract_prefix_latent()` matches OpenPI output within tolerance
- [ ] `sample_low_level_task()` decodes coherent subtask text
- [ ] SharedLatentCache: write from one process, read from another, <0.1ms
- [ ] Dual inference: S1 ≥20Hz, S2 ~4-15Hz, shared memory updates visible to S1
- [ ] S2 training: subtask + FAST loss decreases
- [ ] S1 training with delay augmentation: model robust to stale latents
