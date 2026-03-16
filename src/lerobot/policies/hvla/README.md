# Hierarchical VLA (HVLA)

A dual-system VLA for bimanual robot control. S2 (VLM) provides scene understanding, S1 (action policy) generates actions conditioned on S2's latent.

Inspired by [Helix](https://www.figure.ai/news/helix), [OpenHelix](https://arxiv.org/abs/2505.03912), and [Dual Process VLA](https://arxiv.org/abs/2410.15549).

```
S2 Process (~4-15Hz)                shared memory           S1 Process (~22-30Hz)
┌──────────────────────┐           ┌───────────┐          ┌───────────────────────────┐
│ 4 cameras → SigLIP   │           │ [2048]    │          │ 2-4 cameras → DINOv2/ResNet│
│ task text → Gemma 2B │──latent──→│ + age(s)  │←──read──│ state → state token        │
│ mean-pool → [2048]   │           └───────────┘          │ latent + age → S2 token    │
│ (VLM-only, no action │                                  │ → ACT encoder/decoder      │
│  expert)             │                                  │ → action chunk (14-DOF)    │
└──────────────────────┘                                  └───────────────────────────┘
```

## Quick Start

### 1. Extract S2 latents (offline, one-time)

No OpenPI dependency — uses the HVLA S2 model directly:

```bash
python scripts/extract_s2_latents_hvla.py \
    --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --dataset thewisp/cylinder_ring_assembly \
    --prompt "assemble cylinder into ring" \
    --output ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents.npy \
    --image-keys observation.images.front,observation.images.top,observation.images.left_wrist,observation.images.right_wrist \
    --batch-size 8
```

Produces `s2_latents.npy` shape `[N_frames, 2048]`. ~81ms/frame on GPU.

### 2a. Train S1 — ACT (default, CVAE)

```bash
python scripts/train_act_vlm.py \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents_pt_11997.npy \
    --output-dir outputs/act_vlm_hvla \
    --steps 100000 \
    --batch-size 16 \
    --save-freq 20000 \
    --use-dino-backbone \
    --resize-images 224x224 \
    --num-workers 8 \
    --max-delay 0.15
```

`--max-delay 0.15` enables delay augmentation: each training sample randomly shifts the S2 latent backward by 0-150ms (0-5 frames at 30fps), within episode boundaries. The age (seconds of staleness) is injected as a learned embedding — same scalar used at inference.

### 2b. Train S1 — Flow Matching with Training-Time RTC

```bash
python -u -m lerobot.policies.hvla.s1.flow_matching.train \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents_pt_11997.npy \
    --output-dir outputs/flow_s1_hvla \
    --steps 25000 \
    --batch-size 64 \
    --save-freq 5000 \
    --num-workers 16 \
    --max-delay 0.15 \
    --resize-images 224x224 \
    2>&1 | tee outputs/flow_s1_hvla/train.log
```

Flow matching S1 implements [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964) (Mees et al., 2025): simulated inference delay during training, with ground-truth action prefix inpainting. No architecture changes vs standard flow matching — just masking. At inference, actually-executed actions replace the prefix positions at each denoising step.

- 97M trainable params (DINOv2 ViT-S finetuned + 75M decoder)
- LR 2.5e-5 with cosine decay to 2.5e-6 (matching Pi0)
- bf16 autocast + TF32 matmul
- 5 denoising steps (Euler integration)
- ~370ms/step at bs=64 → **~2.6 hours** on RTX 5090
- 29ms inference latency

### 3a. Inference — ACT (default)

```bash
python -m lerobot.policies.hvla.launch \
    --s1-checkpoint outputs/act_vlm_hvla/checkpoint-100000 \
    --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224 \
    --temporal-ensemble-coeff 0.07
```

### 3b. Inference — Flow Matching with RTC

```bash
python -m lerobot.policies.hvla.launch \
    --s1-type flow \
    --s1-checkpoint outputs/flow_s1_hvla/checkpoint-25000/model.safetensors \
    --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224
```

No `--temporal-ensemble-coeff` needed — RTC provides chunk continuity natively.

Spawns S2 as a separate process communicating via CPU shared memory (0.04ms latency). S1 runs in the main process with robot access.

### 3b. Inference (legacy WebSocket — compatible with OpenPI server)

Start the Pi0.5 server (OpenPI):
```bash
cd ~/Documents/openpi_subtask && .venv/bin/python scripts/async_pi05/pytorch_pi05_server.py \
    --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch \
    --norm-stats ~/.cache/openpi/checkpoints/soarm-pi05-state-11997/assets/thewisp/cylinder_ring_assembly/norm_stats.json \
    --port 8765 --device cuda:0
```

Then run dual_system_infer.py:
```bash
python dual_system_infer.py \
    --s1-checkpoint outputs/act_vlm_hvla/checkpoint-100000 \
    --task "assemble cylinder into ring" \
    --s2-host localhost --s2-port 8765 \
    --resize-images 224x224 \
    --temporal-ensemble-coeff 0.01
```

Both paths pass the S2 latent age to the age embedding. The model is checkpoint-compatible with both.

## Module Structure

```
src/lerobot/policies/hvla/
├── s2/
│   ├── config.py           S2 VLM config (PaliGemma dims, LoRA settings)
│   ├── model.py            VLM-only model: extract_prefix_latent, AR subtask decode
│   ├── preprocessing.py    Camera image resize + normalize to [-1,1]
│   └── tokenizer.py        SentencePiece tokenizer for PaliGemma prompt format
├── s1/
│   ├── protocol.py        S1Policy interface (supports_rtc, needs_temporal_ensemble)
│   ├── __init__.py         Load helpers: load_act_policy / load_flow_matching_policy
│   └── flow_matching/
│       ├── config.py       FlowMatchingS1Config
│       ├── model.py        Flow matching decoder with RTC prefix conditioning
│       └── train.py        Training script with delay augmentation + RTC prefix
├── ipc.py                  SharedLatentCache: CPU shared memory between processes
├── s2_process.py           S2 process entry point (extraction loop)
├── s1_process.py           S1 process entry point (robot control loop)
├── launch.py               Spawns S2 process + runs S1
├── PLAN.md                 Design decisions and architecture rationale
└── README.md               This file
```

## Key Design Decisions

### S2 is VLM-only
S2 uses PaliGemma (SigLIP + Gemma 2B) for scene understanding. The Pi0.5 action expert is NOT loaded — S1 generates all actions. This saves ~40% GPU memory and decouples the action policy from the VLM.

### Latent age embedding
S1 receives the S2 latent's age (seconds of staleness) as a learned additive embedding. The age embedding MLP output layer is **zero-initialized**, so:
- Old checkpoints (without age weights) load fine — missing keys are zero, output unchanged
- `age_embedding(0.0)` = zeros — no-age and age=0 produce identical results
- New training learns to use age via standard backprop through action loss

### Delay augmentation
During S1 training, the S2 latent is randomly shifted backward by 0 to `max_delay` seconds within the same episode. This trains the model to handle stale latents gracefully, matching the real inference scenario where S2 runs at 4-15Hz while S1 runs at 22-30Hz.

### Shared memory IPC
`SharedLatentCache` uses `torch.Tensor.share_memory_()` backed by `/dev/shm`. Each process manages its own CUDA context — no GPU contention from IPC. The [2048] float32 latent (8KB) transfers in <0.1ms.

### Checkpoint compatibility
S2 loads from any Pi0.5 safetensors checkpoint — action expert keys are filtered out, `paligemma_with_expert.paligemma.*` keys are remapped to `paligemma.*`.

## Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| S2 latent extraction | ~81ms | VLM-only, no action expert |
| S1 prep (4 cameras) | ~6ms | CPU resize 720×1280→224×224 |
| S1 inference | ~25ms | ACTWithVLM forward |
| IPC round-trip | <0.1ms | CPU shared memory |
| Robot send_action | <1ms | Feetech serial, non-blocking |

Typical S1 loop: 30-40ms/step (~25-33Hz). S2 runs at 4-15Hz depending on GPU scheduling.

## TODO

### High priority
- [ ] **Exclude RTC prefix from loss** — both arXiv:2512.05964 and Ψ₀ (arXiv:2603.12263) exclude masked prefix tokens from loss computation. Our current implementation includes them (velocity target ≈ 0 for clean positions), wasting model capacity. One-line fix in `model.py:forward()`.
- [ ] **Fix RTC prefix source at inference** — both papers use the **previous chunk's predicted actions** (the overlap portion still being executed) as the prefix, NOT actually-executed robot actions. The previous chunk's predictions are in the model's own output space and are more consistent. Currently we pass `_executed_actions` from the ring buffer; should instead pass `prev_chunk[current_idx : current_idx+d]`. Implementation: inference thread keeps previous chunk + the index it was executing from when new inference started. The overlap region `prev_chunk[start_idx : start_idx + d]` becomes the prefix for the new chunk.
- [ ] **Use dynamic prefix length at inference** — `d` should equal the **actual measured inference delay** in frames, not a fixed `--rtc-prefix-len`. We already compute `elapsed = t_before_send - t_obs` for the time-shift; the same value gives `d = round(elapsed * fps)`. This means prefix length varies per chunk (typically 2-3 at 30Hz with 75-80ms inference). The model handles variable `d` because it trained with `d ~ Uniform(0, d_max)`. The `elapsed` is measured as wall-clock time from when the inference thread captured its observation to when the main loop is about to send the action — same measurement we use for time-shift indexing.
- [ ] **Match training delay distribution to actual inference** — currently `d ~ Uniform(0, 10)` but our actual delay is centered at 2-3 frames (66-100ms). Uniform wastes training signal on unlikely delays. Better: `d ~ Uniform(1, 5)` for 80% of samples, `d=0` for 20% (prefix dropout / episode start). Precedented by the original RTC paper which used "exponentially decreasing weights" in simulation. Also reduce d_max from 10 to 5-6.
- [ ] Evaluate flow matching S1 with RTC on robot (training in progress)
- [ ] Training speed: use `--num-workers 16` (GPU idle 50% at 8 workers due to video decode bottleneck)
- [ ] Pre-decode and cache training images to avoid repeated video decode (main training bottleneck)
- [ ] Persistent S2 process (avoid 45s cold start on every launch)
- [ ] Generic SharedMemoryStore to replace per-type IPC classes

### Medium priority
- [ ] `torch.compile` for S1 training (mode=default, skip DINOv2 — needs investigation)
- [ ] Separate backbone LR (lower for DINOv2 pretrained weights, matching ACT's approach)
- [ ] wandb integration for training monitoring
- [ ] Inference-time RTC guidance (LeRobot's `RTCProcessor`) on top of training-time RTC for extra smoothness
- [ ] Action time-shift: verify `round()` vs `ceil()` with flow matching + RTC on real robot
- [ ] Soft landing: fix occasional motor overload error on disconnect

### Future
- [ ] S2 LoRA training (rank 32 on SigLIP + Gemma 2B) — lower priority since Pi0.5 checkpoint works
- [ ] Co-training S1 + S2 (currently sequential: extract latents → train S1)
- [ ] Adaptive action horizon (short chunks for precision, long for transit)
- [ ] CUDA MPS for reduced GPU contention between S1 and S2 on same GPU
