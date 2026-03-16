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

### 2. Train S1 (with delay augmentation)

```bash
python scripts/train_act_vlm.py \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents.npy \
    --output-dir outputs/act_vlm_hvla \
    --steps 100000 \
    --batch-size 16 \
    --save-freq 20000 \
    --use-dino-backbone \
    --resize-images 224x224 \
    --num-workers 8 \
    --max-delay 0.5
```

`--max-delay 0.5` enables delay augmentation: each training sample randomly shifts the S2 latent backward by 0-500ms (0-15 frames at 30fps), within episode boundaries. The age (seconds of staleness) is injected as a learned embedding — same scalar used at inference.

### 3a. Inference (HVLA local — two processes, shared memory)

```bash
python -m lerobot.policies.hvla.launch \
    --s1-checkpoint outputs/act_vlm_hvla/checkpoint-100000 \
    --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224 \
    --temporal-ensemble-coeff 0.01
```

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
│   └── (re-exports ACTWithVLMPolicy — swappable)
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
