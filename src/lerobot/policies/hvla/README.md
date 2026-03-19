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
    --output-dir outputs/flow_s1_hvla_v5 \
    --steps 50000 \
    --batch-size 64 \
    --save-freq 10000 \
    --num-workers 16 \
    --resize-images 224x224 \
    2>&1 | tee outputs/flow_s1_hvla_v5/train.log
```

Flow matching S1 implements [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964) (Mees et al., 2025): simulated inference delay during training, with ground-truth action prefix inpainting. No architecture changes vs standard flow matching — just masking. At inference, actually-executed actions replace the prefix positions at each denoising step.

- 97M trainable params (DINOv2 ViT-S finetuned + 75M decoder)
- LR 2.5e-5 with cosine decay to 2.5e-6 (matching Pi0)
- bf16 autocast + TF32 matmul
- 15 denoising steps (Euler integration), rtc_max_delay=6
- No S2 latent delay augmentation or age embedding (matching old ACT setup that worked)
- ~870ms/step at bs=64 → **~12 hours** for 50k steps on RTX 5090
- ~54ms inference latency (15 steps)

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
    --s1-checkpoint outputs/flow_s1_hvla_v5/checkpoint-50000/model.safetensors \
    --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224 \
    --s1-query-interval 5
```

No `--temporal-ensemble-coeff` needed — RTC provides chunk continuity natively. `--s1-query-interval 5` executes 5 actions (~167ms) before re-querying S1. `--denoise-steps` overrides the config default (15) if needed.

S2 auto-discovery: the launcher first tries to attach to an existing S2 process via shared memory. If found, S1 starts instantly (no 45s cold start). If not found, spawns S2 from `--s2-checkpoint`.

### 3c. Persistent S2 (recommended for iteration)

Start S2 once in a separate terminal — it stays hot between S1 restarts:
```bash
# Terminal 1: start S2 (one-time, stays running)
python -m lerobot.policies.hvla.s2_standalone \
    --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring"
```

```bash
# Terminal 2: start/restart S1 freely (connects to S2 instantly)
python -m lerobot.policies.hvla.launch \
    --s1-type flow \
    --s1-checkpoint outputs/flow_s1_hvla_v5/checkpoint-50000/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224 \
    --s1-query-interval 5
```

S1 auto-discovers the running S2 via well-known shared memory names. No `--s2-checkpoint` needed when S2 is already running.

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

### No delay augmentation (v5)
The old ACT-based S1 trained without delay augmentation and successfully used S2's latent. Flow matching v1-v4 introduced delay augmentation (shifting the S2 latent backward by 0-150ms), which caused S1 to learn to ignore the noisy/misaligned latent. v5 reverts to aligned latents (no delay, no age embedding) to match the working ACT setup.

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

### Critical
- [ ] **Fix Corrupt JPEG data** — frequent "Corrupt JPEG data: premature end of data segment" from OpenCV cameras during inference. Likely cause: USB bandwidth saturation with 4 cameras at 720p30 + camera buffer contention. This corrupts images fed to S1, potentially causing bad action predictions and contributing to jitter. Investigate: (1) reduce camera resolution or fps, (2) stagger camera reads, (3) use V4L2 direct instead of OpenCV, (4) deep-copy camera frames in main loop before sharing.

### High priority
- [ ] **S1 inference thread latency** — obs→infer gap is 3-45ms (avg ~20ms) due to `threading.Event.wait()` OS scheduler latency. The inference thread sleeps up to 33ms waiting for the next main loop tick. Options: (1) busy-wait with CPU yield, (2) double-buffered obs with atomic swap, (3) redesign so inference thread owns obs capture. This adds 0-33ms to chunk_age, reducing action freshness.
- [ ] **S1 GPU priority over S2** — S1 inference doubles from 21ms to 50ms due to GPU contention with S2. S2 throttle (100ms sleep) helps but doesn't eliminate contention. Options: (1) CUDA MPS for proper time-slicing, (2) increase S2 throttle (trade S2 freshness for S1 speed), (3) Triton kernel optimization ([realtime-vla](https://github.com/Dexmal/realtime-vla), arXiv:2510.26742).
- [ ] **Image resize optimization** — switched to cv2.resize (0.7ms/4 images), but consider: GPU resize (1.2ms, avoids CPU→GPU transfer of full-res), or cameras outputting 224x224 directly.
- [ ] Persistent S2 process (avoid 45s cold start on every launch)
- [ ] Generic SharedMemoryStore to replace per-type IPC classes
- [ ] Pre-decode and cache training images to avoid repeated video decode (main training bottleneck)

### Medium priority
- [ ] `torch.compile` for S1 training (mode=default, skip DINOv2 — needs investigation)
- [ ] Separate backbone LR (lower for DINOv2 pretrained weights, matching ACT's approach)
- [ ] wandb integration for training monitoring
- [ ] Inference-time RTC guidance (LeRobot's `RTCProcessor`) on top of training-time RTC for extra smoothness
- [ ] Soft landing: fix occasional motor overload error on disconnect

### Future
- [ ] S2 LoRA training (rank 32 on SigLIP + Gemma 2B) — lower priority since Pi0.5 checkpoint works
- [ ] Co-training S1 + S2 (currently sequential: extract latents → train S1)
- [ ] Adaptive action horizon (short chunks for precision, long for transit)
- [ ] Switch S1 to using Pi0-style action expert architecture (proven at scale, matches Ψ₀)
- [ ] ONNX/TensorRT export for S1 (potential 2× inference speedup)
