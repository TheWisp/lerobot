# RLT for HVLA S1 — Design Document

Online RL fine-tuning of a frozen HVLA S1 flow matching policy using the RL Token method ([Xu et al., 2026](https://www.pi.website/research/rlt)).

## Motivation

HVLA S1 learns from demonstrations but struggles with precision-critical phases (sub-mm alignment, tight insertions). RLT adds a lightweight RL layer on top of the frozen S1 that refines actions through online practice, using the S1's own representations as state for sample-efficient learning.

## Architecture Mapping

| RLT Paper (pi0.6)              | HVLA S1                                  |
|--------------------------------|------------------------------------------|
| VLM: SigLIP + Gemma 4B        | S2: SigLIP + Gemma 2B (separate process) |
| Action expert: 860M diffusion  | S1: 97M flow matching (DINOv2 + decoder) |
| VLA hidden dim: 2048           | S1 hidden dim: 768                       |
| RL token: 1 x 2048            | **RL token: 1 x 768**                    |
| 50 Hz, 14-DOF                 | 30 Hz, 14-DOF                            |
| H=50 VLA chunk, C=10 RL chunk | H=50 VLA chunk, C=10 RL chunk            |

## System Overview

```
Frozen S1 (97M, bf16)                    Trainable RL head (fp32)
+---------------------------------+      +----------------------------+
| DINOv2 ViT-S (4 cams)          |      |                            |
| + state token                   |      |  Actor MLP (2-layer, 256)  |
| + S2 latent token               |      |  input: z_rl + s^p + a_ref |
| => obs_encoder => context [1026, 768]  |  output: action chunk      |
|         |                       |      |                            |
|  RL token encoder (frozen)      |      |  Critic MLP x2 (TD3)       |
|  => z_rl [1, 768]        ------+----->|  input: z_rl + s^p + a     |
|                                 |      |  output: Q scalar          |
|  flow matching decoder          |      +----------------------------+
|  => reference chunk a_ref [C,14]|
+---------------------------------+
```

All components run in a single process on one GPU.

## Phase 1: RL Token Training (offline, on demo data)

Train an encoder-decoder pair that compresses S1's observation encoder output into a single 768-dim vector.

### RL Token Encoder

Appends a learned readout embedding `e_rl` to the frozen S1 context tokens, then processes through a small transformer:

```
z = S1.obs_encoder(batch)              # [B, N_ctx, 768], N_ctx ~ 1026
z_aug = cat([z, e_rl.expand(B,1,768)]) # [B, N_ctx+1, 768]
z_rl = g_phi(z_aug)[:, -1, :]          # [B, 768] — readout position
```

Encoder transformer `g_phi`: 2 layers, 4 heads, FFN 2048.

### RL Token Decoder

Autoregressive reconstruction of the original context tokens from `z_rl`:

```
L_rto = E_D [ sum_i || h_phi(d_phi([z_rl, z_hat_{1:i-1}]))_i - sg(z_i) ||^2 ]
```

Decoder transformer `d_phi`: 2 layers, 4 heads, FFN 2048. Linear output projection `h_phi`: 768 -> 768.

`sg()` = stop-gradient on the frozen S1 embeddings.

### Optional: S1 fine-tune on task data

If the S1 checkpoint wasn't trained on the target task, simultaneously fine-tune S1 with its original flow matching loss (controlled by weight alpha). After this phase, both S1 and the RL token encoder are frozen.

### Training details

- Dataset: 1-10 hours of task-specific teleop demos (same data used for S1 training)
- Steps: 2000-10000 gradient steps
- Optimizer: Adam, LR 1e-4
- S1 backbone: frozen throughout (only phi is trained; alpha=0 if S1 already task-tuned)

## Phase 2: Online RL (on robot)

Freeze S1 and the RL token encoder. Train lightweight actor and critic MLPs online using off-policy TD3.

### State Representation

```
x_t = (z_rl(s_t), s^p_t)    # z_rl: 768, s^p: 14 (joint positions)
                              # total: 782
```

### Actor MLP

```
pi_theta(a_{1:C} | x, a_ref) = N(mu_theta(x, a_ref), sigma^2 I)
```

- Input: `x_t` (782) + reference action chunk `a_ref` (C x 14 = 140) = **922 dims**
- Architecture: Linear(922, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 140)
- Output: action chunk mean `mu` in R^{C x d} (reshaped to [C, 14])
- Fixed small sigma (e.g. 0.1, tuned per task)
- **Reference action dropout**: 50% of training batches replace `a_ref` with zeros, forcing the actor to maintain an independent action pathway

### Critic MLP (TD3 ensemble of 2)

```
Q_psi(x, a_{1:C}) -> R
```

- Input: `x_t` (782) + action chunk `a_{1:C}` (140) = **922 dims**
- Architecture: Linear(922, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 1)
- Two independent Q-networks; use min(Q1, Q2) for target values

### Actor Loss

```
L_pi(theta) = E_{s~B} [ -Q_psi(x, a) + beta * ||a - a_ref||^2 ]
              where a ~ pi_theta(. | x, a_ref)
```

`beta` controls BC regularization strength — keeps the RL policy close to the frozen S1's suggestions. This is the key mechanism that turns RL into *local refinement* rather than unconstrained search.

### Critic Loss (C-step TD backup)

```
Q_hat = sum_{t'=1}^{C} gamma^{t'-1} r_{t'} + gamma^C * E_{a'~pi} [Q_{psi'}(x', a')]
L_Q(psi) = E_{b~B} [ (Q_hat - Q_psi(x, a))^2 ]
```

### Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| RL chunk length C | 10 | Paper (matches H=50 ratio) |
| Discount gamma | 0.99 | Standard |
| UTD ratio | 5 | Paper Appendix B |
| Actor/Critic LR | 3e-4 | TD3 default |
| Target network tau | 0.005 | TD3 default |
| BC regularizer beta | 1.0 | Paper Eq. 5 (tune per task) |
| Actor noise sigma | 0.1 | Tuned per task |
| Ref action dropout | 50% | Paper Sec. IV-B |
| Subsample stride | 2 | Paper Appendix B (~15 transitions/chunk) |
| Warmup episodes N_warm | 10-20 | Paper: fill buffer with VLA rollouts |
| Reward | Sparse +1 | Human operator presses key on success |

### Action Execution

Every C=10 control steps (0.33s at 30Hz):

1. Capture observation, read S2 latent from shared memory
2. S1 obs encoder -> context tokens (frozen, bf16)
3. RL token encoder -> z_rl (frozen, bf16)
4. S1 flow matching decoder -> reference chunk a_ref (frozen, bf16, 10 denoise steps)
5. Actor MLP -> action chunk a (fp32)
6. Execute a on robot for C steps
7. Subsample transitions (stride 2) into replay buffer
8. Run G = UTD x (new transitions) gradient updates on actor + critic

S1 inference (~30ms) dominates. Actor MLP forward is <0.1ms. Gradient updates interleave between action chunks.

## Integration with Existing Systems

### Human Intervention (already implemented)

The existing SPACE-toggle intervention system (`--teleop-config`) works as-is:

- During intervention, the human's actions replace the actor's output
- Intervention actions are stored in the replay buffer with the corresponding VLA reference (the VLA keeps running during intervention, per Algorithm 1 line 11)
- On resume, the actor picks up from the current state
- The `InferenceThread` pause/resume mechanism handles the S1 side

### Reward Signal

New: operator presses a key (e.g. `R` or `Enter`) to signal binary success (+1) at episode end. No per-step reward — just terminal sparse reward.

### Episode Structure

Two modes from the paper:

**Critical-phase only**: Episodes start at a controlled state right before the hard part. Operator resets to this state between episodes. Fastest learning (isolates the precision segment).

**Full-task**: S1 VLA handles the easy initial phase. Operator triggers switch to RL policy when the critical phase begins (e.g. second SPACE press or a separate key). At test time, train a phase classifier to automate this switch.

### Recording

Use existing `--record-dataset` to log full episodes. Use `--intervention-dataset` for intervention fragments. RL-specific logging:

- Replay buffer: store (x_t, a_{1:C}, a_ref, r, x_{t+C}) tuples
- Q-value curves, actor loss, critic loss -> tensorboard/wandb
- Episode success rate over time

## Implementation Plan

### Step 1: RL Token Encoder-Decoder

New file: `src/lerobot/policies/hvla/rlt/token.py`

- `RLTokenEncoder(nn.Module)`: 2-layer transformer + learned readout
- `RLTokenDecoder(nn.Module)`: 2-layer transformer + linear projection
- `train_rl_token.py`: offline training script on demo data

### Step 2: Actor-Critic MLPs

New file: `src/lerobot/policies/hvla/rlt/actor_critic.py`

- `RLTActor(nn.Module)`: 2-layer MLP, Gaussian output
- `RLTCritic(nn.Module)`: 2-layer MLP, TD3 ensemble of 2
- Target network management (soft update)

### Step 3: Replay Buffer

New file: `src/lerobot/policies/hvla/rlt/replay_buffer.py`

- Stores (z_rl, s^p, a_{1:C}, a_ref, r, z_rl', s^p') tuples
- Pre-computed z_rl (no need to re-run encoder on replay)
- Capacity: ~50k transitions (small — a few hours of 30Hz data)

### Step 4: Online Training Loop

New file: `src/lerobot/policies/hvla/rlt/train_online.py`

Extends the existing `s1_process.py` control loop:

- Wraps frozen S1 + RL token encoder + actor
- Adds critic training interleaved with execution
- Adds reward collection (keyboard input)
- Reuses intervention, recording, S2 integration

### Step 5: Evaluation & Phase Switching

- Success rate tracking (rolling window)
- Optional: train binary classifier on intervention labels to auto-switch between base S1 and RL policy

## File Structure

```
src/lerobot/policies/hvla/
  rlt/
    __init__.py
    token.py            # RLTokenEncoder, RLTokenDecoder
    actor_critic.py     # RLTActor, RLTCritic (TD3)
    replay_buffer.py    # Chunk-level replay buffer
    config.py           # RLTConfig dataclass
    train_token.py      # Phase 1: offline RL token training
    train_online.py     # Phase 2: online RL on robot
  docs/
    rlt_design.md       # This file
```

## Key Risks

1. **S1 is flow matching, not autoregressive** — The paper uses pi0.6's diffusion-based action expert. Our flow matching decoder is similar (both iterative denoising), so the reference action chunk interface is the same. No expected issues.

2. **30 Hz vs 50 Hz** — Fewer transitions per wall-clock second. Mitigated by the high UTD=5 ratio. May need slightly more robot time to converge.

3. **S2 latent freshness** — S2 runs at ~10Hz in a separate process. During fast RL episodes, the latent may be 100-200ms stale. The RL token captures this in z_rl, so the actor/critic see a consistent representation. Not a concern — S1 already handles this.

4. **Reconstruction quality** — 1026 context tokens compressed to 1 x 768 is aggressive. The paper validates that this bottleneck retains manipulation-relevant information. If reconstruction loss plateaus high, consider 2 x 768 or reducing context token count (e.g. spatial pooling on DINOv2 patches before RL token).

## References

- Xu, Springenberg et al. "RL Token: Bootstrapping Online RL with Vision-Language-Action Models." Physical Intelligence, 2026.
- Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods" (TD3), 2018.
- Luo et al. "Efficient Online Reinforcement Learning with Offline Data" (Cal-QL), 2023.
