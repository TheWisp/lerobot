"""Configuration for RLT (RL Token) online fine-tuning of frozen HVLA S1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLTConfig:
    """All hyperparameters for RLT training (Phase 1 + Phase 2).

    Phase 1: Offline RL token encoder-decoder training on demo data.
    Phase 2: Online RL with lightweight actor-critic on robot.
    """

    # --- RL Token (Phase 1) ---
    rl_token_dim: int = 768               # bottleneck dim (matches S1 hidden_dim)
    token_encoder_layers: int = 2         # small transformer encoder
    token_decoder_layers: int = 2         # small transformer decoder
    token_num_heads: int = 4
    token_ffn_dim: int = 2048
    token_dropout: float = 0.1
    token_lr: float = 1e-4
    token_train_steps: int = 5000
    token_batch_size: int = 64
    s1_finetune_weight: float = 0.0       # alpha: 0 = freeze S1 during token training

    # --- Actor-Critic (Phase 2) ---
    actor_hidden_dim: int = 256           # 2-layer MLP hidden size
    actor_num_layers: int = 2
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    num_critics: int = 2                  # TD3 ensemble size

    rl_chunk_length: int = 10             # C: RL action chunk length
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    discount: float = 0.99               # gamma
    tau: float = 0.005                    # target network soft update rate
    beta: float = 0.1                     # BC regularizer weight (paper uses 1.0 with delta EE)
    actor_sigma: float = 0.02            # fixed Gaussian std for exploration (normalized space)
                                          # Paper uses 0.1 but with delta EE position (~1cm scale).
                                          # Joint angles have ~24° std, so 0.02 ≈ 0.5° exploration.
    ref_action_dropout: float = 0.5       # probability of zeroing reference chunk
    utd_ratio: int = 5                    # gradient updates per new transition
    subsample_stride: int = 2             # stride for replay buffer subsampling

    # --- Replay buffer ---
    replay_capacity: int = 200_000        # max transitions (~1.2GB)

    # --- Warmup ---
    warmup_episodes: int = 10             # VLA executes, actor/critic train on VLA data
