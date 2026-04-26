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
    # Dim of S1 context tokens fed into the encoder. ``None`` means "same
    # as rl_token_dim" (symmetric setup — the original). Set explicitly
    # when widening the bottleneck beyond S1's hidden_dim (e.g. to match
    # the paper's 2048). Encoder then adds an input projection
    # (context_dim → rl_token_dim) and decoder a matching output
    # projection so reconstruction lands back in context_dim space.
    context_dim: int | None = None
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
    exploration_sigma: float = 0.02      # ε_expl in TD3: Gaussian std added to the ACTION
                                          # the robot executes. Drives exploration and
                                          # shows up as joint jitter. Paper uses 0.1 with
                                          # delta EE (~1cm); joint angles have ~24° std,
                                          # so 0.02 ≈ 0.5° jitter.
                                          #
                                          # Setting this to 0 disables joint jitter but
                                          # does NOT disable target policy smoothing —
                                          # that's controlled separately by target_sigma.
    target_sigma: float = 0.1            # ε̃ in TD3: Gaussian std added to the TARGET
                                          # action inside the Bellman backup for target
                                          # policy smoothing (Fujimoto 2018). Prevents
                                          # the critic from latching onto over-estimates
                                          # of a single deterministic action. Decoupled
                                          # from exploration_sigma so that exploration=0
                                          # doesn't also disable TD3's stability trick.
                                          # Paper σ̃=0.2 on MuJoCo continuous control;
                                          # 0.1 is half-paper, picked for tighter
                                          # robotic-arm action-space tolerance. Earlier
                                          # 0.02 (matching exploration_sigma) was way
                                          # too narrow — covered only ~0.5° per joint,
                                          # missed real Q-spike widths and contributed
                                          # to the Q explosion seen in v2_widened runs.
    target_noise_clip: float = 0.5       # TD3 target noise clip: ε̃ clamped to [-c, c].
                                          # Paper default: 0.5.
    ref_action_dropout: float = 0.5       # probability of zeroing reference chunk
    utd_ratio: int = 5                    # gradient updates per new transition
    subsample_stride: int = 2             # stride for replay buffer subsampling
    # Global-norm clip on the critic gradient before opt.step(). Caps update
    # magnitude without cancelling direction. SB3's TD3 default is 10; Ville's
    # reproduction uses 20 (paired with 4 critics). We start at 10 as a
    # conservative default — logged norm from training tells us if this
    # triggers too often (loosen) or never (blowup defense is loose; tighten).
    critic_grad_clip: float = 10.0
    # Reward applied to the last transition of an aborted episode (operator
    # marks the trajectory as a "disaster — never do this"). Default -1.0
    # gives the critic explicit discrimination between "didn't reach the
    # goal" (reward=0) and "actively bad / OOD" (reward=-1) — the same
    # transitions that previously read 0 are now ABORT'd to read negative,
    # so the critic learns to drive Q DOWN on those states. Magnitude is
    # tunable: smaller (-0.5) is gentler, larger (-2.0) is harsher.
    abort_reward: float = -1.0

    # --- Replay buffer ---
    replay_capacity: int = 200_000        # max transitions (~1.2GB)

    # --- Warmup ---
    warmup_episodes: int = 10             # VLA executes, actor/critic train on VLA data

    def is_warmup(self, episode: int) -> bool:
        """True when the 0-indexed episode is part of warmup.

        With ``warmup_episodes=10``, episodes 0..9 are warmup (10 total) and
        ep10 is the first non-warmup episode. Guard against off-by-one by
        using this helper instead of inlining the comparison.
        """
        return episode < self.warmup_episodes
