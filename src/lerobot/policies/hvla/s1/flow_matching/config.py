"""Configuration for Flow Matching S1 action policy.

Action expert conditioned on:
  - DINOv2 image features (same backbone as ACTWithVLM)
  - S2 latent [2048] + age embedding
  - Optional robot state, with its own dataset-defined feature layout
  - Training-time RTC: simulated delay + inpainting prefix

No VLM in S1 — S2 handles scene understanding via shared latent.

References:
  - Flow Matching: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
  - Pi0 action expert: Black et al., "π₀: A Vision-Language-Action Flow Model", 2024
  - SmolVLA: Luo et al., "SmolVLA: A Small Vision-Language-Action Model", 2025
  - Training-time RTC: Mees et al., "Training-Time Action Conditioning for Efficient
    Real-Time Chunking", arXiv:2512.05964, 2025
  - Inference-time RTC: Moeglich et al., "Real-Time Execution of Action Chunking
    Flow Policies", arXiv:2506.07339, 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import ClassVar


@dataclass
class FlowMatchingS1Config:
    """Config for FlowMatchingS1Policy.

    Architecture follows Pi0/SmolVLA action expert design:
    - Observation encoder (DINOv2 + state + S2 latent) → context tokens
    - Flow matching decoder with cross-attention to context
    - Action+timestep fusion via concat → MLP(SiLU) (matching Pi0/SmolVLA)
    """

    FEATURE_CONTRACT_VERSION: ClassVar[int] = 1

    # --- Dataset-derived feature contract ---
    # These fields are resolved from dataset metadata during training and
    # persisted in config.json. They are model-shape metadata, not user-facing
    # hyperparameters.
    action_dim: int | None = None
    action_feature_names: list[str] = field(default_factory=list)
    # Train arm positions as offsets from the current named state positions.
    # Gripper positions remain absolute.  This keeps all embodiment dimensions
    # while anchoring the first decoded arm command to the measured posture.
    use_relative_actions: bool = False
    chunk_size: int = 50  # predict 50 future actions (~1.67s at 30Hz)
    n_action_steps: int = 50  # execute full chunk (RTC handles continuity)

    # --- Model architecture ---
    # Targeting ~30M parameters, excluding the vision backbone.
    hidden_dim: int = 768  # transformer hidden dimension
    num_heads: int = 8  # attention heads
    num_encoder_layers: int = 4  # observation encoder depth
    num_decoder_layers: int = 6  # flow matching decoder depth
    dim_feedforward: int = 2048  # FFN intermediate size
    dropout: float = 0.1

    # --- Image backbone ---
    use_dino_backbone: bool = True  # DINOv2 vision backbone
    # Training resolves these from dataset metadata. A visual checkpoint must
    # persist them; inference never guesses camera names from an embodiment.
    image_features: dict = field(default_factory=dict)
    image_resize_shape: tuple[int, int] | None = None
    dino_model: str = "dinov2_vits14"  # ViT-S/14 (22M); 384-d patch tokens
    freeze_backbone: bool = False  # finetune DINOv2 (required for good performance)
    backbone_gradient_checkpointing: bool = True  # saves ~40% activation memory for DINOv2
    backbone_dim: int = 384  # DINOv2 ViT-S output dim (768 for ViT-B)

    # --- S2 conditioning ---
    s2_latent_dim: int = 2048  # S2 prefix latent dimension
    s2_proj_hidden: int = 1024  # S2 projection MLP intermediate
    use_s2_age_embedding: bool = False  # disabled — old ACT worked without it

    # --- Flow matching ---
    num_inference_steps: int = 15  # persisted inference default; higher costs more latency
    time_sampling_beta_alpha: float = 1.5  # Beta distribution for training time sampling
    time_sampling_beta_beta: float = 1.0
    time_min: float = 0.001  # minimum timestep
    time_max: float = 1.0  # maximum timestep

    # --- Training-time RTC (arXiv:2512.05964, Ψ₀ arXiv:2603.12263) ---
    # Simulates inference delay during training by replacing the first D actions
    # in x_t with ground-truth (unnoised) actions, and setting their per-position
    # timestep to t=0 (clean). Prefix positions excluded from loss.
    # At inference, overlap actions from the previous chunk serve as prefix.
    # d sampled as Uniform(1, rtc_max_delay) with rtc_drop_prob chance of d=0.
    rtc_max_delay: int = 6  # max simulated delay in frames (15 denoise steps ≈ 5 frames)
    rtc_drop_prob: float = 0.2  # probability of no prefix (simulates first chunk)

    # --- Soft RTC (arXiv:2605.25537) ---
    # Hard RTC pins positions [0, d) completely and leaves position d fully free,
    # so the conditioning weight jumps 1 -> 0 at exactly the first executed
    # action. Measured on checkpoint-50000, that is where the trajectory
    # reverses: the step across the boundary opposes the prefix in ~70-80% of
    # chunks, against +0.455 agreement inside the chunk body.
    #
    # Soft RTC replaces the binary mask with continuous weights w_j: still 1 on
    # the committed prefix, then decaying to 0 across a soft window
    # [d, e(d)) where e(d) = min(d + rtc_soft_len, rtc_soft_hmax). Those tokens
    # are partly prior-informed and stay in the loss with weight (1 - w_j), so
    # the model is trained to continue from a prefix rather than to ignore it.
    #
    # rtc_soft_len = 0 reproduces Hard RTC exactly — same x_t, same loss, same
    # sampler — which is what test_soft_rtc_zero_len_matches_hard pins down.
    rtc_soft_len: int = 0  # L: soft-window length after the committed prefix
    rtc_soft_hmax: int = 8  # cap on e(d), the far end of the soft window


    # --- Robot state ---
    robot_state_feature: bool | None = None
    state_dim: int | None = None
    state_feature_names: list[str] = field(default_factory=list)
    # Dataset-native units (OpenArm position observations are degrees). Zero
    # preserves checkpoints and training commands produced before this option.
    # Degrees, dataset-native units, applied to ``*.pos`` features only.
    #
    # Defaults on rather than off. A joint held still across a whole recording
    # gets a std at the 1e-6 numerical floor, and dividing by that amplifies a
    # difference smaller than the sensor can resolve. Measured on
    # GPU/0803_20260803_174402: left_joint_3.pos (mean 0.9508, std 1.0e-06) read
    # 0.732 on the rig -- 0.22 degrees off -- and normalized to 218,569 sigma
    # unfloored. With a floor of 0.5 the worst channel on that same frame is
    # 13.7 sigma, and it is a torque one, which this floor does not cover.
    #
    # 0.5 rather than something smaller because the measured distribution is
    # bimodal — across four recorded datasets no ``.pos`` dimension has a std
    # between 0.5 and 1.0, so the threshold sits in an empty gap and separates
    # "held still" from "moving" without splitting either. It is not a no-op:
    # a permanently-closed gripper (std 0.001-0.05) is floored too, which is
    # the intent. Set to 0.0 to restore the unfloored behaviour.
    state_position_std_floor: float = 0.5

    # --- Training ---
    # LR references: Pi0=2.5e-5, ACT=1e-5, SmolVLA=1e-4, Pi0.5+LoRA=1.2e-4
    lr: float = 2.5e-5  # peak LR (cosine schedule)
    lr_decay: float = 2.5e-6  # final LR after cosine decay
    weight_decay: float = 1e-4
    warmup_steps: int = 1000

    @property
    def num_images(self) -> int:
        return len(self.image_features)

    def validate_feature_contract(self, *, require_names: bool = False) -> None:
        """Reject unresolved or internally inconsistent tensor metadata."""
        if (
            type(self.state_position_std_floor) not in (int, float)
            or not math.isfinite(self.state_position_std_floor)
            or self.state_position_std_floor < 0
        ):
            raise ValueError("Flow S1 state_position_std_floor must be a finite non-negative value")

        if type(self.action_dim) is not int or self.action_dim <= 0:
            raise ValueError("Flow S1 action_dim must be resolved from a dataset or checkpoint")
        if (
            not isinstance(self.action_feature_names, list)
            or any(not isinstance(name, str) or not name for name in self.action_feature_names)
            or len(set(self.action_feature_names)) != len(self.action_feature_names)
        ):
            raise ValueError("Flow S1 action feature names must be unique, non-empty strings")
        if self.action_feature_names and len(self.action_feature_names) != self.action_dim:
            raise ValueError(
                f"Flow S1 records {len(self.action_feature_names)} action names "
                f"for action_dim={self.action_dim}"
            )
        if require_names and not self.action_feature_names:
            raise ValueError("Flow S1 requires ordered action feature names; dimensions alone are unsafe")

        if type(self.robot_state_feature) is not bool:
            raise ValueError("Flow S1 robot_state_feature must be resolved from a dataset or checkpoint")
        if self.robot_state_feature:
            if type(self.state_dim) is not int or self.state_dim <= 0:
                raise ValueError("Flow S1 state_dim must be positive when observation.state is enabled")
            if (
                not isinstance(self.state_feature_names, list)
                or any(not isinstance(name, str) or not name for name in self.state_feature_names)
                or len(set(self.state_feature_names)) != len(self.state_feature_names)
            ):
                raise ValueError("Flow S1 state feature names must be unique, non-empty strings")
            if self.state_feature_names and len(self.state_feature_names) != self.state_dim:
                raise ValueError(
                    f"Flow S1 records {len(self.state_feature_names)} state names "
                    f"for state_dim={self.state_dim}"
                )
            if require_names and not self.state_feature_names:
                raise ValueError(
                    "Flow S1 requires ordered state feature names when observation.state is enabled"
                )
        elif self.state_dim not in (None, 0) or self.state_feature_names:
            raise ValueError("Flow S1 disables observation.state but records a non-empty state contract")
        elif self.state_position_std_floor > 0:
            raise ValueError("Flow S1 cannot apply a state position std floor without observation.state")

        if self.use_relative_actions:
            if not self.robot_state_feature:
                raise ValueError("Flow S1 relative actions require observation.state")
            missing_state_names = sorted(set(self.action_feature_names) - set(self.state_feature_names))
            if missing_state_names:
                raise ValueError(
                    "Flow S1 relative actions require every named action to have a matching state "
                    f"position; missing {missing_state_names}"
                )
            relative_names = [
                name
                for name in self.action_feature_names
                if name.endswith(".pos") and "gripper" not in name.lower()
            ]
            if not relative_names:
                raise ValueError("Flow S1 relative actions found no non-gripper *.pos action features")

        if not isinstance(self.image_features, dict) or any(
            not isinstance(name, str) or not name.startswith("observation.images.")
            for name in self.image_features
        ):
            raise ValueError("Flow S1 image features must use observation.images.* keys")
        if self.image_resize_shape is not None and (
            not isinstance(self.image_resize_shape, tuple)
            or len(self.image_resize_shape) != 2
            or any(type(size) is not int or size <= 0 for size in self.image_resize_shape)
        ):
            raise ValueError("Flow S1 image_resize_shape must be a positive (height, width) tuple")

    @classmethod
    def from_checkpoint_dict(cls, data: dict) -> FlowMatchingS1Config:
        """Load a complete feature contract without embodiment guesses."""
        data = dict(data)
        version = data.get("feature_contract_version")
        if version is not None and version != cls.FEATURE_CONTRACT_VERSION:
            raise ValueError(
                f"HVLA checkpoint uses unsupported feature_contract_version={version!r}. "
                "Migrate it after verifying action/state order and camera metadata."
            )

        # Checkpoints produced by the first feature-contract implementation
        # predate the version marker and robot_state_feature flag, but already
        # carry complete ordered state metadata. This inference is exact: no
        # robot identity or runtime feature order is involved.
        if "robot_state_feature" not in data:
            state_dim = data.get("state_dim")
            state_names = data.get("state_feature_names")
            if (
                type(state_dim) is int
                and state_dim > 0
                and isinstance(state_names, list)
                and len(state_names) == state_dim
            ):
                data["robot_state_feature"] = True
            elif state_dim in (None, 0) and state_names == []:
                data["robot_state_feature"] = False

        # Early stateless checkpoints retained an unused state_proj and its
        # old dimension even though robot_state_feature was explicitly false.
        # The ordered empty state contract is authoritative; normalize away
        # the unused layer instead of feeding state at inference.
        if data.get("robot_state_feature") is False and data.get("state_feature_names") == []:
            data["state_dim"] = 0

        required = {
            "action_dim",
            "action_feature_names",
            "robot_state_feature",
            "state_dim",
            "state_feature_names",
            "image_features",
            "image_resize_shape",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(
                f"HVLA checkpoint feature contract is ambiguous or missing fields: {missing}. "
                "Backfill it from the run's training dataset with: python -m "
                "lerobot.policies.hvla.scripts.hvla_migrate_checkpoints <run_dir>"
            )

        init_fields = {item.name for item in fields(cls) if item.init}
        values = {key: value for key, value in data.items() if key in init_fields}
        resize = values.get("image_resize_shape")
        if resize is not None:
            values["image_resize_shape"] = tuple(resize)
        config = cls(**values)
        config.validate_feature_contract(require_names=True)
        if config.use_dino_backbone and not config.image_features:
            raise ValueError("HVLA visual checkpoint does not record any image features")
        return config
