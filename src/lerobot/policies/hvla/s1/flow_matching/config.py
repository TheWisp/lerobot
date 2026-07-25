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

from dataclasses import dataclass, field, fields


@dataclass
class FlowMatchingS1Config:
    """Config for FlowMatchingS1Policy.

    Architecture follows Pi0/SmolVLA action expert design:
    - Observation encoder (DINOv2 + state + S2 latent) → context tokens
    - Flow matching decoder with cross-attention to context
    - Action+timestep fusion via concat → MLP(SiLU) (matching Pi0/SmolVLA)
    """

    # --- Dataset-derived tensor layout ---
    # These fields are resolved from dataset metadata during training and
    # persisted in config.json. They are model-shape metadata, not user-facing
    # hyperparameters.
    action_dim: int | None = None
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

    # --- Robot state ---
    robot_state_feature: bool | None = None
    state_dim: int | None = None

    # --- Training ---
    # LR references: Pi0=2.5e-5, ACT=1e-5, SmolVLA=1e-4, Pi0.5+LoRA=1.2e-4
    lr: float = 2.5e-5  # peak LR (cosine schedule)
    lr_decay: float = 2.5e-6  # final LR after cosine decay
    weight_decay: float = 1e-4
    warmup_steps: int = 1000

    @property
    def num_images(self) -> int:
        return len(self.image_features)

    def validate_feature_contract(self) -> None:
        """Reject unresolved or internally inconsistent tensor dimensions."""
        if type(self.action_dim) is not int or self.action_dim <= 0:
            raise ValueError("Flow S1 action_dim must be resolved from a dataset or checkpoint")

        if type(self.robot_state_feature) is not bool:
            raise ValueError("Flow S1 robot_state_feature must be resolved from a dataset or checkpoint")
        if self.robot_state_feature:
            if type(self.state_dim) is not int or self.state_dim <= 0:
                raise ValueError("Flow S1 state_dim must be positive when observation.state is enabled")
        elif self.state_dim not in (None, 0):
            raise ValueError("Flow S1 disables observation.state but records a non-zero state_dim")

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
        """Restore the tensor dimensions and cameras saved by training."""
        data = dict(data)
        if "robot_state_feature" not in data:
            state_dim = data.get("state_dim")
            data["robot_state_feature"] = type(state_dim) is int and state_dim > 0
        if data.get("robot_state_feature") is False:
            data["state_dim"] = 0

        init_fields = {item.name for item in fields(cls) if item.init}
        values = {key: value for key, value in data.items() if key in init_fields}
        resize = values.get("image_resize_shape")
        if resize is not None:
            values["image_resize_shape"] = tuple(resize)
        config = cls(**values)
        config.validate_feature_contract()
        if config.use_dino_backbone and not config.image_features:
            raise ValueError("HVLA visual checkpoint does not record any image features")
        return config
