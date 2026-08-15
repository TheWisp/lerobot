"""Training script for Flow Matching S1 with Training-Time RTC.

Implements the training procedure from:
  "Training-Time Action Conditioning for Efficient Real-Time Chunking"
  (arXiv:2512.05964, Mees et al., 2025)

Key differences from standard flow matching training:
  - Simulated inference delay: randomly replace first D actions with GT (unnoised)
  - Per-position timestep: prefix positions get t=0, future positions get t~Beta
  - Prefix dropout: with probability p, no prefix (simulates first chunk)
  - S2 latent delay augmentation (independent from RTC delay)

Usage:
    python -m lerobot.policies.hvla.s1.flow_matching.train \\
        --dataset-repo-id thewisp/cylinder_ring_assembly \\
        --s2-latent-path ~/.cache/.../s2_latents_pt_11997.npy \\
        --output-dir outputs/flow_s1_hvla \\
        --steps 100000 --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.common.resource_telemetry import ResourceSampler
from lerobot.common.training_log import TrainingHealthTracker
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.hvla.s1.flow_matching import vision_encoders
from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import (
    NORMALIZED_STATE_CLAMP,
    OBS_IMAGES,
    FlowMatchingS1Policy,
)
from lerobot.policies.hvla.s1.protocol import S2_AGE_KEY, S2_LATENT_KEY
from lerobot.policies.input_contract import log_contract
from lerobot.utils.feature_utils import camera_name, resolve_camera_keys

logger = logging.getLogger(__name__)
TRAINING_TARGET_CONTRACT_VERSION = 2


def checkpoint_config_dict(config: FlowMatchingS1Config) -> dict:
    """Serialize a config into what a checkpoint's ``config.json`` must contain.

    Postcondition: the result satisfies
    :meth:`FlowMatchingS1Config.from_checkpoint_dict`, which is what makes a
    checkpoint loadable. Lives at module level rather than inside ``train()``
    so a test can hold the writer and the reader against each other; while it
    was a closure, dropping a contract field here would have shipped
    unloadable checkpoints with the whole suite still green.
    """
    return {
        "type": "hvla_flow_s1",
        "training_target_contract_version": TRAINING_TARGET_CONTRACT_VERSION,
        "feature_contract_version": FlowMatchingS1Config.FEATURE_CONTRACT_VERSION,
        "action_dim": config.action_dim,
        "action_feature_names": config.action_feature_names,
        "use_relative_actions": config.use_relative_actions,
        "robot_state_feature": config.robot_state_feature,
        "state_dim": config.state_dim,
        "state_feature_names": config.state_feature_names,
        "state_position_std_floor": config.state_position_std_floor,
        "chunk_size": config.chunk_size,
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "num_encoder_layers": config.num_encoder_layers,
        "num_decoder_layers": config.num_decoder_layers,
        "dim_feedforward": config.dim_feedforward,
        "s2_latent_dim": config.s2_latent_dim,
        "s2_proj_hidden": config.s2_proj_hidden,
        "num_inference_steps": config.num_inference_steps,
        "rtc_max_delay": config.rtc_max_delay,
        "rtc_drop_prob": config.rtc_drop_prob,
        "rtc_soft_len": config.rtc_soft_len,
        "rtc_soft_hmax": config.rtc_soft_hmax,
        "use_dino_backbone": config.use_dino_backbone,
        "backbone_dim": config.backbone_dim,
        "freeze_backbone": config.freeze_backbone,
        "image_features": config.image_features,
        "image_resize_shape": config.image_resize_shape,
        "dino_model": config.dino_model,
    }


def validate_resume_training_contract(checkpoint_data: dict, current_config: FlowMatchingS1Config) -> None:
    """Reject resumes whose target meaning differs from the current trainer."""
    if checkpoint_data.get("training_target_contract_version") != TRAINING_TARGET_CONTRACT_VERSION:
        raise ValueError(
            "This checkpoint predates episode-safe action targets and must not be resumed; "
            "start a fresh training run"
        )
    if bool(checkpoint_data.get("use_relative_actions", False)) != current_config.use_relative_actions:
        raise ValueError(
            "Cannot change use_relative_actions while resuming: action normalization and target "
            "semantics differ; start a fresh training run"
        )


def configure_from_dataset_features(
    config: FlowMatchingS1Config,
    features: dict,
    *,
    resize_to: tuple[int, int] | None,
    cameras: list[str] | None = None,
) -> None:
    """Resolve the S1 input/output contract from LeRobot dataset metadata.

    The metadata is the only source of truth here: robot type, motor count,
    state layout, and camera names are deliberately not inferred from names.

    ``cameras`` restricts which visual features become model inputs, named
    either bare (``top_l``) or fully (``observation.images.top_l``). Default is
    every camera in the dataset. An unknown name is an error rather than a
    silent no-op, because a typo would otherwise train a model on more cameras
    than intended and only surface at deployment.
    """
    try:
        action_feature = features["action"]
    except KeyError as exc:
        raise ValueError("HVLA Flow S1 training requires an 'action' feature") from exc

    action_shape = tuple(action_feature.get("shape", ()))
    if len(action_shape) != 1 or action_shape[0] <= 0:
        raise ValueError(f"HVLA Flow S1 requires a 1-D action feature, got shape={action_shape}")
    config.action_dim = int(action_shape[0])
    config.action_feature_names = list(action_feature.get("names") or [])
    if len(config.action_feature_names) != config.action_dim:
        raise ValueError(
            "Action metadata must provide one ordered name per value: "
            f"{len(config.action_feature_names)} names for {config.action_dim} values"
        )

    state_feature = features.get("observation.state")
    if state_feature is None:
        config.robot_state_feature = False
        config.state_dim = 0
        config.state_feature_names = []
    else:
        state_shape = tuple(state_feature.get("shape", ()))
        if len(state_shape) != 1 or state_shape[0] <= 0:
            raise ValueError(
                f"HVLA Flow S1 requires a 1-D observation.state feature, got shape={state_shape}"
            )
        config.robot_state_feature = True
        config.state_dim = int(state_shape[0])
        config.state_feature_names = list(state_feature.get("names") or [])
        if len(config.state_feature_names) != config.state_dim:
            raise ValueError(
                "State metadata must provide one ordered name per value: "
                f"{len(config.state_feature_names)} names for {config.state_dim} values"
            )

    image_keys = [
        key
        for key, feature in features.items()
        if key.startswith("observation.images.")
        and len(tuple(feature.get("shape", ()))) == 3
        and feature.get("dtype") in {"image", "video"}
    ]
    if not image_keys:
        raise ValueError(
            "HVLA Flow S1 training requires at least one visual feature under observation.images.*"
        )

    if cameras is not None:
        # Name resolution is shared with lerobot-train's dataset.cameras so a name that
        # works for one trainer works for the other, and both refuse the same typos with
        # the same message. Only the discovery rule above is HVLA's own.
        available = len(image_keys)
        image_keys = resolve_camera_keys({key: features[key] for key in image_keys}, cameras)
        logger.info(
            "Cameras: using %d of %d (%s)",
            len(image_keys),
            available,
            ", ".join(camera_name(key) for key in image_keys),
        )

    image_size = resize_to[0] if resize_to is not None else None
    config.image_features = dict.fromkeys(image_keys, image_size)

    # Built as PolicyFeature only to report through the same helper as
    # lerobot-train, so both trainers' logs read alike. This one resolves by
    # literal key, so unlike make_policy it cannot pick a column up by accident.
    inputs: dict[str, PolicyFeature] = {
        cam: PolicyFeature(type=FeatureType.VISUAL, shape=(image_size, image_size))
        for cam in config.image_features
    }
    if config.robot_state_feature:
        inputs["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(config.state_dim,))
    outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(config.action_dim,))}
    log_contract(inputs, outputs, logger=logger)
    config.image_resize_shape = resize_to
    config.validate_feature_contract(require_names=True)


class FlowMatchingDataset(torch.utils.data.Dataset):
    """Dataset with S2 latent loading and delay augmentation.

    Training-time RTC is handled inside the model's forward pass (not here),
    because it operates on the noisy action sequence during flow matching.
    This dataset provides: observations, S2 latent + age, target actions.
    """

    def __init__(
        self,
        lerobot_dataset,
        s2_latents: np.ndarray,  # [N_frames, 2048]
        chunk_size: int = 50,
        max_delay_seconds: float = 0.15,
        fps: float = 30.0,
        resize_to: tuple[int, int] | None = None,
        image_keys: list[str] | None = None,
        exclude_flags: list[str] | None = None,
        external_images: bool = False,
        action_feature_names: list[str] | None = None,
        state_feature_names: list[str] | None = None,
        state_position_std_floor: float = 0.5,
        use_relative_actions: bool = False,
        statistics_indices: Sequence[int] | torch.Tensor | None = None,
        augment_indices: Sequence[int] | torch.Tensor | None = None,
    ):
        self.dataset = lerobot_dataset
        # --data-path gpu: images are produced per BATCH by GpuImagePipeline in
        # the training loop, so the per-sample read must not decode video. The
        # parquet row alone carries everything else this wrapper needs -- state
        # and actions come from preloaded tensors, and the global `index` rides
        # through collation for the pipeline.
        self.external_images = external_images
        self.s2_latents = s2_latents
        self.chunk_size = chunk_size
        self.max_delay_frames = int(max_delay_seconds * fps)
        self.fps = fps
        self.resize_to = resize_to
        self.image_keys = image_keys
        self.action_feature_names = action_feature_names
        self.state_feature_names = state_feature_names
        self.state_position_std_floor = state_position_std_floor
        self.use_relative_actions = use_relative_actions
        # Membership, not a boolean: the validation DataLoader reads the same
        # dataset object through a Subset, so a flag would augment the held-out
        # frames too and make the generalisation curve measure the wrong thing.
        self._augment_indices = None if augment_indices is None else {int(i) for i in augment_indices}

        if not math.isfinite(state_position_std_floor) or state_position_std_floor < 0:
            raise ValueError("State position std floor must be a finite non-negative value")

        # Flags to exclude, resolved through the same helper the generic training
        # path uses, so a flag name means the same thing in both. This trainer
        # builds its own chunks instead of going through delta_timestamps, so
        # the reader's flag boundary never reaches it and has to be applied
        # here -- with identical semantics rather than a similar rule.
        self._flagged_indices = None
        if exclude_flags:
            from lerobot.datasets.dataset_reader import _int_column
            from lerobot.utils.feature_utils import resolve_flag_masks

            masks = resolve_flag_masks(lerobot_dataset.meta.features, exclude_flags)
            hf = lerobot_dataset.hf_dataset
            absolute = _int_column(hf, "index")
            selected = np.zeros(len(absolute), dtype=bool)
            for key, mask in masks.items():
                selected |= (_int_column(hf, key) & mask) != 0
            flagged = absolute[selected]
            if flagged.size:
                self._flagged_indices = np.sort(flagged)

        # Build episode boundaries for clipping. LeRobot v3 no longer exposes
        # ``episode_data_index`` on LeRobotDataset, so the episode_index column
        # is the authoritative source for current datasets. Falling back to a
        # single global interval silently joins the tail of one demonstration
        # to the head of the next and creates impossible action targets.
        self._episode_starts = {}
        self._episode_ends = {}
        if hasattr(lerobot_dataset, "episode_data_index"):
            for ep_idx in range(len(lerobot_dataset.episode_data_index["from"])):
                start = lerobot_dataset.episode_data_index["from"][ep_idx].item()
                end = lerobot_dataset.episode_data_index["to"][ep_idx].item()
                for i in range(start, end):
                    self._episode_starts[i] = start
                    self._episode_ends[i] = end
        elif (
            hasattr(lerobot_dataset, "hf_dataset")
            and "episode_index" in lerobot_dataset.hf_dataset.column_names
        ):
            episode_indices = lerobot_dataset.hf_dataset["episode_index"]
            start = 0
            for end in range(1, len(episode_indices) + 1):
                is_boundary = end == len(episode_indices)
                if not is_boundary:
                    previous = episode_indices[end - 1]
                    current = episode_indices[end]
                    if isinstance(previous, torch.Tensor):
                        previous = previous.item()
                    if isinstance(current, torch.Tensor):
                        current = current.item()
                    is_boundary = current != previous
                if is_boundary:
                    for i in range(start, end):
                        self._episode_starts[i] = start
                        self._episode_ends[i] = end
                    start = end

        # Refuse rather than degrade. The original defect existed precisely
        # because a missing-boundary case fell through to a silent global
        # interval; a loud failure is the only way that cannot recur.
        if len(lerobot_dataset) and len(self._episode_ends) != len(lerobot_dataset):
            raise ValueError(
                "HVLA Flow S1 training requires episode boundaries for every frame; "
                "provide LeRobot v3's episode_index column or the legacy "
                "episode_data_index mapping"
            )

        if statistics_indices is None:
            self._statistics_indices = torch.arange(len(lerobot_dataset), dtype=torch.long)
        else:
            self._statistics_indices = torch.as_tensor(statistics_indices, dtype=torch.long)
            if self._statistics_indices.ndim != 1 or len(self._statistics_indices) == 0:
                raise ValueError("Normalization statistics require a non-empty 1-D frame-index list")
            if self._statistics_indices.min() < 0 or self._statistics_indices.max() >= len(lerobot_dataset):
                raise ValueError("Normalization statistics frame indices are outside the dataset")

        # Preload all actions into memory.
        # Avoids calling dataset[i] 50 times per sample for chunk construction
        import logging as _log

        _log.getLogger(__name__).info("Preloading actions for chunk construction...")
        if hasattr(lerobot_dataset, "hf_dataset") and "action" in lerobot_dataset.hf_dataset.column_names:
            action_data = lerobot_dataset.hf_dataset["action"]
            if isinstance(action_data[0], torch.Tensor):
                self._all_actions = torch.stack(list(action_data)).float()
            else:
                import numpy as _np

                self._all_actions = torch.tensor(_np.array(action_data), dtype=torch.float32)
        else:
            # Fallback: load one by one
            self._all_actions = torch.stack(
                [lerobot_dataset[i]["action"] for i in range(len(lerobot_dataset))]
            )
        _log.getLogger(__name__).info("Actions preloaded: %s", self._all_actions.shape)

        # Preload states before action normalization.  Relative targets need the
        # current raw named position as their reference; their statistics must
        # be computed over valid chunk pairs, not just one-step frame pairs.
        if (
            hasattr(lerobot_dataset, "hf_dataset")
            and "observation.state" in lerobot_dataset.hf_dataset.column_names
        ):
            state_data = lerobot_dataset.hf_dataset["observation.state"]
            if isinstance(state_data[0], torch.Tensor):
                self._all_states_raw = torch.stack(list(state_data)).float()
            else:
                import numpy as _np

                self._all_states_raw = torch.tensor(_np.array(state_data), dtype=torch.float32)
            statistics_states = self._all_states_raw[self._statistics_indices]
            self.state_mean = statistics_states.mean(dim=0)
            self.state_std = statistics_states.std(dim=0).clamp(min=1e-6)
            if state_position_std_floor > 0:
                if state_feature_names is None or len(state_feature_names) != self._all_states_raw.shape[1]:
                    raise ValueError(
                        "A positive state position std floor requires one ordered state feature name "
                        f"per state value; got {0 if state_feature_names is None else len(state_feature_names)} "
                        f"names for {self._all_states_raw.shape[1]} values"
                    )
                position_mask = torch.tensor(
                    [name.endswith(".pos") for name in state_feature_names], dtype=torch.bool
                )
                if not position_mask.any():
                    raise ValueError(
                        "A positive state position std floor was requested but no state feature name "
                        "ends in '.pos'"
                    )
                raw_position_std = self.state_std[position_mask].clone()
                self.state_std[position_mask] = self.state_std[position_mask].clamp(
                    min=state_position_std_floor
                )
                floored_count = int((raw_position_std < state_position_std_floor).sum().item())
                _log.getLogger(__name__).info(
                    "State position std floor: %.6g (dataset units), applied to %d/%d position features",
                    state_position_std_floor,
                    floored_count,
                    int(position_mask.sum().item()),
                )
            normalized = (self._all_states_raw - self.state_mean) / self.state_std
            # Same bound the policy applies at inference, so the model is trained
            # on exactly the inputs it will be served. Without this the clamp
            # would be a train/serve skew on the frames it touches.
            clamped_frames = int((normalized.abs() > NORMALIZED_STATE_CLAMP).any(dim=1).sum().item())
            self._all_states = normalized.clamp(-NORMALIZED_STATE_CLAMP, NORMALIZED_STATE_CLAMP)
            _log.getLogger(__name__).info(
                "States preloaded and normalized: %s; %d/%d frames (%.2f%%) had at least one "
                "feature beyond +/-%g sigma and were clamped",
                self._all_states.shape,
                clamped_frames,
                normalized.shape[0],
                100.0 * clamped_frames / max(normalized.shape[0], 1),
                NORMALIZED_STATE_CLAMP,
            )
        else:
            self._all_states_raw = None
            self._all_states = None
            self.state_mean = None
            self.state_std = None

        if use_relative_actions:
            if self._all_states_raw is None:
                raise ValueError("Relative action training requires observation.state")
            if action_feature_names is None or len(action_feature_names) != self._all_actions.shape[1]:
                raise ValueError(
                    "Relative action training requires one ordered action feature name per action value"
                )
            if state_feature_names is None or len(state_feature_names) != self._all_states_raw.shape[1]:
                raise ValueError(
                    "Relative action training requires one ordered state feature name per state value"
                )
            missing = sorted(set(action_feature_names) - set(state_feature_names))
            if missing:
                raise ValueError(
                    f"Relative action training requires matching named state positions; missing {missing}"
                )
            self._relative_action_state_indices = torch.tensor(
                [state_feature_names.index(name) for name in action_feature_names], dtype=torch.long
            )
            self._relative_action_mask = torch.tensor(
                [name.endswith(".pos") and "gripper" not in name.lower() for name in action_feature_names],
                dtype=torch.bool,
            )
            if not self._relative_action_mask.any():
                raise ValueError("Relative action training found no non-gripper *.pos action features")

            action_sum = torch.zeros(self._all_actions.shape[1], dtype=torch.float64)
            action_square_sum = torch.zeros_like(action_sum)
            action_count = 0
            episode_bounds = sorted(
                {(self._episode_starts[i], self._episode_ends[i]) for i in self._episode_starts}
            )
            statistics_mask = torch.zeros(len(self.dataset), dtype=torch.bool)
            statistics_mask[self._statistics_indices] = True
            for start, end in episode_bounds:
                selected = statistics_mask[start:end]
                if selected.any() and not selected.all():
                    raise ValueError("Relative action normalization statistics must select complete episodes")
                if not selected.any():
                    continue
                for offset in range(min(self.chunk_size, end - start)):
                    targets = self._all_actions[start + offset : end]
                    current_positions = self._all_states_raw[
                        start : end - offset, self._relative_action_state_indices
                    ]
                    relative = torch.where(
                        self._relative_action_mask,
                        targets - current_positions,
                        targets,
                    ).double()
                    action_sum += relative.sum(dim=0)
                    action_square_sum += relative.square().sum(dim=0)
                    action_count += relative.shape[0]
            self.action_mean = (action_sum / action_count).float()
            variance = (action_square_sum - action_sum.square() / action_count) / max(action_count - 1, 1)
            self.action_std = variance.clamp(min=0).sqrt().float().clamp(min=1e-6)
            _log.getLogger(__name__).info(
                "Relative action mode: %d/%d dimensions anchored; stats from %d valid chunk tokens",
                int(self._relative_action_mask.sum()),
                len(self._relative_action_mask),
                action_count,
            )
        else:
            self._relative_action_state_indices = None
            self._relative_action_mask = None
            statistics_actions = self._all_actions[self._statistics_indices]
            self.action_mean = statistics_actions.mean(dim=0)
            self.action_std = statistics_actions.std(dim=0).clamp(min=1e-6)
            self._all_actions = (self._all_actions - self.action_mean) / self.action_std

        _log.getLogger(__name__).info(
            "Action norm stats: mean=[%.1f..%.1f] std=[%.1f..%.1f]",
            self.action_mean.min(),
            self.action_mean.max(),
            self.action_std.min(),
            self.action_std.max(),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.external_images:
            sample = dict(self.dataset.hf_dataset[int(idx)])
        else:
            sample = self.dataset[idx]
        ep_start = self._episode_starts.get(idx, 0)
        ep_end = self._episode_ends.get(idx, len(self.dataset))
        # A flagged frame ends the chunk exactly as the episode end does:
        # positions from it onward clamp to the last good action and are marked
        # padding. Same rule as DatasetReader._get_query_indices and for the
        # same reason -- truncating keeps the supervised actions contiguous,
        # where masking a gap would train the model to jump across data we
        # chose not to trust.
        if self._flagged_indices is not None:
            position = int(np.searchsorted(self._flagged_indices, idx, side="left"))
            if position < self._flagged_indices.size:
                ep_end = min(ep_end, int(self._flagged_indices[position]))

        # --- Build action chunk: [chunk_size, action_dim] (already normalized) ---
        # Clamped at both ends. When the start frame is itself flagged, ep_end
        # equals idx and the upper clamp alone yields idx - 1, which at idx 0
        # is -1 and silently reads the last row of the whole dataset. The lower
        # clamp keeps the read inside this episode; every position is padding
        # in that case anyway.
        chunk_floor = self._episode_starts.get(idx, 0)
        indices = torch.arange(idx, idx + self.chunk_size).clamp(max=ep_end - 1).clamp(min=chunk_floor)
        actions = self._all_actions[indices]
        if self.use_relative_actions:
            current_positions = self._all_states_raw[idx, self._relative_action_state_indices]
            actions = torch.where(
                self._relative_action_mask,
                actions - current_positions,
                actions,
            )
            actions = (actions - self.action_mean) / self.action_std
        sample["action"] = actions
        sample["action_is_pad"] = torch.arange(self.chunk_size) >= (ep_end - idx)

        # --- Use normalized state ---
        if self._all_states is not None:
            sample["observation.state"] = self._all_states[idx]

        # --- S2 latent with delay augmentation (skip if training without S2) ---
        if self.s2_latents is not None:
            k = np.random.randint(0, self.max_delay_frames + 1)
            delayed_idx = max(idx - k, ep_start)
            s2_latent = torch.from_numpy(self.s2_latents[delayed_idx]).float()
            age_seconds = k / self.fps

            sample[S2_LATENT_KEY] = s2_latent
            sample[S2_AGE_KEY] = torch.tensor([age_seconds], dtype=torch.float32)

        # The underlying dataset decodes every camera it has. Anything not
        # selected travels into the batch at SOURCE resolution and is collated
        # through shared memory for nothing -- with one of four cameras selected
        # that is three full-size frames per sample, which exhausts a
        # container's /dev/shm and kills the loader workers.
        #
        # Deliberately outside the resize guard below: a run that does not
        # resize still pays the full collation cost, which is the case this
        # exists to prevent.
        if self.image_keys and not self.external_images:
            for key in [k for k in sample if k.startswith("observation.images.")]:
                if key not in self.image_keys:
                    del sample[key]

        # --- Augment (training frames only), then resize ---
        if self.image_keys:
            import torchvision.transforms.functional as TF

            augment = self._augment_indices is not None and idx in self._augment_indices
            for key in self.image_keys:
                image = sample.get(key)
                if not isinstance(image, torch.Tensor):
                    continue
                if augment:
                    image = self._augment_image(image)
                if self.resize_to is not None:
                    image = TF.resize(
                        image,
                        list(self.resize_to),
                        interpolation=TF.InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                sample[key] = image

        return sample

    # Crop keeps at least this fraction of the frame area. Small on purpose:
    # enough to stop the model keying on absolute pixel position, not so much
    # that it implies a camera that moved.
    AUG_MIN_AREA = 0.85

    def _augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """Jitter one camera's frame. Preconditions: CHW tensor, no batch dim.

        Drawn independently per camera because the cameras are physically
        separate; a shared crop would model a rig that cannot exist. The crop
        relies on the caller's resize to restore the target size, so it is
        skipped when no resize is configured -- otherwise frames in a batch
        would disagree on shape.
        """
        import torchvision.transforms.functional as TF

        if image.dtype != torch.float32:
            image = image.float() / 255.0

        if self.resize_to is not None:
            height, width = image.shape[-2:]
            keep = math.sqrt(float(np.random.uniform(self.AUG_MIN_AREA, 1.0)))
            crop_h, crop_w = max(1, int(round(height * keep))), max(1, int(round(width * keep)))
            top = int(np.random.randint(0, height - crop_h + 1))
            left = int(np.random.randint(0, width - crop_w + 1))
            image = image[..., top : top + crop_h, left : left + crop_w]

        image = TF.adjust_brightness(image, float(np.random.uniform(0.7, 1.3)))
        image = TF.adjust_contrast(image, float(np.random.uniform(0.7, 1.3)))
        image = TF.adjust_saturation(image, float(np.random.uniform(0.7, 1.3)))
        image = TF.adjust_hue(image, float(np.random.uniform(-0.03, 0.03)))
        return image.clamp(0.0, 1.0)


def _resolve_data_path(choice, config, dataset, resize_to, device, batch_size):
    """Return a GpuImagePipeline for the GPU data path, or None for the CPU one.

    ``auto`` (the default) uses the GPU path wherever it is supported and falls
    back to the CPU path with the reason logged. ``gpu`` and ``cpu`` are
    honoured exactly: an explicit ``gpu`` that cannot be satisfied stops the
    run rather than quietly training on the other path, because a run that
    asked for one path and silently got the other is how three benchmark runs
    were measured wrong in a single day.

    The auto criteria are checked facts, not guesses: CUDA is the device; the
    mask recipe is one GpuMaskComposite implements (it refuses the rest); CUDA
    decode of this
    dataset's own video reproduces the CPU decoder's pixels (some codecs decode
    to garbage without erroring); and the estimated peak working set fits in
    free VRAM with headroom.
    """
    assert choice in ("auto", "cpu", "gpu"), f"unknown data path {choice!r}"
    if choice == "cpu":
        logger.info("Data path: CPU (requested)")
        return None
    try:
        if not str(device).startswith("cuda"):
            raise NotImplementedError(f"device is {device}, not CUDA")
        from lerobot.datasets.gpu_data_pipeline import GpuImagePipeline

        # Constructing the pipeline calibrates and VERIFIES the GPU decode of
        # this dataset's own video against the CPU decoder, per camera, and
        # raises if it cannot be reproduced.
        pipeline = GpuImagePipeline(
            dataset, list(config.image_features.keys()), resize_to=resize_to, device=device
        )
        # Measured, not estimated: prepare one real batch and read the peak.
        # An arithmetic estimate of the working set was 4.4x under the observed
        # 7769 MB, and a gate that admits the GPU path on a machine where it
        # will not fit is worse than no gate. This costs one batch at startup.
        indices = torch.arange(min(batch_size, dataset.num_frames))
        trial = {"index": indices}
        # A pipeline that composites needs the mask rows in its trial batch; one
        # that only decodes and resizes has no mask_key at all.
        for key in getattr(pipeline, "mask_key", {}).values():
            trial[key] = [dataset.hf_dataset[int(i)][key] for i in indices]
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.mem_get_info()[0]
        pipeline.prepare(trial)
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        if peak > before:
            raise NotImplementedError(
                f"a batch needs {peak / (1 << 30):.1f} GiB, {before / (1 << 30):.1f} GiB was free"
            )
        logger.info("GPU data path working set: %.1f GiB per batch", peak / (1 << 30))
    except Exception as e:
        if choice == "gpu":
            raise
        logger.warning("Data path: CPU (GPU path unavailable — %s: %s)", type(e).__name__, e)
        return None
    logger.info("Data path: GPU (NVDEC decode + on-device composite/resize)")
    return pipeline
def seed_training(seed: int | None) -> torch.Generator | None:
    """Seed model initialization, augmentation, and DataLoader sampling.

    ``None`` intentionally preserves the legacy unseeded behavior. Paired
    experiments must pass the same explicit seed to both runs.
    """
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_data_worker(worker_id: int) -> None:
    """Derive NumPy/Python worker RNGs from PyTorch's per-worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_train_validation_frames_by_episode(
    episode_indices: Sequence[int] | torch.Tensor,
    *,
    validation_fraction: float,
    seed: int | None,
) -> tuple[list[int], list[int], list[int]]:
    """Return train frames, validation frames, and held-out episode IDs.

    Frames from one demonstration must never appear in both splits.  A random
    frame split would leak adjacent images and overlapping action chunks into
    validation, making its loss nearly indistinguishable from training loss.
    """
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and in [0, 1)")
    values = [
        int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in episode_indices
    ]
    if not values:
        raise ValueError("Cannot split an empty dataset")

    groups: list[tuple[int, int, int]] = []
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] != values[end - 1]:
            groups.append((values[start], start, end))
            start = end

    if validation_fraction == 0:
        return list(range(len(values))), [], []
    if len(groups) < 2:
        raise ValueError("Episode-held-out validation requires at least two episodes")

    validation_count = min(max(1, round(len(groups) * validation_fraction)), len(groups) - 1)
    rng = np.random.default_rng(0 if seed is None else seed)
    validation_group_indices = set(rng.permutation(len(groups))[:validation_count].tolist())
    train_frames: list[int] = []
    validation_frames: list[int] = []
    validation_episode_ids: list[int] = []
    for group_index, (episode_id, group_start, group_end) in enumerate(groups):
        target = validation_frames if group_index in validation_group_indices else train_frames
        target.extend(range(group_start, group_end))
        if group_index in validation_group_indices:
            validation_episode_ids.append(episode_id)
    rng.shuffle(validation_frames)
    return train_frames, validation_frames, sorted(validation_episode_ids)


def train(args):
    """Main training loop."""
    import sys

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sampler import make_start_sampler
    from lerobot.datasets.sampling_trace import (
        DIRNAME as SAMPLING_TRACE_DIRNAME,
        save_sampling_trace,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    logging.getLogger().handlers[0].stream = sys.stderr  # ensure unbuffered

    if args.validation_batches <= 0:
        raise ValueError("validation_batches must be positive")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-create train.log in output dir (appends on resume)
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("Command: %s", " ".join(sys.argv))
    data_generator = seed_training(args.seed)
    logger.info("Training seed: %s", args.seed if args.seed is not None else "unseeded (legacy)")

    # Parse resize before resolving the feature contract.
    resize_to = None
    if args.resize_images:
        h, w = (int(x) for x in args.resize_images.split("x"))
        resize_to = (h, w)

    # Load config
    encoder = vision_encoders.resolve(args.vision_encoder)
    config = FlowMatchingS1Config(
        chunk_size=args.chunk_size,
        num_inference_steps=args.num_inference_steps,
        rtc_max_delay=args.rtc_max_delay,
        rtc_drop_prob=args.rtc_drop_prob,
        hidden_dim=args.hidden_dim,
        num_decoder_layers=args.num_decoder_layers,
        dino_model=args.vision_encoder,
        # Derived, not asked for: the encoder determines its own token width,
        # and a hand-set backbone_dim that disagrees is a shape error one batch
        # into training.
        backbone_dim=encoder.embed_dim,
        state_position_std_floor=args.state_position_std_floor,
        use_relative_actions=args.use_relative_actions,
        freeze_backbone=args.freeze_backbone,
        image_augmentation=args.image_augmentation,
        backbone_lr_scale=args.backbone_lr_scale,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )
    logger.info(
        "Vision encoder: %s (%s, %d-d patch tokens, patch %d)",
        args.vision_encoder,
        encoder.label,
        encoder.embed_dim,
        encoder.patch_size,
    )
    # Load dataset
    logger.info("Loading dataset: %s", args.dataset_repo_id)
    # Saved masks are part of the dataset, not a training option: a dataset that
    # carries mask columns was deliberately masked, and reading it without them
    # trains on pixels nobody chose. This script builds LeRobotDataset directly
    # rather than through datasets/factory.py, so it does NOT inherit the
    # `apply_saved_masks: True` default in configs/default.py — which is how a
    # masked-dataset run silently consumed raw frames and looked entirely
    # normal doing it. Stated explicitly here, and logged, so the run's own log
    # answers "did this train on masks?".
    lerobot_dataset = LeRobotDataset(args.dataset_repo_id, apply_saved_masks=not args.ignore_saved_masks)
    from lerobot.datasets.mask_compositing import MASK_NAMESPACE

    _mask_keys = [k for k in lerobot_dataset.meta.features if k.startswith(f"{MASK_NAMESPACE}.")]
    if args.ignore_saved_masks:
        logger.warning("Saved masks IGNORED by request (--ignore-saved-masks); training on raw frames")
    elif _mask_keys:
        logger.info("Saved masks ACTIVE for %s", ", ".join(sorted(_mask_keys)))
    else:
        logger.info("Dataset carries no saved masks; training on raw frames")
    configure_from_dataset_features(
        config,
        lerobot_dataset.meta.features,
        resize_to=resize_to,
        cameras=args.cameras,
    )

    gpu_pipeline = _resolve_data_path(
        args.data_path, config, lerobot_dataset, resize_to, device, args.batch_size
    )

    logger.info(
        "Config: action=%d, state=%d, cameras=%s, chunk=%d, hidden=%d, "
        "dec_layers=%d, rtc_max_delay=%d, rtc_drop=%.2f, denoise_steps=%d",
        config.action_dim,
        config.state_dim,
        list(config.image_features),
        config.chunk_size,
        config.hidden_dim,
        config.num_decoder_layers,
        config.rtc_max_delay,
        config.rtc_drop_prob,
        config.num_inference_steps,
    )
    logger.info("State position std floor: %.6g (dataset-native units)", config.state_position_std_floor)
    logger.info("Relative arm actions: %s", config.use_relative_actions)

    # Load S2 latents (optional — train without S2 conditioning if omitted)
    s2_latents = None
    if args.s2_latent_path:
        logger.info("Loading S2 latents from %s", args.s2_latent_path)
        s2_latents = np.load(args.s2_latent_path)
        logger.info("S2 latents shape: %s", s2_latents.shape)
    else:
        logger.info("No S2 latent path provided — training without S2 conditioning")

    train_frame_indices, validation_frame_indices, validation_episode_ids = (
        split_train_validation_frames_by_episode(
            lerobot_dataset.hf_dataset["episode_index"],
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    )
    logger.info(
        "Episode split: train=%d frames | validation=%d frames across %d held-out episodes",
        len(train_frame_indices),
        len(validation_frame_indices),
        len(validation_episode_ids),
    )

    # Wrap dataset
    exclude_flags = [f.strip() for f in (args.exclude_flags or "").split(",") if f.strip()]
    dataset = FlowMatchingDataset(
        lerobot_dataset,
        s2_latents=s2_latents,
        chunk_size=config.chunk_size,
        max_delay_seconds=args.max_delay,
        resize_to=resize_to,
        image_keys=list(config.image_features.keys()),
        exclude_flags=exclude_flags,
        external_images=gpu_pipeline is not None,
        action_feature_names=config.action_feature_names,
        state_feature_names=config.state_feature_names,
        state_position_std_floor=config.state_position_std_floor,
        use_relative_actions=config.use_relative_actions,
        statistics_indices=train_frame_indices,
        augment_indices=train_frame_indices if config.image_augmentation else None,
    )
    logger.info(
        "Image augmentation: %s (training frames only)",
        "on — crop >=%.0f%% area + brightness/contrast/saturation/hue jitter" % (100 * FlowMatchingDataset.AUG_MIN_AREA)
        if config.image_augmentation
        else "off",
    )
    # Logged whether or not anything was excluded, and worded exactly as the
    # generic trainer's line, so the two read the same in a log.
    _flagged = dataset._flagged_indices
    logger.info(
        "Flags to exclude: %s -- %d of %d frames (%.2f%%). "
        "Each ends the action window of any chunk reaching it.",
        ", ".join(exclude_flags) if exclude_flags else "nothing",
        0 if _flagged is None else int(_flagged.size),
        len(lerobot_dataset),
        0.0
        if _flagged is None or not len(lerobot_dataset)
        else 100.0 * int(_flagged.size) / len(lerobot_dataset),
    )
    # Two independent reasons a start is not drawn, carried by one sampler:
    # the frame carries an excluded flag, or its episode is held out for
    # validation. The holdout goes through `episode_indices_to_use` because the
    # split is by episode and that is the parameter for it -- a Subset would
    # renumber positions underneath a sampler that speaks absolute frames.
    _held_out = set(validation_episode_ids)
    training_episode_ids = [
        episode for episode in range(lerobot_dataset.meta.total_episodes)
        if episode not in _held_out
    ]
    sampler = make_start_sampler(
        lerobot_dataset.meta.episodes["dataset_from_index"],
        lerobot_dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=training_episode_ids,
        excluded_frames=dataset._flagged_indices,
        trace_dir=output_dir / SAMPLING_TRACE_DIRNAME,
        shuffle=True,
        seed=args.seed,
    )
    logger.info(
        "Sampling from %d starts (%d episodes held out for validation, %d frames flagged)",
        len(sampler),
        len(_held_out),
        0 if _flagged is None else int(_flagged.size),
    )
    training_dataset = dataset
    dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_data_worker if args.seed is not None else None,
        generator=data_generator,
    )
    validation_dataloader = None
    if validation_frame_indices:
        validation_dataloader = DataLoader(
            Subset(dataset, validation_frame_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 4),
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_data_worker if args.seed is not None else None,
        )

    # Generation probes over fixed frames, so the curve is comparable across
    # steps rather than re-sampled each time.
    #
    # These must NOT be single-process. Decoding in the main process leaves a
    # torchcodec thread pool behind (dataset_reader fans frame reads out over
    # threads), and the training loader forks fresh workers at every epoch
    # boundary; a child forked from a process with live decoder threads
    # inherits locks no surviving thread will release and dies with "Could not
    # push packet to decoder: Invalid data found". That killed two runs at the
    # first epoch boundary after an evaluation.
    #
    # Suppressing augmentation by mutating the shared dataset still works with
    # workers, because the fork happens when the iterator is created — after
    # evaluate_generation has already cleared the flag.
    generalization_train_loader = None
    generalization_val_loader = None
    if args.eval_generation_batches > 0:
        probe_rng = np.random.default_rng((0 if args.seed is None else args.seed) + 20_000)
        probe_size = args.eval_generation_batches * args.batch_size

        def _probe_loader(frames):
            if not frames:
                return None
            order = probe_rng.permutation(len(frames))[:probe_size]
            return DataLoader(
                Subset(dataset, [frames[int(i)] for i in order]),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, min(args.num_workers, 4)),
                pin_memory=True,
                drop_last=False,
            )

        generalization_train_loader = _probe_loader(train_frame_indices)
        generalization_val_loader = _probe_loader(validation_frame_indices)

    # Build model
    logger.info("Building FlowMatchingS1 model...")

    # TF32 matmul precision — free ~2× speedup on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")

    policy = FlowMatchingS1Policy(config).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(
        "Total params: %d (%.1fM) | Trainable: %d (%.1fM)",
        total_params,
        total_params / 1e6,
        trainable_params,
        trainable_params / 1e6,
    )

    # Optimizer with cosine LR schedule (matching Pi0/SmolVLA). The vision
    # backbone is a separate param group so its LR can be damped without
    # freezing it outright — full-rate fine-tuning memorises the recorded
    # scenes (see FlowMatchingS1Config.freeze_backbone).
    backbone_params, expert_params = [], []
    for parameter_name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        target = backbone_params if ".backbone." in f".{parameter_name}" else expert_params
        target.append(parameter)
    if config.backbone_lr_scale != 1.0 and not config.freeze_backbone and not backbone_params:
        raise ValueError(
            "backbone_lr_scale is set but no backbone parameters were matched; "
            "the module layout changed and the scale would silently do nothing"
        )
    param_groups = [{"params": expert_params, "lr": config.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.lr * config.backbone_lr_scale})
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    logger.info(
        "Optimizer: expert %.1fM params @ lr %.2e | backbone %.1fM params @ lr %.2e (scale %.3g)%s",
        sum(p.numel() for p in expert_params) / 1e6,
        config.lr,
        sum(p.numel() for p in backbone_params) / 1e6,
        config.lr * config.backbone_lr_scale,
        config.backbone_lr_scale,
        " [FROZEN]" if config.freeze_backbone else "",
    )

    # Cosine decay: warmup → peak_lr → decay to lr_decay
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)  # linear warmup
        progress = (step - config.warmup_steps) / max(args.steps - config.warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = config.lr_decay / config.lr
        return min_ratio + (1 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        import json

        resume_dir = Path(args.resume)
        # Support both standard (pretrained_model/) and legacy (flat) formats
        pretrained_dir = resume_dir / "pretrained_model"
        training_state_dir = resume_dir / "training_state"
        if pretrained_dir.is_dir():
            model_path = pretrained_dir / "model.safetensors"
            opt_path = training_state_dir / "optimizer.pt"
        else:
            model_path = resume_dir / "model.safetensors"
            opt_path = resume_dir / "optimizer.pt"

        resume_config_path = (
            pretrained_dir / "config.json" if pretrained_dir.is_dir() else resume_dir / "config.json"
        )
        if not resume_config_path.exists():
            raise ValueError("Cannot resume HVLA training without the checkpoint's config.json")
        resume_config = json.loads(resume_config_path.read_text())
        validate_resume_training_contract(resume_config, config)

        if model_path.exists():
            import safetensors.torch as sft

            state_dict = sft.load_file(str(model_path))
            policy.load_state_dict(state_dict, strict=False)
            logger.info("Resumed model from %s", model_path)
        if opt_path.exists():
            opt_state = torch.load(str(opt_path), weights_only=True, map_location=device)
            optimizer.load_state_dict(opt_state["optimizer"])
            scheduler.load_state_dict(opt_state["scheduler"])
            start_step = opt_state.get("step", 0)
            logger.info("Resumed optimizer from %s (step %d)", opt_path, start_step)
        else:
            # Try to infer step from directory name
            try:
                start_step = int(resume_dir.name.split("-")[-1])
                for _ in range(start_step):
                    scheduler.step()
                logger.info("Resumed from step %d (no optimizer state, LR schedule advanced)", start_step)
            except ValueError:
                pass

    # Mixed precision
    use_amp = device.type == "cuda"
    if use_amp:
        logger.info("Using bf16 mixed precision + TF32 matmul")
    logger.info(
        "LR schedule: warmup %d → peak %.1e → cosine decay → %.1e",
        config.warmup_steps,
        config.lr,
        config.lr_decay,
    )

    @torch.no_grad()
    def evaluate_validation(step: int) -> float | None:
        if validation_dataloader is None:
            return None
        was_training = policy.training
        policy.eval()
        losses = []
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()] if use_amp else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed((0 if args.seed is None else args.seed) + 10_000)
            for batch_index, batch in enumerate(validation_dataloader):
                if batch_index >= args.validation_batches:
                    break
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, _ = policy(batch)
                losses.append(float(loss))
        policy.train(was_training)
        validation_loss = float(np.mean(losses))
        logger.info(
            "validation step %d | held-out episodes=%d | batches=%d | flow_loss: %.6f",
            step,
            len(validation_episode_ids),
            len(losses),
            validation_loss,
        )
        return validation_loss

    @torch.no_grad()
    def evaluate_generation(loader, batch_limit: int, label: str, step: int) -> dict | None:
        """Open-loop chunk error, scaled by the error of using no vision at all.

        Flow-matching validation loss averages over random denoising timesteps,
        is dominated by the easy ones, and barely moves while the model is busy
        memorising which recorded scene it is looking at. This instead runs the
        real sampler and compares the generated chunk against the recorded one.

        The scale is the null model "always emit the dataset-mean action", which
        is the zero vector in normalized space and needs no images. ``ratio``
        near 1.0 means the model is worth no more than ignoring its inputs, and
        the train-to-held-out gap in that ratio is the overfitting signal.

        Augmentation is suppressed for the duration so the two splits are
        compared on identical footing. The flag is cleared before the loader is
        iterated, so the workers forked at that moment inherit the cleared
        value; the probe loaders deliberately use workers rather than decoding
        here (see where they are built).
        """
        if loader is None or batch_limit <= 0:
            return None
        was_training = policy.training
        policy.eval()
        previous_augment = dataset._augment_indices
        dataset._augment_indices = None
        errors, nulls = [], []
        try:
            devices = (
                [device.index if device.index is not None else torch.cuda.current_device()]
                if use_amp
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed((0 if args.seed is None else args.seed) + 20_000)
                for batch_index, batch in enumerate(loader):
                    if batch_index >= batch_limit:
                        break
                    batch = {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    # The sampler lives on the inner module, which expects the
                    # image list the policy's training forward assembles; go
                    # through the same mapping rather than a raw batch.
                    model_batch = dict(batch)
                    if config.image_features:
                        model_batch[OBS_IMAGES] = [batch[key] for key in config.image_features]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        generated = policy.model.sample_actions(
                            model_batch, num_steps=config.num_inference_steps
                        )
                    target = batch["action"].float()
                    generated = generated.float()
                    valid = ~batch["action_is_pad"]
                    if not bool(valid.any()):
                        continue
                    errors.append(float(torch.linalg.vector_norm(generated - target, dim=-1)[valid].mean()))
                    nulls.append(float(torch.linalg.vector_norm(target, dim=-1)[valid].mean()))
        finally:
            dataset._augment_indices = previous_augment
            policy.train(was_training)
        if not errors:
            return None
        error = float(np.mean(errors))
        null = float(np.mean(nulls))
        result = {"chunk_error": error, "null_error": null, "ratio": error / max(null, 1e-9)}
        logger.info(
            "generation step %d | %-8s | chunk_err %.4f | null %.4f | ratio %.3f",
            step,
            label,
            error,
            null,
            result["ratio"],
        )
        return result

    def evaluate_generalization(step: int) -> dict:
        """Both splits plus their gap — the number that exposes memorisation."""
        train_side = evaluate_generation(
            generalization_train_loader, args.eval_generation_batches, "train", step
        )
        heldout_side = evaluate_generation(
            generalization_val_loader, args.eval_generation_batches, "held-out", step
        )
        summary = {"train": train_side, "held_out": heldout_side}
        if train_side and heldout_side:
            gap = heldout_side["ratio"] - train_side["ratio"]
            summary["ratio_gap"] = gap
            logger.info(
                "generation step %d | GAP held-out minus train ratio: %+.3f "
                "(larger = memorising the recorded placements)",
                step,
                gap,
            )
        return summary

    # Save norm stats for inference (must denormalize model output)
    norm_stats = {
        "action_mean": dataset.action_mean,
        "action_std": dataset.action_std,
    }
    if dataset.state_mean is not None:
        norm_stats["state_mean"] = dataset.state_mean
        norm_stats["state_std"] = dataset.state_std

    validation_loss_cache: dict[int, float | None] = {}

    def save_checkpoint(step):
        import json

        import safetensors.torch as sft

        if step not in validation_loss_cache:
            validation_loss_cache[step] = evaluate_validation(step)
        validation_loss = validation_loss_cache[step]
        generalization = evaluate_generalization(step)

        ckpts_dir = output_dir / "checkpoints"
        ckpts_dir.mkdir(exist_ok=True)
        ckpt_dir = ckpts_dir / f"checkpoint-{step}"

        # Save in standard LeRobot format: pretrained_model/ + training_state/
        pretrained_dir = ckpt_dir / "pretrained_model"
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        training_state_dir = ckpt_dir / "training_state"
        training_state_dir.mkdir(parents=True, exist_ok=True)

        # Model weights
        sft.save_file(
            dict(policy.state_dict().items()),
            str(pretrained_dir / "model.safetensors"),
        )

        # Norm stats (HVLA-specific, alongside model weights)
        torch.save(norm_stats, str(pretrained_dir / "norm_stats.pt"))

        # config.json — identifies this as an HVLA checkpoint
        (pretrained_dir / "config.json").write_text(json.dumps(checkpoint_config_dict(config), indent=2))

        # train_config.json — training args for reproducibility
        train_config = {
            "dataset": {"repo_id": args.dataset_repo_id},
            "s2_latent_path": args.s2_latent_path,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_delay": args.max_delay,
            "resize_images": args.resize_images,
            "state_position_std_floor": args.state_position_std_floor,
            "use_relative_actions": args.use_relative_actions,
            "validation_fraction": args.validation_fraction,
            "validation_batches": args.validation_batches,
            "validation_episode_ids": validation_episode_ids,
            "validation_flow_loss": validation_loss,
            # Open-loop generation error per split, scaled by the vision-free
            # null. Recorded per checkpoint so a finished run can be compared
            # without re-running the sampler.
            "generalization": generalization,
            "freeze_backbone": config.freeze_backbone,
            "backbone_lr_scale": config.backbone_lr_scale,
            "image_augmentation": config.image_augmentation,
            "dino_model": config.dino_model,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "seed": args.seed,
        }
        (pretrained_dir / "train_config.json").write_text(json.dumps(train_config, indent=2))

        # The same artifact the generic trainer writes, in the same format, so
        # one reader answers "what did this run draw" for either.
        save_sampling_trace(
            output_dir / SAMPLING_TRACE_DIRNAME,
            draw_counts=sampler.draw_counts,
            episode_from=lerobot_dataset.meta.episodes["dataset_from_index"],
            episode_to=lerobot_dataset.meta.episodes["dataset_to_index"],
            excluded_frames=dataset._flagged_indices,
            step=step,
        )

        # Training state (optimizer, scheduler, step)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            str(training_state_dir / "optimizer.pt"),
        )
        (training_state_dir / "training_step.json").write_text(json.dumps({"step": step}))

        # Update 'last' symlink
        last_link = ckpts_dir / "last"
        if last_link.exists() or last_link.is_symlink():
            # safe-destruct: symlink update (not user data)
            last_link.unlink()
        last_link.symlink_to(ckpt_dir.name)

        logger.info("Saved checkpoint (step %d): %s", step, ckpt_dir)

    # Training loop
    policy.train()
    step = start_step
    # Advance the sampler to the epoch the run had reached. The order is a pure
    # function of (seed, epoch) now, so without this a resumed run replays the
    # identical first epoch -- which the DataLoader's own shuffling never did,
    # because it was never reproducible in the first place. Sample-exact resume
    # within an epoch is the generic trainer's; this only stops the repeat.
    if start_step > 0 and len(dataloader) > 0:
        sampler.set_epoch(start_step // len(dataloader))
    # On the GPU path the image half of a batch is GPU work, so it is produced
    # one batch ahead on a side stream rather than inline -- inline puts it in
    # series with the model step, so the device does both but never at once.
    # The prefetcher restarts the loader itself, which is why the epoch-boundary
    # branch in the loop below is CPU-path only.
    if gpu_pipeline is not None:
        from lerobot.datasets.gpu_data_pipeline import GpuBatchPrefetcher

        data_iter = GpuBatchPrefetcher(
            dataloader,
            gpu_pipeline,
            device,
            depth=args.prefetch_depth,
        )
    else:
        data_iter = iter(dataloader)
    logger.info("Starting training from step %d to %d...", step, args.steps)
    # Resource telemetry rides this trainer's structured record rather than a
    # flat line, because that is the format it prints. The fields are flat
    # finite numbers, which is all the record accepts.
    resources = ResourceSampler()
    resources.start()
    health = TrainingHealthTracker(
        batch_size=args.batch_size,
        total_steps=args.steps,
        peak_memory_gb=(
            (lambda: torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else None
        ),
        reset_peak_memory=(
            (lambda: torch.cuda.reset_peak_memory_stats(device)) if device.type == "cuda" else None
        ),
    )

    while step < args.steps:
        with health.measure_data_loading():
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # The prefetcher has already moved and prepared the GPU path's
            # batches; only the CPU path's still need the transfer here.
            if gpu_pipeline is None:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward with bf16 autocast
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, loss_dict = policy(batch)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        health.step()
        # The first five updates contain one-time startup work. Logging steps
        # 6-10 gives an early provisional ETA after that cold window; step 100
        # replaces it from the representative 11-100 window, after which the
        # ordinary EMA continues every 100 steps.
        is_initial_eta_step = 6 <= step <= 10
        is_log_step = is_initial_eta_step or step % 100 == 0

        if is_log_step:
            cur_lr = optimizer.param_groups[0]["lr"]
            # These scalar reads synchronize CUDA once per log window. Measure
            # the window after that synchronization so throughput and ETA
            # include real GPU execution without stalling every training step.
            loss_value = loss.item()
            flow_loss_value = float(loss_dict["flow_loss"])
            grad_norm_value = grad_norm.item()
            sample = health.sample(
                step=step,
                reseed_eta=step == 100 and start_step <= 10,
                values={
                    # Telemetry first, so a future field that collides with a
                    # training metric loses to it rather than silently
                    # replacing it. Nothing about this run may depend on the
                    # sampler.
                    **resources.drain(),
                    "loss": loss_value,
                    "flow_loss": flow_loss_value,
                    "grdn": grad_norm_value,
                    "lr": cur_lr,
                    # Per-phase preparation cost, so a slow GPU path can be
                    # attributed to decode or resize rather than guessed at.
                    **(gpu_pipeline.report() if gpu_pipeline is not None else {}),
                },
            )
            if sample.omitted_fields:
                logger.warning(
                    "Non-finite training metrics at step %d (omitted from structured record): %s",
                    step,
                    ", ".join(sample.omitted_fields),
                )
            logger.info(
                "step %d/%d | loss: %.4f | flow_loss: %.4f | grdn: %.3f | lr: %.1e "
                "| updt_s: %.3f | data_s: %.3f | %.0fms | %s",
                step,
                args.steps,
                loss_value,
                flow_loss_value,
                grad_norm_value,
                cur_lr,
                sample.values["updt_s"],
                sample.values["data_s"],
                sample.values["step_time_ms"],
                sample.record,
            )

        if step % args.save_freq == 0:
            with health.exclude_time():
                save_checkpoint(step)
        elif args.eval_freq > 0 and step % args.eval_freq == 0:
            # Between checkpoints, so the generalisation gap is a curve rather
            # than two or three points. Excluded from the throughput window for
            # the same reason checkpoint I/O is.
            with health.exclude_time():
                if step not in validation_loss_cache:
                    validation_loss_cache[step] = evaluate_validation(step)
                evaluate_generalization(step)

        if is_log_step or step == 5:
            # Exclude logging and checkpoint I/O from the next training
            # window's throughput/ETA estimate. At step 5 there was no
            # record: this reset deliberately discards the cold-start window,
            # so the provisional step-6 ETA contains only step 6.
            health.reset()

    # Final save
    save_checkpoint(step)
    resources.stop()
    logger.info("Training complete.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Flow Matching S1 with Training-Time RTC")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument(
        "--s2-latent-path",
        default=None,
        help="Path to S2 latents .npy file. If omitted, S1 trains without S2 conditioning.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=2,
        help=(
            "Batches prepared ahead on the GPU data path. 2 is double buffering: "
            "one in flight while the model consumes the other. Higher costs that "
            "many batches of VRAM and buys nothing once preparation is shorter "
            "than the step. Ignored on the CPU path."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of complete episodes held out from optimization (0 disables validation)",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=4,
        help="Fixed number of held-out batches evaluated whenever a checkpoint is saved",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed the sampling order is derived from. The permutation is a pure "
            "function of (seed, epoch), so the same seed replays the same order."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=50, help="Action horizon (50 at 30Hz = 1.67s)")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=15,
        help="Denoising steps at inference (15 → ~130ms, expected_d≈4 at 30fps)",
    )
    parser.add_argument(
        "--rtc-max-delay",
        type=int,
        default=6,
        help="Max simulated delay in frames for training-time RTC (15 denoise steps ≈ 5 frames delay)",
    )
    parser.add_argument(
        "--rtc-drop-prob", type=float, default=0.2, help="Probability of no prefix (simulates first chunk)"
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.0,
        help="Max S2 latent delay in seconds (0 = always use aligned latent)",
    )
    parser.add_argument("--resize-images", type=str, default="224x224")
    parser.add_argument(
        "--cameras",
        type=lambda v: [c.strip() for c in v.split(",") if c.strip()],
        default=None,
        help="Comma-separated cameras to train on (default: every camera in the "
        "dataset). Names may be bare (top_l) or full (observation.images.top_l). "
        "Recorded in the checkpoint, so inference requests exactly these.",
    )
    parser.add_argument(
        "--vision-encoder",
        type=str,
        default=vision_encoders.DEFAULT_ENCODER,
        choices=sorted(vision_encoders.VISION_ENCODERS),
        help="Patch-token backbone. DINOv3 weights are gated: accept the licence "
        "upstream and log in first; they are not redistributed here.",
    )
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument(
        "--exclude-flags",
        type=str,
        default=None,
        help=(
            "Comma-separated flags whose frames must not be learned, e.g. "
            "'blurry,fumble'. A flagged frame ends the action chunk of any window "
            "reaching it, exactly as an episode end does. Omitted trains on every frame."
        ),
    )
    parser.add_argument(
        "--data-path",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help=(
            "Where the image half of a batch is produced. 'cpu' is the "
            "DataLoader-worker path (decode+composite+resize in workers). "
            "'gpu' decodes with NVDEC and composites/resizes on-device. "
            "'auto' (default) takes the GPU path where it is supported and "
            "verified — CUDA present, recipe supported, and CUDA decode of "
            "this dataset's video proven to match the CPU decoder's pixels — "
            "and falls back to the CPU path with the reason logged. An "
            "explicit 'gpu' never falls back; it fails instead."
        ),
    )
    parser.add_argument(
        "--ignore-saved-masks",
        action="store_true",
        help=(
            "Train on raw frames even though the dataset carries saved masks. The default "
            "is to apply them, because a dataset with mask columns was masked on purpose; "
            "this is the escape hatch for comparing against the unmasked pixels."
        ),
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Hold DINOv2 at its pretrained weights. Fine-tuning is more accurate on the "
            "recorded placements and worse on new ones"
        ),
    )
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=1.0,
        help="Multiplier on the backbone learning rate; the middle ground between "
        "fine-tuning and freezing. Ignored when --freeze-backbone is set",
    )
    parser.add_argument(
        "--image-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Random crop plus brightness/contrast/saturation/hue jitter, training frames only",
    )
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Peak learning rate (cosine schedule)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in the action expert")
    parser.add_argument(
        "--eval-generation-batches",
        type=int,
        default=2,
        help=(
            "Batches per split for the open-loop generation metric (0 disables). Each batch "
            "costs one full sampler run, so this is cheap per checkpoint and far too "
            "expensive per step"
        ),
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Evaluate every N steps (0 = only when a checkpoint is saved)",
    )
    parser.add_argument(
        "--state-position-std-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum z-score scale for observation.state features named *.pos, in dataset-native "
            "units (OpenArm joint positions use degrees; 0 preserves legacy behavior)"
        ),
    )
    parser.add_argument(
        "--use-relative-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train non-gripper *.pos actions as target minus the matching current named state "
            "position; gripper targets remain absolute"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Explicit RNG seed for reproducible paired runs (omitted preserves legacy behavior)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint dir (e.g., outputs/flow_s1_hvla_v2/checkpoint-5000)",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
