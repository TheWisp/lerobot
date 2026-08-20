# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the model trains on IS the saved recipe's composite.

The chain under test is the real one end to end: masks written through the
production save path (generate_episode_masks with a deterministic fake
segmenter), read back by LeRobotDataset(apply_saved_masks=True) exactly as
factory.make_dataset builds it for lerobot-train, batched, converted, run
through the policy preprocessor, and captured at policy.forward's boundary
with a hook. The expectation is computed independently: raw frames from a
compositor-less dataset, composited by mask_compositing directly.

If any stage silently dropped, reordered, or re-fetched pixels, these tests
would see raw (or wrong-recipe) frames at the boundary.
"""

import numpy as np
import pytest
import torch

from lerobot.datasets.dataset_postprocess import generate_episode_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mask_compositing import (
    composite_from_store,
    load_recipe_from_disk,
    mask_feature_of,
)


class _StripeAdapter:
    """Deterministic fake segmenter: top half = tray, bottom-left = ball."""

    def set_control(self, _):
        pass

    def set_camera(self, cam):
        pass

    def reset(self):
        pass

    def segment(self, rgb):
        h, w = rgb.shape[:2]
        tray = np.zeros((h, w), np.float32)
        tray[: h // 2] = 1.0
        ball = np.zeros((h, w), np.float32)
        ball[h // 2 :, : w // 2] = 1.0
        return {"tray": tray, "ball": ball}


OBJECTS = [
    {"name": "tray", "sign": "+", "treatment": {"key": "tint", "params": {"color": [255, 0, 0]}}},
    {"name": "ball", "sign": "+", "treatment": {"key": "none"}},
]
BACKGROUND = {"key": "solid", "params": {"color": [0, 255, 0]}}


@pytest.fixture
def masked_dataset_root(tmp_path, info_factory, lerobot_dataset_factory):
    """A real on-disk video dataset whose episode 0 carries saved masks,
    written through the production save path. Feature names follow the real
    recorder conventions (observation.state / observation.images.*), which
    the policies hard-code."""
    root = tmp_path / "ds"
    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {
        "observation.images.top": {
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
        "observation.images.wrist": {
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
    }
    info = info_factory(
        total_episodes=1,
        total_frames=12,
        total_tasks=1,
        motor_features=motors,
        camera_features=cams,
    )
    ds = lerobot_dataset_factory(root=root, total_episodes=1, total_frames=12, info=info)
    result = generate_episode_masks(
        ds,
        episode=0,
        objects=OBJECTS,
        cameras=None,
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment=BACKGROUND,
        adopt=True,
        device="cpu",
        adapter=_StripeAdapter(),
    )
    assert not result.get("cancelled")
    return root, ds.repo_id


def _to_uint8_hwc(frames: torch.Tensor) -> np.ndarray:
    rgb = frames
    if rgb.shape[0] in (1, 3, 4):
        rgb = rgb.permute(1, 2, 0)
    if rgb.is_floating_point():
        rgb = (rgb * 255).round().clamp(0, 255).to(torch.uint8)
    return np.ascontiguousarray(rgb.numpy())


def _stored_row(ds: LeRobotDataset, abs_idx: int, cam: str) -> str:
    cell = ds.hf_dataset[abs_idx][mask_feature_of(cam)]
    if isinstance(cell, (list, tuple)):
        cell = cell[0] if cell else ""
    return "" if cell is None else str(cell)


def _expected_composite(ds: LeRobotDataset, raw_item: dict, cam: str, spec: dict) -> np.ndarray:
    abs_idx = int(raw_item["index"].item())
    return composite_from_store(_to_uint8_hwc(raw_item[cam]), _stored_row(ds, abs_idx, cam), spec, episode=0)


def test_dataset_items_carry_the_composite_exactly(masked_dataset_root):
    """apply_saved_masks=True serves composite_from_store(raw), bit-exact."""
    root, repo_id = masked_dataset_root
    raw = LeRobotDataset(repo_id, root=root, return_uint8=True)
    comp = LeRobotDataset(repo_id, root=root, return_uint8=True, apply_saved_masks=True)

    for idx in (0, 5, 11):
        raw_item, comp_item = raw[idx], comp[idx]
        for cam in comp.meta.camera_keys:
            spec = load_recipe_from_disk(root, cam)
            assert spec is not None, cam
            expected = _expected_composite(raw, raw_item, cam, spec)
            got = _to_uint8_hwc(comp_item[cam])
            assert np.array_equal(got, expected), f"{cam} idx={idx}"
            # And it is genuinely different from raw — the solid background
            # guarantees visible change wherever nothing was segmented.
            assert not np.array_equal(got, _to_uint8_hwc(raw_item[cam]))
            # Mask columns are consumed at load: no RLE strings in items,
            # so default_collate keeps working in the train loop.
            assert mask_feature_of(cam) not in comp_item
            assert mask_feature_of(cam) not in raw_item


def test_float_path_matches_uint8_path(masked_dataset_root):
    """The default float pipeline composites the same pixels as uint8."""
    root, repo_id = masked_dataset_root
    comp_u8 = LeRobotDataset(repo_id, root=root, return_uint8=True, apply_saved_masks=True)
    comp_f = LeRobotDataset(repo_id, root=root, return_uint8=False, apply_saved_masks=True)
    for cam in comp_u8.meta.camera_keys:
        a = _to_uint8_hwc(comp_u8[3][cam])
        b = _to_uint8_hwc(comp_f[3][cam])
        assert np.array_equal(a, b), cam


def test_what_enters_the_policy_is_the_composite(masked_dataset_root):
    """The tensor policy.forward receives equals preprocessor(composite),
    reproducing lerobot_train's exact batch flow on a real ACT policy."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    root, repo_id = masked_dataset_root
    set_seed(0)

    act_cfg = PreTrainedConfig.get_choice_class("act")(
        chunk_size=4,
        n_action_steps=4,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        device="cpu",
    )

    meta = LeRobotDataset(repo_id, root=root).meta
    from lerobot.datasets.factory import resolve_delta_timestamps

    delta = resolve_delta_timestamps(act_cfg, meta)
    comp = LeRobotDataset(
        repo_id, root=root, return_uint8=True, apply_saved_masks=True, delta_timestamps=delta
    )
    raw = LeRobotDataset(repo_id, root=root, return_uint8=True, delta_timestamps=delta)

    policy = make_policy(cfg=act_cfg, ds_meta=meta)
    preprocessor, _ = make_pre_post_processors(policy_cfg=act_cfg, dataset_stats=meta.stats)

    def batch_of(ds, indices):
        items = [ds[i] for i in indices]
        batch = torch.utils.data.default_collate(items)
        # lerobot_train's uint8 conversion, verbatim.
        for cam_key in ds.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        return preprocessor(batch)

    indices = [1, 7]
    got_batch = batch_of(comp, indices)

    # Expected: raw items, composited independently, then the SAME pipeline.
    expected_items = []
    for i in indices:
        item = raw[i]
        for cam in raw.meta.camera_keys:
            spec = load_recipe_from_disk(root, cam)
            composited = _expected_composite(raw, item, cam, spec)
            item[cam] = torch.from_numpy(composited).permute(2, 0, 1).contiguous()
        expected_items.append(item)
    exp_batch = torch.utils.data.default_collate(expected_items)
    for cam_key in raw.meta.camera_keys:
        exp_batch[cam_key] = exp_batch[cam_key].to(dtype=torch.float32) / 255.0
    exp_batch = preprocessor(exp_batch)

    # Capture at the model boundary: the batch policy.forward actually
    # receives. The train loop calls policy.forward(batch) directly (not
    # __call__), so nn.Module hooks never fire there — wrap the method itself,
    # which intercepts the exact call shape lerobot_train uses.
    captured = {}
    orig_forward = policy.forward

    def _capturing_forward(batch, *args, **kwargs):
        captured["batch"] = batch
        return orig_forward(batch, *args, **kwargs)

    policy.forward = _capturing_forward
    loss, _ = policy.forward(got_batch)
    assert torch.isfinite(loss)

    seen = captured["batch"]
    for cam in raw.meta.camera_keys:
        assert torch.equal(seen[cam], exp_batch[cam]), f"model input differs from composite: {cam}"
    # Raw pixels must NOT be what the model sees.
    raw_batch = batch_of(raw, indices)
    for cam in raw.meta.camera_keys:
        assert not torch.equal(seen[cam], raw_batch[cam]), f"model saw RAW pixels: {cam}"
