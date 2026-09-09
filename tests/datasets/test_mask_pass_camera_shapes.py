# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A stored mask row decodes only against the shape it was written at.

`generate_episode_masks` reads every selected camera's stored rows before it
segments, because the write rule fills a gap and leaves what is already there
alone -- so each row has to be merged with the one it replaces. A row is a flat
run-length string carrying no dimensions of its own, and the only record of its
geometry is the camera's own `mask_size`.

The cameras on one robot are not one resolution. Reading them all with a single
shape decoded the narrow camera's rows against the wide one's, and `decode_mask`
refuses a pixel count it cannot fill -- so the second pass over such a dataset
aborted partway through, at the first frame where the narrow camera already had
a mask. The first pass always survived: nothing was stored yet, so nothing was
decoded.

Only the mismatch is observable. Where two cameras happen to share a pixel
count, the row survives a wrong shape unharmed -- the runs are held flat and
re-encoded in the same order, so the string comes back identical.
"""

import numpy as np
import pytest

from lerobot.datasets.dataset_postprocess import generate_episode_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mask_codec import decode_frame
from lerobot.datasets.mask_compositing import mask_feature_of

# Different pixel counts, not just different shapes: a shape read from the wrong
# camera is only detectable when the counts disagree.
CAMERAS = {
    "observation.images.top": (48, 64),
    "observation.images.wrist": (64, 96),
}
BACKGROUND = {"key": "none", "params": {}}


class _StripeAdapter:
    """One mask per named object, sized to the frame it was handed -- which is
    what makes the two cameras store rows of genuinely different lengths."""

    def __init__(self, names):
        self._names = list(names)

    def set_control(self, _control):
        pass

    def set_camera(self, _cam):
        pass

    def reset(self):
        pass

    def segment(self, rgb):
        h, w = rgb.shape[:2]
        out = {}
        for i, name in enumerate(self._names):
            m = np.zeros((h, w), np.float32)
            m[:, i * (w // 4) : (i + 1) * (w // 4)] = 1.0
            out[name] = m
        return out


def _pass(ds, names, *, adopt=False):
    return generate_episode_masks(
        ds,
        episode=0,
        objects=[{"name": n, "sign": "+", "treatment": {"key": "none"}} for n in names],
        cameras=None,
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment=BACKGROUND,
        adopt=adopt,
        device="cpu",
        adapter=_StripeAdapter(names),
    )


@pytest.fixture
def mixed_resolution_masks(tmp_path, info_factory, lerobot_dataset_factory):
    """Two cameras of different sizes, one segmentation pass already stored."""
    root = tmp_path / "ds"
    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {
        cam: {"shape": (h, w, 3), "names": ["height", "width", "channels"], "info": None}
        for cam, (h, w) in CAMERAS.items()
    }
    info = info_factory(
        total_episodes=1, total_frames=12, total_tasks=1, motor_features=motors, camera_features=cams
    )
    ds = lerobot_dataset_factory(root=root, total_episodes=1, total_frames=12, info=info)
    assert not _pass(ds, ["tray"], adopt=True).get("cancelled")
    return root, ds.repo_id


def _spec(ds, cam):
    return ds.meta.features[mask_feature_of(cam)]


def _row(ds, cam, frame):
    cell = ds.hf_dataset[frame][mask_feature_of(cam)]
    if isinstance(cell, (list, tuple)):
        cell = cell[0] if cell else ""
    return "" if cell is None else str(cell)


def test_the_cameras_really_do_store_different_shapes(mixed_resolution_masks):
    """Guards the test below: with one resolution it asserts nothing."""
    root, repo_id = mixed_resolution_masks
    ds = LeRobotDataset(repo_id, root=root)
    sizes = {cam: tuple(_spec(ds, cam)["mask_size"]) for cam in CAMERAS}
    assert len(set(sizes.values())) == len(CAMERAS), f"cameras share a mask_size: {sizes}"
    assert len({h * w for h, w in sizes.values()}) == len(CAMERAS), (
        f"cameras share a pixel COUNT: {sizes}; a shape taken from the wrong camera would fit"
    )
    for cam in CAMERAS:
        assert _row(ds, cam, 0), f"{cam} stored no mask, so nothing is decoded on the next pass"


def test_a_later_pass_decodes_each_cameras_rows_with_that_cameras_shape(mixed_resolution_masks):
    """The regression. A new label leaves a gap on every frame, so the pass runs
    the frame loop instead of skipping the episode, and every stored row is
    decoded on the way through."""
    root, repo_id = mixed_resolution_masks
    ds = LeRobotDataset(repo_id, root=root)

    result = _pass(ds, ["tray", "ball"])
    assert not result.get("skipped"), "the episode was already covered; no row was decoded"
    assert not result.get("cancelled")

    ds = LeRobotDataset(repo_id, root=root)
    for cam, shape in CAMERAS.items():
        spec = _spec(ds, cam)
        assert tuple(spec["mask_size"]) == shape, f"{cam} stored masks at {spec['mask_size']}, not {shape}"
        decoded = decode_frame(_row(ds, cam, 0), list(spec["mask_labels"]), tuple(spec["mask_size"]))
        assert set(decoded) == {"tray", "ball"}, f"{cam} carries {sorted(decoded)}"
        for name, mask in decoded.items():
            assert mask.shape == shape, f"{cam}/{name} decodes to {mask.shape}, not {shape}"
