# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The camera block lands in the right features' info and survives a reload."""

from types import SimpleNamespace

import numpy as np

from lerobot.cameras.intrinsics import CameraIntrinsics
from lerobot.datasets.camera_calibration import attach_camera_intrinsics
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

H, W = 96, 128


def _dataset(root):
    ds = LeRobotDataset.create(
        repo_id="tests/camera_calibration",
        fps=10,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["a"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["a"]},
            "observation.images.top": {
                "dtype": "video",
                "shape": (H, W, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.top_depth": {
                "dtype": "video",
                "shape": (H, W, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            },  # fmt: skip
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (H, W, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.far_depth": {
                "dtype": "video",
                "shape": (H * 2, W * 2, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            },  # fmt: skip
        },
        use_videos=True,
    )
    for i in range(3):
        ds.add_frame(
            {
                "observation.state": np.array([float(i)], np.float32),
                "action": np.array([0.0], np.float32),
                "task": "calib",
                "observation.images.top": np.zeros((H, W, 3), np.uint8),
                "observation.images.top_depth": np.full((H, W, 1), 500, np.uint16),
                "observation.images.wrist": np.zeros((H, W, 3), np.uint8),
                "observation.images.far_depth": np.full((H * 2, W * 2, 1), 500, np.uint16),
            }
        )
    ds.save_episode()
    ds.finalize()
    return ds


def test_block_goes_to_the_colour_feature_and_its_aligned_depth(tmp_path):
    ds = _dataset(tmp_path / "ds")
    top = SimpleNamespace(
        intrinsics=CameraIntrinsics(W, H, 100.0, 101.0, 63.5, 47.5),
        serial_number="0042",
        device_name="Fake D435",
    )
    wrist = SimpleNamespace(intrinsics=None)  # a UVC camera without a calibration
    far = SimpleNamespace(
        intrinsics=CameraIntrinsics(W, H, 100.0, 101.0, 63.5, 47.5)
    )  # wrong size for its depth

    written = attach_camera_intrinsics(ds.meta, {"top": top, "wrist": wrist, "far": far})

    assert written == ["observation.images.top", "observation.images.top_depth"]
    reloaded = LeRobotDatasetMetadata("tests/camera_calibration", root=ds.root)
    block = reloaded.features["observation.images.top"]["info"]["camera"]
    assert block["intrinsics"]["fx"] == 100.0 and block["serial"] == "0042" and block["model"] == "Fake D435"
    assert block["source"] == "device" and "aligned_to" not in block
    depth_block = reloaded.features["observation.images.top_depth"]["info"]["camera"]
    assert depth_block["aligned_to"] == "observation.images.top"
    assert depth_block["intrinsics"] == block["intrinsics"]
    assert reloaded.features["observation.images.top_depth"]["info"]["is_depth_map"] is True  # siblings kept
    assert "camera" not in reloaded.features["observation.images.wrist"].get("info", {})
    assert "camera" not in reloaded.features["observation.images.far_depth"]["info"]


def test_existing_block_is_kept_unless_overwriting(tmp_path):
    ds = _dataset(tmp_path / "ds")
    first = SimpleNamespace(intrinsics=CameraIntrinsics(W, H, 100.0, 100.0, 64.0, 48.0))
    second = SimpleNamespace(intrinsics=CameraIntrinsics(W, H, 200.0, 200.0, 64.0, 48.0))
    attach_camera_intrinsics(ds.meta, {"top": first})
    assert attach_camera_intrinsics(ds.meta, {"top": second}) == []
    assert ds.meta.features["observation.images.top"]["info"]["camera"]["intrinsics"]["fx"] == 100.0
    attach_camera_intrinsics(ds.meta, {"top": second}, overwrite=True)
    assert ds.meta.features["observation.images.top"]["info"]["camera"]["intrinsics"]["fx"] == 200.0


def test_feature_names_route_a_per_arm_camera_to_its_prefixed_features(tmp_path):
    """A bimanual robot keeps per-arm cameras under bare names but publishes
    them prefixed; the map says which feature the block belongs to."""
    wrist = SimpleNamespace(intrinsics=CameraIntrinsics(W, H, 300.0, 300.0, 64.0, 48.0), serial_number="s")
    features = {
        "observation.images.right_wrist": {"shape": [H, W, 3], "info": {}},
        "observation.images.right_wrist_depth": {"shape": [H, W, 1], "info": {"is_depth_map": True}},
    }
    meta = SimpleNamespace(
        features=features, info=SimpleNamespace(to_dict=lambda: {"features": features}), root=tmp_path
    )
    assert attach_camera_intrinsics(meta, {"wrist": wrist}) == []  # the bare name matches nothing
    written = attach_camera_intrinsics(meta, {"wrist": wrist}, feature_names={"wrist": "right_wrist"})
    assert written == ["observation.images.right_wrist", "observation.images.right_wrist_depth"]
    assert (
        features["observation.images.right_wrist_depth"]["info"]["camera"]["aligned_to"]
        == "observation.images.right_wrist"
    )
