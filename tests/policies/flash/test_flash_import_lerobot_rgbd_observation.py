"""An external LeRobot clip can teach scene relations without importing actions."""

import json

import cv2
import numpy as np
import pytest

from benchmarks.flash_import_lerobot_rgbd_observation import import_episode
from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.observation_dataset import episode_camera_samples, scene_from_camera


def test_moving_camera_rgbd_episode_imports_as_actionless_scene_with_aligned_masks(tmp_path):
    source_root = tmp_path / "source"
    features = {
        "observation.images.ego": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.depth": {
            "dtype": "video",
            "shape": (32, 32, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        },
        "observation.images.mask": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.camera_pose": {"dtype": "float32", "shape": (4, 4), "names": None},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    source = LeRobotDataset.create(
        repo_id="local/source_rgbd",
        fps=15,
        features=features,
        root=source_root,
        robot_type="test",
        depth_encoder=DepthEncoderConfig(depth_min=0.0),
    )
    for index in range(3):
        rgb = np.full((64, 64, 3), 40 + index * 10, dtype=np.uint8)
        mask = np.zeros_like(rgb)
        mask[8:32, 8:32, 0] = 255
        mask[36:60, 36:60, 2] = 255
        depth = np.full((32, 32, 1), 800 + index * 10, dtype=np.uint16)
        depth[0:4, 0:4] = 0
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index * 0.02
        source.add_frame(
            {
                "task": "synthetic interaction",
                "observation.images.ego": rgb,
                "observation.images.depth": depth,
                "observation.images.mask": mask,
                "observation.camera_pose": pose,
                "observation.state": np.array([index, 0], dtype=np.float32),
                "action": np.array([1, 0], dtype=np.float32),
            }
        )
    source.save_episode()
    source.finalize()
    k = np.array([[90.0, 0, 16], [0, 90.0, 16], [0, 0, 1]])
    with pytest.raises(ValueError, match="same-rays"):
        import_episode(
            "local/source_rgbd",
            0,
            tmp_path / "misaligned",
            repo_id="local/misaligned",
            rgb_feature="observation.images.ego",
            depth_feature="observation.images.depth",
            camera_intrinsics=k,
            camera_pose_feature="observation.camera_pose",
            source_root=source_root,
        )
    root = import_episode(
        "local/source_rgbd",
        0,
        tmp_path / "observation",
        repo_id="local/aligned_observation",
        rgb_feature="observation.images.ego",
        depth_feature="observation.images.depth",
        camera_intrinsics=k,
        camera_pose_feature="observation.camera_pose",
        mask_feature="observation.images.mask",
        mask_channels={"object": 0, "target": 2},
        same_rays=True,
        source_root=source_root,
    )
    dataset = LeRobotDataset("local/aligned_observation", root=root)
    assert len(dataset) == 3
    assert "action" not in dataset.meta.features
    assert "observation.state" not in dataset.meta.features
    metadata = json.loads((root / "flash/v1/source.json").read_text())
    assert metadata["status"] == "perception_only"
    assert metadata["rgb_depth_alignment"] == "same_rays_area_downsample"
    samples, _ = episode_camera_samples(root, dataset, metadata, 0)
    assert len(samples) == 3
    for index, sample in enumerate(samples):
        row = dataset[sample.row_index]
        assert tuple(row["observation.images.rgb"].shape) == (3, 32, 32)
        assert row["flash.source_frame_index"].item() == index
        assert sample.source_time_s == pytest.approx(index / 15, abs=1e-5)
        pose, version = scene_from_camera(row, dataset.meta.features, metadata)
        assert pose[0, 3] == pytest.approx(index * 0.02)
        assert version == "source-world"
        depth = row["observation.images.depth"].numpy().squeeze()
        assert np.all(depth[0:4, 0:4] == 0)
        assert np.median(depth[5:, 5:]) == pytest.approx(800 + index * 10, abs=3)
        object_mask = cv2.imread(str(root / f"flash/v1/imported-masks/object/{index:06d}.png"), 0)
        target_mask = cv2.imread(str(root / f"flash/v1/imported-masks/target/{index:06d}.png"), 0)
        assert object_mask[10, 10] == 255 and object_mask[25, 25] == 0
        assert target_mask[25, 25] == 255 and target_mask[10, 10] == 0
