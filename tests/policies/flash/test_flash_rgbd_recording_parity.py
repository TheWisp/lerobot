"""Robot actions must not change the shared RGB-D vision and scene evidence."""

import json

import cv2
import numpy as np
import pytest

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.observation_dataset import episode_camera_samples, register_rgbd_dataset


def _episode(root, *, with_actions, robot_base_frame=False):
    features = {
        "observation.images.rgb": {
            "dtype": "image",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.depth": {
            "dtype": "image",
            "shape": (64, 64, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True, "depth_unit": "mm"},
        },
    }
    if with_actions:
        features.update(
            {
                "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
                "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
            }
        )
    repo_id = "local/robot_rgbd" if with_actions else "local/video_rgbd"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        features=features,
        root=root,
        robot_type="test_robot" if with_actions else "external_rgbd_observation",
        depth_encoder=DepthEncoderConfig(depth_min=0.0),
    )
    mask_dir = root / "masks"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18:42, 18:42] = 255
    for index in range(3):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[mask != 0] = [160, 65, 35]
        depth = np.full((64, 64, 1), 850, dtype=np.uint16)
        row = {"task": "move object", "observation.images.rgb": rgb, "observation.images.depth": depth}
        if with_actions:
            row.update(
                {
                    "observation.state": np.array([index], dtype=np.float32),
                    "action": np.array([index + 1], dtype=np.float32),
                }
            )
        dataset.add_frame(row)
        cv2.imwrite(str(mask_dir / f"{index:06d}.png"), mask)
    dataset.save_episode()
    dataset.finalize()
    register_rgbd_dataset(
        root,
        repo_id=repo_id,
        camera_intrinsics=np.array([[100, 0, 32], [0, 100, 32], [0, 0, 1]], dtype=float),
        depth_unit="millimetres",
        stationary_camera=not robot_base_frame,
        fixed_scene_from_camera=np.eye(4) if robot_base_frame else None,
        scene_frame="robot_base" if robot_base_frame else None,
    )
    return mask_dir


def test_registration_requires_explicit_camera_geometry_and_does_not_overwrite(tmp_path):
    root = tmp_path / "dataset"
    masks = _episode(root, with_actions=False)
    assert masks.is_dir()
    with pytest.raises(ValueError, match="declare a camera trajectory"):
        register_rgbd_dataset(
            root, repo_id="local/video_rgbd", camera_intrinsics=np.eye(3), depth_unit="millimetres"
        )
    with pytest.raises(ValueError, match="depth unit conflicts"):
        register_rgbd_dataset(
            root,
            repo_id="local/video_rgbd",
            camera_intrinsics=np.eye(3),
            depth_unit="metres",
            stationary_camera=True,
        )
    with pytest.raises(ValueError, match="uses its camera frame"):
        register_rgbd_dataset(
            root,
            repo_id="local/video_rgbd",
            camera_intrinsics=np.eye(3),
            depth_unit="millimetres",
            stationary_camera=True,
            scene_frame="robot_base",
        )
    with pytest.raises(FileExistsError):
        register_rgbd_dataset(
            root,
            repo_id="local/video_rgbd",
            camera_intrinsics=np.eye(3),
            depth_unit="millimetres",
            stationary_camera=True,
        )
    with pytest.raises(ValueError, match="depth_unit"):
        register_rgbd_dataset(
            root, repo_id="local/video_rgbd", camera_intrinsics=np.eye(3), depth_unit="unknown"
        )


def test_camera_sample_index_does_not_decode_rgbd_rows(tmp_path, monkeypatch):
    root = tmp_path / "observation"
    _episode(root, with_actions=False)
    dataset = LeRobotDataset("local/video_rgbd", root=root)
    source = json.loads((root / "flash/v1/source.json").read_text())

    def reject_decoding(_self, _index):
        raise AssertionError("indexing the episode must not decode RGB-D frames")

    monkeypatch.setattr(LeRobotDataset, "__getitem__", reject_decoding)
    samples, trace_sha = episode_camera_samples(root, dataset, source, 0)
    assert trace_sha is None
    assert [sample.row_index for sample in samples] == [0, 1, 2]
    assert [sample.source_time_s for sample in samples] == pytest.approx([0.0, 0.1, 0.2])
