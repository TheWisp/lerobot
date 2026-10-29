"""Robot actions must not change the shared RGB-D vision and scene evidence."""

import gzip
import json

import cv2
import numpy as np
import pytest

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.api.run import _flash_recordings
from lerobot.policies.flash import observation_processing
from lerobot.policies.flash.observation_dataset import register_rgbd_dataset
from lerobot.policies.flash.observation_replay import load_observation_replay
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def _episode(root, *, with_actions):
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
        stationary_camera=True,
    )
    return mask_dir


class _MeasuredBackend:
    def __init__(self, masks):
        self.masks = masks
        self.last_masks = {}

    def initialize(self, references):
        self.references = references

    def update(self, frame):
        self.last_masks = self.masks.masks(frame)
        pose = np.eye(4)
        pose[2, 3] = 0.85
        return tuple(
            PoseEstimate(
                role=reference.role,
                pose=pose,
                source_time_s=frame.source_time_s,
                available_time_s=frame.available_time_s,
                evidence=Evidence.OBSERVED,
                reference_version=reference.version,
                translation_supported=(True,) * 3,
                rotation_supported=(True,) * 3,
                direct_translation_observed=True,
                estimator="test_rgbd_measurement",
                last_direct_observation_s=frame.source_time_s,
                source_frame_id=frame.frame_id,
            )
            for reference in self.references
        )

    def close(self):
        pass


def test_vision_and_group_evidence_match_with_and_without_robot_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        observation_processing,
        "build_vision_backend",
        lambda _config, masks, *, local_reference: _MeasuredBackend(masks),
    )
    config = tmp_path / "tracking.json"
    config.write_text(json.dumps(observation_processing.default_processing_config()))
    evidence = []
    for with_actions in (False, True):
        root = tmp_path / ("robot" if with_actions else "video")
        masks = _episode(root, with_actions=with_actions)
        sidecar = observation_processing.extract(root, {"object": masks}, config)
        replay = load_observation_replay(root)
        assert replay["evidence_available"]
        assert replay["robot_actions_available"] is with_actions
        assert not replay["gripper_prediction_available"]
        assert [row["snapshot"]["camera"]["source_time_s"] for row in replay["frames"]] == pytest.approx(
            [0.0, 0.1, 0.2]
        )
        assert [
            item["robot_actions_available"]
            for item in _flash_recordings((root.parent,))
            if item["dataset_root"] == str(root.resolve())
        ] == [with_actions]
        episode_evidence = []
        for index in range(3):
            with gzip.open(sidecar / f"{index:06d}.json.gz", "rt") as stream:
                episode_evidence.append(json.load(stream))
        evidence.append(episode_evidence)
    assert evidence[0] == evidence[1]
    assert all("groups" in frame and "masks_rle" in frame for frame in evidence[0])

    # Adding vision evidence to a robot demo must not hide its action editor.
    source_dir = root / "flash/v1/recorded-sources"
    source_dir.mkdir()
    (source_dir / "episode_000000.json").write_text(
        json.dumps({"schema_version": 1, "episode": 0, "repo_id": "local/robot_rgbd"})
    )
    (root / "flash/v1/episode_000000.jsonl").write_text('{"frame_index":0,"poses":{}}\n')
    listing = [item for item in _flash_recordings((root.parent,)) if item["dataset_root"] == str(root)]
    assert len(listing) == 1
    assert listing[0]["kind"] == "recorded_source_demo"
    assert listing[0]["observation_evidence_available"]


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
