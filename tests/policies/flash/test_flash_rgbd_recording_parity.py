"""Robot actions must not change the shared RGB-D vision and scene evidence."""

import gzip
import json

import cv2
import numpy as np
import pytest

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.api.run import _flash_recordings, _flash_scene_on_motor_replay, _flash_tracker_replay
from lerobot.policies.flash import observation_processing
from lerobot.policies.flash.observation_dataset import episode_camera_samples, register_rgbd_dataset
from lerobot.policies.flash.observation_replay import load_observation_replay
from lerobot.policies.flash.recording_scene_replay import attach_scene_evidence
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


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


def test_motor_replay_keeps_uncalibrated_scene_separate(tmp_path):
    extension = tmp_path / "flash/v1"
    sidecar = extension / "observation-evidence/episode_000000"
    sidecar.mkdir(parents=True)
    (sidecar / "manifest.json").write_text("{}")
    (extension / "source.json").write_text(
        json.dumps({"status": "robot_recording", "coordinate_frame": "camera"})
    )
    original = {"frames": [{"frame_index": 0, "snapshot": {"accepted": {"object": {"pose": "recorded"}}}}]}
    replay = _flash_scene_on_motor_replay(tmp_path, 0, [], original)
    assert replay["frames"] == original["frames"]
    assert "not calibrated into the robot base" in replay["scene_evidence"]["reason"]


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


def test_motor_replay_rejects_different_camera_calibration():
    camera = {
        "frame_id": 4,
        "source_time_s": 1.0,
        "base_from_camera": np.eye(4).tolist(),
        "intrinsics": np.eye(3).tolist(),
    }
    changed_camera = {**camera, "base_from_camera": np.eye(4).tolist()}
    changed_camera["base_from_camera"][0][3] = 0.01
    motor = {"frames": [{"frame_index": 0, "snapshot": {"camera": camera, "accepted": {}}}]}
    scene = {
        "evidence_available": True,
        "frames": [
            {
                "frame_index": 0,
                "source_camera_frame_id": 4,
                "snapshot": {"camera": changed_camera, "scene_support": {"groups": {}}, "accepted": {}},
            }
        ],
    }
    with pytest.raises(ValueError, match="disagree on base_from_camera"):
        attach_scene_evidence(
            motor, [{"frame_index": 0, "camera_frame_id": 4, "camera_source_time_s": 1.0}], scene
        )


def test_robot_trace_advances_vision_only_for_distinct_camera_captures(tmp_path, monkeypatch):
    monkeypatch.setattr(
        observation_processing,
        "build_vision_backend",
        lambda _config, masks, *, local_reference: _MeasuredBackend(masks),
    )
    root = tmp_path / "robot"
    masks = _episode(root, with_actions=True, robot_base_frame=True)
    trace = root / "flash/v1/episode_000000.jsonl"
    ticks = [
        {
            "episode_index": 0,
            "frame_index": index,
            "camera_frame_id": camera_id,
            "camera_source_time_s": capture_time,
            "poses": {
                "object": {"pose": np.eye(4).tolist(), "evidence": "predicted", "estimator": "recorded"}
            }
            if index == 0
            else {},
        }
        for index, (camera_id, capture_time) in enumerate(((41, 20.0), (41, 20.0), (43, 20.1)))
    ]
    trace.write_text("\n".join(json.dumps(tick) for tick in ticks) + "\n")
    config = tmp_path / "tracking.json"
    config.write_text(json.dumps(observation_processing.default_processing_config()))
    evidence = observation_processing.extract(root, {"object": masks}, config)
    report = json.loads((evidence / "observation.json").read_text())
    assert report["frames"] == 2
    assert report["source_row_indices"] == [0, 2]
    assert report["source_camera_frame_ids"] == [41, 43]
    replay = load_observation_replay(root)
    assert replay["evidence_available"]
    assert [frame["source_row_index"] for frame in replay["frames"]] == [0, 2]
    assert [frame["snapshot"]["camera"]["source_time_s"] for frame in replay["frames"]] == [20.0, 20.1]
    assert [frame["evidence"]["frame_index"] for frame in replay["frames"]] == [0, 1]
    motor = _flash_tracker_replay(root, 0)
    assert motor["scene_evidence"] == {
        "available": True,
        "camera_frames": 2,
        "motor_frames_aligned": 3,
        "source": "offline_rgbd_reprocessing",
    }
    assert [frame["scene_evidence_frame_index"] for frame in motor["frames"]] == [0, 0, 1]
    assert [frame["snapshot"]["camera"]["frame_id"] for frame in motor["frames"]] == [41, 41, 43]
    assert motor["frames"][1]["snapshot"]["reprocessed_observed_masks"]["object"]["frame_id"] == 41
    assert "observed_masks" not in motor["frames"][1]["snapshot"]
    assert motor["frames"][0]["snapshot"]["accepted"]["object"]["estimator"] == "recorded"
    assert "reprocessed_scene" in motor["frames"][0]["snapshot"]["layers"]["object"]
    assert "geometry" not in motor
    assert motor["reprocessed_scene_geometry"]["points_object_m"]["object"]
    assert motor["frames"][0]["snapshot"]["reprocessed_scene_support"]["frame_id"] == 41
    assert "scene_support" not in motor["frames"][0]["snapshot"]

    # The trace is part of the evidence identity. A changed camera join cannot
    # silently keep showing overlays from the previous selection of frames.
    ticks[2]["camera_frame_id"] = 44
    trace.write_text("\n".join(json.dumps(tick) for tick in ticks) + "\n")
    assert not load_observation_replay(root)["evidence_available"]
    ticks[2]["camera_frame_id"] = 43
    ticks[1]["camera_source_time_s"] = 20.01
    trace.write_text("\n".join(json.dumps(tick) for tick in ticks) + "\n")
    with pytest.raises(ValueError, match="reused camera frame"):
        load_observation_replay(root)
