"""An external RGB-D import must preserve depth validity and source timing."""

import asyncio
import gzip
import hashlib
import json

import cv2
import numpy as np
import pytest

from benchmarks.flash_import_external_rgbd import import_rgbd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.api.run import (
    FlashObservationAnnotationRequest,
    _flash_recordings,
    flash_observation_annotations,
)
from lerobot.policies.flash.observation_annotations import save_annotations
from lerobot.policies.flash.observation_dataset import camera_feature_keys
from lerobot.policies.flash.observation_replay import load_observation_replay


def test_import_preserves_invalid_depth_and_source_clock(tmp_path):
    recording = tmp_path / "recording"
    (recording / "rgb").mkdir(parents=True)
    (recording / "depth").mkdir()
    times = np.array([0.0, 0.04, 0.11], dtype=np.float64)
    np.savetxt(recording / "times.txt", times)
    np.savetxt(recording / "cam_K.txt", np.array([[90, 0, 32], [0, 90, 32], [0, 0, 1]]))
    for index in range(3):
        rgb = np.full((64, 64, 3), index * 30, dtype=np.uint8)
        depth = np.full((64, 64), 850 + index, dtype=np.uint16)
        depth[20:30, 30:40] = 0
        cv2.imwrite(str(recording / "rgb" / f"{index:06d}.png"), rgb)
        cv2.imwrite(str(recording / "depth" / f"{index:06d}.png"), depth)

    root = import_rgbd(
        recording,
        tmp_path / "dataset",
        repo_id="local/external_rgbd_test",
        task="test",
        source="synthetic RGB-D source",
        source_frame_offset=100,
    )
    dataset = LeRobotDataset("local/external_rgbd_test", root=root)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
    for index in range(3):
        row = dataset[index]
        saved_depth = row["observation.images.depth"].numpy().squeeze()
        assert np.array_equal(saved_depth == 0, depth == 0)
        assert np.isclose(row["flash.source_time_s"].item(), times[index], atol=1e-6)
        assert row["flash.source_frame_index"].item() == 100 + index
    source = json.loads((root / "flash" / "v1" / "source.json").read_text())
    assert source["status"] == "perception_only"
    assert "robot_actions" in source["missing_for_spatial_skill"]
    listing = _flash_recordings((tmp_path,))
    assert [(item["kind"], item["episode"]) for item in listing] == [("observation_demo", 0)]
    replay = load_observation_replay(root)
    assert replay["kind"] == "observation_demo"
    assert len(replay["frames"]) == 3
    assert not replay["robot_actions_available"]
    assert not replay["gripper_prediction_available"]

    sidecar = root / "flash" / "v1" / "observation-evidence" / "episode_000000"
    sidecar.mkdir(parents=True)
    (sidecar / "manifest.json").write_text(
        json.dumps({"complete": True, "total_frames": 3, "source_sha256": "test-evidence"})
    )
    (sidecar / "observation.json").write_text(
        json.dumps(
            {
                "surface_reference_max_points_per_view": 100,
                "evidence_identity": "test-evidence",
                "source_metadata_sha256": hashlib.sha256(
                    (root / "flash" / "v1" / "source.json").read_bytes()
                ).hexdigest(),
            }
        )
    )
    pose = np.eye(4).tolist()
    for frame_id in range(3):
        evidence = {
            "frame_index": frame_id,
            "estimates": {"object": {"pose": pose, "evidence": "observed", "estimator": "test"}},
            "masks_rle": {"object": [[0, 64 * 64]]},
            "groups": {},
            "proposals": {},
        }
        with gzip.open(sidecar / f"{frame_id:06d}.json.gz", "wt") as stream:
            json.dump(evidence, stream)
    replay = load_observation_replay(root)
    assert replay["evidence_available"]
    assert replay["frames"][1]["snapshot"]["accepted"]["object"]["evidence"] == "observed"
    assert replay["frames"][1]["snapshot"]["observed_masks"]["object"]["boundary_xy"]
    assert replay["geometry"]["points_object_m"]["object"]
    for frame_id in range(3):
        evidence_path = sidecar / f"{frame_id:06d}.json.gz"
        with gzip.open(evidence_path, "rt") as stream:
            evidence = json.load(stream)
        evidence["estimates"]["reference"] = {
            "pose": pose,
            "evidence": "observed" if frame_id == 0 else "predicted",
            "estimator": "test",
        }
        evidence["masks_rle"]["reference"] = [[0, 64 * 64]]
        with gzip.open(evidence_path, "wt") as stream:
            json.dump(evidence, stream)
    replay = load_observation_replay(root)
    keyframe = {
        "frame_index": 1,
        "moving_role": "object",
        "reference_role": "reference",
        "required": True,
        "allowed_turn": None,
    }
    request = FlashObservationAnnotationRequest(
        dataset_root=str(root),
        episode=0,
        annotation={
            "schema_version": 1,
            "source_metadata_sha256": hashlib.sha256(
                (root / "flash" / "v1" / "source.json").read_bytes()
            ).hexdigest(),
            "keyframes": [keyframe],
        },
    )
    saved = asyncio.run(flash_observation_annotations(request))
    assert saved["keyframes"] == [keyframe]
    reviewed = load_observation_replay(root)
    assert reviewed["annotations"]["keyframes"] == [keyframe]
    relation = reviewed["relationship_evidence"][0]
    assert (relation["frame_index"], relation["moving_role"], relation["reference_role"]) == (
        1,
        "object",
        "reference",
    )
    assert not relation["both_observed"]
    assert relation["reference_evidence"] == "predicted"
    assert np.allclose(relation["relative_pose"], pose)
    reverse = {
        **keyframe,
        "moving_role": "reference",
        "reference_role": "object",
        "allowed_turn": {
            "pivot_reference_m": [0.0, 0.0, 0.0],
            "axis_reference": [0.0, 0.0, 1.0],
            "min_angle_rad": -1.0,
            "max_angle_rad": 1.0,
            "source_evidence": "observed",
        },
    }
    multiple = save_annotations(root, 0, {**saved, "keyframes": [keyframe, reverse]}, replay)
    assert len(load_observation_replay(root)["annotations"]["keyframes"]) == 2
    assert any(
        item["moving_role"] == "reference" and item["reference_role"] == "object"
        for item in load_observation_replay(root)["relationship_evidence"]
    )
    with pytest.raises(ValueError, match="already marked"):
        save_annotations(root, 0, {**multiple, "keyframes": [keyframe, keyframe]}, replay)
    with pytest.raises(ValueError, match="increasing angle range|turn range"):
        save_annotations(
            root,
            0,
            {
                **multiple,
                "keyframes": [{**reverse, "allowed_turn": {**reverse["allowed_turn"], "min_angle_rad": 2.0}}],
            },
            replay,
        )
    with pytest.raises(ValueError, match="different tracked objects"):
        save_annotations(root, 0, {**saved, "keyframes": [{**keyframe, "reference_role": "object"}]}, replay)
    with pytest.raises(ValueError, match="outside this episode"):
        save_annotations(root, 0, {**saved, "keyframes": [{**keyframe, "frame_index": 3}]}, replay)
    source["source"] = "revised source metadata"
    (root / "flash" / "v1" / "source.json").write_text(json.dumps(source))
    assert not load_observation_replay(root)["evidence_available"]
    assert load_observation_replay(root)["annotations"] is None


def test_observation_camera_names_come_from_metadata():
    features = {
        "observation.images.top": {"dtype": "video", "shape": [64, 64, 3]},
        "observation.images.range": {"dtype": "video", "shape": [64, 64, 1], "info": {"is_depth_map": True}},
    }
    assert camera_feature_keys(features, {}) == ("observation.images.top", "observation.images.range")
