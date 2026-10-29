"""An alternative camera trajectory must remain a versioned, reviewable comparison."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.observation_dataset import CameraSample
from lerobot.policies.flash.observation_processing import _camera_pose_candidates, extract


def test_camera_comparison_cannot_replace_normal_evidence(tmp_path):
    with pytest.raises(ValueError, match="explicit output sidecar"):
        extract(tmp_path, {}, tmp_path / "config.json", camera_pose_audit=tmp_path / "audit.json")
    normal = tmp_path / "flash/v1/observation-evidence/episode_000000"
    with pytest.raises(ValueError, match="cannot replace normal observation evidence"):
        extract(
            tmp_path, {}, tmp_path / "config.json", output=normal, camera_pose_audit=tmp_path / "audit.json"
        )


def test_camera_candidate_needs_matching_source_and_masks_and_heldout_improvement(tmp_path):
    root = tmp_path / "episode"
    source_path = root / "flash/v1/source.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps({"camera_pose_feature": "camera"}))
    masks = {role: root / role for role in ("moving", "target")}
    for directory in masks.values():
        directory.mkdir()
        for frame_id in range(3):
            cv2.imwrite(str(directory / f"{frame_id:06d}.png"), np.zeros((8, 8), np.uint8))
    digest = hashlib.sha256()
    for frame_id in range(3):
        for role, directory in sorted(masks.items()):
            digest.update(role.encode() + b"\0" + (directory / f"{frame_id:06d}.png").read_bytes())
    candidate = np.eye(4)
    candidate[0, 3] = 0.02
    audit = {
        "schema": "flash-camera-pose-audit-v1",
        "source": str(root),
        "episode": 0,
        "source_metadata_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "recorded_pose_trace_sha256": "recorded-pose-trace",
        "mask_sha256": digest.hexdigest(),
        "frames": [
            {"frame": 0, "time_s": 0.0, "pnp": None},
            {
                "frame": 1,
                "time_s": 1 / 15,
                "pnp": {
                    "heldout_source": {"median_px": 10.0},
                    "heldout_fit": {"median_px": 1.0},
                    "training_inliers": 12,
                    "candidate_world_from_camera": candidate.tolist(),
                },
            },
            {
                "frame": 2,
                "time_s": 2 / 15,
                "pnp": {
                    "heldout_source": {"median_px": 0.1},
                    "heldout_fit": {"median_px": 0.2},
                    "training_inliers": 12,
                    "candidate_world_from_camera": candidate.tolist(),
                },
            },
        ],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    samples = [CameraSample(i, i / 15, None) for i in range(3)]
    poses, decisions = _camera_pose_candidates(
        path, root, source_path, masks, 0, samples, "recorded-pose-trace"
    )
    assert poses[0] is None and poses[2] is None
    assert np.allclose(poses[1], candidate)
    assert [decision["selected"] for decision in decisions] == [
        "source",
        "background_rgb_candidate",
        "source",
    ]
    cv2.imwrite(str(masks["moving"] / "000001.png"), np.full((8, 8), 255, np.uint8))
    with pytest.raises(ValueError, match="masks no longer match"):
        _camera_pose_candidates(path, root, source_path, masks, 0, samples, "recorded-pose-trace")
    with pytest.raises(ValueError, match="does not match this recorded episode"):
        _camera_pose_candidates(path, root, source_path, masks, 0, samples, "changed-pose-trace")
