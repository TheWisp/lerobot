"""The demo scene uses only aligned, measured RGB-D evidence."""

from __future__ import annotations

import gzip
import hashlib
import json

import cv2
import numpy as np

from lerobot.policies.flash.demo_replay import _add_saved_extraction_layers
from lerobot.policies.flash.demo_scene import (
    reference_surfaces,
    scene_evidence_frame,
    scene_evidence_status,
)
from lerobot.policies.flash.scene_evidence_writer import _mask_runs


def test_reference_surface_uses_masked_depth_and_does_not_bridge_depth_edges(tmp_path) -> None:
    root = tmp_path
    (root / "recording/rgb").mkdir(parents=True)
    (root / "recording/depth").mkdir(parents=True)
    mask_dir = root / "flash/v1/masks"
    mask_dir.mkdir(parents=True)
    bundle_dir = root / "flash/v1/prepared/example"
    bundle_dir.mkdir(parents=True)
    np.savetxt(root / "recording/cam_K.txt", np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]]))
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[:, :, :] = (10, 20, 30)
    depth = np.full((4, 4), 1000, np.uint16)
    depth[:, 2:] = 1500
    mask = np.full((4, 4), 255, np.uint8)
    mask[3, 3] = 0
    assert cv2.imwrite(str(root / "recording/rgb/000000.jpg"), rgb)
    assert cv2.imwrite(str(root / "recording/depth/000000.png"), depth)
    assert cv2.imwrite(str(mask_dir / "example.png"), mask)
    object_from_camera = np.eye(4)
    object_from_camera[0, 3] = 0.2
    bundle = bundle_dir / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "references": [
                    {
                        "role": "test-object",
                        "snippets": [
                            {
                                "frame_index": 0,
                                "mask_artifact": "flash/v1/masks/example.png",
                                "object_from_camera": object_from_camera.tolist(),
                            }
                        ],
                    }
                ]
            }
        )
    )

    surface = reference_surfaces(root, bundle, step_px=1)["test-object"]
    points = np.asarray(surface["positions_object_m"]).reshape(-1, 3)
    faces = np.asarray(surface["indices"]).reshape(-1, 3)
    assert len(points) == 15  # The masked-out pixel never enters the model.
    assert np.any(np.all(np.isclose(points, [0.2, 0, 1]), axis=1))
    assert np.any(np.all(np.isclose(points, [0.23, 0, 1.5]), axis=1))
    assert all(np.ptp(points[face, 2]) == 0 for face in faces)


def test_extraction_layer_preserves_group_prediction_and_lost_state(tmp_path) -> None:
    root = tmp_path
    bundle_dir = root / "flash/v1/prepared/example"
    bundle_dir.mkdir(parents=True)
    cache_dir = root / "flash/v1/preparation-cache"
    cache_dir.mkdir(parents=True)
    pose = np.repeat(np.eye(4)[None], 3, axis=0)
    np.savez(
        cache_dir / "trace.npz",
        frames=3,
        object_poses=pose,
        object_evidence=np.array(["observed", "predicted", "lost"]),
        object_estimator=np.array(["appearance_rgbd", "environment_group_motion", "none"]),
        object_last_direct_s=np.array([1.0, 1.0, np.nan]),
    )
    (bundle_dir / "preparation-report.json").write_text(
        json.dumps(
            {
                "pose_trace_cache": "flash/v1/preparation-cache/trace.npz",
            }
        )
    )
    result = {
        "geometry": {"points_object_m": {"object": [[0, 0, 1]]}},
        "frames": [{"snapshot": {"recorded_at_s": float(i)}} for i in range(3)],
    }

    _add_saved_extraction_layers(root, bundle_dir / "bundle.json", result)
    layers = [frame["snapshot"]["layers"]["object"]["demo_extraction"] for frame in result["frames"]]
    assert result["extraction_history"] == "recorded_for_prepared_version"
    assert layers[1]["estimator"] == "environment_group_motion"
    assert layers[1]["evidence"] == "predicted"
    assert layers[2]["pose"] is None
    assert layers[2]["evidence"] == "lost"


def test_rebuilt_scene_evidence_preserves_mask_pixels_and_rejects_stale_bundle(tmp_path) -> None:
    bundle_dir = tmp_path / "prepared/example"
    sidecar = bundle_dir / "scene-evidence"
    sidecar.mkdir(parents=True)
    bundle = bundle_dir / "bundle.json"
    annotation = bundle_dir / "annotation.json"
    bundle.write_text('{"version":1}')
    annotation.write_text('{"role":"object"}')
    mask = np.array([[False, True, True], [False, False, True]])
    runs = _mask_runs(mask)
    assert runs == [[1, 2], [5, 1]]
    identity = hashlib.sha256(bundle.read_bytes() + annotation.read_bytes()).hexdigest()
    (sidecar / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha256": identity,
                "frames_completed": 1,
                "total_frames": 1,
                "complete": True,
            }
        )
    )
    with gzip.open(sidecar / "000000.json.gz", "wt") as stream:
        json.dump({"frame_index": 0, "masks_rle": {"object": runs}, "groups": {}}, stream)
    assert scene_evidence_status(bundle)["available"]
    assert scene_evidence_frame(bundle, 0)["masks_rle"]["object"] == runs
    assert not scene_evidence_frame(bundle, 1)["available"]
    bundle.write_text('{"version":2}')
    assert not scene_evidence_status(bundle)["available"]
