"""Recorded RGB-D -> real UFM -> shared groups -> actual surface estimator.

Opt in with FLASH_RUN_RECORDED_TESTS=1. No robot, network, saved estimates or
gripper state is used. The annotation is consumed only AFTER tracking.
"""

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_experiment_inputs import samples
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.motion_groups import GroupConfig
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_frame_session import SurfaceFrameSession
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

ANNOTATION = Path(__file__).with_name("fixtures") / "gamepad_stationary.json"


def track_sequence(sequence, role, model, *, matching_config=None, surface_config=None, vision_config=None):
    """Same consumers/order as run_case's group_growing option, with diagnostics."""
    config = vision_config or VisionPipelineConfig()
    kernels = compiled_kernels() if config.compiled_surface_fit else None
    with (
        SearchPool(config.shape.batch_query_workers) if config.reuse_surface_search else nullcontext()
    ) as pool:
        return _track_sequence(sequence, role, model, matching_config, surface_config, config, kernels, pool)


def _track_sequence(sequence, role, model, matching_config, surface_config, config, kernels, pool):
    rows, digest = [], hashlib.sha256()
    with SurfaceFrameSession(
        [role],
        model,
        config,
        kernels,
        pool,
        surface_experiment=surface_config or GrowingSurfaceConfig(retain_unobserved_pose=True),
        group_experiment=GroupConfig(),
        shared_matching_config=matching_config or SharedMatchingConfig(),
        single_method="group_growing",
        record_correspondences=True,
    ) as engine:
        for frame, masks in sequence:
            for value in (frame.rgb, frame.depth, frame.intrinsics, frame.base_from_camera, *masks.values()):
                digest.update(value.tobytes())
            digest.update(np.float64(frame.source_time_s).tobytes())
            row, _ = engine.step(frame, masks)
            result = row["objects"][role]
            rows.append(
                {
                    "status": result["group_growing_status"],
                    "camera_from_reference": result["group_growing"],
                    "diagnostics": result["group_growing_details"],
                    "time_s": frame.source_time_s,
                    "frame_id": frame.frame_id,
                }
            )
            if frame.frame_id % 40 == 0:
                print(f"recorded integration: frame {frame.frame_id}", flush=True)
        estimator = engine.candidates["group_growing"][role]
        return {
            "rows": rows,
            "reference": estimator.reference.tolist(),
            "input_sha256": digest.hexdigest(),
            "config": {
                "matching": asdict(engine.shared.config),
                "surface": asdict(estimator.config),
                "groups": asdict(engine.scene_groups.config),
                "vision": asdict(config),
            },
            "matching_calls": engine.shared.calls,
        }


def stationary_errors(report, last_frame):
    rows = report["rows"][: last_frame + 1]
    assert all(row["camera_from_reference"] is not None for row in rows), "pose disappeared before grasp"
    poses = np.asarray([r["camera_from_reference"] for r in rows])
    center = np.asarray(report["reference"]).mean(0, keepdims=True)
    locations = np.asarray([move(center, p)[0] for p in poses])
    angles = Rotation.from_matrix(poses[:, :3, :3] @ poses[0, :3, :3].T).magnitude()
    return np.linalg.norm(locations - locations[0], axis=1) * 1000, np.rad2deg(angles)


@pytest.fixture(scope="module")
def recording_result(tmp_path_factory):
    if os.environ.get("FLASH_RUN_RECORDED_TESTS") != "1":
        pytest.skip("set FLASH_RUN_RECORDED_TESTS=1 for the real recorded/model integration test")
    import torch
    from huggingface_hub import try_to_load_from_cache

    annotation = json.loads(ANNOTATION.read_text())
    if not torch.cuda.is_available():
        pytest.skip("real UFM requires CUDA")
    weights = try_to_load_from_cache(
        "infinity1096/UFM-Base-DINOv2L-init", "model.safetensors", revision=annotation["ufm_revision"]
    )
    if not isinstance(weights, str):
        pytest.skip("pinned UFM checkpoint is not cached; this test does not download it")
    if not Path(annotation["case"]["mask_evidence_dir"]).expanduser().is_dir():
        pytest.skip("recorded gamepad RGB-D/masks are not installed")
    output = tmp_path_factory.mktemp("gamepad-integration")
    model = load_ufm(revision=annotation["ufm_revision"])
    result = track_sequence(
        islice(samples(annotation["case"]), annotation["run_last_frame"] + 1), annotation["role"], model
    )
    result["annotation"] = annotation
    (output / "result.json").write_text(json.dumps(result) + "\n")
    print(f"Recorded integration diagnostics: {output / 'result.json'}", flush=True)
    assert result["input_sha256"] == annotation["input_sha256"], "the recorded test inputs changed"
    return result


@pytest.mark.timeout(600)
def test_recorded_gamepad_stays_still_through_pregrasp_occlusion(recording_result):
    annotation = recording_result["annotation"]
    displacement, rotation = stationary_errors(recording_result, annotation["stationary_last_frame"])
    failures = {
        "max_center_mm": float(displacement.max()),
        "center_frame": int(displacement.argmax()),
        "max_rotation_deg": float(rotation.max()),
        "rotation_frame": int(rotation.argmax()),
    }
    assert (
        displacement.max() <= annotation["max_center_displacement_mm"]
        and rotation.max() <= annotation["max_rotation_deg"]
    ), failures


@pytest.mark.timeout(600)
def test_recorded_gamepad_is_not_frozen_after_pickup(recording_result):
    # A necessary positive control, NOT a claim of accurate post-grasp 6D pose.
    # The final frame visibly has the gamepad lifted and transported beside the cube.
    # Availability after pickup is measured by the full benchmark separately.
    # A missing intervening pose must not masquerade as a failure to move.
    rows = recording_result["rows"]
    first, last = rows[0]["camera_from_reference"], rows[-1]["camera_from_reference"]
    assert first is not None and last is not None, "movement needs both endpoint estimates"
    center = np.asarray(recording_result["reference"]).mean(0, keepdims=True)
    displacement = np.linalg.norm(move(center, np.asarray(last)) - move(center, np.asarray(first))) * 1000
    assert displacement > 30, "freezing at the initial pose must not pass"
    assert recording_result["matching_calls"] == recording_result["annotation"]["run_last_frame"]
