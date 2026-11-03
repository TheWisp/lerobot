"""Recorded RGB-D -> real UFM -> shared groups -> actual surface estimator.

Opt in with FLASH_RUN_RECORDED_TESTS=1. No robot, network, saved estimates or
gripper state is used. The annotation is consumed only AFTER tracking.
"""

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_experiment_groups import GroupConfig, GroupHints
from benchmarks.flash_experiment_inputs import samples
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.shared_image_tracking import SharedFrameTracker, SharedMatchingConfig
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_correspondence import robust_fit
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession, load_ufm
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
    shared = SharedFrameTracker(matching_config or SharedMatchingConfig(), model)
    preparation = SurfaceTrackingSession(
        model,
        shared_matching=shared,
        record_correspondences=True,
        motion_config=config.motion,
        observation_config=config.observation,
        shape_config=config.shape,
        fit_kernels=kernels,
        search_pool=pool,
    )
    estimator = GrowingSurfaceSession(
        config.shape,
        surface_config or GrowingSurfaceConfig(retain_unobserved_pose=True),
        fit_kernels=kernels,
        search_pool=pool,
    )
    groups = GroupHints(
        GroupConfig(), shared_image_points=True, uncertainty_config=estimator.config.uncertainty
    )
    rows, digest = [], hashlib.sha256()
    for frame, masks in sequence:
        for value in (frame.rgb, frame.depth, frame.intrinsics, frame.base_from_camera, *masks.values()):
            digest.update(value.tobytes())
        digest.update(np.float64(frame.source_time_s).tobytes())
        scene = shared.update(frame, masks)
        prepared = preparation.prepare(
            frame.rgb, frame.metric_depth().astype(np.float32), masks[role], frame.intrinsics, frame.frame_id
        )
        if (
            estimator.connected_observation is not None
            and prepared.get("match_source_frame") != estimator.connected_frame
            and prepared["status"] is None
        ):
            a, b = shared.pairs(estimator.connected_observation, prepared["current"], frame.intrinsics)
            step, _, _ = robust_fit(a, b, config.motion)
            prepared.update(step=step, point_pairs=(a, b), match_source_frame=estimator.connected_frame)
        hint, group_details = groups.predict(
            frame,
            scene,
            masks,
            role,
            estimator.reference_frame,
            object_usable=prepared["current"]["usable"],
            observation=prepared["current"] if prepared["status"] is None else None,
            shape_config=config.shape,
        )
        result = estimator.update(
            prepared,
            frame.source_time_s,
            hint=hint,
            base_from_camera=frame.base_from_camera,
            candidate_check=partial(
                groups.compare_candidate,
                frame,
                masks[role],
                hint,
                observation=prepared["current"] if estimator.config.surface_priority else None,
                shape_config=config.shape,
            )
            if hint
            else None,
        )
        groups.observe(
            frame,
            scene,
            masks,
            role,
            result,
            estimator.reference,
            object_usable=prepared["current"]["usable"],
        )
        result["diagnostics"]["group"] = group_details
        result["time_s"] = frame.source_time_s
        result["frame_id"] = frame.frame_id
        rows.append(result)
        if frame.frame_id % 40 == 0:
            print(f"recorded integration: frame {frame.frame_id}", flush=True)
    return {
        "rows": rows,
        "reference": estimator.reference.tolist(),
        "input_sha256": digest.hexdigest(),
        "config": {
            "matching": asdict(shared.config),
            "surface": asdict(estimator.config),
            "groups": asdict(groups.config),
            "vision": asdict(config),
        },
        "matching_calls": shared.calls,
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
