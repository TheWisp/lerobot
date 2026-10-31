import gzip
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.reference_view_candidates import (
    MeasuredReferenceView,
    review_reference_view_candidates,
)
from lerobot.policies.flash.reference_view_registration import fit_reference_view_candidates
from lerobot.policies.flash.sealed_reference_view_review import (
    _gripper_motion_prior,
    review_sealed_reference_view,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation
from lerobot.policies.flash.surface_reference import (
    RegisteredSurfaceModel,
    registered_surface_agreement,
)
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def test_gripper_motion_prior_uses_only_grasp_seed_and_motor_motion(tmp_path):
    bundle = tmp_path / "flash/v1/prepared/skill/bundle.json"
    bundle.parent.mkdir(parents=True)
    camera_times = np.array([0.0, 1.0, 2.0])
    np.savez(
        bundle.parent / "held-attachment-trace.npz",
        camera_times_s=camera_times,
        gripper_from_object=np.repeat((np.eye(4) * 100)[None], 3, axis=0),
    )
    arm = np.repeat(np.eye(4)[None], 3, axis=0)
    arm[:, 0, 3] = [0.0, 0.1, 0.2]
    np.savez(tmp_path / "showservo_demo.npz", t=camera_times, tips=arm)
    grasp_seed = np.eye(4)
    grasp_seed[0, 3] = 0.2
    camera = np.eye(4)
    camera[0, 3] = 0.05
    annotation = {"held_role": "object", "grasp_time_s": 0.0, "release_time_s": 2.0}
    report = {"held_attachment_trace": "held-attachment-trace.npz"}
    result = _gripper_motion_prior(tmp_path, bundle, annotation, report, "object", 1, grasp_seed, camera)
    assert result is not None
    np.testing.assert_allclose(result[:3, 3], [0.25, 0.0, 0.0])
    assert _gripper_motion_prior(tmp_path, bundle, annotation, report, "other", 1, grasp_seed, camera) is None
    assert (
        _gripper_motion_prior(tmp_path, bundle, annotation, report, "object", 2, grasp_seed, camera) is None
    )


@pytest.fixture
def recorded_views():
    fixture = Path(__file__).parent / "fixtures" / "gamepad_reference_depth_20261028.npz"
    with np.load(fixture) as data:
        intrinsics = data["intrinsics"]
        base_from_camera = data["base_from_camera"]
        first_points = data["reference_points_object_m"].copy()
        measured = {}
        estimates = {}
        for index in (223, 290, 300, 430):
            depth = data[f"depth_{index}"].astype(np.float32) / 1000
            mask = data[f"mask_{index}"].astype(bool)
            observed = observation(depth, mask, intrinsics, asdict(SurfaceObservationConfig()))
            measured[index] = MeasuredReferenceView(
                index, depth, observed["valid_image"], observed["points"], int(mask.sum())
            )
            estimates[index] = {
                "object": PoseEstimate(
                    "object",
                    data[f"pose_{index}"],
                    float(index),
                    float(index),
                    Evidence.OBSERVED,
                    "recorded",
                    (True, True, True),
                    (True, True, True),
                    source_frame_id=index,
                )
            }
    model = RegisteredSurfaceModel("object", "recorded", (first_points,))
    return intrinsics, base_from_camera, model, estimates, measured


def test_candidate_review_keeps_first_reference_and_separates_depth_directions(recorded_views):
    intrinsics, base_from_camera, model, estimates, measured = recorded_views
    original = model.view_points_object_m[0].copy()
    summaries, details = review_reference_view_candidates(
        estimates,
        (model,),
        {"object": [290, 300]},
        lambda role, index: measured[index],
        intrinsics,
        base_from_camera,
        256,
    )
    assert len(summaries) == len(details) == 2
    assert all(item["status"] == "registration_hypothesis" for item in summaries)
    assert all(item["cross_frame"]["frames"] == 3 for item in summaries)
    assert all("with_candidate" in row for item in details for row in item["comparisons"])
    # A closer patch at frame 430 can coexist with a geometrically wrong
    # projected surface. Both directions must remain visible to the reviewer.
    at_430 = next(row for row in details[0]["comparisons"] if row["frame_index"] == 430)
    assert at_430["with_candidate"]["reference_to_depth_p90_m"] > 0.1
    for comparison in (at_430["saved_views"], at_430["with_candidate"]):
        assert comparison["reference_in_front_p90_m"] is not None
        assert comparison["reference_behind_p90_m"] is not None
        assert comparison["reference_in_front_p90_m"] >= 0
        assert comparison["reference_behind_p90_m"] >= 0
    np.testing.assert_array_equal(model.view_points_object_m[0], original)
    assert len(model.view_points_object_m) == 1


def test_candidate_review_accepts_existing_multiple_registered_views(recorded_views):
    intrinsics, base_from_camera, model, estimates, measured = recorded_views
    second = model.view_points_object_m[0].copy()
    second[:, 0] -= 0.03
    multi = RegisteredSurfaceModel("object", "recorded", (model.view_points_object_m[0], second))
    summaries, details = review_reference_view_candidates(
        estimates,
        (multi,),
        {"object": [290]},
        lambda role, index: measured[index],
        intrinsics,
        base_from_camera,
        256,
    )
    assert summaries[0]["status"] == "registration_hypothesis"
    assert details[0]["comparisons"]
    assert len(multi.view_points_object_m) == 2
    # The baseline must include every saved view, not only the primary one.
    rows = details[0]["comparisons"]
    assert all("saved_views" in row and "with_candidate" in row for row in rows)
    other = rows[0]["frame_index"]
    camera_from_object = np.linalg.inv(base_from_camera) @ estimates[other]["object"].pose
    first_score, saved_score = registered_surface_agreement(
        multi,
        camera_from_object,
        measured[other].points_camera_m,
        measured[other].clean_depth_pixels,
        measured[other].depth_m,
        intrinsics,
    )
    assert saved_score.reference_to_depth_p90_m > first_score.reference_to_depth_p90_m
    assert rows[0]["saved_views"]["reference_to_depth_p90_m"] == pytest.approx(
        saved_score.reference_to_depth_p90_m
    )


def test_independent_view_fit_reports_alternatives_without_using_the_pose_trace(recorded_views):
    intrinsics, base_from_camera, model, estimates, measured = recorded_views
    with np.load(Path(__file__).parent / "fixtures" / "gamepad_reference_depth_20261028.npz") as fixture:
        mask = fixture["mask_290"].astype(bool)
    original = model.view_points_object_m[0].copy()
    config = ShapeConfig(max_fit_points=256, max_evaluations=30, batch_jacobian=False)
    first = fit_reference_view_candidates(
        model.view_points_object_m[0], measured[290], mask, intrinsics, config
    )
    prior = np.linalg.inv(base_from_camera) @ estimates[290]["object"].pose
    with_prior = fit_reference_view_candidates(
        model.view_points_object_m[0],
        measured[290],
        mask,
        intrinsics,
        config,
        prior_poses_camera_from_object={"recorded_pose_prior": prior},
    )
    assert len(first) == 4
    assert len(with_prior) == 5
    assert {item.seed for item in first} == {f"geometry_{index}" for index in range(4)}
    assert "recorded_pose_prior" in {item.seed for item in with_prior}
    for item in first:
        matching = next(candidate for candidate in with_prior if candidate.seed == item.seed)
        np.testing.assert_allclose(item.camera_from_object, matching.camera_from_object)
        assert item.fit_cost >= 0
        assert item.measured_to_reference_p90_m >= 0
        assert item.reference_depth_pixels > 0
        assert item.reference_in_front_p90_m is not None
        assert item.reference_behind_p90_m is not None
    np.testing.assert_array_equal(model.view_points_object_m[0], original)


def test_candidate_review_rejects_unobserved_pose_and_misaligned_frame(recorded_views):
    intrinsics, base_from_camera, model, estimates, measured = recorded_views
    observed = estimates[290]["object"]
    estimates[290]["object"] = PoseEstimate(
        observed.role,
        observed.pose,
        observed.source_time_s,
        observed.available_time_s,
        Evidence.PREDICTED,
        observed.reference_version,
        observed.translation_supported,
        observed.rotation_supported,
        source_frame_id=290,
    )
    summaries, _ = review_reference_view_candidates(
        estimates,
        (model,),
        {"object": [290]},
        lambda role, index: measured[index],
        intrinsics,
        base_from_camera,
        256,
    )
    assert summaries[0]["status"] == "unregistered"
    assert "observed" in summaries[0]["reason"]
    with pytest.raises(ValueError, match="different source frame"):
        review_reference_view_candidates(
            estimates,
            (model,),
            {"object": [300]},
            lambda role, index: measured[290],
            intrinsics,
            base_from_camera,
            256,
        )
    estimates[300]["object"] = PoseEstimate(
        observed.role,
        observed.pose,
        300.0,
        300.0,
        Evidence.OBSERVED,
        observed.reference_version,
        observed.translation_supported,
        observed.rotation_supported,
        source_frame_id=290,
    )
    with pytest.raises(ValueError, match="pose has a different source frame"):
        review_reference_view_candidates(
            estimates,
            (model,),
            {"object": [300]},
            lambda role, index: measured[index],
            intrinsics,
            base_from_camera,
            256,
        )


def test_old_prepared_demo_rebuilds_first_view_from_saved_mask_without_mutating_bundle(tmp_path, monkeypatch):
    root = tmp_path / "demo"
    bundle = root / "flash/v1/prepared/example/bundle.json"
    bundle.parent.mkdir(parents=True)
    mask_path = root / "flash/v1/masks/object.png"
    mask_path.parent.mkdir(parents=True)
    depth_path = root / "recording/depth"
    depth_path.mkdir(parents=True)
    calibration = root / "calibration.json"
    identity = np.eye(4)
    calibration.write_text(json.dumps({"camera": {"T_base_cam": identity.tolist()}}))
    (bundle.parent / "annotation.json").write_text(
        json.dumps({"episode": 0, "calibration_path": str(calibration)})
    )
    (bundle.parent / "preparation-report.json").write_text(
        json.dumps({"pose_trace_cache": "flash/v1/preparation-cache/trace.npz"})
    )
    np.savetxt(root / "recording/cam_K.txt", [[100, 0, 16], [0, 100, 16], [0, 0, 1]])
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 255
    assert cv2.imwrite(str(mask_path), mask)
    evidence_root = bundle.parent / "scene-evidence"
    evidence_root.mkdir()
    for index in range(3):
        assert cv2.imwrite(str(depth_path / f"{index:06d}.png"), np.full(mask.shape, 1000, np.uint16))
        evidence = {
            "frame_index": index,
            "image_size": list(mask.shape),
            "masks_rle": {"object": [[row * 32 + 8, 16] for row in range(8, 24)]},
        }
        with gzip.open(evidence_root / f"{index:06d}.json.gz", "wt") as stream:
            json.dump(evidence, stream)
    cache = root / "flash/v1/preparation-cache/trace.npz"
    cache.parent.mkdir(parents=True)
    np.savez(
        cache,
        frames=3,
        object_evidence=["observed"] * 3,
        object_poses=np.repeat(identity[None], 3, axis=0),
        object_translation_supported=np.ones((3, 3), bool),
        object_rotation_supported=np.ones((3, 3), bool),
        object_estimator=["surface_rgbd"] * 3,
    )
    reference = ObjectReference(
        "object",
        "v1",
        (
            ReferenceSnippet(
                "test/demo", 0, 0, "object", identity, "flash/v1/masks/object.png", "calibration"
            ),
        ),
    )
    prepared = SimpleNamespace(
        skill=SimpleNamespace(source_episode=0), references=(reference,), surface_models=()
    )
    monkeypatch.setattr("lerobot.policies.flash.prepared.PreparedFlashSkill.load", lambda _: prepared)
    original_files = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
    result = review_sealed_reference_view(root, bundle, 0, "object", 1)
    assert result["first_view_points"] >= 3
    assert result["candidate"]["status"] == "registration_hypothesis"
    assert result["candidate"]["cross_frame"]["frames"] == 2
    assert "independent_fits" in result
    assert sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file()) == original_files

    second = ReferenceSnippet(
        "test/demo", 0, 1, "object", identity, "flash/v1/masks/object.png", "calibration"
    )
    prepared.references = (ObjectReference("object", "v1", (*reference.snippets, second)),)
    points = np.array([[x, y, 1.0] for x in np.linspace(-0.05, 0.05, 8) for y in np.linspace(-0.05, 0.05, 8)])
    prepared.surface_models = (RegisteredSurfaceModel("object", "v1", (points, points.copy())),)
    reviewed_again = review_sealed_reference_view(root, bundle, 0, "object", 1)
    assert reviewed_again["replacing_saved_view"] is True
    assert reviewed_again["saved_view_count"] == 1
    assert reviewed_again["candidate"]["status"] == "registration_hypothesis"
