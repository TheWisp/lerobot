"""Known geometry and event boundaries for the offline bridge experiment."""

import csv
from pathlib import Path

import numpy as np
import pytest
from audit_motion_fusion import audit
from bridge_shape import CompleteSurfaceFit, MotionSupportedSurfaceFit, SupportedSurfaceFit, attachment_at
from compare import move
from scipy.spatial.transform import Rotation
from shape_grounding import ShapeConfig, WholeSurfaceFit
from test_shape_grounding import fixture, observation

from lerobot.policies.flash.surface_motion import PendingMotionCorrection, SurfaceMotionConfig


def test_recorded_partial_views_cannot_become_fresh_grasp_or_group_anchors():
    # Original aligned recording, frames 0, 200, 230, 270, 399; source geometry
    # from frame zero. 200/230/270 show only a fragment and/or detached depth.
    data = np.load(Path(__file__).parent / "results/bridges/admission_regression.npz")
    accepted = []
    for frame in data["frames"]:
        observation = {
            "depth": data[f"depth_{frame}"],
            "mask": data[f"mask_{frame}"],
            "obs": {"points": data[f"points_{frame}"]},
        }
        fit = CompleteSurfaceFit(
            data["reference_points"], observation, data["intrinsics"], ShapeConfig(), SurfaceMotionConfig()
        )
        if fit.covers_reference(data[f"pose_{frame}"]):
            accepted.append(int(frame))
    assert accepted == [0, 399]


def test_saved_consumer_follows_recorded_motor_motion_and_does_not_bypass_pose_owner():
    root = Path(__file__).parent / "results/bridges/bridge-guarded-final"
    trace = dict(np.load(root / "trace.npz"))
    with (root / "trace.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    result = audit(trace, rows)
    assert result["held_motion_steps_verified"] == 284
    # Restore the original bug at the replay-output boundary: a raw visual
    # proposal replaces the authoritative state without an admitted correction.
    trace["poses"] = trace["poses"].copy()
    trace["poses"][300] = trace["baseline_poses"][300]
    with pytest.raises(AssertionError):
        audit(trace, rows)


def test_repair_requires_visible_extent_and_accepts_genuine_complete_observation():
    source, k = fixture()
    complete = CompleteSurfaceFit(source, observation(source, k), k, ShapeConfig(), SurfaceMotionConfig())
    hidden = source[source[:, 0] > 0.015]
    partial = CompleteSurfaceFit(source, observation(hidden, k), k, ShapeConfig(), SurfaceMotionConfig())
    assert complete.covers_reference(np.eye(4))
    assert not partial.covers_reference(np.eye(4))
    wrong = np.eye(4)
    wrong[2, 3] = -0.02
    repaired, detail = complete.correct(wrong)
    assert detail["complete_surface"]
    assert np.mean(np.linalg.norm(move(source, repaired) - source, axis=1)) < 0.01
    untouched, detail = partial.correct(wrong)
    assert not detail["complete_surface"]
    np.testing.assert_allclose(untouched, wrong)


def test_supported_surface_corrects_depth_offset_but_does_not_chase_sub_noise_error():
    source, k = fixture()
    prior = np.eye(4)
    prior[2, 3] = -0.012
    fit = SupportedSurfaceFit(source, observation(source, k), k, ShapeConfig(), SurfaceMotionConfig())
    corrected, detail = fit.correct(prior)
    before = np.mean(np.linalg.norm(move(source, prior) - source, axis=1))
    after = np.mean(np.linalg.norm(move(source, corrected) - source, axis=1))
    assert detail["visual_changed"]
    assert after < before * 0.7
    small = np.eye(4)
    small[2, 3] = -0.001
    unchanged, detail = fit.correct(small)
    np.testing.assert_allclose(unchanged, small, atol=1e-12)
    assert not detail["visual_changed"]


def test_partial_surface_does_not_spin_held_object_to_fit_sampling_noise():
    source, k = fixture()
    patch = source[np.linalg.norm(source[:, :2], axis=1) < 0.008]
    data = observation(patch, k)
    data["depth"][~data["mask"]] = 0.4  # real foreground hiding the rest, not far background
    fit = SupportedSurfaceFit(source, data, k, ShapeConfig(), SurfaceMotionConfig())
    pose, detail = fit.correct(np.eye(4))
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-12)
    assert detail["supported_modes"] < 3


def test_depth_correction_is_confirmed_after_measured_gripper_movement():
    source, k = fixture()
    first = SupportedSurfaceFit(source, observation(source, k), k, ShapeConfig(), SurfaceMotionConfig())
    prior = np.eye(4)
    prior[2, 3] = -0.02
    candidate, _ = first.correct(prior)

    def score(pose):
        return np.sum(first.bounded_residual(pose) ** 2)

    pending = PendingMotionCorrection()
    pending.propose("grasp", np.eye(4), prior, candidate, score, 1)
    assert pending.pending is not None
    moved = np.eye(4)
    moved[:3, :3] = Rotation.from_euler("x", 5, degrees=True).as_matrix()
    second = SupportedSurfaceFit(
        source, observation(move(source, moved), k), k, ShapeConfig(), SurfaceMotionConfig()
    )

    def score(pose):
        return np.sum(second.bounded_residual(pose) ** 2)

    corrected = pending.check("grasp", moved, moved @ prior, score, 1)
    assert corrected is not None
    truth = move(source, moved)
    assert np.mean(np.linalg.norm(move(source, corrected) - truth, axis=1)) < 0.01


def test_current_geometry_can_correct_an_inaccurate_motion_prediction():
    source, k = fixture()
    prior = np.eye(4)
    prior[:3, :3] = Rotation.from_euler("z", 8, degrees=True).as_matrix()
    prior[:3, 3] = [0.006, 0, 0]
    fit = MotionSupportedSurfaceFit(source, observation(source, k), k, ShapeConfig(), prior, 0.005)
    corrected, _ = fit.correct(prior)
    error_before = np.mean(np.linalg.norm(move(source, prior) - source, axis=1))
    error_after = np.mean(np.linalg.norm(move(source, corrected) - source, axis=1))
    assert error_after < error_before
    # A rigid-gripper-only output cannot pass this correction requirement.
    assert not np.allclose(corrected, prior)


def test_motion_prior_disambiguates_geometry_without_a_symmetry_rule():
    _, k = fixture()
    theta = np.arange(128) * 2 * np.pi / 128
    source = np.column_stack((0.03 * np.cos(theta), 0.03 * np.sin(theta), np.full(128, 0.55)))
    prior = np.eye(4)
    opposite = np.eye(4)
    opposite[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    data = observation(source, k)
    visual = WholeSurfaceFit(source, data, k, ShapeConfig())
    constrained = MotionSupportedSurfaceFit(source, data, k, ShapeConfig(), prior, 0.005)
    assert np.isclose(visual.score(prior), visual.score(opposite), atol=1e-10)
    assert constrained.score(opposite) > constrained.score(prior) + 1


def test_motion_constraint_is_independent_of_arbitrary_reference_origin():
    source, k = fixture()
    prior = np.eye(4)
    shifted = np.eye(4)
    shifted[:3, 3] = [0.003, 0.002, 0.001]
    data = observation(source, k)
    original = MotionSupportedSurfaceFit(source, data, k, ShapeConfig(), prior, 0.005)
    origin_change = np.eye(4)
    origin_change[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.3, -0.1]).as_matrix()
    origin_change[:3, 3] = [1.5, -2.0, 3.0]
    changed = MotionSupportedSurfaceFit(
        move(source, origin_change), data, k, ShapeConfig(), prior @ np.linalg.inv(origin_change), 0.005
    )
    assert np.isclose(original.score(shifted), changed.score(shifted @ np.linalg.inv(origin_change)))


def test_release_removes_gripper_relationship_at_the_recorded_event():
    skill = {
        "attachment_transitions": [
            {"event_id": "closed", "action": "attach", "role": "tool"},
            {"event_id": "opened", "action": "detach", "role": "tool"},
        ],
        "intervals": [
            {
                "samples": [
                    {"time_s": 1.5, "event_id": "closed"},
                    {"time_s": 3.5, "event_id": "opened"},
                ]
            }
        ],
    }
    assert not attachment_at(skill, "tool", 1.49)
    assert attachment_at(skill, "tool", 1.5)
    assert attachment_at(skill, "tool", 3.49)
    assert not attachment_at(skill, "tool", 3.5)
    assert not attachment_at(skill, "another_object", 2.0)


def test_runner_hands_partial_view_from_gripper_to_points_at_release(tmp_path, monkeypatch):
    """Exercise the actual experiment composition, including its event switch."""
    import json
    import sys

    import bridge_shape
    import cv2

    recording = tmp_path / "recording"
    recording.mkdir()
    k = np.array([[100.0, 0, 48], [0, 100.0, 48], [0, 0, 1]])
    times = np.array([0.0, 0.033, 0.066])
    np.savetxt(recording / "cam_K.txt", k)
    np.savetxt(recording / "times.txt", times)
    calibration = np.load(Path(__file__).parent.parent / "flash-tracker-comparison/pose_traces.npz")
    ee = np.tile(np.eye(4), (3, 1, 1))
    ee[:, 0, 3] = [0.0, 0.01, 0.3]  # gripper departs; object moves only another centimetre
    np.savez(tmp_path / "showservo_demo.npz", t=times, t0=0.0, tips=calibration["base_from_camera"] @ ee)
    texture = np.random.default_rng(10).integers(0, 255, (96, 96), dtype=np.uint8)
    gray = np.zeros_like(texture)
    gray[20:76, 20:40] = texture[20:76, 20:40]
    new_view = np.zeros_like(texture)
    new_view[20:76, 50:74] = texture[20:76, 50:74]
    images = []
    for i in range(3):
        rgb = cv2.warpAffine(
            np.repeat((gray if i == 0 else new_view)[:, :, None], 3, axis=2),
            np.float32([[1, 0, i], [0, 1, 0]]),
            (96, 96),
        )
        mask = np.zeros(gray.shape, bool)
        mask[20:76, 20 + i : 76 + i] = True
        if i >= 1:
            # The old surface disappears during the grasp. The new partial
            # surface supplies motion but cannot independently refit the pose.
            mask[:, :48] = False
        y, x = np.nonzero(mask)
        points = np.column_stack(((x - 48) / 100, (y - 48) / 100, np.ones(len(x))))
        depth = np.full(gray.shape, 1.5, np.float32)
        depth[mask] = 1.0
        images.append(
            {
                "rgb": rgb,
                "mask": mask,
                "depth": depth,
                "usable": mask,
                "obs": {"points": points[::4], "valid_image": mask},
            }
        )
    source = images[0]["obs"]["points"]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    np.savez(
        baseline / "trace.npz",
        frames=np.arange(3),
        poses=np.tile(np.eye(4), (3, 1, 1)),
        reference_points=source,
        landmarks_reference=source[[0, -1, len(source) // 2]],
    )
    from dataclasses import asdict

    (baseline / "summary.json").write_text(
        json.dumps({"config": asdict(ShapeConfig()), "reference_frame": 0})
    )
    skill = {
        "attachment_transitions": [
            {"event_id": "grasp", "action": "attach", "role": "object"},
            {"event_id": "release", "action": "detach", "role": "object"},
        ],
        "intervals": [
            {"samples": [{"event_id": "grasp", "time_s": 0.0}, {"event_id": "release", "time_s": 0.066}]}
        ],
    }
    (tmp_path / "skill.json").write_text(json.dumps(skill))
    monkeypatch.setattr(bridge_shape, "RECORDING", recording)
    monkeypatch.setattr(bridge_shape, "frame_data", lambda index, intrinsics: images[index])
    output = tmp_path / "result"
    trial = Path(__file__).resolve().parents[3] / "src/lerobot/policies/flash/docs/trial_config.example.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bridge_shape",
            "--baseline",
            str(baseline),
            "--output",
            str(output),
            "--skill",
            str(tmp_path / "skill.json"),
            "--trial-config",
            str(trial),
            "--role",
            "object",
            "--guarded",
        ],
    )
    bridge_shape.main()
    trace = np.load(output / "trace.npz")
    rows = list(csv.DictReader((output / "trace.csv").open()))
    assert rows[2]["direct_qualified"] == "0"
    assert rows[2]["prior_source"] == "object_point_motion"
    np.testing.assert_allclose(trace["poses"][2, :3, 3], [0.02, 0, 0], atol=0.005)
    assert rows[2]["last_visual_update_s"] == rows[1]["last_visual_update_s"]
