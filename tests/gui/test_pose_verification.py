"""Correct measured pose before control while preserving the raw evidence in replay."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot.gui.api import manipulation as p
from lerobot.showservo import pose_verification


def setup_case(monkeypatch, tmp_path):
    fixture = Path(__file__).parents[1] / "showservo/fixtures/pose_disagreement_20261012.npz"
    with np.load(fixture) as z:
        ref, pose, depth, mask = (z[k] for k in ("reference", "latest_pose", "latest_depth", "latest_mask"))
        intr = json.loads(str(z["latest_intr"]))
    rgb = np.zeros((*depth.shape, 3), np.uint8)
    teach = p.runtime._Teach(
        "test",
        (0, 0, 0, 0),
        rgb,
        depth,
        intr,
        {"concept": "cube", "xyz": ref, "n_points": len(ref), "mode": "features"},
    )
    track = p.runtime._Track(on=True, job="verify")
    run = p.recording._Run(tmp_path, {})
    for key, value in {
        "teach": teach,
        "track": track,
        "demo": None,
        "targets": {},
        "test": None,
        "run": run,
        "recording": None,
        "groups_with_acts": True,
        "groups_feed": p.runtime._GroupsFeed(frames={"cube": [(0.0, np.eye(4)), (2.0, np.eye(4))]}),
    }.items():
        monkeypatch.setattr(p.runtime._state, key, value)
    r = {
        "ok": True,
        "algo": "p2p",
        "state": "tracking",
        "session": "one",
        "delta": pose,
        "model_xyz": ref,
        "mask": mask,
        "n_inliers": len(ref),
        "n_matches": len(ref),
        "n_tracks": len(ref),
        "other_delta_0": pose.copy(),
        "other_model_0": ref.copy(),
        "other_fit_uv_0": np.zeros((6, 2)),
        "other_fit_inlier_0": np.ones(6, bool),
    }
    job = p.runtime._Job("verify", "track", "cube", rgb, depth, intr, 1.0, result=r)
    monkeypatch.setattr(p.camera, "_t_base_cam", lambda: np.eye(4))
    return job, teach, run, track


def test_worker_result_corrects_bad_pose_before_control_and_records_both(monkeypatch, tmp_path):
    job, teach, run, track = setup_case(monkeypatch, tmp_path)
    shown = []
    monkeypatch.setattr(
        p.rendering, "_render_live", lambda *args: shown.extend(args[-1]["pose_verification"]) or b"image"
    )
    before = job.result["delta"].copy()
    asyncio.run(p.tracking._apply_track_result(job))
    p.runtime._RUN_EXECUTOR.submit(lambda: None).result(timeout=5)
    check = run.frames[0]["pose_verification"][0]
    assert run.frames[0]["t_verified"] <= run.frames[0]["t_rendered"] <= run.frames[0]["t_result"]
    assert shown == [check]
    assert check["accepted"] and check["surface"]["state"] == "disagreement"
    assert "group_transport" in check  # primary and secondary roles use the same fusion rule
    assert run.meta["pose_disagreements"][0]["frame"] == 0
    assert track.last["pose_verification"] == [check]
    corrected = np.asarray(check["constraint"]["pose"])
    assert check["constraint"]["state"] == "corrected" and not check["raw_p2p_used"]
    assert not np.allclose(corrected, before)
    assert check["constraint"]["validation_normal_after_deg"] < 3
    np.testing.assert_array_equal(p.runtime._state.test.result["delta_cam"], corrected)
    np.testing.assert_array_equal(track.hold_views[-1][1], corrected)
    np.testing.assert_array_equal(job.result["delta"], before)
    with np.load(tmp_path / "frames/000000.npz") as z:
        np.testing.assert_array_equal(z["delta"], before)
        np.testing.assert_array_equal(z["delta_used"], corrected)
        for key in ("other_delta_0", "other_model_0", "other_fit_uv_0", "other_fit_inlier_0"):
            np.testing.assert_array_equal(z[key], job.result[key])


def test_stale_group_pose_is_explicitly_unavailable(monkeypatch, tmp_path):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    p.runtime._state.groups_feed.frames["cube"] = [(0.0, np.eye(4))]
    check = p.tracking._verify_track_poses(job, job.result, teach, False)[0]
    assert check["groups"]["state"] == "unavailable"
    assert check["groups"]["time_skew_s"] == 1
    assert not check["accepted"] and check["disagreement"]


def test_group_authority_uses_the_pose_frames_evidence_not_a_later_drawing(monkeypatch, tmp_path):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    job.created = 1.0
    feed = p.runtime._state.groups_feed
    feed.frames["cube"] = [(1.0, np.eye(4)), (1.1, np.eye(4))]
    feed.evidence["cube"] = [
        (1.0, {"group": 0, "own": False, "group_supported": False}),
        (1.1, {"group": 0, "own": False, "group_supported": True}),
    ]
    feed.drawing = {"objects": {"cube": {"group": 0}}, "groups": {"0": {"supported": True}}}
    check = p.tracking._verify_track_poses(job, job.result, teach, False)[0]
    assert check["groups"]["state"] == "unavailable"
    assert check["groups"]["context"]["group"] == 0
    job.created = 1.1
    check = p.tracking._verify_track_poses(job, job.result, teach, False)[0]
    assert check["groups"]["state"] == "baseline"


def test_group_conflict_with_partial_surface_cannot_publish_a_control_pose(monkeypatch, tmp_path):
    job, _, _, track = setup_case(monkeypatch, tmp_path)
    shifted = np.eye(4)
    shifted[0, 3] = 0.012
    job.result["delta"] = shifted
    group = {
        **pose_verification.difference(shifted, np.eye(4), np.zeros(3), 0.003, 3.0),
        "predicted_pose": np.eye(4).tolist(),
    }
    monkeypatch.setattr(pose_verification.GroupReference, "compare", lambda *args, **kwargs: group)
    monkeypatch.setattr(
        p.tracking,
        "_surface_decision",
        lambda *args: {"state": "corrected", "pose": shifted.tolist(), "observed_dof": 3},
    )
    monkeypatch.setattr(p.rendering, "_render_live", lambda *args: b"image")
    asyncio.run(p.tracking._apply_track_result(job))
    check = track.last["pose_verification"][0]
    assert check["groups"]["state"] == "disagreement"
    assert check["constraint"]["state"] == "rejected"
    assert not check["accepted"] and p.runtime._state.test is None


def test_visible_rgbd_surface_can_resolve_wrong_group_motion():
    import cv2

    x, y = np.meshgrid(np.linspace(-0.02, 0.02, 21), np.linspace(-0.02, 0.02, 21))
    reference = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.4)))
    intr = {"fx": 400.0, "fy": 400.0, "cx": 100.0, "cy": 100.0}
    moved = np.eye(4)
    moved[0, 3] = 0.024
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (102, 78), (146, 122), 1, -1)
    depth = np.zeros_like(mask, dtype=float)
    depth[mask > 0] = 0.4
    group = {
        "state": "disagreement",
        "predicted_pose": np.eye(4).tolist(),
    }
    decision = {"state": "corrected", "pose": moved.tolist(), "observed_dof": 3}
    accepted = pose_verification.with_group_consistency(
        decision,
        group,
        np.zeros(3),
        tol_m=0.003,
        tol_deg=3.0,
        corrected_group_check=group,
        observation=(reference, depth, mask, intr),
        noise_m=0.003,
    )
    assert accepted["pose"] == decision["pose"]
    assert accepted["group_override"]["pose_support"] > accepted["group_override"]["group_support"]

    # A partly visible surface cannot settle a disagreement this way.
    mask[:, 124:] = 0
    obscured = pose_verification.with_group_consistency(
        decision,
        group,
        np.zeros(3),
        tol_m=0.003,
        tol_deg=3.0,
        corrected_group_check=group,
        observation=(reference, depth, mask, intr),
        noise_m=0.003,
    )
    assert obscured["state"] == "rejected" and obscured["pose"] is None


def test_surface_correction_is_checked_against_group_motion_before_control(monkeypatch, tmp_path):
    job, teach, _, track = setup_case(monkeypatch, tmp_path)
    job.result["delta"] = np.eye(4)
    p.runtime._state.groups_feed.frames["cube"] = [(1.0, np.eye(4)), (1.5, np.eye(4))]
    angles = iter((5.0, 10.0))

    def correction(*_args):
        from scipy.spatial.transform import Rotation

        angle = next(angles)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("x", angle, degrees=True).as_matrix()
        return {"state": "corrected", "pose": pose.tolist(), "observed_dof": 3}

    monkeypatch.setattr(p.tracking, "_surface_decision", correction)
    first = p.tracking._verify_track_poses(job, job.result, teach, False)[0]
    track.corrected_verification["cube"].accept()
    job.created = 1.5
    second = p.tracking._verify_track_poses(job, job.result, teach, False)[0]
    assert first["groups"]["state"] == first["corrected_groups"]["state"] == "baseline"
    assert first["constraint"]["state"] == "corrected"
    assert second["groups"]["state"] == "consistent"
    assert second["corrected_groups"]["state"] == "disagreement"
    assert second["constraint"]["state"] == "rejected" and second["constraint"]["pose"] is None
    assert second["disagreement"]
    assert track.corrected_verification["cube"].stamp == 1.0


def test_secondary_object_compares_raw_pose_in_its_original_designation_frame(monkeypatch, tmp_path):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    anchor = np.eye(4)
    anchor[:3, 3] = [0.04, -0.02, 0.01]
    target = p.runtime._TargetTrack(obj="cube", anchor=anchor, last={"state": "tracking"})
    monkeypatch.setattr(p.runtime._state, "demo", SimpleNamespace(objects={"cube": {}}))
    monkeypatch.setattr(p.runtime._state, "targets", {"cube": target})
    monkeypatch.setattr(p.planning, "_session_object", lambda _: "gamepad")
    monkeypatch.setattr(p.tracking, "_view_points", lambda *_: teach.keypoints["xyz"])
    job.result.update(
        others=[{"name": "cube", "ok": True, "session": "two"}],
        other_delta_0=job.result["delta"] @ np.linalg.inv(anchor),
        other_mask_0=job.result["mask"],
    )
    checks = p.tracking._verify_track_poses(job, job.result, teach, False)
    assert [c["object"] for c in checks] == ["gamepad", "cube"]
    assert not checks[0]["accepted"] and checks[1]["accepted"]
    np.testing.assert_allclose(checks[1]["p2p_pose"], job.result["delta"], atol=1e-12)
    assert checks[1]["surface"]["state"] == "disagreement"


def test_disagreement_is_logged_once_on_entry_even_outside_recording(monkeypatch, tmp_path, caplog):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    monkeypatch.setattr(p.runtime._state, "run", None)
    p.tracking._verify_track_poses(job, job.result, teach, True)
    p.tracking._verify_track_poses(job, job.result, teach, True)
    warnings = [r for r in caplog.records if "Pose disagreement" in r.message]
    assert len(warnings) == 1 and "cube P2P vs surface" in warnings[0].message
    assert "accepted=True" in warnings[0].message


def test_a_broken_surface_check_cannot_publish_a_control_pose_or_hold_sample(monkeypatch, tmp_path):
    job, _, run, _ = setup_case(monkeypatch, tmp_path)

    def broken(*args, **kwargs):
        raise ValueError("malformed diagnostic input")

    monkeypatch.setattr(pose_verification, "surface", broken)
    monkeypatch.setattr(p.rendering, "_render_live", lambda *args: b"image")
    asyncio.run(p.tracking._apply_track_result(job))
    p.runtime._RUN_EXECUTOR.submit(lambda: None).result(timeout=5)
    check = run.frames[0]["pose_verification"][0]
    assert not check["disagreement"] and check["surface"]["state"] == "unavailable"
    assert "malformed diagnostic input" in check["surface"]["reason"]
    assert p.runtime._state.test is None
    assert not p.runtime._state.track.hold_views
    assert not check["accepted"]
    assert check["constraint"]["state"] == "rejected"


def test_missing_geometry_cannot_bypass_rejection_through_hold_history(monkeypatch, tmp_path):
    job, _, _, track = setup_case(monkeypatch, tmp_path)
    job.depth_m[:] = 0
    monkeypatch.setattr(p.rendering, "_render_live", lambda *args: b"image")
    asyncio.run(p.tracking._apply_track_result(job))
    assert p.runtime._state.test is None and not track.hold_views
    assert track.last["state"] == "untrusted"


def test_geometry_result_from_a_replaced_teach_cannot_move_the_arm(monkeypatch, tmp_path):
    job, _, _, track = setup_case(monkeypatch, tmp_path)
    original = p.tracking._verify_track_poses

    def replaced(*args):
        result = original(*args)
        p.runtime._state.teach = None
        return result

    monkeypatch.setattr(p.tracking, "_verify_track_poses", replaced)
    asyncio.run(p.tracking._apply_track_result(job))
    assert p.runtime._state.test is None and not track.hold_views


def test_secondary_object_uses_corrected_absolute_pose_and_rejects_missing_depth(monkeypatch, tmp_path):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    reference = teach.keypoints["xyz"]
    anchor = np.eye(4)
    anchor[:3, 3] = [0.04, -0.02, 0.01]
    target = p.runtime._TargetTrack(obj="cube", anchor=anchor)
    found = {"delta": anchor.copy()}
    monkeypatch.setattr(p.runtime._state, "run", None)
    monkeypatch.setattr(p.runtime._state, "demo", SimpleNamespace(objects={"cube": {}}))
    monkeypatch.setattr(p.runtime._state, "targets", {"cube": target})
    monkeypatch.setattr(p.runtime._state, "located", {"cube": found})
    monkeypatch.setattr(p.planning, "_session_object", lambda _: "gamepad")
    monkeypatch.setattr(p.tracking, "_view_points", lambda *_: reference)
    monkeypatch.setattr(p.tracking, "_object_surface", lambda *_: reference)
    monkeypatch.setattr(p.acquisition, "_remember_seen", lambda *args: None)
    monkeypatch.setattr(p.rendering, "_render_live", lambda *args: b"image")
    job.result.update(
        others=[{"name": "cube", "ok": True, "n_visible": 25, "n_tracks": 25}],
        other_delta_0=job.result["delta"] @ np.linalg.inv(anchor),
        other_mask_0=job.result["mask"],
        other_fit_uv_0=None,
        other_fit_inlier_0=None,
    )
    asyncio.run(p.tracking._apply_track_result(job))
    check = p.runtime._state.track.last["pose_verification"][1]
    assert check["accepted"] and check["constraint"]["state"] == "corrected"
    assert "group_transport" in check
    np.testing.assert_allclose(found["delta"], check["constraint"]["pose"])
    np.testing.assert_allclose(target.hold_views[-1][1], found["delta"])
    held = found["delta"].copy()
    p.runtime._state.track.job = job.id
    job.created += 1
    job.depth_m[:] = 0
    asyncio.run(p.tracking._apply_track_result(job))
    assert len(target.hold_views) == 1
    np.testing.assert_array_equal(found["delta"], held)
    assert target.last["surface_constraint"]["state"] == "rejected"


def test_reference_acquisition_and_recorded_hold_receive_the_same_correction(monkeypatch, tmp_path):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    reference, raw = teach.keypoints["xyz"], job.result["delta"].copy()
    monkeypatch.setattr(p.tracking, "_reference_surface", lambda *args: reference)
    job.kind = "locate"
    job.extra = {"ref_recording": "recorded", "ref_frame": 0}
    job.arrays = {"ref_mask": job.result["mask"]}
    answer = {"ok": True, "ref_ok": True, "ref_delta": raw, "mask": job.result["mask"]}
    corrected = p.tracking._constrain_find_result(job, answer)
    assert corrected["ref_ok"] and corrected["surface_constraint"]["state"] == "corrected"
    np.testing.assert_array_equal(corrected["raw_ref_delta"], raw)
    np.testing.assert_array_equal(answer["ref_delta"], raw)
    assert not np.allclose(corrected["ref_delta"], raw)
    demo = SimpleNamespace(
        recording="recorded", objects={"cube": {"deltas": [raw, raw], "masks": [answer["mask"]] * 2}}
    )
    monkeypatch.setattr(p.tracking, "_object_surface", lambda *args: reference)
    monkeypatch.setattr(p.holds, "_recording_intr", lambda *args: job.intr)
    monkeypatch.setattr(
        p.recording,
        "_stream_frame",
        lambda rec, frame: (job.rgb, job.depth_m if frame == 0 else np.zeros_like(job.depth_m)),
    )
    views = p.holds._demo_surface_views(demo, "cube", [(0, 10), (1, 11)])
    np.testing.assert_allclose(views[0][1]["pose"], corrected["ref_delta"])
    assert views[1][1]["pose"] is None
    job.depth_m[:] = 0
    assert not p.tracking._constrain_find_result(job, answer)["ref_ok"]


def test_held_reference_geometry_uses_the_same_refreshed_mask_as_matching(monkeypatch, tmp_path):
    job, _, _, _ = setup_case(monkeypatch, tmp_path)
    job.kind = "locate"
    job.extra = {"ref_recording": "recorded", "ref_frame": 17}
    original = job.result["mask"]
    refreshed = np.zeros_like(original)
    job.arrays = {"ref_mask": original}
    used = []

    def surface(recording, frame, mask):
        used.append((recording, frame, mask))
        return np.zeros((12, 3))

    monkeypatch.setattr(p.tracking, "_reference_surface", surface)
    monkeypatch.setattr(p.tracking, "_surface_decision", lambda *a, **kw: {"pose": np.eye(4).tolist()})
    p.tracking._constrain_find_result(
        job,
        {
            "ok": True,
            "ref_ok": True,
            "ref_delta": np.eye(4),
            "ref_mask": refreshed,
            "mask": original,
        },
    )
    assert used[0][:2] == ("recorded", 17)
    assert used[0][2] is refreshed
    assert job.arrays["ref_mask"] is original, "the saved segmentation stays intact"


def test_unverified_demo_reference_aborts_before_the_act_can_move(monkeypatch, tmp_path):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    demo = SimpleNamespace(
        objects={"cube": {"deltas": [job.result["delta"]], "seen": [True]}}, reference_constraints={}
    )
    monkeypatch.setattr(p.runtime._state, "demo", demo)
    monkeypatch.setattr(p.runtime._state, "act", p.runtime._Act(on=True))
    monkeypatch.setattr(p.planning, "_act_objects", lambda _: ["cube"])
    monkeypatch.setattr(p.planning, "_pose_frame", lambda *_: 0)
    monkeypatch.setattr(
        p.holds,
        "_demo_surface_views",
        lambda *args: [
            (
                0,
                p.tracking._surface_decision(
                    job.result["delta"], teach.keypoints["xyz"], job.depth_m, job.result["mask"], job.intr
                ),
            )
        ],
    )
    started = []

    async def motion(*args):
        started.append(True)

    from lerobot.gui.api.manipulation import engine_act

    monkeypatch.setattr(engine_act, "_act_engine_task", motion)
    asyncio.run(p.execution._act_task(1.0))
    assert started == [True]
    assert demo.reference_constraints["cube"]["state"] == "corrected"
    np.testing.assert_array_equal(demo.objects["cube"]["deltas"][0], job.result["delta"])
    started.clear()
    job.depth_m[:] = 0
    asyncio.run(p.execution._act_task(1.0))
    assert not started and not p.runtime._state.act.on
    assert "demo reference frame" in p.runtime._state.act.reason


def test_planning_does_not_reintroduce_a_false_demo_tilt_by_inverting_raw_pose(monkeypatch):
    from scipy.spatial.transform import Rotation

    raw = np.eye(4)
    raw[:3, :3] = Rotation.from_euler("x", 15, degrees=True).as_matrix()
    demo = SimpleNamespace(
        name="d",
        recording="recorded",
        objects={"cube": {"frame": 0, "status": "done", "deltas": [raw]}},
        reference_constraints={
            "cube": {"frame": 0, "pose": np.eye(4).tolist(), "raw_pose": raw.tolist(), "state": "corrected"}
        },
    )
    monkeypatch.setattr(p.planning, "_session_object", lambda _: "cube")
    monkeypatch.setattr(p.planning, "_pose_frame", lambda *_: 0)
    found = {"object": "cube", "ok": True, "delta": np.eye(4), "inliers": 100, "card_points": 100}
    teach = SimpleNamespace(keypoints={"ref": found})
    monkeypatch.setattr(p.runtime._state, "located", {"cube": {**found, "view": ["d", 0]}})
    for pose, why in [
        p.planning._reference_motion(demo, teach),
        p.planning._object_motion(demo, "cube", np.eye(4)),
    ]:
        assert not why
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-12)


def test_warning_distinguishes_missing_evidence_from_acceptance(monkeypatch):
    captured = []
    monkeypatch.setattr(p.replay, "_replay_text", lambda _im, text, *_: captured.append(text))
    checks = [
        {
            "object": "cube",
            "accepted": True,
            "groups": {"state": "disagreement", "rotation_deg": 12.0, "centre_mm": 0.2},
            "surface": {"state": "inconclusive"},
        }
    ]
    p.rendering._draw_pose_verification(np.zeros((480, 848, 3), np.uint8), checks, 100)
    assert any(s == "cube P2P vs groups: DISAGREEMENT 12.0 deg / 0.2 mm; P2P accepted" for s in captured)
    assert any("RGB-D refit: inconclusive" in s for s in captured)
    captured.clear()
    p.rendering._draw_pose_verification(np.zeros((480, 848, 3), np.uint8), None, 100)
    assert captured == ["Pose verification: not recorded"]
