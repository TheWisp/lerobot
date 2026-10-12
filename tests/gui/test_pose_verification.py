"""Recording and display of disagreements must not silently change the act's pose."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot.gui.api import pregrasp as p


def setup_case(monkeypatch, tmp_path):
    fixture = Path(__file__).parents[1] / "showservo/fixtures/pose_disagreement_20261012.npz"
    with np.load(fixture) as z:
        ref, pose, depth, mask = (z[k] for k in ("reference", "latest_pose", "latest_depth", "latest_mask"))
        intr = json.loads(str(z["latest_intr"]))
    rgb = np.zeros((*depth.shape, 3), np.uint8)
    teach = p._Teach(
        "test",
        (0, 0, 0, 0),
        rgb,
        depth,
        intr,
        {"concept": "cube", "xyz": ref, "n_points": len(ref), "mode": "features"},
    )
    track = p._Track(on=True, job="verify")
    run = p._Run(tmp_path, {})
    for key, value in {
        "teach": teach,
        "track": track,
        "demo": None,
        "targets": {},
        "test": None,
        "run": run,
        "recording": None,
        "groups_with_acts": True,
        "groups_feed": p._GroupsFeed(frames={"cube": [(0.0, np.eye(4)), (2.0, np.eye(4))]}),
    }.items():
        monkeypatch.setattr(p._state, key, value)
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
    job = p._Job("verify", "track", "cube", rgb, depth, intr, 1.0, result=r)
    monkeypatch.setattr(p, "_t_base_cam", lambda: np.eye(4))
    return job, teach, run, track


def test_worker_result_records_and_warns_on_an_accepted_bad_pose_without_replacing_it(monkeypatch, tmp_path):
    job, teach, run, track = setup_case(monkeypatch, tmp_path)
    shown = []
    monkeypatch.setattr(
        p, "_render_live", lambda *args: shown.extend(args[-1]["pose_verification"]) or b"image"
    )
    before = job.result["delta"].copy()
    asyncio.run(p._apply_track_result(job))
    p._RUN_EXECUTOR.submit(lambda: None).result(timeout=5)
    check = run.frames[0]["pose_verification"][0]
    assert shown == [check]
    assert check["accepted"] and check["surface"]["state"] == "disagreement"
    assert run.meta["pose_disagreements"][0]["frame"] == 0
    assert track.last["pose_verification"] == [check]
    np.testing.assert_array_equal(p._state.test.result["delta_cam"], before)
    np.testing.assert_array_equal(job.result["delta"], before)
    with np.load(tmp_path / "frames/000000.npz") as z:
        for key in ("other_delta_0", "other_model_0", "other_fit_uv_0", "other_fit_inlier_0"):
            np.testing.assert_array_equal(z[key], job.result[key])


def test_stale_group_pose_is_explicitly_unavailable(monkeypatch, tmp_path):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    p._state.groups_feed.frames["cube"] = [(0.0, np.eye(4))]
    check = p._verify_track_poses(job, job.result, teach, False)[0]
    assert check["groups"]["state"] == "unavailable"
    assert check["groups"]["time_skew_s"] == 1
    assert not check["accepted"] and check["disagreement"]


def test_secondary_object_compares_raw_pose_in_its_original_designation_frame(monkeypatch, tmp_path):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    anchor = np.eye(4)
    anchor[:3, 3] = [0.04, -0.02, 0.01]
    target = p._TargetTrack(obj="cube", anchor=anchor, last={"state": "tracking"})
    monkeypatch.setattr(p._state, "demo", SimpleNamespace(objects={"cube": {}}))
    monkeypatch.setattr(p._state, "targets", {"cube": target})
    monkeypatch.setattr(p, "_session_object", lambda _: "gamepad")
    monkeypatch.setattr(p, "_view_points", lambda *_: teach.keypoints["xyz"])
    job.result.update(
        others=[{"name": "cube", "ok": True, "session": "two"}],
        other_delta_0=job.result["delta"] @ np.linalg.inv(anchor),
        other_mask_0=job.result["mask"],
    )
    checks = p._verify_track_poses(job, job.result, teach, False)
    assert [c["object"] for c in checks] == ["gamepad", "cube"]
    assert not checks[0]["accepted"] and checks[1]["accepted"]
    np.testing.assert_allclose(checks[1]["p2p_pose"], job.result["delta"], atol=1e-12)
    assert checks[1]["surface"]["state"] == "disagreement"


def test_disagreement_is_logged_once_on_entry_even_outside_recording(monkeypatch, tmp_path, caplog):
    job, teach, _, _ = setup_case(monkeypatch, tmp_path)
    monkeypatch.setattr(p._state, "run", None)
    p._verify_track_poses(job, job.result, teach, True)
    p._verify_track_poses(job, job.result, teach, True)
    warnings = [r for r in caplog.records if "Pose disagreement" in r.message]
    assert len(warnings) == 1 and "cube P2P vs surface" in warnings[0].message
    assert "accepted=True" in warnings[0].message


def test_a_broken_diagnostic_is_recorded_as_unavailable_without_losing_the_control_pose(
    monkeypatch, tmp_path
):
    job, _, run, _ = setup_case(monkeypatch, tmp_path)

    def broken(*args, **kwargs):
        raise ValueError("malformed diagnostic input")

    monkeypatch.setattr(p.pose_verification, "surface", broken)
    monkeypatch.setattr(p, "_render_live", lambda *args: b"image")
    asyncio.run(p._apply_track_result(job))
    p._RUN_EXECUTOR.submit(lambda: None).result(timeout=5)
    check = run.frames[0]["pose_verification"][0]
    assert not check["disagreement"] and check["surface"]["state"] == "unavailable"
    assert "malformed diagnostic input" in check["surface"]["reason"]
    np.testing.assert_array_equal(p._state.test.result["delta_cam"], job.result["delta"])


def test_warning_distinguishes_missing_evidence_from_acceptance(monkeypatch):
    captured = []
    monkeypatch.setattr(p, "_replay_text", lambda _im, text, *_: captured.append(text))
    checks = [
        {
            "object": "cube",
            "accepted": True,
            "groups": {"state": "disagreement", "rotation_deg": 12.0, "centre_mm": 0.2},
            "surface": {"state": "inconclusive"},
        }
    ]
    p._draw_pose_verification(np.zeros((480, 848, 3), np.uint8), checks, 100)
    assert any(s == "cube P2P vs groups: DISAGREEMENT 12.0 deg / 0.2 mm; P2P accepted" for s in captured)
    assert any("RGB-D refit: inconclusive" in s for s in captured)
    captured.clear()
    p._draw_pose_verification(np.zeros((480, 848, 3), np.uint8), None, 100)
    assert captured == ["Pose verification: not recorded"]
