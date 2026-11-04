"""Acceptance of full run_case outputs, scored again from independent labels.

The costly real SAM-mask/UFM/group/fitter replay runs in the benchmark runner;
these checks consume its raw poses, never its claimed success/accuracy scores.
Set FLASH_RECOVERY_SEQUENCE_ROOT to a completed experiment directory.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.flash_full_pose_score import score_transforms
from benchmarks.flash_recorded_recovery import load
from lerobot.policies.flash.surface_observation import move, unproject

ROOT = os.environ.get("FLASH_RECOVERY_SEQUENCE_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="Requires a completed real recorded-data benchmark run")


@pytest.mark.parametrize(
    "case_id",
    [
        "flash-demo-gamepad-cube",
        "ycb-mustard_easy_00_02",
        "rig-yellow-block",
        "laval-clock-interaction-hard",
        "dexycb-drill-20201002_111923",
    ],
)
def test_combined_replay_never_accepts_a_failed_kalman_test(case_id):
    provenance = Path(ROOT) / f"{case_id}.provenance.json"
    if not provenance.exists():
        pytest.skip("Older run does not declare a combined filter experiment")
    mode = json.loads(provenance.read_text())["filter"]
    if mode in {"reset", "reset_broader"}:
        pytest.skip("Explicit legacy-reset comparison, not the proposed no-reset filter")
    assert mode in {"strict", "strict_broader", "velocity"}
    run = Path(ROOT) / case_id
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == json.loads((run / "case.json").read_text())["expected_frames"]
    decisions = 0
    for row in rows:
        for item in row["objects"].values():
            k = item["group_growing_details"].get("kalman", {})
            if k.get("accepted"):
                decisions += 1
                assert k["reason"] == "corrected", {"frame": row["frame"], "decision": k["reason"]}
                assert k["innovation_squared"] <= k["innovation_limit"], row["frame"]
    assert decisions > 0, "Rejecting everything is not a working correction pipeline"


@pytest.mark.parametrize(
    "name", ["gamepad_271", "mustard_180", "yellow_68", "clock_39", "clock_100", "drill_46"]
)
def test_full_sequence_recovers_at_reviewed_frame(name):
    data, meta = load(name)
    run = Path(ROOT) / meta["case"]
    case = json.loads((run / "case.json").read_text())
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == case["expected_frames"], "Incomplete replay cannot pass as a full-sequence result"
    assert [row["frame"] for row in rows] == list(range(len(rows)))
    role = "gamepad" if name.startswith("gamepad_") else case["roles"][0]
    row = rows[meta["frame"]]
    assert row["time_s"] == pytest.approx(meta["time_s"])
    item = row["objects"][role]
    assert item["group_growing_details"]["reference_frame"] == meta["reference_frame"]
    assert item["group_growing"] is not None, item["group_growing_details"].get("reason")
    pose = np.array(item["group_growing"])
    if "label_pose" in data:
        cloud, angle, _ = score_transforms(
            pose[None], np.array([True]), data["label_pose"][None], data["reference"]
        )
        assert cloud[0] <= 20 and angle[0] <= 20, {"cloud_mm": cloud[0], "angle_deg": angle[0]}
    elif "evaluation_visible_patch" in data:
        source, measured = data["reference"], data["evaluation_visible_patch"]
        _, _, a = np.linalg.svd(source - source.mean(0), full_matrices=False)
        _, _, b = np.linalg.svd(measured - measured.mean(0), full_matrices=False)
        angle = np.rad2deg(np.arccos(np.clip(abs((pose[:3, :3] @ a[-1]) @ b[-1]), 0, 1)))
        assert angle <= 20, {"visible_face_normal_disagreement_deg": angle}
    else:
        # Unlabelled block: only position consistency, as in the checkpoint test.
        f = meta["frame"]
        depth, mask = data[f"depth_{f}"], data[f"mask_{f}"]
        measured = unproject(depth, data[f"intrinsics_{f}"])[mask & (depth > 0) & np.isfinite(depth)]
        assert np.linalg.norm(np.median(move(data["reference"], pose), 0) - np.median(measured, 0)) <= 0.02
