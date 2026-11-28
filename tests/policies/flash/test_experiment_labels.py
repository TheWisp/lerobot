"""Independent labels, coordinate conventions, frame alignment and saved-result scoring."""

import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_experiment_labels import load_pose_labels
from benchmarks.flash_rescore_experiment import rescore
from benchmarks.flash_tracker_experiment import score_case


def labelled_case(tmp_path):
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, :3] = Rotation.from_euler(
        "xyz", [[0, 0, 0], [10, 30, 20], [20, 45, 30], [40, 50, 60]], degrees=True
    ).as_matrix()
    poses[:, :3, 3] = [[0.1, 0.2, 0.8], [0.2, 0.1, 0.7], [0.3, -0.1, 0.8], [0.4, -0.2, 0.9]]
    conversion = np.diag([1, -1, -1, 1])
    source = tmp_path / "poses.npy"
    np.save(source, (conversion @ poses).reshape(-1, 16))
    spec = {"format": "poses_npy", "path": str(source), "camera_from_label_camera": conversion.tolist()}
    rows = []
    for i in range(4):
        relative = poses[i] @ np.linalg.inv(poses[1])
        rows.append(
            {
                "frame": i,
                "time_s": i / 30,
                "objects": {
                    "part": {
                        "ufm_chain": relative.tolist() if i in (1, 2) else None,
                        "surface_corrected": relative.tolist(),
                        "prepare_ms": 2,
                        "finish_ms": 3,
                    }
                },
            }
        )
    case = {"id": "labelled", "kind": "recorded_rgbd", "roles": ["part"], "evaluation": {"part": spec}}
    return case, rows, poses


def test_camera_conversion_and_late_initialization_use_independent_labels(tmp_path):
    case, rows, _ = labelled_case(tmp_path)
    source = np.array([[0.2, 0.1, 0.7], [0.25, 0.1, 0.7], [0.2, 0.15, 0.7]])
    summary = score_case(case, rows, {"part": source}, {"part": 1})
    methods = summary["roles"]["part"]["methods"]
    assert summary["accuracy_labels"]
    assert methods["ufm_chain"]["pose_error"]["within_10mm_10deg"] == 1
    assert methods["ufm_chain"]["total_after_initialization"] == 2
    assert methods["surface_corrected"]["pose_error"]["cloud_mm_median"] < 1e-9
    # Omitting the OpenGL-to-RGB-D conversion must give a clearly wrong score.
    case["evaluation"]["part"].pop("camera_from_label_camera")
    wrong = score_case(case, rows, {"part": source}, {"part": 1})
    assert wrong["roles"]["part"]["methods"]["surface_corrected"]["pose_error"]["cloud_mm_median"] > 100


def test_labels_are_role_specific(tmp_path):
    case, rows, _ = labelled_case(tmp_path)
    case["roles"].append("unlabelled")
    for row in rows:
        row["objects"]["unlabelled"] = row["objects"]["part"].copy()
    summary = score_case(case, rows, {r: np.eye(3) for r in case["roles"]}, dict.fromkeys(case["roles"], 1))
    assert summary["roles"]["part"]["accuracy_labels"]
    assert not summary["roles"]["unlabelled"]["accuracy_labels"]
    assert "pose_error" not in summary["roles"]["unlabelled"]["methods"]["surface_corrected"]


def test_rigid_labelled_points_recover_known_motion(tmp_path):
    case, rows, poses = labelled_case(tmp_path)
    # A planar patch defines rigid motion with labelled, noncollinear points.
    local = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]])
    points = np.array([local @ pose[:3, :3].T + pose[:3, 3] for pose in poses], dtype=np.float32)
    source = tmp_path / "points.npz"
    np.savez(source, truth_xyz=points, frames=np.arange(4), capture_s=1000 + np.arange(4) / 30)
    case["evaluation"]["part"] = {"format": "rigid_points_npz", "path": str(source)}
    labels, _ = load_pose_labels(case, rows)
    np.testing.assert_allclose(labels["part"], poses @ np.linalg.inv(poses[0]), atol=1e-6)
    # Displaced known labels must not be silently accepted as a rigid trajectory.
    points[2, 0, 0] += 0.01
    np.savez(source, truth_xyz=points, frames=np.arange(4), capture_s=np.arange(4) / 30)
    with pytest.raises(ValueError, match="exact rigid motion"):
        load_pose_labels(case, rows)
    np.savez(source, truth_xyz=points, frames=np.arange(1, 5), capture_s=np.arange(4) / 30)
    with pytest.raises(ValueError, match="frame indices"):
        load_pose_labels(case, rows)
    np.savez(source, truth_xyz=points, frames=np.arange(4), capture_s=np.arange(4) / 15)
    with pytest.raises(ValueError, match="timestamps"):
        load_pose_labels(case, rows)


@pytest.mark.parametrize("fault", ["order", "count", "scale", "role"])
def test_invalid_pose_labels_are_rejected(tmp_path, fault):
    case, rows, poses = labelled_case(tmp_path)
    if fault == "order":
        rows.reverse()
    elif fault == "count":
        np.save(tmp_path / "poses.npy", poses[:2])
    elif fault == "scale":
        poses[:, :3, :3] *= 2
        np.save(tmp_path / "poses.npy", poses)
    else:
        case["evaluation"]["missing"] = case["evaluation"]["part"]
    with pytest.raises(ValueError):
        load_pose_labels(case, rows)


def test_rescore_publishes_metrics_without_changing_predictions_or_video(tmp_path):
    case, rows, _ = labelled_case(tmp_path)
    evaluation = case.pop("evaluation")
    run, exported = tmp_path / "run", tmp_path / "export"
    run.mkdir()
    exported.mkdir()
    (run / "case.json").write_text(json.dumps(case))
    (run / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    np.savez(run / "reference_points.npz", part=np.eye(3))
    previous = score_case(case, rows, {"part": np.eye(3)}, {"part": 1})
    previous.update(input_sha256="same-input", offline_wall_s=12)
    (run / "summary.json").write_text(json.dumps(previous))
    manifest = {"source_run": str(run), "input_sha256": "same-input", "frames": 4}
    (exported / "manifest.json").write_text(json.dumps(manifest))
    (exported / "video.mp4").write_bytes(b"unchanged video")
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (run / "frames.jsonl", run / "reference_points.npz", exported / "video.mp4")
    }
    result = rescore(run, evaluation=evaluation, browser_export=exported)
    assert result["accuracy_labels"] and result["offline_wall_s"] == 12
    assert json.loads((exported / "summary.json").read_text()) == result
    assert json.loads((run / "case.json").read_text())["evaluation"] == evaluation
    assert json.loads(next((run / "score-history").glob("*/summary.json")).read_text()) == previous
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    manifest["input_sha256"] = "wrong-export"
    (exported / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different predictions"):
        rescore(run, browser_export=exported)
