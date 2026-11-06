"""Geometry agreement must distinguish symmetry from a genuinely misplaced surface."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_full_pose_score import score_transforms
from benchmarks.flash_surface_score import LabelledSurface, score_surface_trace
from benchmarks.flash_tracker_experiment import score_case
from tests.policies.flash.test_experiment_labels import labelled_case


def cylinder():
    angle, z = np.meshgrid(np.arange(360) * np.pi / 180, np.linspace(-0.05, 0.05, 11))
    return np.column_stack((0.025 * np.cos(angle.ravel()), 0.025 * np.sin(angle.ravel()), z.ravel()))


def test_axial_turn_preserves_surface_but_translation_does_not():
    model = cylinder()
    measured = model[model[:, 0] > 0]  # Only one observed side is reconstructed.
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    surface = LabelledSurface(model)
    assert surface.distances_mm(measured, pose, np.eye(4)).max() < 1e-9
    cloud, angle, _ = score_transforms(pose[None], np.array([True]), np.eye(4)[None], measured)
    assert angle[0] == pytest.approx(90)
    assert cloud[0] > 30  # Primary identity-sensitive pose score stays unchanged.
    pose[0, 3] = 0.1
    assert surface.distances_mm(measured, pose, np.eye(4)).min() > 49


def test_asymmetric_surface_turn_is_not_excused():
    model = np.array([[0, 0, 0], [0.1, 0, 0], [0.08, 0.02, 0], [0, 0.03, 0.01]])
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    error = LabelledSurface(model).distances_mm(model, pose, np.eye(4))
    assert np.median(error) > 20


def test_score_uses_label_coordinates_late_initialization_and_preserves_pose_score(tmp_path):
    case, rows, truth = labelled_case(tmp_path)
    model = cylinder()
    file = tmp_path / "model.npy"
    np.save(file, model)
    reference = model @ truth[1, :3, :3].T + truth[1, :3, 3]
    before = score_case(case, rows, {"part": reference}, {"part": 1})
    case["evaluation_surfaces"] = {"part": {"points_m": str(file)}}
    after = score_case(case, rows, {"part": reference}, {"part": 1})
    for method in ("ufm_chain", "surface_corrected"):
        old = before["roles"]["part"]["methods"][method]
        new = after["roles"]["part"]["methods"][method]
        assert new["pose_error"] == old["pose_error"]
        score = new["surface_error"]
        assert score["median_mm"] < 1e-9
        assert score["frames_total"] == 2
        assert score["frames_scored"] == (1 if method == "ufm_chain" else 2)
        assert score["within_10mm_percent"] == 100


def test_added_bad_geometry_is_scored_and_absent_surface_is_not_substituted(tmp_path):
    model = cylinder()
    file = tmp_path / "model.npy"
    np.save(file, model)
    spec = {"points_m": str(file)}
    rows = [{"group_growing": np.eye(4).tolist(), "group_growing_details": {"map_version_after": 2}}] * 3
    truth = np.repeat(np.eye(4)[None], 3, axis=0)
    absent = score_surface_trace(spec, truth, rows, "group_growing", model, run=tmp_path, role="part")
    assert absent["frames_scored"] == 0
    assert absent["missing_surface_frames"] == 2
    folder = tmp_path / "surface_models/group_growing/part"
    folder.mkdir(parents=True)
    np.savez(folder / "000002.npz", points=np.concatenate((model, model + [0.2, 0, 0])))
    score = score_surface_trace(spec, truth, rows, "group_growing", model, run=tmp_path, role="part")
    assert score["frames_scored"] == 2
    assert score["within_10mm_percent"] == pytest.approx(50)
    assert score["p90_mm"] > 150


def test_no_full_model_has_no_surface_accuracy_score():
    result = score_surface_trace(None, None, [{}, {}], "ufm_chain", np.eye(3))
    assert result["frames_scored"] == 0
    assert "median_mm" not in result
    assert result["reason"] == "No full labelled surface model"
