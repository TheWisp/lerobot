"""Check benchmark scores against independent trajectories and source files."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from benchmarks.flash_surface_benchmark import (
    diagnostic,
    lift_measured_depth,
    prepare_local,
    score,
    timing,
    validate_prediction,
)
from benchmarks.flash_surface_lk import run as run_lk
from benchmarks.flash_surface_synthetic import build as build_synthetic
from benchmarks.flash_surface_ycbineoat import build as build_ycb


def test_missing_visible_points_stay_in_denominator_and_coherent_drift_is_wrong():
    frames = np.arange(3)
    queries = np.array([[0, 1, 2], [0, 3, 4]], dtype=float)
    truth_xy = np.array([[[1, 2], [3, 4]], [[2, 2], [4, 4]], [[3, 2], [5, 4]]], dtype=float)
    truth_xyz = np.concatenate((truth_xy * 0.001, np.ones((3, 2, 1))), axis=-1)
    reference = {
        "frames": frames,
        "queries": queries,
        "truth_visible": np.ones((3, 2), dtype=bool),
        "truth_xy": truth_xy,
        "truth_xyz": truth_xyz,
    }
    # Both points shift by the same wrong amount, so rigidity alone cannot catch it.
    prediction = {
        "frames": frames.copy(),
        "queries": queries.copy(),
        "xy": truth_xy + [10, 0],
        "xyz": truth_xyz + [0.01, 0, 0],
        "visible": np.ones((3, 2), dtype=bool),
    }
    prediction["visible"][1, 0] = False
    report = score(reference, prediction)
    assert report["eligible_point_frames"] == 6
    assert report["predicted_visible"] == 5
    assert report["coverage_px"]["8"] == 0
    assert report["coverage_mm"]["5"] == 0
    assert report["error_px_when_available"]["median"] == 10
    # Losing a visible point cannot make coverage improve.
    prediction["xy"][:] = truth_xy
    assert score(reference, prediction)["coverage_px"]["1"] == pytest.approx(5 / 6)


def test_prediction_must_retain_query_identity():
    reference = {"frames": np.arange(2), "queries": np.array([[0, 2, 3], [0, 4, 5]])}
    prediction = {
        "frames": reference["frames"],
        "queries": reference["queries"][::-1],
        "xy": np.zeros((2, 2, 2)),
        "visible": np.ones((2, 2), bool),
    }
    with pytest.raises(ValueError, match="exact source frames and query identities"):
        validate_prediction(reference, prediction)


def test_timing_reports_old_results_even_when_published_often():
    capture = np.array([0, 1 / 30, 2 / 30, 3 / 30], float)
    report = timing(capture, np.array([0, 0, 2]), np.array([0.05, 0.06, 0.13]))
    assert report["publications"] == 3
    assert report["unique_source_publications"] == 2
    assert report["publication_age_ms"]["max"] == pytest.approx(63.3333333)
    assert report["consumer_ticks_before_first_result"] > 0
    with pytest.raises(ValueError, match="cannot precede"):
        timing(capture, np.array([2]), np.array([0.01]))


def test_local_preparation_uses_real_source_times_and_sealed_queries(tmp_path):
    recording = tmp_path / "recording"
    masks = tmp_path / "masks"
    for folder in (recording / "rgb", recording / "depth", masks):
        folder.mkdir(parents=True)
    np.savetxt(recording / "cam_K.txt", np.array([[20, 0, 4], [0, 20, 4], [0, 0, 1]]))
    np.savetxt(recording / "times.txt", [4.0, 4.034, 4.068])
    for frame in range(3):
        cv2.imwrite(str(recording / "rgb" / f"{frame:06d}.jpg"), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(recording / "depth" / f"{frame:06d}.png"), np.full((8, 8), 900, np.uint16))
        cv2.imwrite(str(masks / f"{frame:06d}.png"), np.full((8, 8), 255, np.uint8))
    manifest = prepare_local(recording, masks, 1, 2, 9, tmp_path / "experiment")
    with np.load(tmp_path / "experiment" / "queries.npz", allow_pickle=False) as prepared:
        assert prepared["frames"].tolist() == [1, 2]
        assert prepared["capture_s"].tolist() == pytest.approx([0, 0.034])
        assert len({tuple(row) for row in prepared["queries"]}) == 9
        assert np.allclose(prepared["initial_depth_m"], 0.9)
    assert json.loads((tmp_path / "experiment" / "manifest.json").read_text()) == manifest
    assert manifest["ground_truth"].startswith("unlabelled")


def test_sensor_depth_lift_does_not_score_its_own_depth_as_independent_agreement(tmp_path):
    depth = np.full((8, 8), 1.0, dtype=np.float32)
    intrinsics = np.array([[20, 0, 4], [0, 20, 4], [0, 0, 1]], float)
    xy = np.array([[3.0, 2.0], [99.0, 99.0]], np.float32)
    xyz = lift_measured_depth(xy, depth, intrinsics)
    np.testing.assert_allclose(xyz[0], [-0.05, -0.1, 1.0])
    assert np.isnan(xyz[1]).all()
    recording, masks = tmp_path / "recording", tmp_path / "masks"
    for directory in (recording / "depth", masks):
        directory.mkdir(parents=True)
    cv2.imwrite(str(recording / "depth" / "000000.png"), np.full((8, 8), 1000, np.uint16))
    cv2.imwrite(str(masks / "000000.png"), np.full((8, 8), 255, np.uint8))
    reference = {"frames": np.array([0]), "queries": np.array([[0, 3, 2]])}
    prediction = {
        "frames": reference["frames"],
        "queries": reference["queries"],
        "xy": xy[:1, None],
        "xyz": xyz[:1, None],
        "visible": np.ones((1, 1), bool),
        "geometry_source": np.asarray("sensor_at_prediction"),
    }
    report = diagnostic(recording, masks, reference, prediction, 0.01)
    assert report["mask_fraction"] == 1
    assert report["within_depth_tolerance_fraction"] is None
    prediction["geometry_source"] = np.asarray("model_3d")
    assert diagnostic(recording, masks, reference, prediction, 0.01)["within_depth_tolerance_fraction"] == 1


def test_ycb_pose_reference_transports_same_material_point(tmp_path):
    source = tmp_path / "source"
    for directory in ("rgb", "depth", "gt_mask", "annotated_poses"):
        (source / directory).mkdir(parents=True)
    intrinsics = np.array([[20, 0, 8], [0, 20, 8], [0, 0, 1]], float)
    np.savetxt(source / "cam_K.txt", intrinsics)
    for index in range(3):
        name = f"{1_000_000_000 + index * 33_000_000}.png"
        rgba = np.full((16, 16, 4), 120, np.uint8)
        cv2.imwrite(str(source / "rgb" / name), rgba)
        cv2.imwrite(str(source / "depth" / name), np.full((16, 16), 1000, np.uint16))
        cv2.imwrite(str(source / "gt_mask" / name), np.full((16, 16), 255, np.uint8))
        pose = np.eye(4)
        pose[0, 3] = index * 0.05
        np.savetxt(source / "annotated_poses" / f"{index:07d}.txt", pose)
    output = tmp_path / "experiment"
    manifest = build_ycb(source, 0, 2, 1, 0.01, output)
    with np.load(output / "queries.npz") as reference:
        np.testing.assert_allclose(reference["truth_xy"][:, 0, 0], reference["queries"][0, 1] + [0, 1, 2])
        np.testing.assert_allclose(
            reference["truth_xyz"][:, 0, 0], reference["truth_xyz"][0, 0, 0] + [0, 0.05, 0.1]
        )
        assert reference["truth_visible"].all()
    assert manifest["labelled_visible_fraction"] == 1


def test_lk_adapter_tracks_known_pixel_translation(tmp_path):
    recording, masks = tmp_path / "recording", tmp_path / "masks"
    for directory in (recording / "rgb", recording / "depth", masks):
        directory.mkdir(parents=True)
    np.savetxt(recording / "cam_K.txt", [[80, 0, 32], [0, 80, 32], [0, 0, 1]])
    np.savetxt(recording / "times.txt", [0, 1 / 30, 2 / 30])
    rng = np.random.default_rng(1729)
    base = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    for index in range(3):
        shifted = cv2.warpAffine(base, np.float32([[1, 0, index], [0, 1, 0]]), (64, 64))
        cv2.imwrite(str(recording / "rgb" / f"{index:06d}.jpg"), shifted)
        cv2.imwrite(str(recording / "depth" / f"{index:06d}.png"), np.full((64, 64), 1000, np.uint16))
        mask = np.zeros((64, 64), np.uint8)
        mask[20:44, 20:44] = 255
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    experiment = tmp_path / "experiment"
    prepare_local(recording, masks, 0, 2, 9, experiment)
    run_lk(experiment, experiment / "lk.npz")
    with np.load(experiment / "lk.npz") as result:
        assert result["visible"].all()
        displacement = result["xy"][2] - result["xy"][0]
        assert np.median(np.linalg.norm(displacement - [2, 0], axis=1)) < 0.5
    paced_report = run_lk(experiment, experiment / "lk_paced.npz", paced=True)
    with np.load(experiment / "lk_paced.npz") as paced:
        indices = paced["publication_source_index"]
        assert indices[0] == 0 and indices[-1] == 2
        assert np.all(np.diff(indices) > 0)
        assert np.all(paced["publication_s"] >= np.asarray([0, 1 / 30, 2 / 30])[indices])
        assert paced_report["pacing"]["consumer_ticks"] > 1
        assert paced_report["pacing"]["skipped_source_frames"] == 3 - len(indices)


def test_exact_motion_control_distinguishes_visible_drift_from_good_tracking(tmp_path):
    output = tmp_path / "plane"
    build_synthetic(output, "translation", "textured", frames=5, points=9)
    with np.load(output / "queries.npz", allow_pickle=False) as saved:
        ref = {key: saved[key] for key in saved.files}
    displacement = ref["truth_xy"][-1, :, 0] - ref["truth_xy"][0, :, 0]
    np.testing.assert_allclose(displacement, 40.0, atol=0.1)
    np.testing.assert_allclose(ref["truth_xyz"][-1, :, 0] - ref["truth_xyz"][0, :, 0], 0.08)
    assert ref["truth_visible"].all()
    prediction = {
        "frames": ref["frames"],
        "queries": ref["queries"],
        "xy": ref["truth_xy"].copy(),
        "xyz": ref["truth_xyz"].copy(),
        "visible": np.ones_like(ref["truth_visible"]),
    }
    assert score(ref, prediction)["coverage_px"]["1"] == 1.0
    prediction["xy"] += [3, 0]
    prediction["xyz"] += [0.006, 0, 0]
    assert score(ref, prediction)["coverage_px"]["2"] == 0.0
    assert score(ref, prediction)["coverage_mm"]["5"] == 0.0
