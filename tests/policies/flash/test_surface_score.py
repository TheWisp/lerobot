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
    result = score_surface_trace(None, np.eye(4)[None], [{}, {}], "ufm_chain", np.eye(3))
    assert result["frames_scored"] == 0
    assert "median_mm" not in result
    assert result["reason"] == "No full labelled surface model"


def test_runtime_and_full_run_score_the_same_actual_surfaces(tmp_path):
    import json

    from benchmarks.flash_tracker_experiment import run_case
    from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
    from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig
    from tests.policies.flash.test_tracker_experiment import demo

    case = demo(tmp_path)
    x, y = np.meshgrid((np.arange(64) - 32) / 100, (np.arange(64) - 32) / 100)
    model = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    np.save(tmp_path / "model.npy", model)
    np.save(tmp_path / "poses.npy", np.repeat(np.eye(4)[None], 2, axis=0))
    case["evaluation"] = {"item": {"format": "poses_npy", "path": str(tmp_path / "poses.npy")}}
    case["evaluation_surfaces"] = {"item": {"points_m": str(tmp_path / "model.npy")}}
    reports = []
    for archive in (True, False):
        destination = tmp_path / str(archive)
        report = run_case(
            case,
            destination,
            None,
            VisionPipelineConfig(),
            None,
            None,
            single_method="conditioned_growing",
            surface_experiment=GrowingSurfaceConfig(),
            shared_matching_config=SharedMatchingConfig(backend="lk"),
            record_surfaces=archive,
        )
        metrics = report["roles"]["item"]["methods"]["conditioned_growing"]
        assert metrics["surface_error"]["frames_scored"] == 1
        assert metrics["surface_error"]["median_mm"] < 1
        assert list((destination / "surface_models").rglob("*.npz"))
        rows = [json.loads(line) for line in (destination / "frames.jsonl").read_text().splitlines()]
        assert rows[-1]["objects"]["item"]["conditioned_growing_details"]["map_version_after"] > 0
        reports.append(metrics)
    assert reports[0] == reports[1]
    assert not (tmp_path / "False/surface_patches").exists()


def test_unlabelled_runs_explicitly_report_surface_accuracy_unavailable(tmp_path):
    case, rows, _ = labelled_case(tmp_path)
    case.pop("evaluation")
    report = score_case(case, rows, {"part": np.eye(3)}, {"part": 1})
    for method in report["roles"]["part"]["methods"].values():
        assert method["surface_error"]["reason"] == "No reference poses"
        assert "median_mm" not in method["surface_error"]


def test_live_runtime_cli_uses_the_shared_quality_scorer(tmp_path, monkeypatch):
    import json
    import sys

    from benchmarks.flash_live_runtime import main
    from lerobot.policies.flash.surface_session_config import defaults
    from tests.policies.flash.test_tracker_experiment import demo

    case = demo(tmp_path)
    model = np.array([[x / 100, y / 100, 1.0] for x in range(-32, 32) for y in range(-32, 32)])
    np.save(tmp_path / "model.npy", model)
    np.save(tmp_path / "poses.npy", np.repeat(np.eye(4)[None], 2, axis=0))
    case["evaluation"] = {"item": {"format": "poses_npy", "path": str(tmp_path / "poses.npy")}}
    case["evaluation_surfaces"] = {"item": {"points_m": str(tmp_path / "model.npy")}}
    (tmp_path / "case.json").write_text(json.dumps(case))
    (tmp_path / "settings.json").write_text(json.dumps(defaults()))
    out = tmp_path / "live"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flash_live_runtime",
            "--case",
            str(tmp_path / "case.json"),
            "--settings",
            str(tmp_path / "settings.json"),
            "--output",
            str(out),
            "--frames",
            "2",
        ],
    )
    main()  # Real source reader, live session, RGB-D scene encoder and evaluator.
    report = json.loads((out / "summary.json").read_text())
    quality = report["quality"]["roles"]["item"]["methods"]["group_growing"]
    assert quality["surface_error"]["frames_scored"] == 1
    assert quality["surface_error"]["median_mm"] < 1
    rows = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    with np.load(out / "reference_points.npz") as references:
        rescored = score_case(case, rows, dict(references), {"item": 0}, run=out)
    assert rescored == report["quality"]
    from benchmarks.flash_runtime_benchmark import compare_runs

    comparison = compare_runs(out, out, "group_growing")
    assert comparison["identical"]
    assert comparison["quality"]["item"]["before"] == quality
    assert comparison["before_processing"] == report["steady"]


def test_live_comparison_hashes_fresh_masks_not_only_saved_seed_masks(tmp_path, monkeypatch):
    import json
    import sys

    from benchmarks import flash_live_runtime
    from benchmarks.flash_runtime_benchmark import compare_runs
    from lerobot.policies.flash.surface_session_config import defaults
    from tests.policies.flash.test_tracker_experiment import demo

    case = demo(tmp_path)
    (tmp_path / "case.json").write_text(json.dumps(case))
    (tmp_path / "settings.json").write_text(json.dumps(defaults()))
    outputs = []
    for inset in (4, 8):

        class Segmenter:
            def __init__(self, prompts, **kwargs):
                self.prompts = prompts

            def masks(self, frame, inset=inset):
                mask = np.zeros(frame.depth.shape, bool)
                mask[inset:-inset, inset:-inset] = True
                return {"item": mask}

        # Only replace the external model. Real source, tracking, encoding,
        # hashing and the comparison consumer run normally.
        monkeypatch.setattr(flash_live_runtime, "Sam3MaskSource", Segmenter)
        out = tmp_path / str(inset)
        outputs.append(out)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "flash_live_runtime",
                "--case",
                str(tmp_path / "case.json"),
                "--settings",
                str(tmp_path / "settings.json"),
                "--output",
                str(out),
                "--frames",
                "2",
                "--sam",
            ],
        )
        flash_live_runtime.main()
    with pytest.raises(ValueError, match="identical frames, depth and masks"):
        compare_runs(*outputs, "group_growing")
