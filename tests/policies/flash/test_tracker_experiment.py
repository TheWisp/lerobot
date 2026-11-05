"""Experiment boundaries: source clocks, failed motion links and independent scoring."""

import gzip
import json

import cv2
import numpy as np
import pytest

from benchmarks.flash_experiment_groups import GroupConfig
from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_tracker_experiment import MotionChain, RetainedMotionReference, run_case, score_case
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.surface_correspondence import SurfaceMotionConfig
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def demo(tmp_path):
    root = tmp_path / "recording"
    evidence = tmp_path / "masks"
    evidence.mkdir()
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    np.savetxt(root / "times.txt", [1000.0, 1000.04])
    np.savetxt(root / "cam_K.txt", [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    (evidence / "manifest.json").write_text(json.dumps({"complete": True, "frames_completed": 2}))
    for i in range(2):
        cv2.imwrite(str(root / "rgb" / f"{i:06d}.jpg"), np.full((64, 64, 3), 120, np.uint8))
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.full((64, 64), 1000, np.uint16))
        row = {
            "frame_index": i,
            "source_time_s": 2 + i * 0.04,
            "image_size": [64, 64],
            "base_from_camera": np.eye(4).tolist(),
            "calibration_version": "test",
            "masks_rle": {"item": [[0, 64 * 64]]},
            "estimates": {"item": {"pose": "deliberately unusable old prediction"}},
        }
        with gzip.open(evidence / f"{i:06d}.json.gz", "wt") as f:
            json.dump(row, f)
    return {
        "id": "fixture",
        "kind": "flash_prepared_demo",
        "artifact_root": str(tmp_path),
        "mask_evidence_dir": str(evidence),
        "expected_frames": 2,
        "roles": ["item"],
    }


@pytest.mark.parametrize("kind,extension", [("flash_prepared_demo", "jpg"), ("recorded_rgbd", "png")])
def test_demo_reads_measured_frames_and_masks_without_saved_pose(tmp_path, kind, extension):
    case = demo(tmp_path)
    case["kind"] = kind
    case["rgb_extension"] = extension
    if extension == "png":
        for source in (tmp_path / "recording/rgb").glob("*.jpg"):
            cv2.imwrite(str(source.with_suffix(".png")), cv2.imread(str(source)))
            source.unlink()
    frames = list(samples(case))
    assert len(frames) == 2
    assert frames[1][0].source_time_s == pytest.approx(0.04)
    assert np.all(frames[0][1]["item"])
    assert frames[0][0].depth_m_per_unit == 0.001


def test_demo_rejects_mask_from_wrong_camera_time(tmp_path):
    case = demo(tmp_path)
    file = tmp_path / "masks/000001.json.gz"
    with gzip.open(file, "rt") as f:
        row = json.load(f)
    row["source_time_s"] += 0.03
    with gzip.open(file, "wt") as f:
        json.dump(row, f)
    with pytest.raises(ValueError, match="mask time"):
        list(samples(case))


def test_chain_distinguishes_missing_image_from_broken_correspondence():
    chain = MotionChain()
    first = {"status": None, "step": None}
    np.testing.assert_equal(chain.update(first), np.eye(4))
    assert chain.update({"status": "no_mask", "step": None}) is None
    step = np.eye(4)
    step[0, 3] = 0.02
    np.testing.assert_equal(chain.update({"status": None, "step": step}), step)
    assert chain.update(first) is None  # a usable image with no measured link
    assert chain.update({"status": None, "step": step}) is None


def test_fresh_demo_experiment_crosses_real_reader_and_surface_fit(tmp_path, monkeypatch):
    from lerobot.policies.flash import surface_tracking

    # Stub only the neural network boundary; run the real rigid fit, surface
    # optimizer, source reader and persisted report on a stationary fixture.
    points = np.random.default_rng(7).uniform(-0.05, 0.05, (24, 3)) + [0, 0, 1]
    monkeypatch.setattr(surface_tracking, "ufm_pairs", lambda *_: (points, points.copy()))
    case = demo(tmp_path)
    report = run_case(case, tmp_path / "result", None, VisionPipelineConfig(), None, None)
    frames = [json.loads(line) for line in (tmp_path / "result/frames.jsonl").read_text().splitlines()]
    assert len(frames) == 2
    np.testing.assert_allclose(frames[1]["objects"]["item"]["ufm_chain"], np.eye(4), atol=1e-7)
    # Preserve the archived fitter, including its errors. A full-image plane
    # can move under its shape objective; the harness must not fix or hide that.
    standalone = surface_tracking.SurfaceTrackingSession(None)
    for frame, masks in samples(case):
        expected = standalone.update(
            frame.rgb,
            frame.depth.astype(np.float32) * frame.depth_m_per_unit,
            masks["item"],
            frame.intrinsics,
            frame.frame_id,
        )
    np.testing.assert_allclose(
        frames[1]["objects"]["item"]["surface_corrected"], expected["camera_from_reference"]
    )
    for name in ("ufm_chain", "surface_corrected"):
        assert report["roles"]["item"]["methods"][name]["available_after_initialization"] == 1
    assert report["accuracy_labels"] is False
    assert "pose_error" not in report["roles"]["item"]["methods"]["ufm_chain"]
    with pytest.raises(FileExistsError):
        run_case(case, tmp_path / "result", None, VisionPipelineConfig(), None, None)


def test_single_method_runs_same_tracker_without_comparison_fits(tmp_path, monkeypatch):
    from benchmarks.flash_runtime_benchmark import compare_runs
    from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
    from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession

    case = demo(tmp_path)
    options = {
        "surface_experiment": GrowingSurfaceConfig(),
        "group_experiment": GroupConfig(),
        "shared_matching_config": SharedMatchingConfig(backend="lk"),
    }
    run_case(case, tmp_path / "all", None, VisionPipelineConfig(), None, None, **options)

    def unexpected_fit(*_):
        raise AssertionError("runtime benchmark must not compute a comparison baseline")

    monkeypatch.setattr(SurfaceTrackingSession, "finish", unexpected_fit)
    result = run_case(
        case,
        tmp_path / "single",
        None,
        VisionPipelineConfig(),
        None,
        None,
        single_method="group_growing",
        record_surfaces=False,
        **options,
    )
    old = [json.loads(line) for line in (tmp_path / "all/frames.jsonl").read_text().splitlines()]
    new = [json.loads(line) for line in (tmp_path / "single/frames.jsonl").read_text().splitlines()]
    for a, b in zip(old, new, strict=True):
        a, b = a["objects"]["item"], b["objects"]["item"]
        for key in ("group_growing", "group_growing_status", "group_growing_details"):
            assert a[key] == b[key]
        assert "surface_corrected" not in b
    assert set(result["roles"]["item"]["methods"]) == {"group_growing"}
    assert result["processing"]["masks"] == "recorded"
    assert not (tmp_path / "single/surface_models").exists()
    assert compare_runs(tmp_path / "all", tmp_path / "single", "group_growing")["identical"]
    # Guard the comparator with changed real outputs: one wrong translation,
    # an unavailable estimate and a spurious surface-addition event.
    changed = new[-1]["objects"]["item"]
    changed["group_growing"] = np.eye(4).tolist()
    changed["group_growing"][0][3] = 0.2
    changed["group_growing_status"] = "lost"
    changed["group_growing_details"]["map_events"].append({"event": "added", "frame": 1})
    (tmp_path / "single/frames.jsonl").write_text("\n".join(json.dumps(r) for r in new) + "\n")
    comparison = compare_runs(tmp_path / "all", tmp_path / "single", "group_growing")
    assert not comparison["identical"]
    assert comparison["changed_frame_objects"] == {"pose": 1, "status": 1, "surface_events": 1}


def test_two_object_parallel_replay_preserves_serial_tracker_results(tmp_path, monkeypatch):
    from threading import Thread

    from benchmarks.flash_runtime_benchmark import compare_runs
    from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig

    case = demo(tmp_path)
    case["roles"] = ["item", "second"]
    for path in sorted((tmp_path / "masks").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            row = json.load(stream)
        row["masks_rle"]["second"] = row["masks_rle"]["item"]
        with gzip.open(path, "wt") as stream:
            json.dump(row, stream)
    options = {
        "surface_experiment": GrowingSurfaceConfig(),
        "group_experiment": GroupConfig(),
        "shared_matching_config": SharedMatchingConfig(backend="lk"),
        "single_method": "group_growing",
        "record_surfaces": False,
    }
    config = VisionPipelineConfig(parallel_surface_objects=True, surface_object_workers=2)
    run_case(case, tmp_path / "serial", None, config, None, None, parallel_objects=False, **options)
    starts = []
    original = Thread.start

    def start(thread):
        starts.append(thread.name)
        return original(thread)

    monkeypatch.setattr(Thread, "start", start)
    run_case(case, tmp_path / "parallel", None, config, None, None, **options)
    assert len([name for name in starts if name.startswith("ThreadPoolExecutor")]) == 2
    assert compare_runs(tmp_path / "serial", tmp_path / "parallel", "group_growing")["identical"]


@pytest.mark.parametrize("pipelined", [False, True])
def test_debugger_benchmark_scores_published_pose_not_rejected_candidate(tmp_path, monkeypatch, pipelined):
    from benchmarks.flash_runtime_benchmark import run_debugger
    from lerobot.policies.flash import surface_tracking

    points = np.random.default_rng(7).uniform(-0.05, 0.05, (24, 3)) + [0, 0, 1]
    monkeypatch.setattr(surface_tracking, "ufm_pairs", lambda *_: (points, points.copy()))
    config = VisionPipelineConfig(
        pipeline_surface_frames=pipelined,
        registration_min_inliers=999,  # A real rejection at the public backend boundary.
    )
    result = run_debugger(demo(tmp_path), tmp_path / "result", object(), config)
    rows = [json.loads(line) for line in (tmp_path / "result/frames.jsonl").read_text().splitlines()]
    assert [r["frame"] for r in rows] == [0, 1]
    assert rows[0]["objects"]["item"]["surface_corrected"] is not None
    assert rows[1]["objects"]["item"]["candidate"] is not None
    assert rows[1]["objects"]["item"]["surface_corrected"] is None
    assert result["roles"]["item"]["methods"]["surface_corrected"]["available_after_initialization"] == 0


def test_runtime_profiling_keeps_real_fits_and_counts_nested_work(tmp_path):
    from benchmarks.flash_runtime_benchmark import FitProfile, compare_runs
    from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig

    case = demo(tmp_path)
    options = {
        "surface_experiment": GrowingSurfaceConfig(),
        "group_experiment": GroupConfig(),
        "shared_matching_config": SharedMatchingConfig(backend="lk"),
        "single_method": "group_growing",
        "record_surfaces": False,
    }
    run_case(case, tmp_path / "plain", None, VisionPipelineConfig(), None, None, **options)
    with FitProfile() as profile:
        run_case(case, tmp_path / "profiled", None, VisionPipelineConfig(), None, None, **options)
    assert compare_runs(tmp_path / "plain", tmp_path / "profiled", "group_growing")["identical"]
    assert profile.stats["fitter.refine"]["calls"] > 0
    solver = profile.stats["fitter.refine"]
    assert 0 <= solver["exclusive_ms"] < solver["inclusive_ms"]


def test_runtime_cli_fixed_recipe_does_not_inherit_scene_settings(tmp_path, monkeypatch):
    import sys

    from benchmarks.flash_pipeline_settings import defaults
    from benchmarks.flash_runtime_benchmark import main

    source = tmp_path / "source"
    source.mkdir()
    (source / "case.json").write_text(json.dumps(demo(tmp_path)))
    recipe = defaults()
    recipe["shared_matching"]["backend"] = "lk"
    recipe["surface_experiment"]["spacing_m"] = 0.004
    settings = tmp_path / "recipe.json"
    settings.write_text(json.dumps(recipe))
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--source-run", str(source), "--settings", str(settings), "--output", str(output)],
    )
    # No historical provenance exists: the supplied recipe must drive the actual loop.
    main()
    result = json.loads((output / "summary.json").read_text())[0]
    assert result["frames"] == 2
    assert result["surface_experiment"]["spacing_m"] == 0.004
    assert result["shared_matching"]["config"]["backend"] == "lk"


@pytest.mark.parametrize("limit", [None, 1])
def test_runtime_mask_source_is_used_and_missing_role_is_empty(tmp_path, limit):
    from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig

    case = demo(tmp_path)

    class Masks:
        calls = []

        def masks(self, frame):
            self.calls.append(frame.frame_id)
            return {}  # Recorded nonempty masks must not leak through.

    source = Masks()
    result = run_case(
        case,
        tmp_path / "result",
        None,
        VisionPipelineConfig(),
        None,
        None,
        single_method="conditioned_growing",
        surface_experiment=GrowingSurfaceConfig(),
        shared_matching_config=SharedMatchingConfig(backend="lk"),
        mask_source=source,
        max_frames=limit,
    )
    assert source.calls == ([0, 1] if limit is None else [0])
    assert result["roles"]["item"]["reference_frame"] is None
    assert result["processing"]["masks"] == "computed"
    rows = [json.loads(line) for line in (tmp_path / "result/frames.jsonl").read_text().splitlines()]
    assert all(r["mask_ms"] > 0 and r["processing_ms"] >= r["mask_ms"] for r in rows)


def test_score_keeps_missing_poses_and_paired_accuracy_separate(tmp_path):
    labels = tmp_path / "annotated_poses"
    labels.mkdir()
    rows = []
    for i in range(3):
        truth = np.eye(4)
        truth[0, 3] = i * 0.05
        np.savetxt(labels / f"{i:06d}.txt", truth)
        wrong = truth.copy()
        wrong[0, 3] += 0.02 if i else 0
        rows.append(
            {
                "objects": {
                    "object": {
                        "ufm_chain": truth.tolist() if i < 2 else None,
                        "surface_corrected": wrong.tolist(),
                        "prepare_ms": 2.0,
                        "finish_ms": 1.0,
                    }
                }
            }
        )
    case = {
        "id": "known-motion",
        "kind": "ycb_annotated_pose",
        "source_root": str(tmp_path),
        "start_frame": 0,
        "roles": ["object"],
    }
    result = score_case(
        case, rows, {"object": np.array([[0, 0, 1.0], [0.1, 0, 1.0], [0, 0.1, 1.0]])}, {"object": 0}
    )["roles"]["object"]
    assert result["paired_frames"] == 1
    assert result["paired_point_error_mm"]["ufm_chain"]["median"] == pytest.approx(0)
    assert result["paired_point_error_mm"]["surface_corrected"]["median"] == pytest.approx(20)
    assert result["methods"]["ufm_chain"]["total_after_initialization"] == 2
    assert result["methods"]["ufm_chain"]["available_after_initialization"] == 1


@pytest.mark.parametrize("groups", [False, True])
def test_growing_experiment_keeps_baselines_and_archives_original_patches(tmp_path, monkeypatch, groups):
    from lerobot.policies.flash import surface_tracking

    points = np.random.default_rng(7).uniform(-0.05, 0.05, (24, 3)) + [0, 0, 1]
    monkeypatch.setattr(surface_tracking, "ufm_pairs", lambda *_: (points, points.copy()))
    case = demo(tmp_path)
    config = VisionPipelineConfig()
    base = run_case(case, tmp_path / "base", None, config, None, None)
    result = run_case(
        case,
        tmp_path / "growth",
        None,
        config,
        None,
        None,
        surface_experiment=GrowingSurfaceConfig(),
        group_experiment=GroupConfig() if groups else None,
    )
    for method in ("ufm_chain", "surface_corrected"):
        assert base["roles"]["item"]["methods"][method] == result["roles"]["item"]["methods"][method]
    rows = [json.loads(line) for line in (tmp_path / "growth/frames.jsonl").read_text().splitlines()]
    methods = ["conditioned_fixed", "conditioned_growing"] + (
        ["group_fixed", "group_growing"] if groups else []
    )
    for method in methods:
        assert rows[0]["objects"]["item"][method + "_status"] == "initialized"
        archive = tmp_path / f"growth/surface_patches/{method}/item/000000.npz"
        with np.load(archive) as patch:
            assert len(patch["measured"]) > 0
            np.testing.assert_array_equal(patch["camera_from_reference"], np.eye(4))
        assert (tmp_path / f"growth/surface_models/{method}/item/000001.npz").exists()
    selected = methods[-1]
    limited = run_case(
        case,
        tmp_path / "limited",
        None,
        config,
        None,
        None,
        surface_experiment=GrowingSurfaceConfig(),
        group_experiment=GroupConfig() if groups else None,
        surface_methods=(selected,),
    )
    assert set(limited["roles"]["item"]["methods"]) == {"ufm_chain", "surface_corrected", selected}
    assert limited["roles"]["item"]["methods"][selected] == result["roles"]["item"]["methods"][selected]
    limited_rows = [json.loads(line) for line in (tmp_path / "limited/frames.jsonl").read_text().splitlines()]
    for full_row, limited_row in zip(rows, limited_rows, strict=True):
        np.testing.assert_array_equal(
            full_row["objects"]["item"][selected], limited_row["objects"]["item"][selected]
        )


def test_retry_connects_to_last_good_image_not_the_unregistered_one(monkeypatch):
    from benchmarks import flash_tracker_experiment as experiment

    old, missed, current = {}, {}, {}

    def prepared(frame, index, step=None):
        return {"status": None, "current": frame, "frame_id": index, "step": step, "intrinsics": np.eye(3)}

    tracker = RetainedMotionReference(None, SurfaceMotionConfig())
    np.testing.assert_equal(tracker.update(prepared(old, 0), None)[0], np.eye(4))
    assert tracker.update(prepared(missed, 1), old)[0] is None
    assert tracker.reference is old
    monkeypatch.setattr(experiment, "crop_pair", lambda *_: (None, None, None))
    points = np.random.default_rng(12).normal(size=(20, 3))

    def pairs(_model, _a, _b, _crop, source, target, *_):
        assert source is old and target is current
        return points, points + [0.03, 0, 0]

    monkeypatch.setattr(experiment, "ufm_pairs", pairs)
    pose, diagnostics = tracker.update(prepared(current, 2), missed)
    np.testing.assert_allclose(pose[:3, 3], [0.03, 0, 0], atol=1e-8)
    assert diagnostics["reference_frame"] == 0
    assert tracker.reference is current


def test_group_benchmark_publishes_pose_when_mask_and_all_point_support_are_empty(tmp_path):
    case = demo(tmp_path)
    file = tmp_path / "masks/000001.json.gz"
    with gzip.open(file, "rt") as stream:
        row = json.load(stream)
    row["masks_rle"] = {"item": []}
    row["base_from_camera"][0][3] = 0.02
    with gzip.open(file, "wt") as stream:
        json.dump(row, stream)
    # Flat RGB has no LK features. The neural matcher has no valid second mask.
    run_case(
        case,
        tmp_path / "run",
        None,
        VisionPipelineConfig(),
        None,
        None,
        surface_experiment=GrowingSurfaceConfig(),
        group_experiment=GroupConfig(),
    )
    rows = [json.loads(line) for line in (tmp_path / "run/frames.jsonl").read_text().splitlines()]
    for method in ("group_fixed", "group_growing"):
        obj = rows[1]["objects"]["item"]
        assert obj[method + "_status"] == "predicted"
        expected = np.eye(4)
        expected[0, 3] = -0.02
        np.testing.assert_allclose(obj[method], expected)
        details = obj[method + "_details"]
        assert details["prediction_source"] == "last_world_pose"
        assert details["map_version_after"] == details["map_version_used"] == 1
        assert not details["group"]["source"]
        assert not details["map_events"]


def test_saved_2d_matches_preserve_raw_predictions_and_rejection_stages(monkeypatch):
    from types import SimpleNamespace

    import torch

    from benchmarks.flash_tracker_experiment import match_record
    from lerobot.policies.flash.surface_correspondence import ufm_pairs

    config = SurfaceMotionConfig(input_size_px=8, grid_step_px=1, ransac_distance_m=0.005)
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 1  # Known one-pixel rightward motion.
    flow[0, 1, 1, 4] = 1  # A wrong match, kept in the raw display.
    visibility = torch.ones(1, 8, 8)
    visibility[0, 1, 1] = 0
    output = SimpleNamespace(
        flow=SimpleNamespace(flow_output=flow), covisibility=SimpleNamespace(mask=visibility)
    )
    model = SimpleNamespace(predict_correspondences_batched=lambda *_: output)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    source = {
        "mask": np.ones((8, 8), bool),
        "usable": np.ones((8, 8), bool),
        "depth": np.ones((8, 8)),
        "frame_id": 2,
    }
    target = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in source.items()}
    target["frame_id"] = 5
    target["mask"][1, 3] = False
    target["usable"][1, 3:5] = False
    k = np.array([[100.0, 0, 4], [0, 100.0, 4], [0, 0, 1.0]])
    rgb = np.zeros((8, 8, 3), np.uint8)
    plain = ufm_pairs(model, rgb, rgb, (0, 0, 8), source, target, k, config)
    evidence = {}
    recorded = ufm_pairs(model, rgb, rgb, (0, 0, 8), source, target, k, config, evidence=evidence)
    for a, b in zip(plain, recorded, strict=True):
        np.testing.assert_array_equal(a, b)  # Recording cannot change the estimator's inputs.
    step = np.eye(4)
    step[0, 3] = 0.01
    saved = match_record(evidence, source, target, k, step, config)
    assert (saved["source_frame"], saved["target_frame"]) == (2, 5)
    assert saved["stage"][:5] == [0, 1, 2, 3, 4]
    # Raw predictions are image measurements, not projections of the fitted rigid motion.
    assert saved["source_uv"][3] == [4, 1]
    assert saved["target_uv"][3] == [5, 2]
    assert len(saved["source_uv"]) == 36
    # Record the full-field path samples once during inference. Re-exporting
    # must not silently replace them with interpolation of the sparse fit grid.
    from benchmarks.flash_flow_paths import FlowPaths

    traced = match_record(evidence, source, target, k, step, config, paths=FlowPaths())
    assert traced["path_sampling"] == "full_ufm_output"
    assert traced["paths"]
    assert "flow" not in traced
    json.dumps(traced, allow_nan=False)
