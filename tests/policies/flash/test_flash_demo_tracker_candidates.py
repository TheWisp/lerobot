"""A reviewed tracker trace changes only a new prepared demo version."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from lerobot.policies.flash.appearance import AppearanceModel
from lerobot.policies.flash.demo_tracker_candidates import (
    apply_candidate,
    import_relative_trace,
    import_relative_traces,
    load_candidate,
    load_candidate_point_tracks,
    load_candidate_set,
    process_demo,
    runner_availability,
)
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def _demo(tmp_path):
    root = tmp_path / "demo"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_episodes": 1}')
    (root / "recording").mkdir()
    np.savetxt(root / "recording/times.txt", [0, 1, 2])
    np.savez(root / "showservo_demo.npz", t=np.array([0.0, 1.0, 2.0]), t0=np.array(0.0))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"camera": {"T_base_cam": _pose(1).tolist()}}))
    versions = {role: f"{role}-v1" for role in ("object", "target")}
    references = tuple(
        ObjectReference(
            role,
            versions[role],
            (ReferenceSnippet("demo", 0, 0, role, np.eye(4), f"{role}.png", "cal-v1"),),
        )
        for role in versions
    )
    models = tuple(
        AppearanceModel(
            ObjectPointModel(role, versions[role], np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]])),
            np.ones((3, 2)),
        )
        for role in versions
    )
    grasp = Interval(
        "grasp",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "object",
        (
            SpatialSample(0, _pose(0), _pose(1), source_frame_index=0),
            SpatialSample(1, _pose(0), _pose(1), source_frame_index=1),
        ),
        reference_version=versions["object"],
    )
    carry = Interval(
        "carry",
        MotionMode.DEMONSTRATED,
        Relation.HELD_TO_TARGET,
        "target",
        (
            SpatialSample(1, _pose(0), _pose(2), _pose(1), source_frame_index=1),
            SpatialSample(2, _pose(0), _pose(2), _pose(1), source_frame_index=2),
        ),
        reference_version=versions["target"],
        held_role="object",
        held_reference_version=versions["object"],
    )
    prepared = PreparedFlashSkill(
        FlashSkill("demo", 0, (grasp, carry)),
        references,
        models,
        {"object": "object", "target": "target"},
        "descriptor-v1",
    )
    bundle = prepared.save(root / "flash/v1/prepared/original")
    (bundle.parent / "annotation.json").write_text(
        json.dumps({"calibration_path": str(calibration), "registration_threshold_m": 0.005})
    )
    (bundle.parent / "keypoints.json").write_text('{"keypoints": []}')
    return root, bundle


def test_tracker_preview_and_apply_keep_original_and_missing_samples(tmp_path) -> None:
    root, bundle = _demo(tmp_path)
    source = tmp_path / "tracker.npz"
    np.savez(
        source,
        frames=np.array([0, 1, 2]),
        poses=np.stack([_pose(0.005), _pose(0.105), _pose(0.2)]),
        lost=np.array([False, False, True]),
    )
    result = import_relative_trace(
        root, bundle, tracker="point2pose", role="object", source_path=source, candidate_id="point2pose-1"
    )
    assert result["valid_frames"] == 2
    manifest, preview = load_candidate(root, bundle, "point2pose-1")
    assert manifest["status"] == "preview"
    assert np.isclose(preview[1][0, 3], 1.1)
    assert 2 not in preview

    revised_bundle, report = apply_candidate(root, bundle, "point2pose-1", output_name="applied")
    original = PreparedFlashSkill.load(bundle)
    revised = PreparedFlashSkill.load(revised_bundle)
    assert np.isclose(original.skill.intervals[0].samples[1].demo_reference[0, 3], 1.0)
    assert np.isclose(revised.skill.intervals[0].samples[1].demo_reference[0, 3], 1.1)
    assert np.isclose(revised.skill.intervals[1].samples[0].demo_held[0, 3], 1.1)
    assert np.isclose(revised.skill.intervals[1].samples[1].demo_held[0, 3], 1.0)
    assert report["changed_held_samples"] == 1
    assert report["preserved_samples"] == 1
    assert report["review_status"] == "accepted"


def test_tracker_import_requires_reference_pose_and_unchanged_bundle(tmp_path) -> None:
    root, bundle = _demo(tmp_path)
    source = tmp_path / "tracker.npz"
    np.savez(
        source, frames=np.array([1, 2]), poses=np.stack([_pose(0), _pose(0.1)]), lost=np.array([False, False])
    )
    with pytest.raises(ValueError, match="appearance frame"):
        import_relative_trace(
            root,
            bundle,
            tracker="point2pose",
            role="object",
            source_path=source,
            candidate_id="missing-reference",
        )
    assert not (root / "flash/v1/tracker-candidates/missing-reference").exists()
    np.savez(
        source, frames=np.array([0, 1]), poses=np.stack([_pose(0), _pose(0.1)]), lost=np.array([False, False])
    )
    import_relative_trace(
        root, bundle, tracker="point2pose", role="object", source_path=source, candidate_id="valid"
    )
    bundle.write_text(bundle.read_text() + "\n")
    with pytest.raises(ValueError, match="changed prepared demo"):
        load_candidate(root, bundle, "valid")


def test_configured_runner_processes_demo_and_imports_review_candidate(tmp_path, monkeypatch) -> None:
    root, bundle = _demo(tmp_path)
    (root / "object.png").write_bytes(b"mask")
    script = tmp_path / "runner.py"
    script.write_text(
        "import sys, numpy as np\n"
        "first, last, output = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]\n"
        "frames = np.arange(first, last + 1)\n"
        "poses = np.repeat(np.eye(4)[None], len(frames), axis=0)\n"
        "poses[:, 0, 3] = frames * 0.01\n"
        "np.savez(output, frames=frames, poses=poses, lost=np.zeros(len(frames), dtype=bool))\n"
    )
    config = tmp_path / "trackers.json"
    config.write_text(
        json.dumps(
            {"pany": {"command": [sys.executable, str(script), "{first_frame}", "{last_frame}", "{output}"]}}
        )
    )
    monkeypatch.setenv("LEROBOT_FLASH_TRACKERS_CONFIG", str(config))
    assert next(item for item in runner_availability() if item["id"] == "pany")["runnable"]
    result = process_demo(root, bundle, tracker="pany", role="object", candidate_id="pany-run")
    assert result["valid_frames"] == 3
    _, poses = load_candidate(root, bundle, "pany-run")
    assert np.isclose(poses[2][0, 3], 1.02)
    assert not (root / "flash/v1/prepared/pany-run").exists()


def test_scene_candidate_keeps_independent_motion_and_applies_one_version(tmp_path):
    root, bundle = _demo(tmp_path)
    sources = {}
    for role, offsets, lost in (
        ("object", [0, 0.1, 0.2], [False, False, True]),
        ("target", [0, -0.3, -0.4], [False, False, False]),
    ):
        path = tmp_path / f"{role}.npz"
        np.savez(path, frames=np.arange(3), poses=np.stack([_pose(x) for x in offsets]), lost=np.array(lost))
        sources[role] = path
    result = import_relative_traces(
        root, bundle, tracker="trackeverything", sources=sources, candidate_id="scene"
    )
    assert result["roles"] == ["object", "target"]
    assert result["coverage_by_role"]["object"]["valid_frames"] == 2
    _, traces = load_candidate_set(root, bundle, "scene")
    assert traces["object"][1][0, 3] == pytest.approx(1.1)
    assert traces["target"][1][0, 3] == pytest.approx(0.7)
    revised_bundle, report = apply_candidate(root, bundle, "scene", output_name="scene-applied")
    revised = PreparedFlashSkill.load(revised_bundle)
    carry = revised.skill.intervals[1]
    assert carry.samples[0].demo_held[0, 3] == pytest.approx(1.1)
    assert carry.samples[0].demo_reference[0, 3] == pytest.approx(0.7)
    assert carry.samples[1].demo_held[0, 3] == pytest.approx(1)  # lost: preserve old estimate
    assert carry.samples[1].demo_reference[0, 3] == pytest.approx(0.6)
    assert report["preserved_samples"] == 1
    assert len(list((root / "flash/v1/prepared").iterdir())) == 2


def test_scene_runner_defaults_to_every_object(tmp_path, monkeypatch):
    root, bundle = _demo(tmp_path)
    for role in ("object", "target"):
        (root / f"{role}.png").write_bytes(b"mask")
    script = tmp_path / "runner.py"
    script.write_text(
        "import sys, numpy as np\n"
        "poses = np.repeat(np.eye(4)[None], 3, axis=0)\n"
        "poses[:, 0, 3] = np.arange(3) * (.01 if sys.argv[1] == 'object' else -.02)\n"
        "np.savez(sys.argv[2], frames=np.arange(3), poses=poses, lost=np.zeros(3, dtype=bool))\n"
    )
    config = tmp_path / "trackers.json"
    config.write_text(json.dumps({"pany": {"command": [sys.executable, str(script), "{role}", "{output}"]}}))
    monkeypatch.setenv("LEROBOT_FLASH_TRACKERS_CONFIG", str(config))
    process_demo(root, bundle, tracker="pany", candidate_id="all")
    _, traces = load_candidate_set(root, bundle, "all")
    assert set(traces) == {"object", "target"}
    assert traces["object"][2][0, 3] == pytest.approx(1.02)
    assert traces["target"][2][0, 3] == pytest.approx(0.96)


def test_raw_point_replay_preserves_model_tracks_separately_from_pose_fit(tmp_path):
    root, bundle = _demo(tmp_path)
    np.savetxt(root / "recording/cam_K.txt", np.eye(3))
    source = tmp_path / "relative.npz"
    poses = np.stack([_pose(0), _pose(0.1), _pose(0.2)])
    np.savez(source, frames=np.arange(3), poses=poses, lost=np.zeros(3, dtype=bool))
    import_relative_trace(
        root, bundle, tracker="trackeverything", role="object", source_path=source, candidate_id="raw-points"
    )
    raw = root / "flash/v1/tracker-runs/raw-points/scene.npz"
    raw.parent.mkdir(parents=True)
    reference = np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]], dtype=float)
    tracked = np.stack([reference, reference + [0.1, 0, 0], reference + [0.2, 0, 0]])
    tracked[1, 2, 0] = 0.2  # raw point did not follow the rigid fit
    visibility = np.ones((3, 3))
    visibility[1, 1] = -1
    np.savez(
        raw,
        frames=np.arange(3),
        roles=np.array(["object"]),
        query_points_2d=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
        query_coords_m=tracked,
        query_visibility_logits=visibility,
        poses=poses[None],
        lost=np.zeros((1, 3), dtype=bool),
        metadata=np.asarray(json.dumps({"query_counts": {"object": 3}})),
    )
    manifest_path = root / "flash/v1/tracker-candidates/raw-points/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_query_trace"] = str(raw)
    manifest_path.write_text(json.dumps(manifest))
    result = load_candidate_point_tracks(root, bundle, "raw-points")
    assert result["fit_threshold_mm"] == 5
    assert result["frames"] == [0, 1, 2]
    points = result["roles"][0]["positions"]
    assert points[1][0] == [0.1, 0.0, 0.0]
    assert points[1][1] is None
    assert points[1][2] == [0.2, 0.1, 100.0]
    manifest["raw_query_trace"] = str(source)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside this dataset"):
        load_candidate_point_tracks(root, bundle, "raw-points")


def test_grown_point_replay_shows_only_visible_mask_and_depth_supported_points(tmp_path):
    import cv2

    root, bundle = _demo(tmp_path)
    np.savetxt(root / "recording/cam_K.txt", np.eye(3))
    (root / "recording/depth").mkdir()
    for frame in range(3):
        cv2.imwrite(str(root / "recording/depth" / f"{frame:06d}.png"), np.full((4, 4), 1000, np.uint16))
    source = tmp_path / "relative.npz"
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    np.savez(source, frames=np.arange(3), poses=poses, lost=np.zeros(3, bool))
    import_relative_trace(
        root,
        bundle,
        tracker="trackeverything",
        role="object",
        source_path=source,
        candidate_id="grown-points",
    )
    raw = root / "flash/v1/tracker-runs/grown-points/scene.npz"
    raw.parent.mkdir(parents=True)
    masks = raw.parent / "segmentation/masks/0"
    masks.mkdir(parents=True)
    for frame in range(3):
        cv2.imwrite(str(masks / f"{frame:06d}.png"), np.full((4, 4), 255, np.uint8))
    queries = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    xyz = np.repeat(np.column_stack((queries[:, 1:], np.ones(4)))[None].astype(float), 3, axis=0)
    xyz[2, 3, 2] = 1.1  # model-visible but contradicted by measured depth
    np.savez(
        raw,
        frames=np.arange(3),
        roles=np.array(["object"]),
        query_roles=np.array(["object"] * 4),
        query_points_2d=queries,
        query_coords_m=xyz,
        query_visibility_logits=np.ones((3, 4)),
        poses=poses[None],
        lost=np.zeros((1, 3), bool),
        metadata=np.asarray(json.dumps({"query_counts": {"object": 4}, "fit_threshold_m": 0.005})),
    )
    manifest_path = root / "flash/v1/tracker-candidates/grown-points/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_query_trace"] = str(raw)
    manifest["point_growth"] = {"surface_tolerance_m": 0.01}
    manifest_path.write_text(json.dumps(manifest))
    points = load_candidate_point_tracks(root, bundle, "grown-points")
    assert points["surface_filtered"]
    assert points["roles"][0]["birth_frames"] == [0, 0, 0, 1]
    assert points["roles"][0]["positions"][0][3] is None
    assert points["roles"][0]["positions"][1][3] == [1.0, 1.0, 0.0]
    assert points["roles"][0]["positions"][2][3] is None


def test_trackeverything_shares_inference_but_fits_each_object_at_its_reference(tmp_path, monkeypatch):
    """Exercise the adapter boundary with independent known motions and seed times."""
    import importlib.util
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    import cv2
    import torch

    spec = importlib.util.spec_from_file_location(
        "flash_te_adapter", Path(__file__).resolve().parents[3] / "benchmarks/flash_trackeverything_pose.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    recording = tmp_path / "recording"
    (recording / "rgb").mkdir(parents=True)
    (recording / "depth").mkdir()
    for i in range(2):
        cv2.imwrite(str(recording / "rgb" / f"{i:06d}.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
        cv2.imwrite(str(recording / "depth" / f"{i:06d}.png"), np.full((4, 4), 1000, dtype=np.uint16))
    np.savetxt(recording / "cam_K.txt", np.eye(3))
    mask = tmp_path / "mask.png"
    cv2.imwrite(str(mask), np.full((4, 4), 255, dtype=np.uint8))
    repository = tmp_path / "trackeverything"
    repository.mkdir()
    (repository / "demo_hf.py").touch()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    motion_a = _pose(0.1)
    motion_a[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    motion_b = _pose(-0.02)
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def infer(self, **kwargs):
            queries = kwargs["query_points_2d"]
            calls.append((queries.copy(), kwargs["mode"]))
            base = np.column_stack([queries[:, 1:], np.ones(len(queries))])
            moved = base.copy()
            moved[:4] = base[:4] @ motion_a[:3, :3].T + motion_a[:3, 3]
            moved[4:] += motion_b[:3, 3]
            points = np.stack([base, moved])
            return SimpleNamespace(
                query_coords=torch.tensor(points[None] / 2), query_visibs=torch.ones((1, 2, 8)), norm_factor=2
            )

    fake = ModuleType("trackeverything.demo_hf")
    fake.TrackEverything = FakeModel
    monkeypatch.setitem(sys.modules, "trackeverything.demo_hf", fake)
    output = tmp_path / "scene.npz"
    adapter.run(
        recording,
        None,
        repository=repository,
        checkpoint=checkpoint,
        first_frame=0,
        last_frame=1,
        output=output,
        query_count=4,
        fit_threshold_m=1e-5,
        min_inliers=3,
        fit_trials=32,
        reference_specs=[
            {"role": "a", "frame_index": 0, "mask_path": str(mask)},
            {"role": "b", "frame_index": 1, "mask_path": str(mask)},
        ],
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0][:, 0], [0, 0, 0, 0, 1, 1, 1, 1])
    assert calls[0][1] == "dense"
    with np.load(output) as trace:
        assert trace["roles"].tolist() == ["a", "b"]
        assert json.loads(str(trace["metadata"]))["fit_threshold_m"] == 1e-5
        assert not trace["lost"].any()
        assert not trace["temporal_conflicts"].any()
        assert trace["query_coords_m"].shape == (2, 8, 3)
        np.testing.assert_allclose(trace["poses"][0, 1], motion_a, atol=1e-6)
        np.testing.assert_allclose(trace["poses"][1, 0], np.linalg.inv(motion_b), atol=1e-6)
        np.testing.assert_allclose(trace["poses"][1, 1], np.eye(4), atol=1e-6)


def test_temporal_conflict_flags_pose_motion_unsupported_by_point_tracks():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "flash_te_adapter", Path(__file__).resolve().parents[3] / "benchmarks/flash_trackeverything_pose.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    reference = np.array([[0.0, 0, 1], [0.1, 0, 1], [0, 0.1, 1], [0.1, 0.1, 1]])
    points = np.stack((reference, reference.copy()))
    visible = np.ones((2, 4))
    poses = np.stack((np.eye(4), _pose(0.05)))
    conflicts = adapter._temporal_motion_conflicts(
        points,
        visible,
        poses,
        np.zeros(2, dtype=bool),
        reference_index=0,
        fit_threshold_m=0.005,
        min_inliers=3,
    )
    assert conflicts.tolist() == [False, True]
    poses[1] = np.eye(4)
    assert not adapter._temporal_motion_conflicts(
        points,
        visible,
        poses,
        np.zeros(2, dtype=bool),
        reference_index=0,
        fit_threshold_m=0.005,
        min_inliers=3,
    ).any()
