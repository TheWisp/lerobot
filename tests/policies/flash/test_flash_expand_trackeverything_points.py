"""Exercise multi-view point seeding across the tracker output boundary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import torch


def test_later_queries_only_count_when_visible_on_the_measured_object(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmarks"))
    from flash_expand_trackeverything_points import run

    recording = tmp_path / "recording"
    masks = tmp_path / "masks"
    (recording / "rgb").mkdir(parents=True)
    (recording / "depth").mkdir()
    masks.mkdir()
    for frame in range(3):
        cv2.imwrite(str(recording / "rgb" / f"{frame:06d}.jpg"), np.zeros((4, 4, 3), np.uint8))
        cv2.imwrite(str(recording / "depth" / f"{frame:06d}.png"), np.full((4, 4), 1000, np.uint16))
        mask = np.full((4, 4), 255, np.uint8)
        if frame == 2:
            mask[2:] = 0
        cv2.imwrite(str(masks / f"{frame:06d}.png"), mask)
    np.savetxt(recording / "cam_K.txt", np.eye(3))
    old_queries = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    old_points = np.repeat(np.column_stack((old_queries[:, 1:], np.ones(3)))[None], 3, axis=0)
    baseline = tmp_path / "baseline.npz"
    np.savez(
        baseline,
        frames=np.arange(3),
        roles=np.array(["object"]),
        query_points_2d=old_queries,
        query_coords_m=old_points,
        query_visibility_logits=np.array([[1, 1, 1], [-1, -1, -1], [-1, -1, -1]]),
        metadata=np.asarray(json.dumps({"query_counts": {"object": 3}})),
    )
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def infer(self, **kwargs):
            queries = kwargs["query_points_2d"]
            calls.append(queries.copy())
            points = np.repeat(np.column_stack((queries[:, 1:], np.ones(len(queries))))[None], 3, axis=0)
            visibility = np.ones(points.shape[:2])
            points[1, len(old_queries), 2] = 2  # visible, but disagrees with measured depth
            visibility[1, len(old_queries) + 1] = -1  # on the surface, but model-hidden
            return SimpleNamespace(
                query_coords=torch.tensor(points[None]),
                query_visibs=torch.tensor(visibility[None]),
                norm_factor=1,
            )

    fake = ModuleType("trackeverything.demo_hf")
    fake.TrackEverything = FakeModel
    monkeypatch.setitem(sys.modules, "trackeverything.demo_hf", fake)
    output = tmp_path / "experiment"
    report = run(
        argparse.Namespace(
            recording=recording,
            baseline=baseline,
            masks=masks,
            role="object",
            repository=tmp_path / "trackeverything",
            checkpoint=tmp_path / "checkpoint.pth",
            output=output,
            query_count=4,
            depth_tolerance_m=0.01,
        )
    )
    assert report["birth_frame"] == 1
    assert set(calls[0][:, 0]) == {0, 1}
    with (output / "coverage.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert int(rows[0]["new_visible"]) == 0  # predictions before birth do not count
    assert int(rows[1]["new_visible"]) == report["new_queries"] - 1
    assert int(rows[1]["new_mask_depth_supported"]) == report["new_queries"] - 2
    assert (output / "seed-mask.png").is_file()


def test_bounded_growth_refits_pose_from_new_ids_when_original_ids_disappear(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmarks"))
    from flash_trackeverything_pose import run

    recording = tmp_path / "recording"
    masks = tmp_path / "masks" / "0"
    for directory in (recording / "rgb", recording / "depth", masks):
        directory.mkdir(parents=True)
    np.savetxt(recording / "cam_K.txt", np.eye(3))
    for frame in range(3):
        cv2.imwrite(str(recording / "rgb" / f"{frame:06d}.jpg"), np.zeros((16, 16, 3), np.uint8))
        cv2.imwrite(str(recording / "depth" / f"{frame:06d}.png"), np.full((16, 16), 1000, np.uint16))
        mask = np.zeros((16, 16), np.uint8)
        mask[:4] = 255 if frame == 0 else 0
        if frame:
            mask[:] = 255
        cv2.imwrite(str(masks / f"{frame:06d}.png"), mask)
    reference = masks / "000000.png"
    repository = tmp_path / "trackeverything"
    repository.mkdir()
    (repository / "demo_hf.py").touch()
    checkpoint = tmp_path / "weights"
    checkpoint.touch()
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def infer(self, **kwargs):
            queries = kwargs["query_points_2d"]
            calls.append(queries.copy())
            xyz = np.repeat(np.column_stack((queries[:, 1:], np.ones(len(queries))))[None], 3, axis=0)
            xyz[2, :, 0] += 0.002
            visibility = np.ones((1, 3, len(queries)))
            if len(queries) > 4:
                visibility[0, 2, :4] = -1
            return SimpleNamespace(
                query_coords=torch.tensor(xyz[None]),
                query_visibs=torch.tensor(visibility),
                norm_factor=1,
            )

    fake = ModuleType("trackeverything.demo_hf")
    fake.TrackEverything = FakeModel
    monkeypatch.setitem(sys.modules, "trackeverything.demo_hf", fake)
    output = tmp_path / "scene.npz"
    run(
        recording,
        None,
        repository=repository,
        checkpoint=checkpoint,
        first_frame=0,
        last_frame=2,
        output=output,
        query_count=4,
        fit_threshold_m=0.001,
        min_inliers=3,
        fit_trials=16,
        reference_specs=[{"role": "object", "frame_index": 0, "mask_path": str(reference)}],
        growth_masks=masks.parent,
        max_points_per_role=7,
        points_per_view=3,
        support_tolerance_m=0.01,
    )
    assert len(calls) >= 2
    with np.load(output) as trace:
        original = int(trace["original_query_count"])
        assert original == 4
        assert 4 < len(trace["query_points_2d"]) <= 7
        assert set(trace["query_points_2d"][original:, 0]) <= {1, 2}
        assert trace["query_roles"].tolist() == ["object"] * len(trace["query_points_2d"])
        assert not trace["lost"].any()
        assert trace["visible_surface_support"].shape == trace["query_visibility_logits"].shape
        assert not trace["visible_surface_support"][0, original:].any()
        assert not trace["visible_surface_support"][2, :original].any()
        assert trace["visible_surface_support"][2, original:].sum() >= 3
        assert trace["new_points"][0].sum() >= 3
        np.testing.assert_allclose(trace["poses"][0, 2, 0, 3], 0.002, atol=1e-6)
        assert json.loads(str(trace["metadata"]))["fit_mode"] == "causal_growing_rgbd"
        assert (
            sum(event["count"] for event in json.loads(str(trace["metadata"]))["growth_events"])
            == len(trace["query_points_2d"]) - original
        )
