"""Compare experimental accelerators to the unchanged production consumer."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation

SOURCE = Path(__file__).resolve().parents[3] / "docs/proofs/flash-native-acceleration/candidates.py"
spec = importlib.util.spec_from_file_location("native_candidates", SOURCE)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


@pytest.fixture(scope="module")
def kernels():
    pytest.importorskip("numba")
    return candidate.compiled_kernels()


def recorded_fit(name):
    data = np.load(Path(__file__).with_name("fixtures") / f"{name}.npz")
    is_ring = "source_surface" in data
    points = data["source_surface" if is_ring else "model_points"]
    depth = data["target_depth_mm" if is_ring else "depth"].astype(np.float32) * 0.001
    mask = data["target_mask" if is_ring else "mask"].astype(bool)
    k = data["intrinsics"]
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    current = {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": obs["points"]}
    initial = np.eye(4) if is_ring else data["candidate"]
    return WholeSurfaceFit(points, current, k, ShapeConfig()), initial


def test_reused_search_matches_exact_queries_and_preserves_batch_order():
    rng = np.random.default_rng(10)
    # Duplicate points, ties, empty/one query and noncontiguous input.
    points = rng.normal(size=(251, 3))
    tree = cKDTree(np.concatenate((points, points[:30])))
    with candidate.SearchPool(4) as pool:
        for queries in (points[:0], points[:1], points[::2], rng.normal(size=(163, 3))):
            expected = tree.query(queries, workers=4)
            actual = pool.query(tree, queries, workers=4)
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)
        # Two simultaneous object fits can use the same pool without reordering.
        with ThreadPoolExecutor(max_workers=2) as callers:
            outputs = list(callers.map(lambda q: pool.query(tree, q, workers=4), (points, points[::-1])))
        for q, output in zip((points, points[::-1]), outputs, strict=True):
            for actual, expected in zip(output, tree.query(q), strict=True):
                np.testing.assert_array_equal(actual, expected)
    with pytest.raises(RuntimeError, match="closed"):
        pool.query(tree, points)


@pytest.mark.parametrize("name", ["ring_visible_motion_20261016", "dowel_held_frame_315"])
def test_pooled_complete_fit_preserves_every_decision(name):
    baseline, initial = recorded_fit(name)
    expected, old = baseline.correct(initial)
    with candidate.SearchPool(4) as pool:
        factory = candidate.fitter_type(search_pool=pool)
        fit = factory(baseline.reference, baseline.observation, baseline.k, baseline.config)
        actual, new = fit.correct(initial)
        np.testing.assert_array_equal(actual, expected)
        assert new == old


@pytest.mark.parametrize("name", ["ring_visible_motion_20261016", "dowel_held_frame_315"])
@pytest.mark.parametrize("depth_case", ["original", "missing", "foreground", "background", "mixed"])
def test_compiled_objective_preserves_depth_mask_and_outline_checks(kernels, name, depth_case):
    baseline, initial = recorded_fit(name)
    depth = baseline.observation["depth"].copy()
    if depth_case == "missing":
        depth[:] = 0
    elif depth_case == "foreground":
        depth *= 0.8
    elif depth_case == "background":
        depth *= 1.2
    elif depth_case == "mixed":
        depth[::3, ::4] = 0
        depth[1::3, 2::4] *= 0.7
    baseline.observation = {**baseline.observation, "depth": depth}
    factory = candidate.fitter_type(kernels=kernels)
    fit = factory(baseline.reference, baseline.observation, baseline.k, baseline.config)
    poses = [initial]
    for offset in (-0.3, -0.01, 0.01, 0.3):
        p = initial.copy()
        p[:3, 3] += [offset, offset / 3, 0]
        p[:3, :3] = Rotation.from_rotvec([0.05, -0.1, 0.02]).as_matrix() @ p[:3, :3]
        poses.append(p)
    for p in poses:
        # Double-precision rounding tolerance, not a physical pose acceptance gate.
        np.testing.assert_allclose(fit.residual(p), baseline.residual(p), rtol=1e-12, atol=1e-12)
    assert not np.array_equal(baseline.residual(poses[0]), baseline.residual(poses[-1]))


@pytest.mark.parametrize("name", ["ring_visible_motion_20261016", "dowel_held_frame_315"])
@pytest.mark.parametrize("parallel", ["none", "search", "candidates"])
def test_compiled_complete_fit_preserves_final_pose_and_solver_decisions(kernels, name, parallel):
    baseline, initial = recorded_fit(name)
    expected, old = baseline.correct(initial)
    with ExitStack() as stack:
        options = {"kernels": kernels}
        if parallel == "search":
            options["search_pool"] = stack.enter_context(candidate.SearchPool(4))
        elif parallel == "candidates":
            options["candidate_pool"] = stack.enter_context(ThreadPoolExecutor(max_workers=4))
        factory = candidate.fitter_type(**options)
        fit = factory(baseline.reference, baseline.observation, baseline.k, baseline.config)
        shifted = initial.copy()
        shifted[:3, 3] += [0.01, -0.02, 0.005]
        poses = [initial, shifted, initial]
        for actual_residual, pose in zip(fit.batch_residuals(poses), poses, strict=True):
            np.testing.assert_array_equal(actual_residual, baseline.residual(pose))
        assert fit.batch_residuals([]) == []
        actual, new = fit.correct(initial)
    np.testing.assert_array_equal(actual, expected)
    assert new == old


def test_compiled_shared_pipeline_preserves_motion_loss_and_recovery(kernels):
    from unittest.mock import patch

    from test_vision_pipeline import Masks, frames, reference, synthetic_pairs

    from lerobot.policies.flash.perception_step import PerceptionPipeline, PerceptionRunner
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend

    outputs = []
    for factory in (WholeSurfaceFit, candidate.fitter_type(kernels=kernels)):
        backend = build_vision_backend(
            VisionPipelineConfig(), Masks(), matcher=object(), local_reference=True
        )
        pipeline = PerceptionPipeline(backend, (reference(),))
        source = iter(frames())
        results = []
        with (
            patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs),
            patch("lerobot.policies.flash.surface_tracking.WholeSurfaceFit", factory),
            PerceptionRunner(pipeline) as runner,
        ):
            while (item := runner.process_next(lambda source=source: next(source, None))) is not None:
                results.append(item[1])
        outputs.append(results)
    for old, new in zip(*outputs, strict=True):
        assert old.frame_id == new.frame_id and old.source_time_s == new.source_time_s
        assert old.failures == new.failures
        np.testing.assert_array_equal(old.masks["object"], new.masks["object"])
        assert old.estimates[0].evidence == new.estimates[0].evidence
        if old.estimates[0].pose is not None:
            np.testing.assert_array_equal(old.estimates[0].pose, new.estimates[0].pose)
    assert outputs[1][2].estimates[0].pose is None
    assert outputs[1][3].estimates[0].pose is not None
    assert (
        np.linalg.norm(outputs[1][3].estimates[0].pose[:3, 3] - outputs[1][0].estimates[0].pose[:3, 3]) > 0.01
    )


def test_batch_adapter_keeps_distinct_objects_and_empty_batches():
    class Model:
        def predict_correspondences_batched(self, a, b):
            delta = (b.float() - a.float()).permute(0, 3, 1, 2)
            return SimpleNamespace(
                flow=SimpleNamespace(flow_output=delta[:, :2]),
                covisibility=SimpleNamespace(mask=a[:, :, :, 0].float()),
            )

    images = [np.full((8, 9, 3), n, np.uint8) for n in (3, 17, 250)]
    targets = images[::-1]
    outputs = candidate.predict_pairs(Model(), images, targets, device="cpu")
    for i, (flow, mask) in enumerate(outputs):
        np.testing.assert_array_equal(flow, np.moveaxis(targets[i].astype(float) - images[i], -1, 0)[:2])
        np.testing.assert_array_equal(mask, images[i][:, :, 0])
    assert candidate.predict_pairs(Model(), [], [], device="cpu") == []
    with pytest.raises(ValueError, match="sizes differ"):
        candidate.predict_pairs(Model(), images, targets[:1], device="cpu")


def test_bad_batch_cardinality_is_rejected():
    import torch

    class Model:
        def predict_correspondences_batched(self, a, b):
            return SimpleNamespace(
                flow=SimpleNamespace(flow_output=torch.zeros((0, 2, 8, 9))),
                covisibility=SimpleNamespace(mask=torch.zeros((0, 8, 9))),
            )

    images = [np.zeros((8, 9, 3), np.uint8)]
    with pytest.raises(ValueError, match="cardinality"):
        candidate.predict_pairs(Model(), images, images, device="cpu")


def test_real_ufm_batch_rounding_difference_reaches_the_actual_fitter():
    import cv2

    from lerobot.policies.flash.surface_correspondence import measured_pairs, robust_fit

    data = np.load(Path(__file__).with_name("fixtures") / "ufm_batch_depth_rounding_20261025.npz")
    a = {"depth": data["source_depth"], "usable": data["source_usable"]}
    b = {"depth": data["depth"], "usable": data["target_usable"]}
    poses = []
    for mode in ("scalar", "batch"):
        keep = data[f"{mode}_keep"]
        source, target = measured_pairs(
            a, b, data["source_uv"][keep], data[f"{mode}_target_uv"][keep], data["intrinsics"]
        )
        np.testing.assert_array_equal(source, data[f"{mode}_source"])
        np.testing.assert_array_equal(target, data[f"{mode}_target"])
        cv2.setRNGSeed(17)
        step, _, _ = robust_fit(source, target)
        current = {
            "obs": {"points": data["target"]},
            "all_surface_points": data["target"],
            "mask": data["mask"],
            "depth": data["depth"],
        }
        fit = WholeSurfaceFit(data["reference"], current, data["intrinsics"], ShapeConfig())
        pose, _ = fit.correct(step)
        poses.append(pose)
    # This recorded case defeats a claim that batching is an exact acceleration:
    # a tiny pixel shift crosses a depth-sampling boundary and changes the fit.
    assert len(data["scalar_source"]) != len(data["batch_source"])
    assert not np.array_equal(poses[0], poses[1])
