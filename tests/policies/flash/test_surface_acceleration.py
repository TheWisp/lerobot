"""Acceleration preserves measured geometry, state order and image ownership."""

from dataclasses import asdict
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from test_vision_pipeline import Masks, frames, reference, synthetic_pairs

from lerobot.policies.flash import surface_grounding
from lerobot.policies.flash.perception_step import PerceptionPipeline, PerceptionRunner
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation
from lerobot.policies.flash.tracking_debug_worker import debug_reference
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend


@pytest.mark.parametrize("fixture", ["ring_visible_motion_20261016", "dowel_held_frame_315"])
def test_batched_solver_matches_scalar_on_recorded_surfaces(fixture, monkeypatch):
    data = np.load(Path(__file__).with_name("fixtures") / f"{fixture}.npz")
    is_ring = "source_surface" in data
    points = data["source_surface" if is_ring else "model_points"]
    depth = data["target_depth_mm" if is_ring else "depth"].astype(np.float32) * 0.001
    mask = data["target_mask" if is_ring else "mask"].astype(bool)
    k = data["intrinsics"]
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    current = {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": obs["points"]}
    scalar = WholeSurfaceFit(points, current, k, ShapeConfig(batch_jacobian=False))
    batched = WholeSurfaceFit(points, current, k, ShapeConfig())
    initial = np.eye(4) if is_ring else data["candidate"]
    poses = []
    for offset in (0, 0.003, -0.005):
        pose = initial.copy()
        pose[:3, 3] += offset
        poses.append(pose)
    for actual, expected in zip(batched.batch_residuals(poses), map(scalar.residual, poses), strict=True):
        np.testing.assert_array_equal(actual, expected)
    old, old_detail = scalar.correct(initial)
    new, new_detail = batched.correct(initial)
    np.testing.assert_array_equal(new, old)
    assert new_detail["cost"] == old_detail["cost"]
    assert new_detail["evaluations"] == old_detail["evaluations"]
    assert new_detail["used_coarse"] == old_detail["used_coarse"]
    # Supported old SciPy installations retain their scalar implementation.
    monkeypatch.setattr(surface_grounding, "_HAS_SOLVER_WORKERS", False)
    fallback, detail = batched.correct(initial)
    np.testing.assert_array_equal(fallback, old)
    assert detail["batched_jacobian"] is False


def backend(pipelined):
    return build_vision_backend(
        VisionPipelineConfig(pipeline_surface_frames=pipelined),
        Masks(),
        matcher=object(),
        local_reference=True,
        clock=lambda: 0.0,
    )


def test_overlap_preserves_poses_masks_loss_and_recovery():
    outputs = []
    with patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs):
        for enabled in (False, True):
            pipeline = PerceptionPipeline(backend(enabled), (reference(),))
            source = iter(frames())
            results = []
            with PerceptionRunner(pipeline) as runner:
                while (item := runner.process_next(lambda source=source: next(source, None))) is not None:
                    results.append(item)
            outputs.append(results)
            pipeline.backend.close()
    for (old_frame, old), (new_frame, new) in zip(*outputs, strict=True):
        assert old_frame.frame_id == new_frame.frame_id == old.frame_id == new.frame_id
        assert old.source_time_s == new.source_time_s
        assert old.failures == new.failures
        np.testing.assert_array_equal(old.masks["object"], new.masks["object"])
        a, b = old.estimates[0], new.estimates[0]
        assert a.evidence == b.evidence
        assert a.reference_version == b.reference_version
        if a.pose is not None:
            np.testing.assert_array_equal(a.pose, b.pose)
    assert outputs[1][2][1].estimates[0].pose is None
    assert outputs[1][3][1].estimates[0].pose is not None


def test_next_matching_overlaps_fit_but_commits_in_order(monkeypatch):
    pipeline = PerceptionPipeline(backend(True), (reference(),))
    sequence = frames()
    sequence = [sequence[0], sequence[1], sequence[3]]
    source = iter(sequence)
    prepared_second = Event()
    original_prepare, original_finish = pipeline.backend.prepare, pipeline.backend.finish
    calls = []

    def prepare(frame):
        result = original_prepare(frame)
        calls.append(("prepared", frame.frame_id))
        if frame.frame_id == 1:
            prepared_second.set()
        return result

    def finish(item):
        if item["frame"].frame_id == 0:
            assert prepared_second.wait(3), "next matching was serialized behind fitting"
            with pytest.raises(RuntimeError, match="pending surface frames"):
                pipeline.add_reference(debug_reference("new", sequence[1]))
        calls.append(("finish", item["frame"].frame_id))
        return original_finish(item)

    monkeypatch.setattr(pipeline.backend, "prepare", prepare)
    monkeypatch.setattr(pipeline.backend, "finish", finish)
    with patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs):
        with PerceptionRunner(pipeline) as runner:
            first = runner.process_next(lambda source=source: next(source, None))
            assert first[0].frame_id == 0
            np.testing.assert_array_equal(first[1].masks["object"], Masks().masks(sequence[0])["object"])
            # Pausing completes the pending delta; resuming matches from that frame.
            drained = runner.drain()
            assert drained[0].frame_id == 1
            last = runner.process_next(lambda source=source: next(source, None))
            assert last[0].frame_id == 3
            assert runner.process_next(lambda source=source: next(source, None)) is None
    assert [n for event, n in calls if event == "finish"] == [0, 1, 3]
    np.testing.assert_allclose(
        last[1].estimates[0].pose[:3, 3] - first[1].estimates[0].pose[:3, 3], [0.012, 0, 0], atol=0.0015
    )
    pipeline.backend.close()


def test_staged_backend_rejects_out_of_order_duplicate_and_unbounded_frames():
    pipeline = PerceptionPipeline(backend(True), (reference(),))
    with patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs):
        a, b = (pipeline.backend.prepare(frame) for frame in frames()[:2])
    with pytest.raises(RuntimeError, match="one frame ahead"):
        pipeline.backend.prepare(frames()[2])
    with pytest.raises(ValueError, match="preparation order"):
        pipeline.backend.finish(b)
    pipeline.backend.finish(a)
    with pytest.raises(ValueError, match="preparation order"):
        pipeline.backend.finish(a)
    pipeline.backend.finish(b)
    pipeline.backend.close()


def test_inference_worker_reports_fit_error_without_waiting_for_another_camera_frame():
    from queue import Queue
    from threading import Thread

    from lerobot.policies.flash.perception_worker import _Failure, _run_backend

    failed = Event()

    def factory():
        result = backend(True)

        def fail(_prepared):
            failed.set()
            raise RuntimeError("deliberate fit failure")

        result.finish = fail
        return result

    inputs, outputs = Queue(), Queue()
    inputs.put(frames()[0])
    # There is no next frame or EOF sentinel. Prefetch must not trap teardown.
    thread = Thread(target=_run_backend, args=(factory, (reference(),), inputs, outputs, True), daemon=True)
    thread.start()
    assert failed.wait(3)
    thread.join(3)
    assert not thread.is_alive()
    messages = []
    while not outputs.empty():
        messages.append(outputs.get_nowait())
    assert any(isinstance(item, _Failure) and "deliberate fit failure" in item.detail for item in messages)


def test_first_debug_selection_and_replacement_both_initialize():
    from lerobot.policies.flash.tracking_debug_worker import replace_debug_pipeline

    frame = frames()[0]
    pipeline, runner = replace_debug_pipeline(None, VisionPipelineConfig(), Masks(), object(), frame)
    source = iter((frame,))
    first = runner.process_next(lambda: next(source, None))
    assert first[1].details["object"]["status"] == "reference"
    replacement, new_runner = replace_debug_pipeline(runner, VisionPipelineConfig(), Masks(), object(), frame)
    assert runner.closed
    assert replacement is not pipeline
    source = iter((frame,))
    result = new_runner.process_next(lambda: next(source, None))
    assert result[1].details["object"]["status"] == "reference"
    new_runner.close()
