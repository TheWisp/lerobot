"""A live perception host must wait for seeded evidence before motion."""

import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.policies.flash.perception import (
    LatestPoseStore,
    LatestRGBDFrameStore,
    ObjectReference,
    ReferenceSnippet,
    RGBDFrame,
)
from lerobot.policies.flash.perception_service import PerceptionService, PerceptionServiceError
from lerobot.policies.flash.perception_worker import PoseProcess
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


class EchoBackend:
    def initialize(self, references):
        self.reference = references[0]

    def update(self, frame):
        return (
            PoseEstimate(
                self.reference.role,
                np.eye(4),
                frame.source_time_s,
                time.perf_counter(),
                Evidence.OBSERVED,
                self.reference.version,
                (True,) * 3,
                (True,) * 3,
            ),
        )


def echo_factory():
    return EchoBackend()


class FakeCameraSource:
    def __init__(self):
        self.frame_id = 0

    def read(self):
        time.sleep(0.015)
        now = time.perf_counter()
        frame = RGBDFrame(
            self.frame_id,
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint16),
            np.eye(3),
            0.001,
            np.eye(4),
            now,
            now,
            "camera-v1",
        )
        self.frame_id += 1
        return frame


def test_latest_frame_survives_camera_buffer_reuse():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.uint16)
    frame = RGBDFrame(1, rgb, depth, np.eye(3), 0.001, np.eye(4), 1.0, 1.0, "camera-v1")
    store = LatestRGBDFrameStore()
    store.publish(frame)
    rgb[:] = 255
    depth[:] = 5
    saved = store.snapshot()
    assert saved.rgb.max() == 0
    assert saved.depth.max() == 1


def test_perception_service_waits_for_real_pose_from_spawned_worker():
    snippet = ReferenceSnippet("recorded", 0, 0, "item", np.eye(4), "mask", "camera-v1")
    reference = ObjectReference("item", "item-v1", (snippet,))
    store = LatestPoseStore()
    published = Event()

    def on_pose_update(version, poses):
        if version > 0:
            assert poses["item"].evidence == Evidence.OBSERVED
            published.set()

    service = PerceptionService(
        FakeCameraSource(),
        PoseProcess(echo_factory, (reference,), store),
        store,
        on_pose_update=on_pose_update,
    )
    try:
        service.start(timeout_s=10)
        service.wait_ready(("item",), max_pose_age_s=1, timeout_s=10)
        assert service.captured > 0
        assert store.snapshot()[1]["item"].evidence == Evidence.OBSERVED
        assert published.wait(5)
        recorded_frame = service.frames.snapshot()
        assert recorded_frame is not None
        assert recorded_frame.rgb.shape == (2, 2, 3)
        assert not recorded_frame.rgb.flags.writeable
        assert service.frames.snapshot(no_later_than_s=recorded_frame.source_time_s - 0.001) is None
        assert service.error is None
    finally:
        service.close()


def test_camera_failure_prevents_pose_readiness():
    class FailedCamera:
        def read(self):
            raise TimeoutError("camera frames stopped")

    snippet = ReferenceSnippet("recorded", 0, 0, "item", np.eye(4), "mask", "camera-v1")
    reference = ObjectReference("item", "item-v1", (snippet,))
    store = LatestPoseStore()
    service = PerceptionService(FailedCamera(), PoseProcess(echo_factory, (reference,), store), store)
    try:
        service.start(timeout_s=10)
        with pytest.raises(PerceptionServiceError, match="live perception failed"):
            service.wait_ready(("item",), max_pose_age_s=1, timeout_s=10)
        assert isinstance(service.error, TimeoutError)
    finally:
        service.close()


def test_transient_camera_timeout_keeps_latest_frame_and_recovers():
    class IntermittentSource(FakeCameraSource):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls == 2:
                time.sleep(0.025)
                raise TimeoutError("one missed camera frame")
            return super().read()

    store = LatestPoseStore()
    worker = SimpleNamespace(store=store, submit=lambda _frame: True, alive=True)
    service = PerceptionService(IntermittentSource(), worker, store, max_camera_silence_s=0.15)
    service._capture_thread.start()
    try:
        deadline = time.monotonic() + 1
        while service.captured < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert service.captured >= 2
        assert service.source.calls >= 3
        assert service.error is None
        assert service.camera_timeouts == 1
    finally:
        service._stop.set()
        service._capture_thread.join(1)


def test_persistent_camera_timeout_fails_at_camera_age_bound():
    class StoppedSource(FakeCameraSource):
        def read(self):
            if self.frame_id:
                time.sleep(0.025)
                raise TimeoutError("camera stopped")
            return super().read()

    store = LatestPoseStore()
    worker = SimpleNamespace(store=store, submit=lambda _frame: True, alive=True)
    service = PerceptionService(StoppedSource(), worker, store, max_camera_silence_s=0.09)
    service._capture_thread.start()
    try:
        deadline = time.monotonic() + 1
        while service.error is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert service.captured == 1
        assert isinstance(service.error, TimeoutError)
        assert service.camera_timeouts >= 2
        assert "RGB-D camera produced no frame for" in str(service.error)
        assert "camera silence limit 0.090s" in str(service.error)
        assert "camera stopped" in str(service.error)
    finally:
        service._stop.set()
        service._capture_thread.join(1)


def test_readiness_uses_capture_time_of_direct_support_after_prediction():
    store = LatestPoseStore()
    worker = SimpleNamespace(store=store, metrics=SimpleNamespace(last_failures={}))
    service = PerceptionService(FakeCameraSource(), worker, store, clock=lambda: 1.1)
    service._started = True
    store.publish(
        (
            PoseEstimate(
                "item",
                np.eye(4),
                1.1,
                1.1,
                Evidence.PREDICTED,
                "v1",
                (True,) * 3,
                (True,) * 3,
                estimator="stationary_prior",
                last_direct_observation_s=1.0,
            ),
        )
    )
    service.wait_ready(("item",), max_pose_age_s=0.2, timeout_s=0.01)

    store.publish(
        (
            PoseEstimate(
                "item",
                np.eye(4),
                1.2,
                1.2,
                Evidence.PREDICTED,
                "v1",
                (True,) * 3,
                (True,) * 3,
                estimator="stationary_prior",
                last_direct_observation_s=0.5,
            ),
        )
    )
    with pytest.raises(PerceptionServiceError, match="no fresh observed pose"):
        service.wait_ready(("item",), max_pose_age_s=0.2, timeout_s=0.01)


def test_readiness_accepts_measured_position_with_predicted_orientation_only_after_publication():
    store = LatestPoseStore()
    worker = SimpleNamespace(store=store, metrics=SimpleNamespace(last_failures={}))
    service = PerceptionService(FakeCameraSource(), worker, store, clock=lambda: 1.1)
    service._started = True
    partial = PoseEstimate(
        "item",
        np.eye(4),
        1.05,
        1.2,
        Evidence.PREDICTED,
        "v1",
        (True,) * 3,
        (True,) * 3,
        direct_translation_observed=True,
    )
    store.publish((partial,))
    with pytest.raises(PerceptionServiceError, match="no fresh observed pose"):
        service.wait_ready(("item",), max_pose_age_s=0.2, timeout_s=0.01)

    store.revise(replace(partial, available_time_s=1.05))
    service.wait_ready(("item",), max_pose_age_s=0.2, timeout_s=0.01)
