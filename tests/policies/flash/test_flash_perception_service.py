"""A live perception host must wait for seeded evidence before motion."""

import time
from threading import Event

import numpy as np
import pytest

from lerobot.policies.flash.perception import LatestPoseStore, ObjectReference, ReferenceSnippet, RGBDFrame
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
