"""Heavy pose estimation must not run in or block the motor owner."""

import time
from dataclasses import replace

import numpy as np
import pytest

from lerobot.policies.flash.perception import LatestPoseStore, ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.perception_worker import PoseProcess, PoseProcessError
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


class EchoBackend:
    def initialize(self, references):
        self.reference = references[0]

    def update(self, frame):
        pose = np.eye(4)
        pose[2, 3] = float(frame.depth[0, 0]) * frame.depth_m_per_unit
        return (
            PoseEstimate(
                self.reference.role,
                pose,
                frame.source_time_s,
                time.monotonic(),
                Evidence.OBSERVED,
                self.reference.version,
                (True, True, True),
                (True, True, True),
            ),
        )


class FailingBackend(EchoBackend):
    def update(self, frame):
        raise RuntimeError("injected perception failure")


class OccludedBackend(EchoBackend):
    def update(self, frame):
        if frame.frame_id == 1:
            return super().update(frame)
        return (
            PoseEstimate(
                self.reference.role,
                None,
                frame.source_time_s,
                time.monotonic(),
                Evidence.LOST,
                self.reference.version,
                (False,) * 3,
                (False,) * 3,
            ),
        )


class EchoMaskBackend(EchoBackend):
    def __init__(self):
        self.last_masks = {}

    def update(self, frame):
        self.last_masks = {self.reference.role: np.ones(frame.depth.shape, dtype=bool)}
        return super().update(frame)


def echo_factory():
    return EchoBackend()


def echo_mask_factory():
    return EchoMaskBackend()


def failing_factory():
    return FailingBackend()


def occluded_factory():
    return OccludedBackend()


def setup():
    snippet = ReferenceSnippet("recorded", 0, 0, "item", np.eye(4), "mask", "camera-v1")
    reference = ObjectReference("item", "item-v1", (snippet,))
    frame = RGBDFrame(
        1,
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.full((4, 4), 1000, dtype=np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        1.0,
        1.0,
        "camera-v1",
    )
    return reference, frame


def poll_until(worker, predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        worker.poll()
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("pose process did not deliver result")


def test_spawned_pose_worker_publishes_copied_snapshot():
    reference, frame = setup()
    store = LatestPoseStore()
    worker = PoseProcess(echo_factory, (reference,), store)
    try:
        worker.start(timeout_s=10)
        assert worker.submit(frame)
        frame.depth[0, 0] = 3000  # camera buffer reuse must not alter the queued frame
        poll_until(worker, lambda: store.snapshot()[0] == 1)
        version, poses = store.snapshot()
        assert version == 1
        assert poses["item"].pose[2, 3] == pytest.approx(1.0)
        assert worker.metrics.delivered == 1
    finally:
        worker.close()


def test_spawned_pose_worker_reports_backend_failure():
    reference, frame = setup()
    worker = PoseProcess(failing_factory, (reference,), LatestPoseStore())
    try:
        worker.start(timeout_s=10)
        assert worker.submit(frame)
        with pytest.raises(PoseProcessError, match="injected perception failure"):
            poll_until(worker, lambda: False)
    finally:
        worker.close()


def test_pose_worker_only_emits_proposals_when_scene_estimator_owns_state():
    reference, first = setup()
    store = LatestPoseStore()
    proposals = []
    worker = PoseProcess(
        occluded_factory, (reference,), store, publish_results=False, on_result=proposals.append
    )
    try:
        worker.start(timeout_s=10)
        assert worker.submit(first)
        poll_until(worker, lambda: worker.metrics.delivered == 1)
        hidden = replace(first, frame_id=2, source_time_s=1.1, available_time_s=1.1)
        assert worker.submit(hidden)
        poll_until(worker, lambda: worker.metrics.delivered == 2)
        assert store.snapshot() == (0, {})
        assert [result.estimates[0].evidence for result in proposals] == [Evidence.OBSERVED, Evidence.LOST]
        assert [result.frame_id for result in proposals] == [1, 2]
        assert worker.metrics.last_source_time_s == hidden.source_time_s
    finally:
        worker.close()


def test_causal_handoff_requires_aligned_masks_from_the_same_pose_frame():
    reference, frame = setup()
    received = []
    worker = PoseProcess(
        echo_mask_factory,
        (reference,),
        LatestPoseStore(),
        require_masks=True,
        on_result=received.append,
    )
    try:
        worker.start(timeout_s=10)
        assert worker.submit(frame)
        poll_until(worker, lambda: bool(received))
        assert received[0].frame_id == frame.frame_id
        assert received[0].masks["item"].shape == frame.depth.shape
    finally:
        worker.close()

    missing_masks = PoseProcess(echo_factory, (reference,), LatestPoseStore(), require_masks=True)
    try:
        with pytest.raises(
            PoseProcessError, match="requires a pose backend that exposes aligned instance masks"
        ):
            missing_masks.start(timeout_s=10)
    finally:
        missing_masks.close()
