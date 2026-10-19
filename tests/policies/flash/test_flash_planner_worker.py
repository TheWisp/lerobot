"""IK preparation continues while the motor owner can consume old actions."""

import time
from threading import Event

import numpy as np

from lerobot.policies.flash.execution import TimedAction
from lerobot.policies.flash.planner_worker import PlannerWorker, PlanRequest
from lerobot.policies.flash.skill import Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def request(revision):
    identity = np.eye(4)
    interval = Interval(
        "carry",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        (SpatialSample(0, identity, identity), SpatialSample(1, identity, identity)),
        speed_m_s=0.1,
    )
    reference = PoseEstimate("object", identity, 0, 0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3)
    return PlanRequest(revision, interval, 0, 1, 0, 1 / 30, 0, np.array([revision]), reference, None)


def test_planner_worker_supersedes_pending_request_without_blocking_motor_owner():
    started = Event()
    release = Event()
    solved = []

    class BlockingPlanner:
        def plan(self, interval, *, seed_deg, **kwargs):
            solved.append(int(seed_deg[0]))
            started.set()
            release.wait(5)
            return [TimedAction(0, {"joint.pos": float(seed_deg[0])}, interval.name)]

    published = []
    worker = PlannerWorker(BlockingPlanner(), lambda revision, actions: published.append(revision))
    worker.start()
    try:
        first = request(1)
        worker.submit(first)
        assert started.wait(5)
        # The worker is blocked in IK, but the caller can replace its pending
        # future request without waiting for that computation to finish.
        second, third = request(2), request(3)
        worker.submit(second)
        worker.submit(third)
        assert worker.superseded == 1
        release.set()
        deadline = time.monotonic() + 5
        while worker.last_result is None or worker.last_result.revision != 3:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert published == [3]
        assert solved == [1, 3]
        assert worker.last_result.error is None
    finally:
        release.set()
        worker.close()
