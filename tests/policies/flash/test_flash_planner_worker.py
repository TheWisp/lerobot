"""IK preparation continues while the motor owner can consume old actions."""

import json
import time
from dataclasses import asdict, replace
from threading import Event

import numpy as np

from lerobot.policies.flash.execution import TimedAction
from lerobot.policies.flash.planner_worker import PlannerWorker, PlanRequest
from lerobot.policies.flash.skill import Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.trajectory import AttachmentState, ConnectionTrajectory, Evidence, PoseEstimate


class ProcessActionPlanner:
    def plan(self, interval, *, seed_deg, **_kwargs):
        return [TimedAction(0, {"joint.pos": float(seed_deg[0])}, interval.name)]


class SlowProcessActionPlanner(ProcessActionPlanner):
    def plan(self, interval, *, seed_deg, **kwargs):
        time.sleep(0.25)
        return super().plan(interval, seed_deg=seed_deg, **kwargs)


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


def test_plan_request_preserves_retimed_connection_in_worker_snapshot():
    base = request(1)
    connection = ConnectionTrajectory.begin(base.interval, np.eye(4), base.reference.pose, None, 0)
    connection.duration_s = 1.5
    connection.joint_retimed = True
    connection.requested_goal_pose[0, 3] = 1.0
    connection.goal_pose[0, 3] = 0.4
    connection.achieved_fraction = 0.4
    copied = PlanRequest(
        base.revision,
        base.interval,
        base.start_tick,
        base.count,
        base.epoch_s,
        base.period_s,
        base.interval_start_s,
        base.seed_deg,
        base.reference,
        base.attachment,
        connection,
    )
    assert copied.connection is not connection
    assert copied.connection.duration_s == 1.5
    assert copied.connection.joint_retimed
    assert copied.connection.goal_pose[0, 3] == 0.4
    assert copied.connection.requested_goal_pose[0, 3] == 1.0
    assert copied.connection.achieved_fraction == 0.4


def test_worker_stamps_queued_commands_with_the_pose_snapshot_used_for_ik():
    class OneActionPlanner:
        def plan(self, interval, **_kwargs):
            return [TimedAction(0, {"joint.pos": 1.0}, interval.name)]

    published = []
    worker = PlannerWorker(
        OneActionPlanner(), lambda revision, actions: published.append((revision, actions))
    )
    worker.start()
    try:
        base = request(1)
        reference_pose = np.eye(4)
        reference_pose[0, 3] = 0.12
        attachment_pose = np.eye(4)
        attachment_pose[1, 3] = -0.03
        reference = replace(
            base.reference,
            pose=reference_pose,
            source_frame_id=5,
            source_time_s=0.1,
            available_time_s=0.2,
        )
        attachment = AttachmentState(attachment_pose, 0.1, 0.2, "held-v1")
        worker.submit(replace(base, reference=reference, attachment=attachment, pose_revision=7))
        deadline = time.monotonic() + 5
        while not published:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        evidence = published[0][1][0].plan_evidence
        assert evidence is not None
        assert evidence.pose_revision == 7
        assert evidence.reference_source_frame_id == 5
        assert evidence.reference_source_time_s == 0.1
        assert evidence.reference_available_time_s == 0.2
        assert evidence.reference_pose[0][3] == 0.12
        assert evidence.attachment_gripper_from_object[1][3] == -0.03
        assert evidence.turn_angle_rad == 0.0
        recorded = json.loads(json.dumps(asdict(evidence)))
        assert recorded["reference_pose"][0][3] == 0.12
        assert recorded["attachment_gripper_from_object"][1][3] == -0.03
    finally:
        worker.close()


def test_isolated_worker_returns_the_same_pose_stamped_plan():
    published = []
    worker = PlannerWorker(
        ProcessActionPlanner(),
        lambda revision, actions: published.append((revision, actions)),
        process_factory=ProcessActionPlanner,
    )
    worker.start()
    try:
        base = request(1)
        reference_pose = np.eye(4)
        reference_pose[2, 3] = 0.07
        reference = replace(
            base.reference,
            pose=reference_pose,
            source_frame_id=5,
            source_time_s=0.1,
            available_time_s=0.2,
        )
        worker.submit(replace(base, reference=reference, pose_revision=7))
        deadline = time.monotonic() + 10
        while not published and worker.fatal_error is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert worker.fatal_error is None
        assert worker.last_result is not None and worker.last_result.error is None
        assert published[0][1][0].plan_evidence.pose_revision == 7
        assert published[0][1][0].plan_evidence.reference_source_frame_id == 5
        assert published[0][1][0].plan_evidence.reference_pose[2][3] == 0.07
    finally:
        worker.close()


def test_isolated_worker_discards_superseded_solve_and_pending_request():
    published = []
    worker = PlannerWorker(
        SlowProcessActionPlanner(),
        lambda revision, actions: published.append(revision),
        process_factory=SlowProcessActionPlanner,
    )
    worker.start()
    try:
        worker.submit(request(1))
        time.sleep(0.1)
        worker.submit(request(2))
        worker.submit(request(3))
        deadline = time.monotonic() + 10
        while worker.last_result is None or worker.last_result.revision != 3:
            assert worker.fatal_error is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert published == [3]
    finally:
        worker.close()


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
