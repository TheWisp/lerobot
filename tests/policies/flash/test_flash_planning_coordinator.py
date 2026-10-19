"""The motor consumes future chunks while pose changes revise only unissued ticks."""

import numpy as np

from lerobot.policies.flash.perception import LatestPoseStore
from lerobot.policies.flash.planner import JointPlanner, PlannerLimits
from lerobot.policies.flash.planner_worker import PlanResult
from lerobot.policies.flash.planning_coordinator import PlanBufferConfig, PlanningCoordinator
from lerobot.policies.flash.progress import Phase
from lerobot.policies.flash.session import FlashSession
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def pose(x=0.0):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix


class LinearKinematics:
    def inverse_kinematics(self, seed, target):
        return np.array([target[0, 3] * 100])

    def forward_kinematics(self, joints):
        return pose(joints[0] / 100)


class FakeRobot:
    def __init__(self):
        self.joint = 0.0
        self.sent = []

    def get_state_observation(self):
        return {"arm.pos": self.joint}

    def send_action(self, action):
        self.joint = action["arm.pos"]
        self.sent.append(dict(action))
        return dict(action)


class InlineWorker:
    def __init__(self, planner, publish):
        self.planner = planner
        self.publish = publish
        self.last_result = None
        self.fatal_error = None
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        actions = self.planner.plan(
            request.interval,
            start_tick=request.start_tick,
            count=request.count,
            epoch_s=request.epoch_s,
            period_s=request.period_s,
            interval_start_s=request.interval_start_s,
            seed_deg=request.seed_deg,
            reference=request.reference,
            attachment=request.attachment,
            connection=request.connection,
            issued_events=request.issued_events,
        )
        self.publish(request.revision, actions)
        self.last_result = PlanResult(request.revision, len(actions), None)


def make_runtime(*, mode=MotionMode.DEMONSTRATED, max_pose_age_s=10.0):
    now = [0.0]
    robot = FakeRobot()
    interval = Interval(
        "contact",
        mode,
        Relation.EE_TO_OBJECT,
        "target",
        (
            SpatialSample(0.0, pose(), pose()),
            SpatialSample(0.2, pose(0.1), pose()),
        ),
        speed_m_s=0.2 if mode == MotionMode.CONNECTION else None,
        reference_version="target-v1",
    )
    skill = FlashSkill("demo/source", 0, (interval,))
    session = FlashSession(
        skill,
        robot,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
    )
    poses = LatestPoseStore()
    poses.publish((target(0.2, 0.0),))
    kinematics = LinearKinematics()
    planner = JointPlanner(
        kinematics, ("arm",), "gripper", PlannerLimits((1000.0,), 0.001, 0.01, max_pose_age_s)
    )
    worker = InlineWorker(planner, session.publish_plan)
    coordinator = PlanningCoordinator(
        session,
        worker,
        poses,
        kinematics,
        ("arm",),
        PlanBufferConfig(
            lead_ticks=2,
            horizon_ticks=8,
            refill_at_ticks=3,
            max_reference_shift_m=0.02,
            max_reference_turn_rad=0.1,
        ),
        lambda now_s: None,
    )
    return now, robot, session, poses, worker, coordinator


def target(x, source_time_s):
    return PoseEstimate(
        "target",
        pose(x),
        source_time_s,
        source_time_s,
        Evidence.OBSERVED,
        "target-v1",
        (True,) * 3,
        (True,) * 3,
    )


def test_coordinator_starts_in_future_and_replans_from_committed_prefix():
    now, robot, session, poses, worker, coordinator = make_runtime()
    assert coordinator.step() is None
    assert worker.requests[0].start_tick == 3
    assert robot.sent == []
    for _ in range(3):
        coordinator.step()
    assert len(robot.sent) == 1
    assert robot.sent[0]["arm.pos"] == 20.0
    poses.publish((target(0.205, now[0]),))
    coordinator.schedule()
    assert worker.requests[-1].revision > worker.requests[0].revision
    assert worker.requests[-1].start_tick > session.actions.snapshot().last_tick
    for _ in range(20):
        if session.progress.phase not in (Phase.WAITING, Phase.MOVING, Phase.VERIFYING):
            break
        coordinator.step()
    assert session.progress.phase == Phase.MOTION_COMPLETE
    assert robot.sent[-1]["arm.pos"] == 30.5


def test_coordinator_interrupts_when_target_exits_explicit_replanning_envelope():
    now, robot, session, poses, worker, coordinator = make_runtime()
    for _ in range(4):
        coordinator.step()
    assert session.progress.phase == Phase.MOVING
    poses.publish((target(0.3, now[0]),))
    coordinator.schedule()
    assert session.progress.phase == Phase.INTERRUPTED
    assert session.actions.snapshot().pending == ()
    assert "configured replanning envelope" in session.progress.events[-1]["reason"]


def test_generated_connection_retargets_and_extends_duration_for_live_distance():
    now, robot, session, poses, worker, coordinator = make_runtime(mode=MotionMode.CONNECTION)
    for _ in range(12):
        coordinator.step()
    poses.publish((target(0.205, now[0]),))
    coordinator.schedule()
    for _ in range(100):
        if session.progress.phase == Phase.MOTION_COMPLETE:
            break
        coordinator.step()
    assert session.progress.phase == Phase.MOTION_COMPLETE
    assert robot.sent[-1]["arm.pos"] == 30.5
    assert len(robot.sent) > 30  # generated travel is longer than the 0.2 s contact demo


def test_lost_target_cancels_old_queue_before_next_motor_send():
    now, robot, session, poses, worker, coordinator = make_runtime()
    for _ in range(4):
        coordinator.step()
    sent_before_loss = len(robot.sent)
    poses.publish(
        (
            PoseEstimate(
                "target", None, now[0], now[0], Evidence.LOST, "target-v1", (False,) * 3, (False,) * 3
            ),
        )
    )
    assert coordinator.step() is None
    assert len(robot.sent) == sent_before_loss
    assert session.progress.phase == Phase.INTERRUPTED


def test_future_chunk_stops_at_camera_evidence_age_boundary():
    now, robot, session, poses, worker, coordinator = make_runtime(max_pose_age_s=0.11)
    coordinator.step()
    assert worker.requests[0].count == 1
    assert [action.tick for action in session.actions.snapshot().pending] == [3]
