"""Behavioral invariants for direct spatial imitation."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.attachment_lifecycle import AttachmentLifecycle
from lerobot.policies.flash.execution import (
    ActionQueue,
    ActionReceipt,
    MissedEventError,
    MotorLoop,
    MotorTick,
    PlanEvidence,
    PlanRejectedError,
    TimedAction,
)
from lerobot.policies.flash.geometry import held_object_goal
from lerobot.policies.flash.perception import (
    AttachmentEstimate,
    HeldAttachmentUpdater,
    LatestPoseStore,
    LatestRGBDFrameStore,
    RGBDFrame,
    TimedEndEffectorHistory,
)
from lerobot.policies.flash.planner import (
    InfeasibleTrajectoryError,
    JointPlanner,
    PlannerLimits,
    _ConnectionSpeedExceededError,
)
from lerobot.policies.flash.pose_disagreement import PoseComparison, PoseComparisonVerdict, SurfaceSupport
from lerobot.policies.flash.progress import MilestoneUpdate, Phase, Progress, Verdict
from lerobot.policies.flash.recording import (
    FlashEpisodeWriter,
    RecordedTick,
    RecorderWorker,
    TickRecordAssembler,
    load_pose_comparisons,
    load_pose_estimates,
)
from lerobot.policies.flash.skill import (
    AttachmentAction,
    AttachmentTransition,
    EndpointRequirement,
    FlashSkill,
    Interval,
    MilestoneSpec,
    MotionMode,
    Relation,
    SpatialSample,
)
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig
from lerobot.policies.flash.trajectory import (
    AttachmentState,
    ConnectionTrajectory,
    Evidence,
    MissingEvidenceError,
    PoseEstimate,
    demonstrated_goal,
)


def pose(x=0.0, y=0.0, z=0.0):
    value = np.eye(4)
    value[:3, 3] = x, y, z
    return value


def sample(t, ee, target, held=None, gripper=None):
    return SpatialSample(t, pose(*ee), pose(*target), pose(*held) if held is not None else None, gripper)


def test_live_held_object_follows_live_target_with_changed_grasp_attachment():
    demo_held = pose(0.2, 0, 0.1)
    demo_target = pose(0.2, 0, 0)
    live_target = pose(0.4, 0.1, 0)
    live_attachment = pose(0.01, 0.02, 0)
    ee_goal = held_object_goal(demo_held, demo_target, live_target, live_attachment)
    np.testing.assert_allclose((ee_goal @ live_attachment)[:3, 3], [0.4, 0.1, 0.1])


def test_demonstrated_interval_keeps_its_local_time_and_updates_live_target():
    interval = Interval(
        "insert",
        MotionMode.DEMONSTRATED,
        Relation.HELD_TO_TARGET,
        "target",
        (
            sample(4.0, (0, 0, 0), (0, 0, 0), (0, 0, 0.1), 0.8),
            sample(5.0, (0, 0, 0), (0, 0, 0), (0, 0, 0), 0.8),
        ),
    )
    first, _ = demonstrated_goal(interval, 0.5, pose(0.4), pose())
    moved, grip = demonstrated_goal(interval, 0.5, pose(0.6), pose())
    np.testing.assert_allclose(moved[:3, 3] - first[:3, 3], [0.2, 0, 0])
    assert grip == 0.8
    assert interval.demo_duration_s == 1.0


def test_connection_duration_adapts_to_distance_without_stretching_next_contact():
    carry = Interval(
        "carry",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        speed_m_s=0.05,
    )
    near = ConnectionTrajectory.begin(carry, pose(), pose(0.1), None, 10)
    far = ConnectionTrajectory.begin(carry, pose(), pose(0.3), None, 10)
    assert near.duration_s == pytest.approx(2)
    assert far.duration_s == pytest.approx(6)
    near.retarget(pose(0.05), pose(0.2), None, 11)
    assert near.duration_s == pytest.approx(3)
    assert near.start_time_s == 11


def test_generated_connection_retimes_short_cartesian_move_with_large_joint_travel():
    class AmplifiedKinematics:
        def inverse_kinematics(self, _seed, target):
            return np.array([target[0, 3] * 10_000])

        def forward_kinematics(self, joints):
            return pose(joints[0] / 10_000)

    interval = Interval(
        "approach",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        speed_m_s=0.04,
        reference_version="object-v1",
    )
    reference = PoseEstimate(
        "object", pose(0.01), 0, 0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
    )
    connection = ConnectionTrajectory.begin(interval, pose(), reference.pose, None, 0)
    assert connection.duration_s == pytest.approx(0.25)
    planner = JointPlanner(
        AmplifiedKinematics(), ("arm",), "gripper", PlannerLimits((180.0,), 0.001, 0.01, 10.0)
    )
    actions = planner.plan(
        interval,
        start_tick=0,
        count=8,
        epoch_s=0,
        period_s=1 / 30,
        interval_start_s=0,
        seed_deg=np.array([0.0]),
        reference=reference,
        attachment=None,
        connection=connection,
    )
    assert connection.joint_retimed
    assert connection.duration_s > 100 / 180
    assert max(np.diff([0.0, *(action.action["arm.pos"] for action in actions)])) <= 180 / 30


def test_generated_connection_subdivides_when_local_ik_cannot_jump_to_endpoint():
    class LocalKinematics:
        def inverse_kinematics(self, seed, target):
            if abs(target[0, 3] - seed[0] / 10) > 0.3:
                return seed
            return np.array([target[0, 3] * 10])

        def forward_kinematics(self, joints):
            return pose(joints[0] / 10)

    interval = Interval(
        "carry",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        speed_m_s=0.5,
        reference_version="object-v1",
    )
    reference = PoseEstimate(
        "object", pose(1), 0, 0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
    )
    connection = ConnectionTrajectory.begin(interval, pose(), reference.pose, None, 0)
    planner = JointPlanner(LocalKinematics(), ("arm",), "gripper", PlannerLimits((1000,), 0.001, 0.01, 10))
    planner._retime_connection(connection, np.array([0.0]), 1 / 30)
    assert connection.joint_retimed


def test_preferred_connection_advances_to_checked_ik_prefix_without_claiming_endpoint(tmp_path):
    class BoundedKinematics:
        def inverse_kinematics(self, seed, target):
            return seed if target[0, 3] > 0.4 else np.array([target[0, 3] * 10])

        def forward_kinematics(self, joints):
            return pose(joints[0] / 10)

    samples = (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0)))
    interval = Interval(
        "hover",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        samples,
        speed_m_s=0.5,
        reference_version="object-v1",
        endpoint_requirement=EndpointRequirement.PREFERRED,
    )
    skill_path = tmp_path / "preferred.json"
    FlashSkill("demo/source", 0, (interval,)).save(skill_path)
    assert FlashSkill.load(skill_path).intervals[0].endpoint_requirement == EndpointRequirement.PREFERRED
    reference = PoseEstimate(
        "object", pose(1), 0, 0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
    )
    planner = JointPlanner(BoundedKinematics(), ("arm",), "gripper", PlannerLimits((1000,), 0.001, 0.01, 10))
    connection = ConnectionTrajectory.begin(interval, pose(), reference.pose, None, 0)
    actions = planner.plan(
        interval,
        start_tick=0,
        count=100,
        epoch_s=0,
        period_s=1 / 30,
        interval_start_s=0,
        seed_deg=np.array([0.0]),
        reference=reference,
        attachment=None,
        connection=connection,
    )
    assert actions[-1].interval_complete
    assert 0.3 < connection.achieved_fraction <= 0.401
    assert connection.goal_pose[0, 3] <= 0.401
    assert connection.requested_goal_pose[0, 3] == 1.0

    strict = Interval(
        "hover",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        samples,
        speed_m_s=0.5,
        reference_version="object-v1",
    )
    with pytest.raises(InfeasibleTrajectoryError, match="no bounded continuous IK path"):
        planner.plan(
            strict,
            start_tick=0,
            count=100,
            epoch_s=0,
            period_s=1 / 30,
            interval_start_s=0,
            seed_deg=np.array([0.0]),
            reference=reference,
            attachment=None,
            connection=ConnectionTrajectory.begin(strict, pose(), reference.pose, None, 0),
        )


def test_connection_retime_makes_progress_when_discrete_tick_is_just_over_speed_limit():
    class IdentityKinematics:
        def inverse_kinematics(self, _seed, target):
            return np.array([target[0, 3]])

        def forward_kinematics(self, joints):
            return pose(joints[0])

    class DiscreteLimitPlanner(JointPlanner):
        def _plan_once(self, interval, **kwargs):
            if kwargs["connection"].duration_s < 0.25 + kwargs["period_s"]:
                raise _ConnectionSpeedExceededError(kwargs["start_tick"] + 2, 1.0001)
            return [TimedAction(kwargs["start_tick"], {"arm.pos": 0.0}, interval.name)]

    interval = Interval(
        "approach",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        speed_m_s=0.04,
        reference_version="object-v1",
    )
    reference = PoseEstimate(
        "object", pose(0.01), 0, 0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
    )
    connection = ConnectionTrajectory.begin(interval, pose(), reference.pose, None, 0)
    planner = DiscreteLimitPlanner(
        IdentityKinematics(),
        ("arm",),
        "gripper",
        PlannerLimits((180.0,), 0.001, 0.01, 10.0, max_connection_retimings=3),
    )
    actions = planner.plan(
        interval,
        start_tick=0,
        count=3,
        epoch_s=0,
        period_s=1 / 30,
        interval_start_s=0,
        seed_deg=np.array([0.0]),
        reference=reference,
        attachment=None,
        connection=connection,
    )
    assert len(actions) == 1
    assert connection.duration_s >= 0.25 + 1 / 30


def test_milestone_requires_fresh_evidence_and_motion_is_not_success():
    interaction = Interval(
        "grasp",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "object",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        completion_milestone="attached",
        recovery_intervals=("grasp",),
    )
    progress = Progress(
        FlashSkill("owner/demo", 0, (interaction,), milestones=(MilestoneSpec("attached", ("held",)),))
    )
    assert progress.begin(10)
    progress.receipt(11, interval_complete=True)
    assert progress.phase == Phase.VERIFYING
    assert not progress.advance(11)
    update = MilestoneUpdate("attached", Verdict.TRUE, "camera:7", 11.2)
    assert progress.update_milestone(update)
    assert not progress.update_milestone(update)
    assert progress.advance(11.3)
    assert progress.phase == Phase.MOTION_COMPLETE
    progress.finish(11.4, Verdict.TRUE, "attachment observed after lift")
    assert progress.phase == Phase.SUCCEEDED


def test_completion_milestone_needs_both_objects_observed_after_final_command():
    interaction = Interval(
        "place",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "target",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        completion_milestone="placed",
    )
    progress = Progress(
        FlashSkill(
            "owner/demo",
            0,
            (interaction,),
            milestones=(MilestoneSpec("placed", ("held", "target"), require_post_motion_evidence=True),),
        )
    )
    progress.update_milestone(
        MilestoneUpdate("placed", Verdict.TRUE, "before", 10.5, oldest_source_time_s=10.5)
    )
    assert progress.begin(10)
    progress.receipt(11, interval_complete=True)
    assert not progress.advance(11)
    progress.update_milestone(
        MilestoneUpdate("placed", Verdict.TRUE, "one-new", 11.2, oldest_source_time_s=10.5)
    )
    assert not progress.advance(11.2)
    progress.update_milestone(
        MilestoneUpdate("placed", Verdict.TRUE, "both-new", 11.3, oldest_source_time_s=11.1)
    )
    assert progress.advance(11.3)


def test_contact_milestone_can_accumulate_evidence_during_motion():
    interval = Interval(
        "lift",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "object",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        completion_milestone="attached",
    )
    progress = Progress(
        FlashSkill("owner/demo", 0, (interval,), milestones=(MilestoneSpec("attached", ("object",)),))
    )
    assert progress.begin(10)
    progress.update_milestone(MilestoneUpdate("attached", Verdict.TRUE, "during-lift", 10.5))
    progress.receipt(11, interval_complete=True)
    assert progress.advance(11)


def test_target_pose_change_withdraws_previous_milestone_verdict():
    interval = Interval(
        "release",
        MotionMode.DEMONSTRATED,
        Relation.HELD_TO_TARGET,
        "target",
        (sample(0, (0, 0, 0), (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0), (0, 0, 0))),
        entry_milestone="aligned",
    )
    skill = FlashSkill(
        "owner/demo", 0, (interval,), milestones=(MilestoneSpec("aligned", ("target", "held")),)
    )
    progress = Progress(skill)
    assert progress.update_milestone(MilestoneUpdate("aligned", Verdict.TRUE, "camera:1", 1.0))
    assert progress.invalidate_changed_roles({"target"}, "pose:2", 1.1) == ("aligned",)
    assert not progress.begin(1.2)
    assert progress.update_milestone(MilestoneUpdate("aligned", Verdict.TRUE, "camera:3", 1.3))
    assert progress.begin(1.4)


def test_recovery_is_explicit_and_bounded():
    interaction = Interval(
        "grasp",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "object",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        recovery_intervals=("grasp",),
    )
    progress = Progress(FlashSkill("owner/demo", 0, (interaction,)))
    progress.retry_from("grasp", 1.0, "missed", max_attempts=1)
    assert progress.attempt == 1
    progress.retry_from("grasp", 2.0, "missed again", max_attempts=1)
    assert progress.phase == Phase.FAILED


def test_recovery_requires_new_milestone_evidence_before_restarting():
    interaction = Interval(
        "approach",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0))),
        speed_m_s=0.1,
        entry_milestone="visible",
        recovery_intervals=("approach",),
    )
    progress = Progress(
        FlashSkill("owner/demo", 0, (interaction,), milestones=(MilestoneSpec("visible", ("object",)),))
    )
    assert progress.update_milestone(MilestoneUpdate("visible", Verdict.TRUE, "camera:1", 1.0))
    assert progress.begin(1.1)
    progress.retry_from("approach", 2.0, "first path blocked", max_attempts=1)
    assert not progress.can_begin()
    assert progress.milestones == {}
    assert not progress.update_milestone(MilestoneUpdate("visible", Verdict.TRUE, "camera:1", 1.0))
    assert progress.update_milestone(MilestoneUpdate("visible", Verdict.TRUE, "camera:2", 2.1))
    assert progress.can_begin()


def test_retarget_keeps_future_prefix_and_required_event():
    queue = ActionQueue()
    queue.publish(
        0,
        [
            TimedAction(0, {"joint.pos": 0}, "grasp"),
            TimedAction(1, {"joint.pos": 1}, "grasp", event_id="close"),
        ],
    )
    assert queue.take(0)[0].action["joint.pos"] == 0
    with pytest.raises(PlanRejectedError, match="drops"):
        queue.publish(1, [TimedAction(1, {"joint.pos": 2}, "grasp")])
    queue.publish(1, [TimedAction(1, {"joint.pos": 2}, "grasp", event_id="close")])
    assert queue.take(1)[0].event_id == "close"
    with pytest.raises(PlanRejectedError, match="repeats"):
        queue.publish(2, [TimedAction(2, {"joint.pos": 3}, "grasp", event_id="close")])

    prefix_queue = ActionQueue()
    prefix_queue.publish(
        4, [TimedAction(0, {"joint.pos": 0}, "carry"), TimedAction(1, {"joint.pos": 1}, "carry")]
    )
    prefix_queue.publish(5, [TimedAction(1, {"joint.pos": 2}, "carry")])
    assert prefix_queue.take(0)[1] == 4
    assert prefix_queue.take(1)[1] == 5


def test_delayed_replan_cannot_lose_the_start_of_a_demonstrated_interval():
    queue = ActionQueue()
    queue.skip(0)
    with pytest.raises(PlanRejectedError, match="expired before required samples"):
        queue.publish(
            0,
            [TimedAction(i, {"joint.pos": float(i)}, "contact") for i in range(4)],
        )
    assert queue.revision == -1
    queue.publish(0, [TimedAction(i, {"joint.pos": float(i)}, "contact") for i in range(1, 4)])
    assert queue.take(1)[0].action["joint.pos"] == 1.0


def test_queue_rejects_a_retarget_jump_in_robot_api_action_space():
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.5})
    queue.publish(0, [TimedAction(0, {"joint.pos": 0.2}, "carry")])
    assert queue.take(0)[0] is not None
    queue.note_issued({"joint.pos": 0.1})
    with pytest.raises(PlanRejectedError, match="jumps"):
        queue.publish(1, [TimedAction(1, {"joint.pos": 1.0}, "carry")])
    queue.publish(1, [TimedAction(1, {"joint.pos": 0.4}, "carry")])
    assert queue.take(1)[0].action["joint.pos"] == 0.4


def test_queue_snapshot_exposes_future_splice_seed_without_mutating_live_queue():
    queue = ActionQueue()
    queue.publish(
        0,
        [
            TimedAction(0, {"joint.pos": 0.1}, "carry"),
            TimedAction(1, {"joint.pos": 0.2}, "carry"),
            TimedAction(2, {"joint.pos": 0.3}, "carry"),
        ],
    )
    queue.take(0)
    queue.note_issued({"joint.pos": 0.1})
    snapshot = queue.snapshot()
    assert snapshot.boundary_before(1) == {"joint.pos": 0.1}
    assert snapshot.boundary_before(3) == {"joint.pos": 0.3}
    snapshot.pending[1].action["joint.pos"] = 99
    assert queue.take(1)[0].action["joint.pos"] == 0.2


def test_motor_overrun_delays_free_space_sequence_without_a_joint_jump():
    class FakeRobot:
        def get_state_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            return action

    now = [0.0]
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.31})
    queue.publish(
        0,
        [TimedAction(i, {"joint.pos": i * 0.1}, "carry", may_skip_if_late=True) for i in range(6)],
    )
    ticks = []
    loop = MotorLoop(
        FakeRobot(),
        queue,
        lambda receipt: None,
        on_tick=ticks.append,
        clock=lambda: now[0],
        sleep=lambda dt: None,
    )
    loop.start()
    assert loop.tick().tick == 0
    now[0] = 0.12
    receipt = loop.tick()
    assert receipt.tick == 3
    assert receipt.requested.action["joint.pos"] == pytest.approx(0.1)
    assert loop.metrics.skipped_ticks == 2
    assert loop.metrics.commanded == 2
    assert ticks[-1].skipped_before == 2
    now[0] = 0.134
    assert loop.tick().requested.action["joint.pos"] == pytest.approx(0.2)


def test_motor_metrics_separate_post_dispatch_callbacks_from_bus_io():
    class FakeRobot:
        def get_state_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            return action

    now = [0.0]
    queue = ActionQueue()
    queue.publish(0, [TimedAction(0, {"joint.pos": 0.1}, "carry")])

    def on_receipt(_receipt):
        now[0] += 0.02

    def on_tick(_tick):
        now[0] += 0.03

    loop = MotorLoop(
        FakeRobot(), queue, on_receipt, on_tick=on_tick, clock=lambda: now[0], sleep=lambda _dt: None
    )
    loop.start()
    loop.tick()
    assert loop.metrics.write_s == [0.0]
    assert loop.metrics.max_receipt_callback_s == pytest.approx(0.02)
    assert loop.metrics.max_tick_callback_s == pytest.approx(0.03)


def test_one_missed_motor_deadline_defers_required_samples_without_a_joint_jump():
    class FakeRobot:
        def get_state_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            return action

    now = [0.0]
    queue = ActionQueue()
    queue.publish(
        0,
        [
            TimedAction(0, {"joint.pos": 0.0}, "contact"),
            TimedAction(1, {"joint.pos": 0.5}, "contact", event_id="grasp"),
            TimedAction(2, {"joint.pos": 1.0}, "contact", interval_complete=True),
        ],
    )
    loop = MotorLoop(FakeRobot(), queue, lambda receipt: None, clock=lambda: now[0], sleep=lambda dt: None)
    loop.start()
    assert loop.tick().tick == 0
    now[0] = 0.07  # Tick 1 is missed, but only by one motor period.
    second = loop.tick()
    assert (second.tick, second.requested.event_id, second.requested.action["joint.pos"]) == (
        2,
        "grasp",
        0.5,
    )
    assert queue.snapshot().issued_events == frozenset({"grasp"})
    now[0] = 0.101
    final = loop.tick()
    assert (final.tick, final.requested.interval_complete, final.requested.action["joint.pos"]) == (
        3,
        True,
        1.0,
    )
    assert loop.metrics.skipped_ticks == 1


def test_multiple_missed_deadlines_preserve_required_contact_event_order():
    class FakeRobot:
        def get_state_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            return action

    now = [0.0]
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.1})
    queue.publish(
        0,
        [
            TimedAction(0, {"joint.pos": 0.0}, "contact"),
            TimedAction(1, {"joint.pos": 0.1}, "contact", event_id="grasp"),
            TimedAction(2, {"joint.pos": 0.2}, "contact", interval_complete=True),
        ],
    )
    loop = MotorLoop(FakeRobot(), queue, lambda receipt: None, clock=lambda: now[0], sleep=lambda dt: None)
    loop.start()
    assert loop.tick().tick == 0
    now[0] = 0.14
    grasp = loop.tick()
    assert (grasp.tick, grasp.requested.event_id, grasp.requested.action["joint.pos"]) == (
        4,
        "grasp",
        0.1,
    )
    now[0] = 0.168
    completion = loop.tick()
    assert (completion.tick, completion.requested.interval_complete) == (5, True)
    assert loop.metrics.skipped_ticks == 3


def test_motor_overrun_rejects_an_unsafe_free_space_jump():
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.1})
    queue.publish(
        0,
        [
            TimedAction(i, {"joint.pos": value}, "carry", may_skip_if_late=True)
            for i, value in enumerate((0, 0.1, 0.5))
        ],
    )
    queue.take(0)
    queue.note_issued({"joint.pos": 0})
    with pytest.raises(MissedEventError, match="would jump"):
        queue.expire_before(2)


def test_slow_state_read_delays_contact_sequence_without_dropping_its_first_sample():
    now = [0.0]

    class SlowReadRobot:
        def __init__(self):
            self.sent = []

        def get_state_observation(self):
            now[0] += 0.08
            return {"joint.pos": 0.0}

        def send_action(self, action):
            self.sent.append(action)
            return action

    robot = SlowReadRobot()
    queue = ActionQueue()
    queue.publish(0, [TimedAction(i, {"joint.pos": float(i)}, "contact") for i in range(4)])
    loop = MotorLoop(robot, queue, lambda receipt: None, clock=lambda: now[0], sleep=lambda dt: None)
    loop.start()
    receipt = loop.tick()
    assert receipt.tick == 2
    assert receipt.requested.action["joint.pos"] == 0.0
    assert robot.sent == [{"joint.pos": 0.0}]
    assert loop.metrics.skipped_ticks == 2


def test_direct_queue_take_cannot_drop_a_required_contact_sample():
    queue = ActionQueue()
    queue.publish(0, [TimedAction(i, {"joint.pos": float(i)}, "contact") for i in range(4)])
    with pytest.raises(MissedEventError, match="missed required contact sample"):
        queue.take(2)


def test_motor_tick_reads_state_and_records_accepted_action_without_camera_wait():
    class FakeRobot:
        def get_state_observation(self):
            calls.append("state")
            return {"joint.pos": 0.0}

        def send_action(self, action):
            calls.append("send")
            return {"joint.pos": min(action["joint.pos"], 1.0)}

    calls = []
    receipts = []
    ticks = []
    now = [0.0]
    queue = ActionQueue()
    queue.publish(0, [TimedAction(0, {"joint.pos": 2.0}, "move")])
    loop = MotorLoop(
        FakeRobot(),
        queue,
        receipts.append,
        on_tick=ticks.append,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
    )
    receipt = loop.tick()
    assert calls == ["state", "send"]
    assert receipt.accepted["joint.pos"] == 1.0
    assert receipts == [receipt]
    assert ticks[0].observed_at_s == 0.0
    assert loop.metrics.commanded == 1
    assert loop.tick() is None
    assert now[0] == pytest.approx(1 / 30)
    assert loop.metrics.starved == 1


def test_kinematics_planner_seeds_from_previous_solution_and_rejects_bad_path():
    class LinearKinematics:
        def __init__(self):
            self.seeds = []

        def inverse_kinematics(self, seed, target):
            self.seeds.append(float(seed[0]))
            return np.array([target[0, 3] * 100])

        def forward_kinematics(self, joints):
            return pose(joints[0] / 100)

    interval = Interval(
        "place",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "target",
        (sample(0, (0, 0, 0), (0, 0, 0)), sample(1, (0.01, 0, 0), (0, 0, 0))),
        reference_version="v1",
    )
    estimate = PoseEstimate(
        "target", pose(), 0, 0, Evidence.OBSERVED, "v1", (True, True, True), (True, True, True)
    )
    kinematics = LinearKinematics()
    limits = PlannerLimits((100.0,), 0.001, 0.01, 2.0)
    planner = JointPlanner(kinematics, ("arm",), "gripper", limits)
    actions = planner.plan(
        interval,
        start_tick=0,
        count=3,
        epoch_s=0,
        period_s=0.5,
        interval_start_s=0,
        seed_deg=np.array([0.0]),
        reference=estimate,
        attachment=None,
    )
    assert [a.action["arm.pos"] for a in actions] == pytest.approx([0, 0.5, 1.0])
    assert kinematics.seeds == pytest.approx([0, 0, 0.5])
    assert actions[-1].interval_complete
    strict = JointPlanner(kinematics, ("arm",), "gripper", PlannerLimits((0.1,), 0.001, 0.01, 2.0))
    with pytest.raises(InfeasibleTrajectoryError, match="joint speed"):
        strict.plan(
            interval,
            start_tick=0,
            count=3,
            epoch_s=0,
            period_s=0.5,
            interval_start_s=0,
            seed_deg=np.array([0.0]),
            reference=estimate,
            attachment=None,
        )


def test_planner_places_semantic_event_on_first_tick_after_demo_frame():
    class StaticKinematics:
        def inverse_kinematics(self, seed, target):
            return np.array([0.0])

        def forward_kinematics(self, joints):
            return pose()

    interval = Interval(
        "place",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "target",
        (
            SpatialSample(0.0, pose(), pose(), gripper=10.0),
            SpatialSample(
                0.05, pose(), pose(), gripper=20.0, event_id="release", requires_milestone="aligned"
            ),
            SpatialSample(0.1, pose(), pose(), gripper=20.0),
        ),
        reference_version="v1",
    )
    estimate = PoseEstimate("target", pose(), 0, 0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3)
    planner = JointPlanner(StaticKinematics(), ("arm",), "gripper", PlannerLimits((100.0,), 0.001, 0.01, 2.0))
    actions = planner.plan(
        interval,
        start_tick=0,
        count=5,
        epoch_s=0,
        period_s=1 / 30,
        interval_start_s=0,
        seed_deg=np.array([0.0]),
        reference=estimate,
        attachment=None,
    )
    assert [action.tick for action in actions if action.event_id] == [2]
    assert [action.action["gripper.pos"] for action in actions[:2]] == [10.0, 10.0]
    assert actions[2].event_id == "release"
    assert actions[2].requires_milestone == "aligned"
    assert actions[2].action["gripper.pos"] == 20.0
    assert actions[-1].interval_complete
    with pytest.raises(InfeasibleTrajectoryError, match="required semantic event was not issued"):
        planner.plan(
            interval,
            start_tick=3,
            count=1,
            epoch_s=0,
            period_s=1 / 30,
            interval_start_s=0,
            seed_deg=np.array([0.0]),
            reference=estimate,
            attachment=None,
        )
    replanned = planner.plan(
        interval,
        start_tick=3,
        count=1,
        epoch_s=0,
        period_s=1 / 30,
        interval_start_s=0,
        seed_deg=np.array([0.0]),
        reference=estimate,
        attachment=None,
        issued_events=frozenset({"release"}),
    )
    assert replanned[0].event_id is None


def test_partial_pose_is_represented_but_cannot_satisfy_strict_six_dof_goal():
    estimate = PoseEstimate(
        "held", pose(), 0, 0, Evidence.OBSERVED, "v1", (True, True, True), (True, False, False)
    )
    assert estimate.pose is not None
    assert not estimate.usable(0.1, 1.0, allow_prediction=False)
    assert estimate.usable(
        0.1,
        1.0,
        allow_prediction=False,
        translation=(True, True, True),
        rotation=(True, False, False),
    )


def test_joint_planner_uses_gripper_attachment_frame_and_checks_reference_version():
    class LinearKinematics:
        def inverse_kinematics(self, _seed, target):
            return np.array([target[0, 3] * 100])

        def forward_kinematics(self, joints):
            return pose(joints[0] / 100)

    interval = Interval(
        "place",
        MotionMode.DEMONSTRATED,
        Relation.HELD_TO_TARGET,
        "target",
        (sample(0, (0, 0, 0), (0, 0, 0), (0.1, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0), (0.1, 0, 0))),
        reference_version="target-v1",
        held_reference_version="held-v1",
    )
    target = PoseEstimate(
        "target", pose(0.4), 0, 0, Evidence.OBSERVED, "target-v1", (True, True, True), (True, True, True)
    )
    attachment = AttachmentState(pose(0.02), 0, 0, "held-v1")
    planner = JointPlanner(
        LinearKinematics(), ("arm",), "gripper", PlannerLimits((2000.0,), 0.001, 0.01, 2.0)
    )
    actions = planner.plan(
        interval,
        start_tick=0,
        count=1,
        epoch_s=0,
        period_s=1 / 30,
        interval_start_s=0,
        seed_deg=np.array([48.0]),
        reference=target,
        attachment=attachment,
    )
    assert actions[0].action["arm.pos"] == pytest.approx(48.0)
    with pytest.raises(MissingEvidenceError, match="object frame"):
        planner.plan(
            interval,
            start_tick=0,
            count=1,
            epoch_s=0,
            period_s=1 / 30,
            interval_start_s=0,
            seed_deg=np.array([48.0]),
            reference=target,
            attachment=AttachmentState(pose(0.02), 0, 0, "other"),
        )


def test_joint_planner_keeps_grasp_attachment_after_observed_pose_age_expires():
    class LinearKinematics:
        def inverse_kinematics(self, _seed, target):
            return np.array([target[0, 3] * 100])

        def forward_kinematics(self, joints):
            return pose(joints[0] / 100)

    interval = Interval(
        "carry",
        MotionMode.DEMONSTRATED,
        Relation.HELD_TO_TARGET,
        "target",
        (sample(0, (0, 0, 0), (0, 0, 0), (0.1, 0, 0)), sample(1, (0, 0, 0), (0, 0, 0), (0.1, 0, 0))),
        reference_version="target-v1",
        held_reference_version="held-v1",
    )
    reference = PoseEstimate(
        "target", pose(0.4), 2, 2, Evidence.OBSERVED, "target-v1", (True, True, True), (True, True, True)
    )
    attachment = AttachmentState(pose(0.02), 0, 0, "held-v1")
    kwargs = {
        "start_tick": 0,
        "count": 1,
        "epoch_s": 2,
        "period_s": 1 / 30,
        "interval_start_s": 2,
        "seed_deg": np.array([48.0]),
        "reference": reference,
        "attachment": attachment,
    }
    default = JointPlanner(
        LinearKinematics(),
        ("arm",),
        "gripper",
        PlannerLimits((2000.0,), 0.001, 0.01, 0.8, max_attachment_age_s=0.8),
    )
    with pytest.raises(MissingEvidenceError, match="attachment is stale"):
        default.plan(interval, **kwargs)

    separate = JointPlanner(
        LinearKinematics(),
        ("arm",),
        "gripper",
        PlannerLimits((2000.0,), 0.001, 0.01, 0.8, max_attachment_age_s=20),
    )
    assert separate.plan(interval, **kwargs)[0].action["arm.pos"] == pytest.approx(48.0)

    # A rigid grasp is an assumption until release or contradictory evidence;
    # an elapsed clock alone does not prove that the object slipped.
    persistent = JointPlanner(
        LinearKinematics(), ("arm",), "gripper", PlannerLimits((2000.0,), 0.001, 0.01, 0.8)
    )
    assert persistent.plan(interval, **kwargs)[0].action["arm.pos"] == pytest.approx(48.0)


def test_attachment_updates_from_observation_but_not_its_own_prediction():
    relation = AttachmentEstimate()
    observed = PoseEstimate(
        "held",
        pose(0.2, 0, 0),
        1.0,
        1.1,
        Evidence.OBSERVED,
        "ref1",
        (True, True, True),
        (True, True, True),
    )
    assert relation.update(pose(0.1, 0, 0), observed)
    prediction = relation.predict(pose(0.3, 0, 0), 1.2, 1.0, "held")
    assert prediction.evidence == Evidence.PREDICTED
    np.testing.assert_allclose(prediction.pose[:3, 3], [0.4, 0, 0])
    assert not relation.update(pose(0.3, 0, 0), prediction)
    assert relation.predict(pose(0.3, 0, 0), 2.2, 1.0, "held") is None
    assert relation.state(100.0, None) is not None


def test_delayed_held_observation_uses_gripper_pose_at_camera_capture():
    history = TimedEndEffectorHistory(capacity=4, max_sample_gap_s=0.04)
    history.append(1.00, pose(0.00))
    history.append(1.03, pose(0.03))
    history.append(1.06, pose(0.06))
    held = PoseEstimate(
        "held", pose(0.115), 1.015, 1.100, Evidence.OBSERVED, "ref1", (True,) * 3, (True,) * 3
    )
    attachment = AttachmentEstimate()
    assert attachment.update_from_history(history, held)
    np.testing.assert_allclose(attachment.gripper_from_object[:3, 3], [0.1, 0, 0])
    assert history.at(1.07) is None  # never extrapolate beyond the latest motor read

    sparse = TimedEndEffectorHistory(capacity=3, max_sample_gap_s=0.01)
    sparse.append(1.0, pose())
    sparse.append(1.03, pose(0.03))
    assert not AttachmentEstimate().update_from_history(sparse, held)


def test_held_attachment_uses_postgrasp_observation_and_stops_on_conflict():
    class LinearKinematics:
        calls = 0

        def forward_kinematics(self, joints):
            self.calls += 1
            return pose(joints[0])

    kinematics = LinearKinematics()
    updater = HeldAttachmentUpdater(
        "held",
        "held-v1",
        kinematics,
        ("arm",),
        history_capacity=5,
        max_motor_sample_gap_s=0.04,
        max_attachment_age_s=1.0,
        max_update_shift_m=0.01,
        max_update_turn_rad=0.1,
    )
    for time_s in (0.0, 0.03, 0.06):
        updater.on_motor_tick(
            MotorTick(0, time_s, {"arm.pos": time_s}, 0.0, {"arm.pos": time_s}, None, observed_at_s=time_s)
        )
    assert kinematics.calls == 0
    pregrasp = PoseEstimate(
        "held", pose(0.142), 0.015, 0.04, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert updater.activate(0.04, pregrasp, np.array([0.04]))
    assert updater.state(0.06) is not None
    assert not updater.has_postgrasp_observation
    assert not updater.on_pose_update(1, {"held": pregrasp})
    observed = PoseEstimate(
        "held", pose(0.147), 0.045, 0.07, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert updater.on_pose_update(2, {"held": observed})
    assert updater.has_postgrasp_observation
    assert updater.state(0.08).gripper_from_object[0, 3] == pytest.approx(0.102)
    conflicting = PoseEstimate(
        "held", pose(0.25), 0.055, 0.09, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert not updater.on_pose_update(3, {"held": conflicting})
    assert updater.state(0.09) is None
    assert "disagrees" in updater.last_rejection


def test_held_attachment_reanchors_after_repeated_image_supported_visual_correction():
    class StationaryKinematics:
        def forward_kinematics(self, _joints):
            return np.eye(4)

    xy = np.linspace(-0.1, 0.1, 11)
    points = np.array([(x, y, 1.0) for x in xy for y in xy])
    updater = HeldAttachmentUpdater(
        "held",
        "held-v1",
        StationaryKinematics(),
        ("arm",),
        history_capacity=5,
        max_motor_sample_gap_s=0.1,
        max_attachment_age_s=1.0,
        max_update_shift_m=0.015,
        max_update_turn_rad=0.1,
        object_points_object_m=points,
        min_visual_support_points=12,
        depth_occlusion_tolerance_m=0.005,
    )
    for time_s in (0.0, 0.03, 0.06, 0.09):
        updater.on_motor_tick(
            MotorTick(0, time_s, {"arm.pos": 0.0}, 0.0, {"arm.pos": 0.0}, None, observed_at_s=time_s)
        )
    pregrasp = PoseEstimate("held", pose(), 0.0, 0.0, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3)
    assert updater.activate(0.02, pregrasp, np.array([0.0]))
    intrinsics = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    mask = np.zeros((64, 64), dtype=bool)
    mask[22:43, 25:46] = True
    for frame_id, time_s in ((1, 0.04), (2, 0.07)):
        frame = RGBDFrame(
            frame_id,
            np.zeros((64, 64, 3), dtype=np.uint8),
            np.ones((64, 64), dtype=np.float32),
            intrinsics,
            1.0,
            np.eye(4),
            time_s,
            time_s,
            "calibration",
        )
        observed = PoseEstimate(
            "held", pose(0.03), time_s, time_s, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
        )
        accepted = updater.on_pose_update(frame_id, {"held": observed}, frame=frame, mask=mask)
        assert accepted == (frame_id == 2)
        if frame_id == 1:
            assert updater.state(time_s) is None
            assert not updater.on_pose_update(frame_id, {"held": observed})
            assert updater.state(time_s) is None
    assert updater.has_postgrasp_observation
    assert updater.state(0.08).gripper_from_object[0, 3] == pytest.approx(0.03)


def test_ambiguous_visual_flip_keeps_depth_supported_gripper_prediction():
    class StationaryKinematics:
        def forward_kinematics(self, _joints):
            return np.eye(4)

    xy = np.linspace(-0.1, 0.1, 11)
    points = np.array([(x, y, 1.0) for x in xy for y in xy])
    updater = HeldAttachmentUpdater(
        "held",
        "held-v1",
        StationaryKinematics(),
        ("arm",),
        history_capacity=5,
        max_motor_sample_gap_s=0.1,
        max_attachment_age_s=1.0,
        max_update_shift_m=0.015,
        max_update_turn_rad=0.1,
        object_points_object_m=points,
        min_visual_support_points=12,
        depth_occlusion_tolerance_m=0.005,
    )
    for time_s in (0.0, 0.03, 0.06, 0.09):
        updater.on_motor_tick(
            MotorTick(0, time_s, {"arm.pos": 0.0}, 0.0, {"arm.pos": 0.0}, None, observed_at_s=time_s)
        )
    pregrasp = PoseEstimate("held", pose(), 0.0, 0.0, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3)
    assert updater.activate(0.02, pregrasp, np.zeros(1))
    frame = RGBDFrame(
        1,
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.ones((64, 64), dtype=np.float32),
        np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]),
        1.0,
        np.eye(4),
        0.04,
        0.04,
        "calibration",
    )
    flipped = pose()
    flipped[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    observed = PoseEstimate(
        "held", flipped, 0.04, 0.04, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert not updater.on_pose_update(1, {"held": observed}, frame=frame, mask=np.ones((64, 64), bool))
    assert updater.state(0.04) is not None
    assert updater.visual_ambiguous
    assert updater.last_rejection.endswith("image comparison ambiguous")
    np.testing.assert_allclose(updater.last_accepted_gripper_from_object, np.eye(4))
    clear = PoseEstimate("held", pose(), 0.07, 0.07, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3)
    assert updater.on_pose_update(2, {"held": clear}, ee_at_capture=np.eye(4))
    assert not updater.visual_ambiguous
    contradicted_frame = RGBDFrame(
        3,
        frame.rgb,
        np.full((64, 64), 1.03, dtype=np.float32),
        frame.intrinsics,
        1.0,
        np.eye(4),
        0.08,
        0.08,
        "calibration",
    )
    contradicted = PoseEstimate(
        "held", flipped, 0.08, 0.08, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert not updater.on_pose_update(
        3, {"held": contradicted}, frame=contradicted_frame, mask=np.ones((64, 64), bool)
    )
    assert updater.state(0.08) is None


def test_small_held_correction_requires_reliable_depth_support():
    class StationaryKinematics:
        def forward_kinematics(self, _joints):
            return np.eye(4)

    xy = np.linspace(-0.1, 0.1, 11)
    points = np.array([(x, y, 1.0) for x in xy for y in xy])
    updater = HeldAttachmentUpdater(
        "held",
        "held-v1",
        StationaryKinematics(),
        ("arm",),
        history_capacity=5,
        max_motor_sample_gap_s=0.1,
        max_attachment_age_s=1.0,
        max_update_shift_m=0.015,
        max_update_turn_rad=0.1,
        object_points_object_m=points,
        min_visual_support_points=12,
        depth_occlusion_tolerance_m=0.005,
        observation_quality=SurfaceObservationConfig(),
    )
    assert updater.activate(
        0.02,
        PoseEstimate("held", pose(), 0.0, 0.0, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3),
        np.zeros(1),
    )
    frame = RGBDFrame(
        7,
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.full((64, 64), 1.01, dtype=np.float32),
        np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]),
        1.0,
        np.eye(4),
        0.04,
        0.04,
        "calibration",
    )
    observed = PoseEstimate(
        "held", pose(0.002), 0.04, 0.04, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert not updater.on_pose_update(
        7, {"held": observed}, frame=frame, mask=np.ones((64, 64), bool), ee_at_capture=np.eye(4)
    )
    evidence = updater.latest_visual_evidence()
    assert evidence["frame_id"] == 7
    assert evidence["verdict"] == "agree"
    assert evidence["visual"]["supported_pixels"] >= 12
    assert evidence["visual"]["depth_residual_p90_m"] > evidence["depth_tolerance_m"]
    assert evidence["visual"]["clean_depth_pixels"] >= 12
    assert evidence["visual"]["clean_depth_residual_p90_m"] > evidence["depth_tolerance_m"]
    assert not evidence["correction_accepted"]
    assert "depth p90" in evidence["rejection_reason"]
    assert updater.state(0.04) is None
    supported_frame = RGBDFrame(
        8,
        frame.rgb,
        np.ones((64, 64), dtype=np.float32),
        frame.intrinsics,
        1.0,
        np.eye(4),
        0.07,
        0.07,
        "calibration",
    )
    supported_observation = PoseEstimate(
        "held", pose(0.002), 0.07, 0.07, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert updater.on_pose_update(
        8,
        {"held": supported_observation},
        frame=supported_frame,
        mask=np.ones((64, 64), bool),
        ee_at_capture=np.eye(4),
    )
    assert updater.state(0.07) is not None
    assert updater.latest_visual_evidence()["correction_accepted"]
    updater.deactivate()
    assert updater.latest_visual_evidence() is None
    assert not updater.activate(0.02)
    assert not updater.on_pose_update(
        7, {"held": observed}, frame=frame, mask=np.ones((64, 64), bool), ee_at_capture=np.eye(4)
    )
    assert updater.latest_visual_evidence()["verdict"] == "no_gripper_prediction"
    assert not updater.latest_visual_evidence()["correction_accepted"]


def test_weak_visual_update_keeps_image_supported_gripper_prediction():
    class StationaryKinematics:
        def forward_kinematics(self, _joints):
            return np.eye(4)

    xy = np.linspace(-0.1, 0.1, 11)
    points = np.array([(x, y, 1.0) for x in xy for y in xy])
    updater = HeldAttachmentUpdater(
        "held",
        "held-v1",
        StationaryKinematics(),
        ("arm",),
        history_capacity=5,
        max_motor_sample_gap_s=0.1,
        max_attachment_age_s=1.0,
        max_update_shift_m=0.03,
        max_update_turn_rad=0.1,
        object_points_object_m=points,
        min_visual_support_points=12,
        depth_occlusion_tolerance_m=0.005,
        observation_quality=SurfaceObservationConfig(),
    )
    assert updater.activate(
        0.02,
        PoseEstimate("held", pose(), 0.0, 0.0, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3),
        np.zeros(1),
    )
    frame = RGBDFrame(
        1,
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.ones((64, 64), dtype=np.float32),
        np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]),
        1.0,
        np.eye(4),
        0.04,
        0.04,
        "calibration",
    )
    behind_surface = pose()
    behind_surface[2, 3] = 0.02
    observed = PoseEstimate(
        "held", behind_surface, 0.04, 0.04, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert not updater.on_pose_update(
        1, {"held": observed}, frame=frame, mask=np.ones((64, 64), bool), ee_at_capture=np.eye(4)
    )
    assert updater.state(0.04) is not None
    assert updater.visual_ambiguous
    np.testing.assert_allclose(updater.last_accepted_gripper_from_object, np.eye(4))
    assert not updater.latest_visual_evidence()["correction_accepted"]


def test_held_attachment_compares_physical_center_when_object_frame_rotates():
    class StationaryKinematics:
        def forward_kinematics(self, _joints):
            return np.eye(4)

    center = np.array([0.5, 0.0, 0.0])
    updater = HeldAttachmentUpdater(
        "held",
        "held-v1",
        StationaryKinematics(),
        ("arm",),
        history_capacity=5,
        max_motor_sample_gap_s=0.04,
        max_attachment_age_s=1.0,
        max_update_shift_m=0.015,
        max_update_turn_rad=np.deg2rad(15),
        object_center_object_m=center,
    )
    for time_s in (0.0, 0.03, 0.06):
        updater.on_motor_tick(
            MotorTick(0, time_s, {"arm.pos": 0.0}, 0.0, {"arm.pos": 0.0}, None, observed_at_s=time_s)
        )
    pregrasp = PoseEstimate(
        "held", np.eye(4), 0.015, 0.035, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert updater.activate(0.04, pregrasp, np.zeros(1))

    rotated = np.eye(4)
    rotated[:3, :3] = Rotation.from_euler("y", 10, degrees=True).as_matrix()
    rotated[:3, 3] = center - rotated[:3, :3] @ center
    assert np.linalg.norm(rotated[:3, 3]) > updater.max_update_shift_m
    observation = PoseEstimate(
        "held", rotated, 0.045, 0.07, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert updater.on_pose_update(1, {"held": observation})

    displaced = rotated.copy()
    displaced[0, 3] += 0.02
    observation = PoseEstimate(
        "held", displaced, 0.055, 0.08, Evidence.OBSERVED, "held-v1", (True,) * 3, (True,) * 3
    )
    assert not updater.on_pose_update(2, {"held": observation})


def test_demonstrated_attachment_events_bind_and_release_live_object():
    class LinearKinematics:
        def forward_kinematics(self, joints):
            return pose(joints[0])

    first = SpatialSample(0, pose(), pose(), event_id="close")
    last = SpatialSample(1, pose(), pose(), event_id="open")
    skill = FlashSkill(
        "demo",
        0,
        (Interval("move", MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object", (first, last)),),
        attachment_transitions=(
            AttachmentTransition("close", AttachmentAction.ATTACH, "object", "object-v1"),
            AttachmentTransition("open", AttachmentAction.DETACH, "object", "object-v1"),
        ),
    )
    updater = HeldAttachmentUpdater(
        "object",
        "object-v1",
        LinearKinematics(),
        ("arm",),
        history_capacity=8,
        max_motor_sample_gap_s=0.04,
        max_attachment_age_s=1,
        max_update_shift_m=0.01,
        max_update_turn_rad=0.1,
    )
    poses = LatestPoseStore()
    pregrasp = PoseEstimate(
        "object", pose(0.115), 0.015, 0.04, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
    )
    poses.publish((pregrasp,))
    lifecycle = AttachmentLifecycle(skill, poses, {"object": updater}, max_pregrasp_age_s=0.1)

    def tick(time_s, event_id=None):
        receipt = (
            ActionReceipt(
                0,
                time_s,
                time_s + 0.01,
                0,
                0,
                {"arm.pos": time_s},
                TimedAction(0, {"arm.pos": time_s}, "move", event_id=event_id),
                {"arm.pos": time_s},
                1,
            )
            if event_id
            else None
        )
        return MotorTick(
            0, time_s, {"arm.pos": time_s}, 0, {"arm.pos": time_s}, receipt, observed_at_s=time_s
        )

    lifecycle.on_motor_tick(tick(0))
    lifecycle.on_motor_tick(tick(0.03))
    lifecycle.on_motor_tick(tick(0.06, "close"))
    assert lifecycle.active_role == "object"
    assert lifecycle.state(0.08) is not None
    assert not updater.has_postgrasp_observation
    lifecycle.on_motor_tick(tick(0.09))
    held = PoseEstimate(
        "object", pose(0.132), 0.075, 0.1, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
    )
    assert lifecycle.on_pose_update(2, {"object": held})
    assert updater.has_postgrasp_observation
    lifecycle.on_motor_tick(tick(0.12, "open"))
    assert lifecycle.state(0.13) is None
    assert lifecycle.last_event == "open"


def test_rgbd_alignment_and_pose_handoff_preserve_source_time():
    frame = RGBDFrame(
        1,
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.full((2, 3), 1000, dtype=np.uint16),
        np.eye(3),
        0.001,
        pose(),
        1.0,
        1.05,
        "cal1",
    )
    np.testing.assert_allclose(frame.metric_depth(), 1.0)
    with pytest.raises(ValueError, match="aligned"):
        RGBDFrame(2, frame.rgb, np.zeros((3, 2), dtype=np.uint16), np.eye(3), 0.001, pose(), 1.0, 1.1, "cal1")
    store = LatestPoseStore()
    fresh = PoseEstimate(
        "target", pose(0.2), 1.0, 1.1, Evidence.OBSERVED, "ref1", (True, True, True), (True, True, True)
    )
    stale = PoseEstimate(
        "target", pose(0.1), 0.9, 1.2, Evidence.OBSERVED, "ref1", (True, True, True), (True, True, True)
    )
    assert store.publish((fresh,)) == 1
    assert store.publish((stale,)) == 1
    assert store.snapshot()[1]["target"].source_time_s == 1.0


def test_tick_record_uses_only_camera_and_pose_evidence_available_at_that_time():
    frames = LatestRGBDFrameStore()
    frames.publish(
        RGBDFrame(
            7,
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint16),
            np.eye(3),
            0.001,
            pose(),
            1.0,
            1.0,
            "camera-v1",
        )
    )
    poses = LatestPoseStore()
    poses.publish(
        (
            PoseEstimate("target", pose(), 1.0, 1.05, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3),
            PoseEstimate("held", pose(), 1.0, 1.2, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3),
        )
    )
    events = [{"type": "interval_start"}]
    now = [1.1]
    assembler = TickRecordAssembler(
        frames,
        poses,
        rgb_feature="observation.images.overhead",
        depth_feature="observation.depth.overhead",
        max_camera_age_s=0.2,
        progress_events=lambda: events,
        clock=lambda: now[0],
    )
    tick = MotorTick(0, 1.1, {"arm.pos": 0.0}, 0.001, {"arm.pos": 0.0}, None)
    recorded = assembler.assemble(tick)
    assert recorded.camera_frame_id == 7
    assert set(recorded.camera_observation) == {
        "observation.images.overhead",
        "observation.depth.overhead",
    }
    assert set(recorded.pose_estimates) == {"target"}
    assert recorded.progress_events == ({"type": "interval_start"},)
    assert assembler.assemble(tick).progress_events == ()
    now[0] = 1.3
    with pytest.raises(RuntimeError, match="no fresh aligned RGB-D"):
        assembler.assemble(tick)


def test_regular_lerobot_frame_and_flash_trace_are_written_together(tmp_path):
    class FakeDataset:
        root = tmp_path
        features = {
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        }

        def __init__(self):
            self.frames = []
            self.saved = False
            self.finalized = False

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved = True

        def finalize(self):
            self.finalized = True

    dataset = FakeDataset()
    writer = FlashEpisodeWriter(
        dataset,
        0,
        "insert object",
        artifacts={"anchor_decisions": "tracker_episode_000000.decisions.jsonl"},
    )
    worker = RecorderWorker(writer)
    worker.start()
    receipt = ActionReceipt(
        tick=1,
        scheduled_s=1.0,
        dispatched_s=1.01,
        read_duration_s=0.001,
        write_duration_s=0.002,
        observed={"joint.pos": 0.0},
        requested=TimedAction(
            1,
            {"joint.pos": 2.0},
            "grasp",
            plan_evidence=PlanEvidence(3, "target", "ref1", 5, 0.8, 0.9),
        ),
        accepted={"joint.pos": 1.0},
        revision=3,
    )
    estimate = PoseEstimate(
        "target",
        pose(0.1),
        0.9,
        1.0,
        Evidence.OBSERVED,
        "ref1",
        (True, True, True),
        (True, True, True),
        source_frame_id=7,
    )
    comparison = PoseComparison(
        PoseComparisonVerdict.AMBIGUOUS,
        0.036,
        0.2,
        SurfaceSupport(100, 40, 0.4, 0.01),
        SurfaceSupport(100, 60, 0.6, 0.06),
    )
    worker.submit(
        RecordedTick(
            receipt,
            {},
            None,
            7,
            pose_estimates={"target": estimate},
            pose_comparisons={"target": comparison},
            motor_tick=MotorTick(
                1,
                1.0,
                {"joint.pos": 0.0},
                0.001,
                {"joint.pos": 2.0},
                receipt,
                skipped_before=1,
                observed_at_s=1.001,
            ),
        )
    )
    worker.finish("unknown", "no post-release observation")
    assert dataset.saved
    assert dataset.finalized
    assert dataset.frames[0]["action"].tolist() == [2.0]
    assert dataset.frames[0]["task"] == "insert object"
    trace = (tmp_path / "flash/v1/episode_000000.jsonl").read_text()
    assert '"pose_version": 7' in trace
    assert '"plan_evidence": {"pose_revision": 3' in trace
    assert '"reference_source_frame_id": 5' in trace
    assert '"reference_version": "ref1"' in trace
    assert '"pose_comparisons": {"target": {"verdict": "ambiguous"' in trace
    assert '"skipped_before": 1' in trace
    assert '"observed_at_s": 1.001' in trace
    assert '"accepted": {"joint.pos": 1.0}' in trace
    assert '"outcome": "unknown"' in (tmp_path / "flash/v1/manifest.json").read_text()
    assert '"timing_regular": false' in (tmp_path / "flash/v1/manifest.json").read_text()
    manifest = json.loads((tmp_path / "flash/v1/manifest.json").read_text())
    assert manifest["episodes"]["0"]["artifacts"]["anchor_decisions"] == (
        "tracker_episode_000000.decisions.jsonl"
    )
    loaded = load_pose_estimates(tmp_path, 0, expected_frames=1)
    assert loaded[0]["target"].reference_version == "ref1"
    assert loaded[0]["target"].source_frame_id == 7
    np.testing.assert_allclose(loaded[0]["target"].pose, pose(0.1))
    comparisons = load_pose_comparisons(tmp_path, 0, expected_frames=1)
    assert comparisons[0]["target"].verdict == PoseComparisonVerdict.AMBIGUOUS
    assert comparisons[0]["target"].candidate.depth_residual_p90_m == pytest.approx(0.01)


def test_idle_motor_ticks_remain_in_regular_dataset_timeline(tmp_path):
    class FakeDataset:
        root = tmp_path
        features = {
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        }

        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            pass

        def finalize(self):
            pass

    class Robot:
        def get_state_observation(self):
            return {"joint.pos": 0.1}

        def send_action(self, action):
            return dict(action)

    now = [0.0]
    captured = []
    queue = ActionQueue()
    loop = MotorLoop(
        Robot(),
        queue,
        lambda _receipt: None,
        on_tick=captured.append,
        clock=lambda: now[0],
        sleep=lambda delay: now.__setitem__(0, now[0] + delay),
    )
    assert loop.tick() is None
    evidence = PlanEvidence(2, "target", "ref1", 4, 0.0, 0.0)
    queue.publish(0, [TimedAction(1, {"joint.pos": 0.8}, "move", plan_evidence=evidence)])
    assert loop.tick() is not None
    assert loop.tick() is None
    assert [tick.action_source for tick in captured] == ["observed_hold", "commanded", "held_goal"]
    dataset = FakeDataset()
    writer = FlashEpisodeWriter(dataset, 0, "motion")
    worker = RecorderWorker(writer)
    worker.start()
    for tick in captured:
        worker.submit(RecordedTick(None, {}, None, None, motor_tick=tick))
    worker.finish("unknown", "unverified")
    np.testing.assert_allclose([frame["action"].tolist() for frame in dataset.frames], [[0.1], [0.8], [0.8]])
    trace = [
        json.loads(line) for line in (tmp_path / "flash/v1/episode_000000.jsonl").read_text().splitlines()
    ]
    assert [entry["motor_tick"] for entry in trace] == [0, 1, 2]
    assert [entry["action_source"] for entry in trace] == ["observed_hold", "commanded", "held_goal"]
    assert [entry["plan_evidence"] for entry in trace] == [None, dict(vars(evidence)), dict(vars(evidence))]
