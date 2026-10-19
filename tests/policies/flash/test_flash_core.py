"""Behavioral invariants for direct spatial imitation."""

import json

import numpy as np
import pytest

from lerobot.policies.flash.attachment_lifecycle import AttachmentLifecycle
from lerobot.policies.flash.execution import (
    ActionQueue,
    ActionReceipt,
    MissedEventError,
    MotorLoop,
    MotorTick,
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
from lerobot.policies.flash.planner import InfeasibleTrajectoryError, JointPlanner, PlannerLimits
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
    FlashSkill,
    Interval,
    MilestoneSpec,
    MotionMode,
    Relation,
    SpatialSample,
)
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


def test_motor_overrun_skips_only_explicit_free_space_samples():
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
    assert loop.metrics.skipped_ticks == 2
    assert loop.metrics.commanded == 2
    assert ticks[-1].skipped_before == 2


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


def test_motor_does_not_send_contact_after_a_state_read_crosses_its_deadline():
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
    with pytest.raises(MissedEventError, match="missed required contact sample"):
        loop.tick()
    assert robot.sent == []


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
        LinearKinematics(), ("arm",), "gripper", PlannerLimits((2000.0,), 0.001, 0.01, 0.8)
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
    writer = FlashEpisodeWriter(dataset, 0, "insert object")
    worker = RecorderWorker(writer)
    worker.start()
    receipt = ActionReceipt(
        tick=1,
        scheduled_s=1.0,
        dispatched_s=1.01,
        read_duration_s=0.001,
        write_duration_s=0.002,
        observed={"joint.pos": 0.0},
        requested=TimedAction(1, {"joint.pos": 2.0}, "grasp"),
        accepted={"joint.pos": 1.0},
        revision=3,
    )
    estimate = PoseEstimate(
        "target", pose(0.1), 0.9, 1.0, Evidence.OBSERVED, "ref1", (True, True, True), (True, True, True)
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
    assert '"reference_version": "ref1"' in trace
    assert '"pose_comparisons": {"target": {"verdict": "ambiguous"' in trace
    assert '"skipped_before": 1' in trace
    assert '"observed_at_s": 1.001' in trace
    assert '"accepted": {"joint.pos": 1.0}' in trace
    assert '"outcome": "unknown"' in (tmp_path / "flash/v1/manifest.json").read_text()
    assert '"timing_regular": false' in (tmp_path / "flash/v1/manifest.json").read_text()
    loaded = load_pose_estimates(tmp_path, 0, expected_frames=1)
    assert loaded[0]["target"].reference_version == "ref1"
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
    queue.publish(0, [TimedAction(1, {"joint.pos": 0.8}, "move")])
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
