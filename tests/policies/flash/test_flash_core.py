"""Behavioral invariants for direct spatial imitation."""

import json

import numpy as np
import pytest

from lerobot.policies.flash.execution import (
    ActionQueue,
    ActionReceipt,
    MotorLoop,
    PlanRejectedError,
    TimedAction,
)
from lerobot.policies.flash.geometry import held_object_goal
from lerobot.policies.flash.perception import (
    AttachmentEstimate,
    LatestPoseStore,
    RGBDFrame,
    TimedEndEffectorHistory,
)
from lerobot.policies.flash.planner import InfeasibleTrajectoryError, JointPlanner, PlannerLimits
from lerobot.policies.flash.progress import MilestoneUpdate, Phase, Progress, Verdict
from lerobot.policies.flash.recording import (
    FlashEpisodeWriter,
    RecordedTick,
    RecorderWorker,
    load_pose_estimates,
)
from lerobot.policies.flash.skill import (
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


def test_queue_rejects_a_retarget_jump_in_robot_api_action_space():
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.5})
    queue.publish(0, [TimedAction(0, {"joint.pos": 0.2}, "carry")])
    assert queue.take(0)[0] is not None
    queue.note_issued({"joint.pos": 0.1})
    with pytest.raises(PlanRejectedError, match="jumps"):
        queue.publish(1, [TimedAction(1, {"joint.pos": 1.0}, "carry")])
    queue.publish(1, [TimedAction(1, {"joint.pos": 0.4}, "carry")])
    assert queue.take(1)[0].action["joint.pos"] == 0.4


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
    now = [0.0]
    queue = ActionQueue()
    queue.publish(0, [TimedAction(0, {"joint.pos": 2.0}, "move")])
    loop = MotorLoop(
        FakeRobot(),
        queue,
        receipts.append,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
    )
    receipt = loop.tick()
    assert calls == ["state", "send"]
    assert receipt.accepted["joint.pos"] == 1.0
    assert receipts == [receipt]
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

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved = True

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
    worker.submit(RecordedTick(receipt, {}, None, 7, pose_estimates={"target": estimate}))
    worker.finish("unknown", "no post-release observation")
    assert dataset.saved
    assert dataset.frames[0]["action"].tolist() == [2.0]
    assert dataset.frames[0]["task"] == "insert object"
    trace = (tmp_path / "flash/v1/episode_000000.jsonl").read_text()
    assert '"pose_version": 7' in trace
    assert '"reference_version": "ref1"' in trace
    assert '"accepted": {"joint.pos": 1.0}' in trace
    assert '"outcome": "unknown"' in (tmp_path / "flash/v1/manifest.json").read_text()
    loaded = load_pose_estimates(tmp_path, 0, expected_frames=1)
    assert loaded[0]["target"].reference_version == "ref1"
    np.testing.assert_allclose(loaded[0]["target"].pose, pose(0.1))


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
