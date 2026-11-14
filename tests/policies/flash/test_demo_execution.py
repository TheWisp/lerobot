"""Demo annotation -> recorded joints -> FK/IK -> bounded robot actions.

Only sensor masks and motor I/O are synthetic. The RGB-D tracker, dataset
reader, calibration conversion and SO-107 IK are the production paths.
"""

import json
from dataclasses import replace
from threading import Event
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.datasets.camera_calibration import attach_robot_geometry
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.demo_document import DemoKeyframe, DemoObject, atomic_json, load_document
from lerobot.policies.flash.demo_execution import PLAN_VERSION, DemoRunTracker, KeyframeExecutor, compile_demo
from lerobot.policies.flash.demo_live_worker import ExecutionLoop
from lerobot.policies.flash.demo_tracking import track_demo
from lerobot.policies.flash.live_surface_session import LIVE_METHOD
from lerobot.policies.flash.surface_session_config import defaults
from lerobot.robots.so107_follower_gravity import SO107FollowerGravity, SO107FollowerGravityRobotConfig
from tests.policies.flash.test_demo_tracking import Masks, rgbd_episode  # noqa: F401
from tests.policies.flash.test_shared_image_tracking import frame


@pytest.fixture
def robot(tmp_path):
    pytest.importorskip("pink")
    geometry = {
        "robot_id": "execution-test",
        "tool_point": {"offset_m": [0.02, -0.025, -0.017]},
        "cameras": {
            "camera": {
                "identity": {"type": "intelrealsense", "serial_number_or_name": "test"},
                "T_base_cam": [[0, -1, 0, 0.1], [1, 0, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
            }
        },
    }
    atomic_json(tmp_path / "geometry.json", geometry)
    names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "forearm_roll",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    atomic_json(
        tmp_path / "motors" / "execution-test.json",
        {
            name: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
            for i, name in enumerate(names)
        },
    )
    result = SO107FollowerGravity(
        SO107FollowerGravityRobotConfig(
            id="execution-test",
            port="/never/open",
            calibration_dir=tmp_path / "motors",
            geometry_path=tmp_path / "geometry.json",
            use_degrees=True,
        )
    )
    # Any accidental physical access fails at the boundary, including tests.
    result.connect = Mock(side_effect=AssertionError("physical connect in simulation"))
    result.send_action = Mock(side_effect=AssertionError("physical send in simulation"))
    return result


def ready_state(model):
    return dict(zip(model.joint_names, [0, -30, 45, 0, -30, 0], strict=True)) | {model.gripper: 90.0}


def plan_for(robot, *, reference="object_1"):
    model = robot.cartesian_model()
    state = ready_state(model)
    tool = model.forward(state)
    camera = np.asarray(robot.geometry.to_dict()["cameras"]["camera"]["T_base_cam"])
    ref = camera if reference != "world" else np.eye(4)
    return {
        "version": PLAN_VERSION,
        "base_from_camera": camera.tolist(),
        "robot_geometry": robot.geometry.to_dict(),
        "robot_cartesian": model.description(),
        "keyframes": [
            {
                "frame": 0,
                "reference": reference,
                "reference_frame": 0,
                "reference_from_tool": (np.linalg.inv(ref) @ tool).tolist(),
                "state": state,
                "gripper": 10,
                "gripper_observed": 10,
            }
        ],
    }


def evidence(pose=None, status="observed_candidate", source=1.0, available=1.0):
    return {
        "object_1": {
            "status": status,
            "pose": (np.eye(4) if pose is None else pose).tolist(),
            "source_time_s": source,
            "available_time_s": available,
        }
    }


def test_live_moved_reference_retargets_real_ik_without_connecting(robot, tmp_path):
    plan = plan_for(robot)
    loop = ExecutionLoop(plan, robot, tmp_path, Event(), simulate=True)
    # A +10 mm camera-Y move is -10 mm robot-X under this calibration.
    moved = np.eye(4)
    moved[1, 3] = 0.01
    initial_tool = loop.model.forward(loop.state)
    for i in range(150):
        now = 1 + i / 30
        loop.observations = evidence(moved, source=now, available=now)
        action = loop.tick(now, 1 / 30)
        if loop.executor.status == "complete":
            break
        if i == 0:
            assert action[loop.model.gripper] == 90  # no early grasp
    assert loop.executor.status == "complete"
    expected = initial_tool[:3, 3] + [-0.01, 0, 0]
    np.testing.assert_allclose(loop.model.forward(loop.state)[:3, 3], expected, atol=0.003)
    assert loop.state[loop.model.gripper] == 10
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


@pytest.mark.parametrize("reason", ["lost", "stale", "future", "unavailable"])
def test_missing_or_wrong_time_evidence_does_not_command_or_advance(robot, reason):
    model = robot.cartesian_model()
    executor = KeyframeExecutor(plan_for(robot), model)
    observations = evidence(
        status=reason if reason in {"lost", "predicted"} else "observed_candidate",
        source=0 if reason == "stale" else 2 if reason == "future" else 1,
        available=2 if reason in {"future", "unavailable"} else 1,
    )
    assert executor.step(1, 1 / 30, ready_state(model), observations) is None
    assert executor.index == 0
    assert executor.status.startswith("waiting")
    # Recovery follows a new accepted frame; no restarting or retry stack.
    assert executor.step(1.1, 1 / 30, ready_state(model), evidence(source=1.1, available=1.1)) is not None


def test_stop_is_terminal_even_when_evidence_returns(robot, tmp_path):
    stop = Event()
    loop = ExecutionLoop(plan_for(robot), robot, tmp_path, stop, simulate=True)
    loop.observations = evidence()
    stop.set()
    assert loop.tick(1, 1 / 30) is None
    assert loop.commands_sent == 0
    loop.executor.stop()
    stop.clear()
    assert loop.tick(1.1, 1 / 30) is None
    assert loop.commands_sent == 0


def test_world_keyframe_and_joint_speed_bound(robot):
    model = robot.cartesian_model()
    plan = plan_for(robot, reference="world")
    target = np.asarray(plan["keyframes"][0]["reference_from_tool"])
    target[0, 3] += 0.02
    plan["keyframes"][0]["reference_from_tool"] = target.tolist()
    executor = KeyframeExecutor(plan, model, joint_speed_deg_s=1)
    state = ready_state(model)
    action = executor.step(1, 1 / 30, state, {})
    assert action is not None
    assert all(abs(action[name] - state[name]) <= 1 / 30 + 1e-9 for name in model.joint_names)
    assert action[model.gripper] == state[model.gripper]


def test_ik_cannot_silently_accept_unreachable_goal(robot):
    model = robot.cartesian_model()
    state = ready_state(model)
    target = model.forward(state)
    target[0, 3] += 5
    with pytest.raises(ValueError, match="limits|reach"):
        model.inverse(state, target)


def test_stalled_ik_is_waiting_without_sending_unchanged_commands(robot):
    model = robot.cartesian_model()
    plan = plan_for(robot, reference="world")
    plan["keyframes"][0]["reference_from_tool"][0][3] += 0.01
    state = ready_state(model)
    # The IK contract allows a constrained solver to return its seed. FK and
    # the controller remain real; only that solver outcome is supplied here.
    model.inverse = Mock(return_value={name: state[name] for name in model.joint_names})
    executor = KeyframeExecutor(plan, model)
    assert executor.step(1, 1 / 30, state, {}) is None
    assert executor.index == 0
    assert executor.status == "waiting: IK made no progress (10.0 mm / 0.0° remaining)"


def test_demo_reference_is_reused_by_actual_shared_tracker(rgbd_episode, tmp_path):  # noqa: F811
    document, _ = load_document(rgbd_episode, 0)
    document.objects = [DemoObject(id="object_1", name="left"), DemoObject(id="object_2", name="right")]
    folder = tmp_path / "tracking"
    settings = defaults()
    track_demo(rgbd_episode, 0, document, folder, settings=settings, mask_source=Masks())
    plan = {"settings": settings, "document": document.model_dump(mode="json")}
    tracker = DemoRunTracker(plan, folder / "reference.npz", Masks())
    try:
        current = replace(frame(100), source_time_s=10.0, available_time_s=10.0)
        measured = tracker.measure_masks(current)
        row, _, _ = tracker.step(current, measured_masks=measured)
        with np.load(folder / "reference.npz") as saved:
            assert saved["depth"].dtype == np.float32
        for role in ("object_1", "object_2"):
            result = row["objects"][role]
            assert result[LIVE_METHOD + "_status"] == "observed_candidate"
            np.testing.assert_allclose(result[LIVE_METHOD], np.eye(4), atol=0.003)
            assert tracker.engine.candidates[LIVE_METHOD][role].reference_frame == 99
    finally:
        tracker.close()


def make_annotated_demo(robot, tmp_path, states=None):
    model = robot.cartesian_model()
    initial_state = ready_state(model)
    names = [*model.joint_names, model.gripper]
    features = {
        key: {"dtype": "float32", "shape": (7,), "names": names} for key in ("observation.state", "action")
    }
    features["observation.images.camera"] = {
        "dtype": "image",
        "shape": (16, 16, 3),
        "names": ["height", "width", "channels"],
    }
    ds = LeRobotDataset.create(
        "test/actions", root=tmp_path / "episode", fps=30, features=features, use_videos=False
    )
    attach_robot_geometry(ds.meta, robot)
    for state in states or [initial_state] * 3:
        ds.add_frame(
            {
                "task": "move",
                "observation.state": np.array([state[n] for n in names], np.float32),
                "action": np.array([state[n] if n != model.gripper else 10 for n in names], np.float32),
                "observation.images.camera": np.zeros((16, 16, 3), np.uint8),
            }
        )
    ds.save_episode()
    ds.finalize()
    ds = LeRobotDataset("test/actions", root=ds.root)
    document, _ = load_document(ds, 0)
    document.objects = [DemoObject(id=f"object_{i}", name=f"item{i}") for i in (1, 2, 3)]
    document.keyframes = [DemoKeyframe(frame=i - 1, reference=f"object_{i}") for i in (1, 2, 3)]
    folder = tmp_path / "tracked"
    atomic_json(
        folder / "input.json",
        {"document": document.model_dump(mode="json"), "settings": defaults(), "depth_source": "sensor"},
    )
    atomic_json(
        folder / "status.json", {"state": "complete", "reference_frames": {o.id: 0 for o in document.objects}}
    )
    np.savez_compressed(
        folder / "reference.npz",
        rgb=np.zeros((16, 16, 3), np.uint8),
        depth=np.ones((16, 16), np.float32),
        intrinsics=np.eye(3),
        **{obj.id: np.ones((16, 16), bool) for obj in document.objects},
    )
    for i in range(3):
        camera_pose = np.eye(4)
        camera_pose[0, 3] = i * 0.02
        atomic_json(
            folder / "frames" / f"{i:06d}.json",
            {"objects": {f"object_{i + 1}": {"status": "observed_candidate", "pose": camera_pose.tolist()}}},
        )
    return ds, document, folder


def test_record_reopen_compile_uses_actual_measured_joints_and_selected_reference(robot, tmp_path):
    ds, document, folder = make_annotated_demo(robot, tmp_path)
    model = robot.cartesian_model()
    state = ready_state(model)
    plan = compile_demo(ds, 0, document, folder, robot, "camera")
    assert len(plan["keyframes"]) == 3
    for key in plan["keyframes"]:
        np.testing.assert_allclose(
            np.asarray(key["demo_base_from_reference"]) @ key["reference_from_tool"],
            model.forward(state),
            atol=1e-8,
        )
        assert key["gripper"] == 10
    bad = folder / "frames" / "000001.json"
    packet = json.loads(bad.read_text())
    packet["objects"]["object_2"]["status"] = "lost"
    atomic_json(bad, packet)
    with pytest.raises(ValueError, match="no usable pose"):
        compile_demo(ds, 0, document, folder, robot, "camera")


def test_held_object_motion_does_not_cancel_lift_and_later_occlusion_keeps_keys(robot, tmp_path):
    model = robot.cartesian_model()
    states = [ready_state(model) for _ in range(3)]
    states[1][model.joint_names[1]] += 5
    states[2][model.joint_names[1]] += 10
    ds, document, folder = make_annotated_demo(robot, tmp_path, states)
    document.keyframes = [DemoKeyframe(frame=i, reference="object_1") for i in range(3)]
    camera = np.asarray(robot.geometry.to_dict()["cameras"]["camera"]["T_base_cam"])
    tools = [model.forward(state) for state in states]
    # Object follows the gripper perfectly. Per-frame object coordinates would
    # erase the demonstrated lift, leaving three identical tool goals.
    for i, tool in enumerate(tools):
        pose = np.linalg.inv(camera) @ tool @ np.linalg.inv(tools[0]) @ camera
        packet = evidence(pose)
        atomic_json(folder / "frames" / f"{i:06d}.json", {"objects": packet})
    plan = compile_demo(ds, 0, document, folder, robot, "camera")
    relative = [np.asarray(key["reference_from_tool"]) for key in plan["keyframes"]]
    assert np.linalg.norm(relative[2][:3, 3] - relative[0][:3, 3]) > 0.02
    for key, tool in zip(plan["keyframes"], tools, strict=True):
        np.testing.assert_allclose(camera @ key["reference_from_tool"], tool, atol=1e-8)
    # The real demo loses its ring at 601, inside a range starting at 570.
    # Later frames need robot samples, not another visible reference.
    for i in (1, 2):
        atomic_json(
            folder / "frames" / f"{i:06d}.json", {"objects": {"object_1": {"status": "lost", "pose": None}}}
        )
    occluded = compile_demo(ds, 0, document, folder, robot, "camera")
    assert occluded["keyframes"] == plan["keyframes"]


def test_range_finishes_through_occlusion_then_waits_for_new_reference(robot, tmp_path):
    from copy import deepcopy

    plan = plan_for(robot)
    first = plan["keyframes"][0]
    second, third = deepcopy(first), deepcopy(first)
    second["frame"] = 1
    second["reference_from_tool"][1][3] -= 0.015  # +15 mm robot X
    third["frame"] = third["reference_frame"] = 10  # gap: new range, same object
    plan["keyframes"] += [second, third]
    loop = ExecutionLoop(plan, robot, tmp_path, Event(), simulate=True)
    initial = loop.model.forward(loop.state)
    loop.observations = evidence()
    loop.tick(1, 1 / 30)
    acquired = deepcopy(loop.executor.evidence)
    # The object can now be hidden or moving with the tool. Neither changes
    # the range-entry reference or subtracts that movement from the goals.
    moved = np.eye(4)
    moved[0, 3] = 0.1
    for i in range(1, 150):
        now = 1 + i / 30
        loop.observations = evidence(moved, source=now, available=now) if i % 2 else {}
        if loop.executor.index == 2:
            break
        assert loop.tick(now, 1 / 30) is not None
        assert loop.executor.evidence == acquired
    assert loop.executor.index == 2
    np.testing.assert_allclose(
        loop.model.forward(loop.state)[:3, 3], initial[:3, 3] + [0.015, 0, 0], atol=0.003
    )
    loop.observations = {}
    assert loop.tick(now + 1 / 30, 1 / 30) is None
    assert loop.executor.target is None
    assert loop.executor.index == 2
    # The next range acquires a different current pose, never the old anchor.
    moved = np.eye(4)
    moved[1, 3] = -0.01  # +10 mm robot X
    loop.observations = evidence(moved, status="predicted", source=now, available=now)
    assert loop.tick(now + 2 / 30, 1 / 30) is not None
    np.testing.assert_allclose(loop.executor.target[:3, 3], initial[:3, 3] + [0.01, 0, 0], atol=1e-8)
    assert loop.executor.evidence["reference_frame"] == 10
    loop.executor.stop()
    assert loop.tick(now + 3 / 30, 1 / 30) is None
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


def test_motor_disconnect_is_independent_of_perception_shutdown(robot, tmp_path):
    """The real dispatch branch owns a fake I/O boundary, never a serial port."""
    import time

    class IO:
        geometry = robot.geometry

        def __init__(self):
            self.is_connected = False
            self.state = ready_state(robot.cartesian_model())
            self.sent = Event()
            self.disconnected = Event()
            self.calls = 0

        def cartesian_model(self):
            return robot.cartesian_model()

        def connect(self, calibrate):
            assert calibrate is False
            self.is_connected = True

        def get_state_observation(self):
            return self.state.copy()

        def send_action(self, action):
            self.calls += 1
            self.state.update(action)
            self.sent.set()

        def disconnect(self):
            self.is_connected = False
            self.disconnected.set()

    boundary = IO()
    stop = Event()
    loop = ExecutionLoop(plan_for(robot, reference="world"), boundary, tmp_path, stop, simulate=False)
    try:
        loop.start()
        assert boundary.sent.wait(2)
        stop.set()  # Same event as camera worker SIGINT/EOF; no need to wait for history flush.
        assert boundary.disconnected.wait(1)
        commands = boundary.calls
        loop.tick(time.perf_counter(), 1 / 30)
        assert boundary.calls == commands
    finally:
        loop.close()
    assert loop.error is None
    assert not boundary.is_connected
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


def test_repeated_camera_capture_keeps_robot_rows_without_retracking(rgbd_episode, tmp_path, monkeypatch):  # noqa: F811
    from lerobot.policies.flash.demo_tracking import DemoFrames

    original = DemoFrames.__iter__

    def repeated(self):
        for index, sample in enumerate(original(self)):
            yield replace(sample, source_time_s=0.0) if index == 1 else sample

    monkeypatch.setattr(DemoFrames, "__iter__", repeated)
    document, _ = load_document(rgbd_episode, 0)
    document.objects = [DemoObject(id="object_1", name="left"), DemoObject(id="object_2", name="right")]
    folder = tmp_path / "tracked"
    track_demo(rgbd_episode, 0, document, folder, mask_source=Masks())
    first = json.loads((folder / "frames/000000.json").read_text())
    repeated = json.loads((folder / "frames/000001.json").read_text())
    assert repeated == {**first, "frame": 1}
    assert repeated["observation_frame"] == 0
    assert json.loads((folder / "status.json").read_text())["done"] == 4


def test_recorded_joint_units_must_match_execution(robot, tmp_path):
    ds, document, folder = make_annotated_demo(robot, tmp_path)
    robot.config.use_degrees = False
    with pytest.raises(ValueError, match="joint units"):
        compile_demo(ds, 0, document, folder, robot, "camera")


def test_grasp_contact_uses_measured_aperture_not_unreachable_command(robot):
    model = robot.cartesian_model()
    plan = plan_for(robot, reference="world")
    plan["keyframes"][0]["gripper_observed"] = 50
    executor = KeyframeExecutor(plan, model)
    state = ready_state(model)
    state[model.gripper] = 90
    action = executor.step(1, 1 / 30, state, {})
    assert action[model.gripper] == 10
    assert executor.index == 0
    # Contact stops the motor at 50, while it keeps commanding closure to 10.
    state[model.gripper] = 50
    action = executor.step(1.1, 1 / 30, state, {})
    assert action[model.gripper] == 10
    assert executor.status == "complete"


def test_simulation_thread_never_opens_or_writes_robot(robot, tmp_path):
    stop = Event()
    loop = ExecutionLoop(plan_for(robot, reference="world"), robot, tmp_path, stop, simulate=True)
    try:
        loop.start()
        stop.wait(0.15)
    finally:
        loop.close()
    assert loop.error is None
    assert loop.commands_sent > 0
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()
