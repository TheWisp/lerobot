"""Demo annotation -> recorded joints -> FK/IK -> bounded robot actions.

Only sensor masks and motor I/O are synthetic. The RGB-D tracker, dataset
reader, calibration conversion and SO-107 IK are the production paths.
"""

import json
from concurrent.futures import Future
from dataclasses import replace
from threading import Event
from time import perf_counter, sleep
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.datasets.camera_calibration import attach_robot_geometry
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.demo_document import DemoKeyframe, DemoObject, atomic_json, load_document
from lerobot.policies.flash.demo_execution import (
    PLAN_VERSION,
    DemoRunTracker,
    compile_demo,
    plan_demo_range,
)
from lerobot.policies.flash.demo_live_worker import ExecutionLoop
from lerobot.policies.flash.demo_tracking import track_demo
from lerobot.policies.flash.live_surface_session import LIVE_METHOD
from lerobot.policies.flash.surface_session_config import defaults
from lerobot.robots.so107_follower_gravity import SO107FollowerGravity, SO107FollowerGravityRobotConfig
from tests.policies.flash.test_demo_tracking import Masks, rgbd_episode  # noqa: F401
from tests.policies.flash.test_shared_image_tracking import frame, masks


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


@pytest.fixture
def motor_robot(robot):
    """Real robot geometry/IK, synthetic motor boundary with readback lag."""
    state = ready_state(robot.cartesian_model())
    state[robot.cartesian_model().joint_names[0]] += 10
    io = SimpleNamespace(
        geometry=robot.geometry,
        cartesian_model=robot.cartesian_model,
        action_features=robot.action_features,
        is_connected=False,
        state=state,
        sent=[],
    )
    io.connect = Mock(side_effect=lambda **_: setattr(io, "is_connected", True))
    io.disconnect = Mock(side_effect=lambda: setattr(io, "is_connected", False))
    io.get_state_observation = lambda: dict(io.state)
    io.get_observation = io.get_state_observation

    def send(action):
        io.sent.append(dict(action))
        # Deliberately imperfect arrival: subsequent planning must use readback.
        io.state.update({name: value + 0.01 for name, value in action.items()})
        return dict(action)

    io.send_action = send
    return io


def test_every_robot_run_resets_before_planning_from_readback(motor_robot, tmp_path, monkeypatch):
    from lerobot.policies.flash import demo_live_worker as worker

    plan = plan_for(motor_robot, reference="world")
    rest = ready_state(motor_robot.cartesian_model())
    original_plan = worker.plan_demo_range
    planning_states = []
    reset_states = []

    def plan_from_readback(*args):
        planning_states.append(dict(args[3]))
        reset_states.append(list(motor_robot.sent))
        return original_plan(*args)

    monkeypatch.setattr(worker, "plan_demo_range", plan_from_readback)
    for attempt in range(2):
        motor_robot.sent.clear()
        motor_robot.state[next(iter(rest))] += 5
        initial = dict(motor_robot.state)
        stop = Event()
        monkeypatch.setattr(stop, "wait", lambda _, stop=stop: stop.is_set())
        loop = ExecutionLoop(
            plan, motor_robot, tmp_path / str(attempt), stop, simulate=False, rest_position=rest
        )
        send = motor_robot.send_action

        def check_reset_status(action, loop=loop, send=send):
            if not motor_robot.sent:
                status = json.loads((loop.directory / "execution.json").read_text())
                assert status["state"] == "resetting to rest position"
                assert status["commands"] == 0
            return send(action)

        monkeypatch.setattr(motor_robot, "send_action", check_reset_status)
        try:
            loop.start()
            deadline = perf_counter() + 5
            while loop.status not in {"complete", "failed"} and perf_counter() < deadline:
                sleep(0.01)
            assert loop.status == "complete", loop.error
            assert loop.commands_sent > 0
        finally:
            loop.close()
            monkeypatch.setattr(motor_robot, "send_action", send)
        reset = reset_states[-1]
        assert len(reset) == 150
        assert reset[-1] == rest
        assert reset[0] != initial and reset[0] != rest
        assert planning_states[-1] == {name: value + 0.01 for name, value in rest.items()}
        assert not motor_robot.is_connected
    assert len(planning_states) == 2
    assert motor_robot.connect.call_count == motor_robot.disconnect.call_count == 2


@pytest.mark.parametrize("failure", ["stop", "write", "read", "missing", "unknown", "already_stopped"])
def test_interrupted_or_invalid_reset_never_dispatches_demo(motor_robot, tmp_path, monkeypatch, failure):
    rest = ready_state(motor_robot.cartesian_model())
    stop = Event()
    monkeypatch.setattr(stop, "wait", lambda _: stop.is_set())
    if failure == "missing":
        rest = {}
    elif failure == "unknown":
        rest["unknown.pos"] = 0
    elif failure == "already_stopped":
        stop.set()
    elif failure == "read":
        monkeypatch.setattr(motor_robot, "get_state_observation", lambda: {})
    send = motor_robot.send_action

    def interrupt(action):
        if failure == "write":
            raise RuntimeError("motor write failed")
        result = send(action)
        if failure == "stop" and len(motor_robot.sent) == 3:
            stop.set()
        return result

    monkeypatch.setattr(motor_robot, "send_action", interrupt)
    loop = ExecutionLoop(
        plan_for(motor_robot, reference="world"),
        motor_robot,
        tmp_path,
        stop,
        simulate=False,
        rest_position=rest,
    )
    try:
        loop.start()
        loop.thread.join(2)
        assert not loop.thread.is_alive()
    finally:
        loop.close()
    assert loop.commands_sent == 0
    assert loop.index == 0
    assert len(motor_robot.sent) == (3 if failure == "stop" else 0)
    assert not motor_robot.is_connected
    status = json.loads((tmp_path / "execution.json").read_text())
    assert status["state"] == ("stopped" if failure in {"stop", "already_stopped"} else "failed")
    if failure in {"missing", "already_stopped"}:
        motor_robot.connect.assert_not_called()
    else:
        motor_robot.disconnect.assert_called_once()


def test_saved_rest_is_simulated_without_physical_access(robot, tmp_path, monkeypatch):
    from lerobot.policies.flash import demo_live_worker as worker

    plan = plan_for(robot, reference="world")
    rest = ready_state(robot.cartesian_model())
    rest[robot.cartesian_model().joint_names[0]] += 5
    stop = Event()
    monkeypatch.setattr(stop, "wait", lambda _: stop.is_set())
    planned_from = []
    original_plan = worker.plan_demo_range

    def plan_after_reset(*args):
        planned_from.append(dict(args[3]))
        return original_plan(*args)

    monkeypatch.setattr(worker, "plan_demo_range", plan_after_reset)
    loop = ExecutionLoop(plan, robot, tmp_path, stop, simulate=True, rest_position=rest)
    try:
        loop.start()
        deadline = perf_counter() + 5
        while loop.status not in {"complete", "failed"} and perf_counter() < deadline:
            sleep(0.01)
        assert loop.status == "complete", loop.error
    finally:
        loop.close()
    assert planned_from == [rest]
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


def test_live_moved_reference_retargets_real_ik_without_connecting(robot, tmp_path):
    plan = plan_for(robot)
    loop = ExecutionLoop(plan, robot, tmp_path, Event(), simulate=True)
    # A +10 mm camera-Y move is -10 mm robot-X under this calibration.
    moved = np.eye(4)
    moved[1, 3] = 0.01
    initial_tool = loop.model.forward(loop.state)
    now = perf_counter()
    loop.observations = evidence(moved, source=now, available=now)
    try:
        loop.start()
        deadline = perf_counter() + 3
        while loop.status != "complete" and perf_counter() < deadline:
            sleep(0.01)
        assert loop.status == "complete", loop.error
    finally:
        loop.close()
    expected = initial_tool[:3, 3] + [-0.01, 0, 0]
    np.testing.assert_allclose(loop.model.forward(loop.state)[:3, 3], expected, atol=0.005)
    assert loop.state[loop.model.gripper] == 10
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


def test_target_needs_a_pose_but_not_an_arbitrary_age_limit(robot):
    model = robot.cartesian_model()
    with pytest.raises(ValueError, match="needs a pose"):
        plan_demo_range(plan_for(robot), model, 0, ready_state(model), {}, 100)
    old = evidence(source=1, available=2)
    assert plan_demo_range(plan_for(robot), model, 0, ready_state(model), old, 100).actions


def test_stop_is_terminal_even_when_evidence_returns(robot, tmp_path):
    stop = Event()
    loop = ExecutionLoop(plan_for(robot), robot, tmp_path, stop, simulate=True)
    stop.set()
    loop.start()
    loop.thread.join(1)
    assert loop.commands_sent == 0
    loop.close()
    assert loop.commands_sent == 0


def test_world_keyframe_uses_robot_owned_ik_and_opens_gripper_during_entry(robot):
    model = robot.cartesian_model()
    plan = plan_for(robot, reference="world")
    target = np.asarray(plan["keyframes"][0]["reference_from_tool"])
    target[0, 3] += 0.02
    plan["keyframes"][0]["reference_from_tool"] = target.tolist()
    state = ready_state(model)
    planned = plan_demo_range(plan, model, 0, state, {}, 1)
    assert planned.actions[0].interval.startswith("entry:")
    assert planned.actions[0].action[model.gripper] > 10
    assert planned.actions[-1].action[model.gripper] == 10
    assert planned.actions[-1].interval_complete
    assert planned.entry_speed_m_s > 0


def test_robot_ik_returns_bounded_approximation_without_editor_pose_gate(robot):
    model = robot.cartesian_model()
    state = ready_state(model)
    target = model.forward(state)
    target[0, 3] += 5
    action = model.inverse(state, target)
    q = model.degrees(action)
    assert np.isfinite(q).all()
    assert np.all(q >= model.lower_deg) and np.all(q <= model.upper_deg)


def test_preflight_rejects_nonrigid_observed_reference(robot):
    model = robot.cartesian_model()
    pose = np.eye(4)
    pose[0, 0] = 2
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        plan_demo_range(plan_for(robot), model, 0, ready_state(model), evidence(pose), 1)


def test_target_update_rewrites_only_unissued_motor_commands(robot, tmp_path):
    from copy import deepcopy

    plan = plan_for(robot)
    key0 = plan["keyframes"][0]
    key1, key2 = deepcopy(key0), deepcopy(key0)
    key1["frame"] = 1
    key2["frame"] = 2
    key1["reference_from_tool"][0][3] += 0.002
    key2["reference_from_tool"][0][3] += 0.004
    plan["keyframes"] += [key1, key2]
    loop = ExecutionLoop(plan, robot, tmp_path, Event(), simulate=True)
    loop._motor = Mock()
    loop._motor.metrics.ticks = 0
    now = perf_counter()
    first = plan_demo_range(plan, loop.model, 0, loop.state, evidence(source=now), now)
    future = Future()
    future.set_result(first)
    loop._future = future
    loop._publish_ready()
    original = loop._queue.snapshot().pending
    assert [a.interval for a in original if a.interval.startswith("demo:")] == ["demo:0", "demo:1", "demo:2"]
    for tick, _command in enumerate(original):
        issued, _ = loop._queue.take(tick)
        loop._queue.note_issued(issued.action)
        loop.state.update(issued.action)
        if issued.interval == "demo:0":
            break
    issued_tick = loop._queue.snapshot().last_tick
    loop.index = 1
    moved = np.eye(4)
    moved[1, 3] = 0.01
    loop.observations = evidence(moved, source=now + 1)
    assert loop._target_changed(loop.observations)
    replacement = plan_demo_range(plan, loop.model, loop.index, loop.state, loop.observations, now + 1)
    future = Future()
    future.set_result(replacement)
    loop._future = future
    loop._motor.metrics.ticks = issued_tick + 1
    loop._publish_ready()
    pending = loop._queue.snapshot().pending
    assert pending[0].tick == issued_tick + 1
    assert [a.interval for a in pending if a.interval.startswith("demo:")] == ["demo:1", "demo:2"]
    assert loop._queue.snapshot().last_tick == issued_tick
    assert pending[-1].action != original[-1].action
    loop._planner.shutdown(wait=True)


def test_camera_updates_do_not_restart_a_partly_sent_approach(robot, tmp_path):
    plan = plan_for(robot)
    plan["keyframes"][0]["reference_from_tool"][0][3] += 0.02
    loop = ExecutionLoop(plan, robot, tmp_path, Event(), simulate=True)
    loop._motor = Mock()
    loop._motor.metrics.ticks = 0
    now = perf_counter()
    first = plan_demo_range(plan, loop.model, 0, loop.state, evidence(source=now), now)
    future = Future()
    future.set_result(first)
    loop._future = future
    loop._publish_ready()
    old_entries = sum(a.interval.startswith("entry:") for a in loop._queue.snapshot().pending)
    assert old_entries > 2
    issued, _ = loop._queue.take(0)
    loop._queue.note_issued(issued.action)
    loop.state.update(issued.action)
    moved = np.eye(4)
    moved[1, 3] = 0.005
    updated = plan_demo_range(plan, loop.model, 0, loop.state, evidence(moved), now + 1)
    future = Future()
    future.set_result(updated)
    loop._future = future
    loop._motor.metrics.ticks = 1
    loop._publish_ready()
    new_entries = sum(a.interval.startswith("entry:") for a in loop._queue.snapshot().pending)
    assert new_entries == old_entries - 1
    loop._planner.shutdown(wait=True)


def test_running_simulation_retargets_while_motor_queue_keeps_moving(robot, tmp_path):
    plan = plan_for(robot)
    plan["keyframes"][0]["reference_from_tool"][0][3] += 0.04
    loop = ExecutionLoop(plan, robot, tmp_path, Event(), simulate=True)
    now = perf_counter()
    loop.observations = evidence(source=now, available=now)
    try:
        loop.start()
        deadline = perf_counter() + 4
        while loop.commands_sent < 3 and perf_counter() < deadline:
            sleep(0.01)
        assert loop.commands_sent >= 3
        moved = np.eye(4)
        moved[1, 3] = 0.01
        loop.observations = evidence(moved, source=perf_counter())
        while loop.status not in ("complete", "failed") and perf_counter() < deadline:
            sleep(0.01)
        assert loop.status == "complete", loop.error
        assert loop._revision >= 2  # Initial queue and at least one replacement.
        assert loop.evidence["base_from_reference"] == (np.asarray(plan["base_from_camera"]) @ moved).tolist()
    finally:
        loop.close()
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


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
        for role in ("object_1", "object_2"):
            transferred = tracker.engine.candidates[LIVE_METHOD][role]
            assert transferred.accepted_frame is None
            assert transferred.accepted_observation is None
            assert transferred.initial_appearance_view is not None
            assert len(transferred.store.points) > 0
        row, _, _ = tracker.step(current, measured_masks=measured)
        with np.load(folder / "reference.npz") as saved:
            assert saved["depth"].dtype == np.float32
        for role in ("object_1", "object_2"):
            result = row["objects"][role]
            assert result[LIVE_METHOD + "_status"] == "observed_candidate"
            np.testing.assert_allclose(result[LIVE_METHOD], np.eye(4), atol=0.003)
            assert tracker.engine.candidates[LIVE_METHOD][role].reference_frame == 99
            assert tracker.engine.candidates[LIVE_METHOD][role].accepted_frame == 100
    finally:
        tracker.close()


@pytest.mark.parametrize("valid_live_depth", [True, False])
def test_demo_live_scene_initializes_from_current_mask_after_large_move(tmp_path, valid_live_depth):
    reference = frame(0)
    old_mask = masks()["left"]
    yy, xx = np.indices(old_mask.shape)
    shaped_depth = 850 + 0.2 * (xx - 30) + 0.03 * (xx - 30) * (yy - 40) + 0.012 * (xx - 30) ** 2
    old_depth = reference.depth.copy()
    old_depth[old_mask] = shaped_depth[old_mask]
    np.savez_compressed(
        tmp_path / "reference.npz",
        rgb=reference.rgb,
        depth=old_depth.astype(np.float32) * 0.001,
        intrinsics=reference.intrinsics,
        object_1=old_mask,
    )
    moved_mask = np.zeros_like(old_mask)
    moved_mask[:, 12:] = old_mask[:, :-12]
    new_depth = reference.depth.copy()
    new_depth[:, 12:] = old_depth[:, :-12]
    if not valid_live_depth:
        new_depth[moved_mask] = 0
    current = replace(
        frame(100),
        rgb=np.zeros_like(reference.rgb),
        depth=new_depth,
        source_time_s=10.0,
        available_time_s=10.0,
    )

    class MovedMasks:
        prompts = {"object_1": "left"}

        def masks(self, _frame):
            return {"object_1": moved_mask}

    tracker = DemoRunTracker({"settings": defaults()}, tmp_path / "reference.npz", MovedMasks())
    try:
        measured = tracker.measure_masks(current)
        transferred = tracker.engine.candidates[LIVE_METHOD]["object_1"]
        assert transferred.accepted_frame is None
        assert transferred.accepted_observation is None
        row, _, _ = tracker.step(current, measured_masks=measured)
        result = row["objects"]["object_1"]
        if valid_live_depth:
            assert result[LIVE_METHOD + "_status"] == "observed_candidate", result[LIVE_METHOD + "_details"]
            assert result[LIVE_METHOD + "_details"]["hint_source"] == "demo_live_mask_center"
            assert abs(result[LIVE_METHOD][0][3] - 0.108) < 0.025
            assert transferred.accepted_frame == 100
            assert transferred.accepted_observation["frame_id"] == 100
        else:
            assert result[LIVE_METHOD + "_status"] != "observed_candidate"
            assert transferred.accepted_frame is None
    finally:
        tracker.close()


def test_new_scene_visible_surface_is_not_vetoed_by_demo_background(tmp_path):
    demo = frame(0)
    demo_mask = np.zeros(demo.depth.shape, bool)
    demo_mask[20:70, 20:50] = True
    live_mask = demo_mask.copy()
    live_mask[20:70, 50:65] = True  # more of the same object is visible now
    demo_depth = np.full(demo.depth.shape, 1000, np.uint16)
    demo_depth[demo_mask] = 850
    live_depth = np.full(demo.depth.shape, 1000, np.uint16)
    live_depth[live_mask] = 850
    np.savez_compressed(
        tmp_path / "reference.npz",
        rgb=demo.rgb,
        depth=demo_depth.astype(np.float32) * 0.001,
        intrinsics=demo.intrinsics,
        object_1=demo_mask,
    )

    class LiveMasks:
        prompts = {"object_1": "toy"}

        def masks(self, _frame):
            return {"object_1": live_mask}

    tracker = DemoRunTracker({"settings": defaults()}, tmp_path / "reference.npz", LiveMasks())
    try:
        current = replace(frame(100), depth=live_depth, source_time_s=10.0, available_time_s=10.0)
        row, _, _ = tracker.step(current)
        result = row["objects"]["object_1"]
        assert result[LIVE_METHOD + "_status"] == "observed_candidate", result[LIVE_METHOD + "_details"]
        track = tracker.engine.candidates[LIVE_METHOD]["object_1"]
        assert track.accepted_frame == 100
        assert track.accepted_observation["frame_id"] == 100
        # The new lobe is in front of the demo's background. The demo view may
        # guide registration but must not become a retained live-scene veto.
        next_frame = replace(frame(101), depth=live_depth, source_time_s=10.1, available_time_s=10.1)
        next_row, _, _ = tracker.step(next_frame)
        assert next_row["objects"]["object_1"][LIVE_METHOD + "_status"] == "observed_candidate"
        assert track.accepted_frame == 101
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


def test_rest_keyframe_uses_recorded_approach_speed(robot, tmp_path):
    model = robot.cartesian_model()
    states = [ready_state(model) for _ in range(3)]
    states[1][model.joint_names[1]] += 4
    states[2][model.joint_names[1]] += 8
    ds, document, folder = make_annotated_demo(robot, tmp_path, states)
    document.keyframes = [
        DemoKeyframe(frame=0, reference="world", label="Rest"),
        DemoKeyframe(frame=2, reference="object_1", label="Pre-grasp"),
    ]
    atomic_json(
        folder / "frames/000002.json",
        {"objects": {"object_1": {"status": "observed_candidate", "pose": np.eye(4).tolist()}}},
    )
    plan = compile_demo(ds, 0, document, folder, robot, "camera")
    points = [model.forward(state)[:3, 3] for state in states]
    expected_path = sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:], strict=False))
    expected_speed = expected_path / (2 / 30)
    assert plan["keyframes"][1]["entry_demo_speed_m_s"] == pytest.approx(expected_speed)
    planned = plan_demo_range(plan, model, 1, states[0], evidence(source=1, available=1), 1)
    assert planned.entry_speed_m_s == pytest.approx(expected_speed)


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
    second["reference_from_tool"][1][3] -= 0.003  # +3 mm robot X in one 30 Hz tick
    third["frame"] = third["reference_frame"] = 10  # gap: new range, same object
    plan["keyframes"] += [second, third]
    model = robot.cartesian_model()
    initial = model.forward(ready_state(model))
    first_range = plan_demo_range(plan, model, 0, ready_state(model), evidence(), 1)
    assert first_range.next_index == 2
    acquired = deepcopy(first_range.evidence)
    assert first_range.actions[-1].action[model.gripper] == 10
    # The reference is captured once. Occlusion and later movement do not
    # change the preflighted 30 Hz commands inside this demonstrated range.
    moved = np.eye(4)
    moved[0, 3] = 0.1
    assert first_range.evidence == acquired
    np.testing.assert_allclose(first_range.target[:3, 3], initial[:3, 3] + [0.003, 0, 0], atol=0.003)
    with pytest.raises(ValueError, match="needs a pose"):
        plan_demo_range(plan, model, 2, ready_state(model), {}, 1)
    moved = np.eye(4)
    moved[1, 3] = -0.01
    second = plan_demo_range(plan, model, 2, ready_state(model), evidence(moved, status="predicted"), 1)
    np.testing.assert_allclose(second.target[:3, 3], initial[:3, 3] + [0.01, 0, 0], atol=1e-8)
    assert second.evidence["reference_frame"] == 10
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


def test_motor_disconnect_is_independent_of_perception_shutdown(robot, tmp_path, monkeypatch):
    """The real dispatch branch owns a fake I/O boundary, never a serial port."""

    class IO:
        geometry = robot.geometry
        action_features = robot.action_features

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

        def get_observation(self):
            return self.get_state_observation()

        def send_action(self, action):
            self.calls += 1
            self.state.update(action)
            self.sent.set()

        def disconnect(self):
            self.is_connected = False
            self.disconnected.set()

    boundary = IO()
    stop = Event()
    monkeypatch.setattr(stop, "wait", lambda _: stop.is_set())
    loop = ExecutionLoop(
        plan_for(robot, reference="world"),
        boundary,
        tmp_path,
        stop,
        simulate=False,
        rest_position=ready_state(robot.cartesian_model()),
    )
    try:
        loop.start()
        assert boundary.sent.wait(2)
        deadline = perf_counter() + 2
        while loop.commands_sent == 0 and perf_counter() < deadline:
            sleep(0.01)
        assert loop.commands_sent > 0, loop.error
        stop.set()  # Same event as camera worker SIGINT/EOF; no need to wait for history flush.
        assert boundary.disconnected.wait(1)
        commands = boundary.calls
        sleep(0.05)
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


def test_gripper_command_follows_entry_motion_without_waiting_for_contact_aperture(robot):
    model = robot.cartesian_model()
    plan = plan_for(robot, reference="world")
    plan["keyframes"][0]["gripper_observed"] = 50
    state = ready_state(model)
    state[model.gripper] = 90
    planned = plan_demo_range(plan, model, 0, state, {}, 1)
    entry = [
        action.action[model.gripper] for action in planned.actions if action.interval.startswith("entry:")
    ]
    assert entry[0] == 90
    assert entry[-1] == 10
    assert all(a >= b for a, b in zip(entry, entry[1:], strict=False))
    assert planned.actions[-1].action[model.gripper] == 10
    assert planned.actions[-1].interval_complete


def test_simulation_thread_never_opens_or_writes_robot(robot, tmp_path):
    stop = Event()
    loop = ExecutionLoop(plan_for(robot, reference="world"), robot, tmp_path, stop, simulate=True)
    try:
        loop.start()
        deadline = perf_counter() + 2
        while loop.status != "complete" and perf_counter() < deadline:
            sleep(0.01)
    finally:
        loop.close()
    assert loop.error is None
    assert loop.commands_sent > 0
    assert len(loop._completed_ranges) == 1
    assert loop._completed_ranges[0]["recorded_demo_duration_s"] == 0
    assert loop._completed_ranges[0]["commanded_demo_duration_s"] >= 0
    trace = json.loads((tmp_path / "motor_trace.json").read_text())
    assert trace["total_ticks"] >= loop.commands_sent > 0
    assert any(sample["requested"] and sample["accepted"] for sample in trace["samples"])
    assert all(sample["observed_at_s"] is not None for sample in trace["samples"])
    assert all(model_name in trace["samples"][-1]["observed"] for model_name in loop.model.joint_names)
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


def test_range_uses_final_motor_readback_without_cartesian_settle_gate(robot, tmp_path, monkeypatch):
    class LaggingRobot(ExecutionLoop._SimulationRobot):
        def send_action(self, action):
            super().send_action(action)
            self.state["wrist_flex.pos"] = action["wrist_flex.pos"] - 5.0
            return action

    monkeypatch.setattr(ExecutionLoop, "_SimulationRobot", LaggingRobot)
    plan = plan_for(robot, reference="world")
    loop = ExecutionLoop(plan, robot, tmp_path, Event(), simulate=True)
    try:
        loop.start()
        deadline = perf_counter() + 2
        while loop.status != "complete" and perf_counter() < deadline:
            sleep(0.01)
        assert loop.status == "complete"
        target = np.asarray(plan["keyframes"][0]["reference_from_tool"])
        actual = robot.cartesian_model().forward(loop.state)
        assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) > 0.003
    finally:
        loop.close()
