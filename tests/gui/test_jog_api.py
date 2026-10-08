# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jog API: the bounded pose walk, the motor->URDF conversion, and the guards that need no arm."""

import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import jog
from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES


def _pose(xyz, rot=np.eye(3)):
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = xyz
    return pose


def test_step_pose_is_bounded_per_tick_and_arrives():
    ref, target = _pose([0, 0, 0]), _pose([0.1, 0, 0])
    stepped = jog._step_pose(ref, target)
    assert np.linalg.norm(stepped[:3, 3]) == pytest.approx(jog.MAX_LINEAR_M_S / jog.HZ)
    for _ in range(200):
        ref = jog._step_pose(ref, target)
    assert np.allclose(ref, target)


def test_step_pose_bounds_rotation_and_finishes_it():
    from scipy.spatial.transform import Rotation

    target = _pose([0, 0, 0], Rotation.from_euler("z", 40, degrees=True).as_matrix())
    ref = _pose([0, 0, 0])
    stepped = jog._step_pose(ref, target)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(stepped[:3, :3]).as_rotvec()))
    assert ang == pytest.approx(math.degrees(jog.MAX_ANGULAR_RAD_S / jog.HZ))
    for _ in range(100):
        ref = jog._step_pose(ref, target)
    assert np.allclose(ref, target, atol=1e-9)


def test_urdf_rad_applies_sign_and_offset():
    q = dict.fromkeys(MOTOR_NAMES, 10.0)
    out = jog._urdf_rad(LEFT_ARM_ALIGNMENT, q)
    a = LEFT_ARM_ALIGNMENT["shoulder_lift"]
    assert out["S2"] == pytest.approx(math.radians(a.sign * 10.0 + a.offset_deg))
    assert out["S1"] == pytest.approx(math.radians(-10.0))


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(jog.router)
    return TestClient(app)


def test_guards_without_an_arm(client):
    assert client.get("/api/jog/meta").json() == {"available": False}
    assert client.get("/api/jog/state").json() == {"connected": False}
    assert (
        client.post("/api/jog/target", json={"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]}).status_code
        == 409
    )
    assert client.post("/api/jog/connect", json={"profile": "x", "arm": "middle"}).status_code == 422
    assert client.post(
        "/api/jog/connect", json={"profile": "no-such-profile-xyz", "arm": "left"}
    ).status_code in (404, 500)
    assert client.post("/api/jog/disconnect").json() == {"status": "ok"}


def test_step_pose_honours_the_speed_it_is_given():
    ref, target = _pose([0, 0, 0]), _pose([0.1, 0, 0])
    stepped = jog._step_pose(ref, target, max_linear_m_s=0.3)
    assert np.linalg.norm(stepped[:3, 3]) == pytest.approx(0.3 / jog.HZ)


def test_limits_are_validated_and_survive_without_an_arm(client):
    r = client.post("/api/jog/limits", json={"linear_mm_s": 120, "angular_deg_s": 60})
    assert r.status_code == 200
    assert r.json()["linear_mm_s"] == pytest.approx(120)
    assert r.json()["angular_deg_s"] == pytest.approx(60)
    assert client.post("/api/jog/limits", json={"linear_mm_s": 5000}).status_code == 422
    assert client.post("/api/jog/limits", json={"angular_deg_s": 0}).status_code == 422
    # A rejected value leaves the accepted ones in place.
    assert jog._jog.max_linear_m_s == pytest.approx(0.12)
    assert jog._jog.max_angular_rad_s == pytest.approx(math.radians(60))


def test_cap_rotation_stops_on_the_same_axis_at_the_cap():
    from scipy.spatial.transform import Rotation

    r_obs = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    far = _pose([0.1, 0.2, 0.3], Rotation.from_euler("x", 110, degrees=True).as_matrix())
    out, clamped = jog._cap_rotation(far, r_obs, math.radians(60))
    assert clamped
    assert np.allclose(out[:3, 3], far[:3, 3])
    rel = Rotation.from_matrix(out[:3, :3] @ r_obs.T).as_rotvec()
    assert np.degrees(np.linalg.norm(rel)) == pytest.approx(60)
    assert np.allclose(rel / np.linalg.norm(rel), [1, 0, 0])
    near = _pose([0, 0, 0], Rotation.from_euler("x", 50, degrees=True).as_matrix())
    same, clamped = jog._cap_rotation(near, r_obs, math.radians(60))
    assert not clamped and same is near


def test_rotation_cap_limit_is_validated(client):
    assert client.post("/api/jog/limits", json={"rotation_cap_deg": 45}).json()[
        "rotation_cap_deg"
    ] == pytest.approx(45)
    assert client.post("/api/jog/limits", json={"rotation_cap_deg": 200}).status_code == 422


def test_gripper_request_is_validated_and_needs_an_arm(client):
    assert client.post("/api/jog/gripper", json={"pos": 50}).status_code == 409
    assert client.post("/api/jog/gripper", json={"pos": 120}).status_code in (409, 422)


class _FakeBus:
    """The reads the jog loop makes on a Feetech bus. ``errors`` maps a motor id to the status byte of its replies."""

    model_ctrl_table = {"sts3215": {"Present_Temperature": (63, 1)}}

    def __init__(self, errors=None):
        self.motors = {m: SimpleNamespace(id=i + 1, model="sts3215") for i, m in enumerate(MOTOR_NAMES)}
        self.errors = dict(errors or {})
        self.packet_handler = SimpleNamespace(
            getRxPacketError=lambda e: f"[RxPacketError] status {e}", getTxRxResult=lambda c: f"comm {c}"
        )

    def _read(self, address, length, motor_id, **kwargs):
        return 30, 0, self.errors.get(motor_id, 0)

    def _is_comm_success(self, comm):
        return comm == 0


class _FakeRobot:
    def __init__(self, q, errors=None):
        self.q, self.sent, self.bus = dict(q), [], _FakeBus(errors)

    def get_observation(self):
        return {f"{m}.pos": v for m, v in self.q.items()}

    def send_action(self, action):
        self.sent.append(dict(action))
        return action


class _FakeKin:
    def forward_kinematics(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3], dtype=float) / 1000.0
        return pose

    def inverse_kinematics(self, seed, pose):
        return np.asarray(seed, dtype=float).copy()


def test_handing_the_arm_back_after_an_act_keeps_the_grasps_closing():
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    q["gripper"] = 80.2  # the jaws stopped by the object
    j = jog._Jog(robot=_FakeRobot(q), kin=_FakeKin(), arm="left", workspace_min=(-1.0, -1.0, -1.0))
    j.mode, j.q_target = "joints", {**q, "gripper": 80.9}  # the act's last command closes past contact
    try:
        jog._joints_stop(j)
        assert j.mode == "cartesian" and j.q_target is None
        assert j.q_cmd["gripper"] == 80.9 and j.grip_target == 80.9, (
            "the observed opening would relax the grasp"
        )
    finally:
        jog._stop_loop(j)


def test_a_gripper_squeezing_an_object_keeps_the_arm_running():
    """The gripper's overload flag on a firm grasp froze teleop mid-demo and every restart after it."""
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    gripper_id = MOTOR_NAMES.index("gripper") + 1
    robot = _FakeRobot(q, errors={gripper_id: jog.OVERLOAD_ERRBIT})
    j = jog._Jog(robot=robot, kin=_FakeKin(), arm="left", workspace_min=(-1.0, -1.0, -1.0))
    try:
        jog._restart_from_present(j)
        deadline = time.monotonic() + 3.0
        while j.ticks < 3 and not j.halted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not j.halted, j.reason
        assert j.ticks >= 3 and j.temps["gripper"] == 30
    finally:
        jog._stop_loop(j)


def test_a_slow_tick_is_kept_with_where_its_time_went():
    """A tick that holds the loop up is kept with how long each part took, so a stalled stream can be traced to the
    bus, the solve or the lock; the state shows the recent ticks' timing."""
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    robot = _FakeRobot(q)
    slow_once = [0.15]

    def get_observation():
        if slow_once:
            time.sleep(slow_once.pop())  # the encoders' read hangs once
        return {f"{m}.pos": v for m, v in robot.q.items()}

    robot.get_observation = get_observation
    j = jog._Jog(
        robot=robot,
        kin=_FakeKin(),
        arm="left",
        alignment=LEFT_ARM_ALIGNMENT,
        workspace_min=(-1.0, -1.0, -1.0),
    )
    j.ref0 = j.ref = j.target = np.eye(4)
    j.ctrl = lambda _keys: {f"{m}.pos": 0.0 for m in MOTOR_NAMES}
    j.ctrl.is_holding = False
    j.q_cmd = dict(q)
    old, jog._jog = jog._jog, j
    j.thread = threading.Thread(target=jog._loop, args=(j,), daemon=True)
    try:
        j.thread.start()
        deadline = time.monotonic() + 3.0
        while j.ticks < 10 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert j.ticks >= 10
        (slow,) = list(j.slow_ticks)
        assert slow["ms"] >= 150.0 and slow["phases_ms"]["read_positions"] >= 150.0
        assert max(v for k, v in slow["phases_ms"].items() if k != "read_positions") < 50.0
        state = jog._state_locked(j)
        assert state["ticks_ms"]["slow"] == 1 and state["ticks_ms"]["max"] >= 150.0
        assert state["slow_ticks"][-1]["phases_ms"]["read_positions"] >= 150.0
    finally:
        jog._jog = old
        jog._stop_loop(j)


def test_a_playback_sends_every_sample_in_order_and_a_slow_tick_delays_the_rest():
    """The act's stream is played back on the loop's own clock: a sample goes out no earlier than its time, one per
    tick, and a tick held up by the bus delays the samples after it instead of skipping them, as a stream set from
    another thread did (the fingers' opening and the lift away then reached the arm in one command)."""
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    robot = _FakeRobot(q)
    reads = [0]

    def get_observation():
        reads[0] += 1
        if reads[0] == 6:
            time.sleep(0.15)  # the bus holds one tick up mid-stream
        return {f"{m}.pos": v for m, v in robot.q.items()}

    robot.get_observation = get_observation
    j = jog._Jog(robot=robot, kin=_FakeKin(), arm="left", workspace_min=(-1.0, -1.0, -1.0))
    j.settle_on = False
    j.mode, j.q_target, j.q_cmd = "joints", dict(q), dict(q)
    old, jog._jog = jog._jog, j
    n, dt = 20, 1.0 / jog.HZ
    samples = [{**q, "shoulder_pan": float(k + 1)} for k in range(n)]
    due = [k * dt for k in range(n)]
    j.thread = threading.Thread(target=jog._loop, args=(j,), daemon=True)
    try:
        j.thread.start()
        t_start = time.time()
        jog.play_joints(samples, due)
        deadline = time.monotonic() + 5.0
        while jog.playback_state()[0] < n - 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        sent, at = jog.playback_state()
        assert sent == n - 1
        pans = [a["shoulder_pan.pos"] for a in robot.sent]
        went_out = [p for i, p in enumerate(pans) if p >= 1.0 and (i == 0 or p != pans[i - 1])]
        assert went_out == [float(k + 1) for k in range(n)], "every sample, in order, none skipped"
        assert all(a - t_start >= d - 0.005 for a, d in zip(at, due, strict=True)), "none before its time"
        assert at[-1] - t_start > due[-1] + 0.1, "the slow tick delayed the samples after it"
    finally:
        jog._jog = old
        jog._stop_loop(j)


def test_an_overloaded_joint_or_any_other_gripper_fault_still_stops_the_arm():
    lift_id, gripper_id = MOTOR_NAMES.index("shoulder_lift") + 1, MOTOR_NAMES.index("gripper") + 1
    with pytest.raises(RuntimeError, match="shoulder_lift"):
        jog._read_temps(_FakeBus({lift_id: jog.OVERLOAD_ERRBIT}))
    with pytest.raises(RuntimeError, match="gripper"):
        jog._read_temps(_FakeBus({gripper_id: jog.OVERLOAD_ERRBIT | 4}))


def _sticky(obs: float, cmd: float, band: float = 0.8) -> float:
    """A joint with static friction: it moves only when its goal is more than ``band`` away, and stops ``band`` short."""
    e = cmd - obs
    return cmd - math.copysign(band, e) if abs(e) > band else obs


def test_the_settle_correction_brings_a_sticking_joint_onto_its_goal():
    """At the act's poses the fingertip settled up to 10 mm off a still target: friction, not gravity."""
    goal, obs, st, prev = {"elbow_flex": 30.0}, {"elbow_flex": 29.2}, jog._Settle(), None
    for _ in range(300):
        st = jog._trim_step(st, goal, prev, obs)
        prev = goal
        obs = {"elbow_flex": _sticky(obs["elbow_flex"], goal["elbow_flex"] + st.trim["elbow_flex"])}
    assert abs(goal["elbow_flex"] - obs["elbow_flex"]) <= jog.TRIM_DEAD_DEG
    assert 0.0 < st.trim["elbow_flex"] <= jog.TRIM_MAX_DEG


def test_a_joint_that_keeps_breaking_free_past_its_goal_is_hunted_only_a_few_times():
    """Unbounded, the elbow broke free back and forth over two degrees: 73 trim changes in 7 s on the rig."""

    def breakaway(obs, cmd):  # sticks until 1.5 deg off, then slides all the way to its goal
        return cmd if abs(cmd - obs) > 1.5 else obs

    goal, obs, st, prev, trims = {"elbow_flex": 30.0}, {"elbow_flex": 29.0}, jog._Settle(), None, []
    for _ in range(300):
        st = jog._trim_step(st, goal, prev, obs)
        prev = goal
        obs = {"elbow_flex": breakaway(obs["elbow_flex"], goal["elbow_flex"] + st.trim["elbow_flex"])}
        trims.append(st.trim["elbow_flex"])
    assert st.flips["elbow_flex"] > jog.TRIM_MAX_FLIPS, (
        "this joint can never settle: each break-free overshoots"
    )
    assert len(set(trims[-200:])) == 1, "so after a few crossings its trim holds still instead of hunting"


def test_a_joint_that_creeps_off_after_settling_is_corrected_again():
    goal, st = {"shoulder_lift": -40.0}, jog._Settle()
    for _ in range(jog.TRIM_REST_TICKS + 5):  # settled: on its goal
        st = jog._trim_step(st, goal, goal, {"shoulder_lift": -40.0})
    held = st.trim.get("shoulder_lift", 0.0)
    for _ in range(20):  # then it slides a degree under its load
        st = jog._trim_step(st, goal, goal, {"shoulder_lift": -41.0})
    assert st.trim["shoulder_lift"] > held, "the drift is corrected, not ignored"


def test_the_settle_correction_waits_for_a_still_goal_and_never_winds_up():
    moved = jog._trim_step(
        jog._Settle({"elbow_flex": 1.0}, 20), {"elbow_flex": 30.5}, {"elbow_flex": 30.0}, {"elbow_flex": 29.0}
    )
    assert moved.trim == {"elbow_flex": 0.0} and moved.rest == 0, (
        "a moving goal starts over: friction flips with travel"
    )
    st = jog._Settle()
    for _ in range(jog.TRIM_REST_TICKS - 1):
        st = jog._trim_step(st, {"elbow_flex": 30.0}, {"elbow_flex": 30.0}, {"elbow_flex": 29.0})
    assert st.trim.get("elbow_flex", 0.0) == 0.0, "nothing before the goal has held still"
    for _ in range(500):  # a blocked joint: it never moves
        st = jog._trim_step(st, {"elbow_flex": 30.0}, {"elbow_flex": 30.0}, {"elbow_flex": 20.0})
    assert st.trim["elbow_flex"] == jog.TRIM_MAX_DEG


class _StickyRobot(_FakeRobot):
    """The follower with friction on every joint: the observation follows the sent goal only past a band."""

    def send_action(self, action):
        self.sent.append(dict(action))
        for m in MOTOR_NAMES:
            self.q[m] = _sticky(self.q[m], float(action[f"{m}.pos"]))
        return action


def test_the_loop_sends_a_still_goal_trimmed_and_the_joint_reaches_it():
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    goal = {**q, "elbow_flex": 30.0, "gripper": 40.0}
    robot = _StickyRobot({**q, "elbow_flex": 29.2, "gripper": 40.0})
    j = jog._Jog(robot=robot, kin=_FakeKin(), arm="left", workspace_min=(-1.0, -1.0, -1.0))
    j.mode, j.q_target, j.q_cmd = "joints", dict(goal), dict(goal)
    j.thread = threading.Thread(target=jog._loop, args=(j,), daemon=True)
    try:
        j.thread.start()
        deadline = time.monotonic() + 5.0
        while abs(robot.q["elbow_flex"] - 30.0) > jog.TRIM_DEAD_DEG and time.monotonic() < deadline:
            time.sleep(0.02)
        assert abs(robot.q["elbow_flex"] - 30.0) <= jog.TRIM_DEAD_DEG, j.settle
        assert robot.sent[-1]["elbow_flex.pos"] > 30.0, "the goal went out raised by the trim"
        assert robot.sent[-1]["gripper.pos"] == 40.0, "the gripper's goal is never trimmed"
        assert j.q_cmd["elbow_flex"] == 30.0, "the recorded command is the goal itself"
    finally:
        jog._stop_loop(j)


def test_the_gripper_letting_go_starts_the_settle_correction_over():
    """An act of 2026-10-08: the gamepad rested on the cube and held the arm 1.6 mm over its still goal, the wrist's
    trim grew to its 3 deg cap, and when the gripper opened the arm fell 7 mm, the jaws slid down around the gamepad
    and lifted it off the cube. The gripper opening clears the trim, as a moving goal does; its small corrections
    while it holds, and closing, do not."""
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    j = jog._Jog(robot=None, kin=_FakeKin(), arm="left", workspace_min=(-1.0, -1.0, -1.0))
    j.q_obs = {**q, "wrist_flex": -40.0}  # held up short of its goal by what the held object rests on
    goal = {f"{m}.pos": v for m, v in {**q, "wrist_flex": -42.0, "gripper": 82.9}.items()}
    for _ in range(jog.TRIM_REST_TICKS + 40):
        sent = jog._trimmed(j, goal)
    assert sent["wrist_flex.pos"] < -42.0 - 1.0, "precondition: the trim grew against the contact"
    sent = jog._trimmed(j, {**goal, "gripper.pos": 82.8})
    assert sent["wrist_flex.pos"] < -42.0 - 1.0, "a gripper holding on keeps the trim"
    sent = jog._trimmed(j, {**goal, "gripper.pos": 90.0})
    assert sent["wrist_flex.pos"] < -42.0 - 1.0, "closing keeps the trim"
    sent = jog._trimmed(j, {**goal, "gripper.pos": 78.0})
    assert sent["wrist_flex.pos"] == -42.0, "the gripper opening: the goal goes out untrimmed"
    assert sent["gripper.pos"] == 78.0


def test_each_joints_range_is_half_its_servos_calibrated_span_either_way():
    """The arm's degrees run from the middle of each servo's calibrated span (the bus's normalisation), so a joint
    reaches half the span either way; the servo stops it there whatever it is asked, as it stopped a wrist asked for
    -101 deg at -93. The gripper, in its own units, has no limit here; no arm, no ranges."""
    cal = {m: SimpleNamespace(range_min=1000, range_max=3000) for m in MOTOR_NAMES}
    cal["wrist_flex"] = SimpleNamespace(range_min=933, range_max=3056)
    bus = SimpleNamespace(
        calibration=cal,
        motors={m: SimpleNamespace(model="sts3215") for m in MOTOR_NAMES},
        model_resolution_table={"sts3215": 4096},
    )
    j = jog._Jog(robot=SimpleNamespace(bus=bus), kin=_FakeKin(), arm="left", workspace_min=(-1.0, -1.0, -1.0))
    old, jog._jog = jog._jog, j
    try:
        lo, hi = jog.servo_ranges()
        wf, gi = MOTOR_NAMES.index("wrist_flex"), MOTOR_NAMES.index("gripper")
        assert hi[wf] == pytest.approx(93.3, abs=0.05) and lo[wf] == -hi[wf]
        assert hi[0] == pytest.approx(1000 * 360 / 4095)
        assert np.isnan(lo[gi]) and np.isnan(hi[gi])
        jog._jog = jog._Jog()
        assert jog.servo_ranges() is None
    finally:
        jog._jog = old


def test_the_settle_correction_never_pushes_a_straining_joint_to_its_overload_trip():
    """Lifting the extended arm, the shoulder stalled short and the correction pushed it until its servo tripped."""
    goal, obs, st, prev, peak = {"shoulder_lift": 30.0}, {"shoulder_lift": 28.0}, jog._Settle(), None, 0.0
    for _ in range(300):  # the joint cannot move; its load climbs with how far its goal is pushed past it
        share = (
            200 + 150 * (goal["shoulder_lift"] + st.trim.get("shoulder_lift", 0.0) - obs["shoulder_lift"])
        ) / 800
        peak = max(peak, share)
        st = jog._trim_step(st, goal, prev, obs, {"shoulder_lift": share})
        prev = goal
    assert peak < jog.TRIM_RELEASE_LOAD, "the push stops before the servo's trip level"
    assert st.trim["shoulder_lift"] < jog.TRIM_MAX_DEG


def test_a_joint_found_straining_lets_go_of_its_trim_and_a_sticking_one_keeps_correcting():
    goal = {"shoulder_lift": 30.0}
    st = jog._Settle(trim={"shoulder_lift": 2.0}, rest=jog.TRIM_REST_TICKS)
    for _ in range(20):
        st = jog._trim_step(st, goal, goal, {"shoulder_lift": 28.0}, {"shoulder_lift": 0.95})
    assert st.trim["shoulder_lift"] < 0.1, "near its trip level the push is released"
    held = jog._trim_step(
        jog._Settle(trim={"shoulder_lift": 1.0}, rest=20),
        goal,
        goal,
        {"shoulder_lift": 28.0},
        {"shoulder_lift": 0.8},
    )
    assert held.trim["shoulder_lift"] == 1.0, "between the two levels it neither grows nor drops"
    grows = jog._trim_step(
        jog._Settle(trim={"shoulder_lift": 1.0}, rest=20),
        goal,
        goal,
        {"shoulder_lift": 28.0},
        {"shoulder_lift": 0.4},
    )
    assert grows.trim["shoulder_lift"] > 1.0, "a joint stuck at moderate load is still corrected"
