"""No hardware: a planned command never counts as observed arrival."""

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.gui.api._guarded_path import ARRIVAL_TIMEOUT_S, HOT_READS_TO_STOP, LEASE_S, GuardedPath

NAMES = ("x", "y", "z", "gripper")


class Kin:
    def forward_kinematics(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q)[:3] / 1000
        return pose


def make_path(q=None, contact=False, **kwargs):
    q = np.array(q if q is not None else [[k * 0.5, 0, 0, 80] for k in range(9)], float)
    kin = Kin()
    poses = np.stack([kin.forward_kinematics(row) for row in q])
    return GuardedPath.prepare(
        "owner",
        NAMES,
        q,
        poses,
        kin,
        (np.full(4, -90.0), np.full(4, 90.0)),
        ((-1, -1, -1), (1, 1, 1)),
        3,
        contact,
        0.0,
        **kwargs,
    )


def tick(path, t, observed, **kw):
    path.heartbeat_at = t
    return path.tick(
        t,
        dict(zip(NAMES, observed, strict=True)),
        Kin().forward_kinematics(observed),
        kw.get("load", dict.fromkeys(NAMES, 0)),
        kw.get("temps", dict.fromkeys(NAMES, 30)),
    )


def test_arrived_path_sends_all_samples_in_order_and_holds_the_grip_command():
    p = make_path()
    observed = p.q[0].copy()
    sent = []
    for i in range(40):
        command = tick(p, i * 0.11, observed)
        assert command is not None
        observed = np.array(list(command.values()))
        observed[-1] = 76  # the held object stops the fingers; this must not relax the closing command
        sent.append(p.index)
        assert command["gripper"] == 80
        if p.status == "done":
            break
    assert p.status == "done" and sorted(set(sent)) == list(range(len(p.q)))


def test_a_stuck_arm_times_out_without_advancing_to_the_end_or_skipping_the_wait():
    p = make_path()
    for t in np.arange(0, 4, 0.1):
        tick(p, t, p.q[0])
        if p.status == "failed":
            break
    assert p.status == "failed" and "did not arrive" in p.reason
    assert p.index < len(p.q) - 1 and t >= ARRIVAL_TIMEOUT_S


def test_best_effort_stops_without_freezing_when_observed_motion_stalls_inside_guard_bounds():
    p = make_path()
    p.constraint = "best_effort"
    p.index = 4
    observed = np.array([0.0, 0.0, 0.0, 80.0])
    assert tick(p, 0.0, observed) is not None
    held = tick(p, ARRIVAL_TIMEOUT_S + 0.1, observed)
    assert p.status == "done" and p.stopped_early and "no observed progress" in p.reason
    assert held == dict(zip(NAMES, observed, strict=True)), "hold the measured joints, not the blocked target"
    assert tick(p, ARRIVAL_TIMEOUT_S + 0.2, observed) == held, "handoff can verify the stopped pose"


def test_best_effort_holds_observed_joints_at_the_following_bound():
    p = make_path()
    p.constraint = "best_effort"
    p.index = 8
    observed = np.array([-1.5, 0.0, 0.0, 80.0])
    held = tick(p, 0.0, observed)
    assert p.status == "done" and p.stopped_early
    assert "following bound" in p.reason and held["x"] == observed[0]
    strict = make_path()
    strict.index = 8
    assert tick(strict, 0.0, observed) is None and strict.status == "failed"


def test_best_effort_does_not_accept_a_large_unexpected_deviation_as_progress():
    p = make_path()
    p.constraint = "best_effort"
    p.index = 8
    assert tick(p, 0.0, np.array([-12.0, 0.0, 0.0, 80.0])) is None
    assert p.status == "failed" and "following error" in p.reason


def test_a_lost_client_cannot_leave_a_path_running():
    p = make_path()
    q = dict(zip(NAMES, p.q[0], strict=True))
    assert (
        p.tick(
            LEASE_S + 0.01,
            q,
            Kin().forward_kinematics(p.q[0]),
            dict.fromkeys(NAMES, 0),
            dict.fromkeys(NAMES, 30),
        )
        is None
    )
    assert p.status == "failed" and "heartbeat" in p.reason and p.index == 0


@pytest.mark.parametrize("fault", ["load", "missing", "nan", "joint_drift", "tip_drift"])
@pytest.mark.parametrize("stream_samples", [False, True])
def test_faults_stop_before_another_path_sample(fault, stream_samples):
    p = make_path()
    p.stream_samples = stream_samples
    q = p.q[0].copy()
    pose = Kin().forward_kinematics(q)
    temps, load = dict.fromkeys(NAMES, 30), dict.fromkeys(NAMES, 0)
    if fault == "load":
        load["y"] = -700
    if fault == "missing":
        del temps["gripper"]
    if fault == "nan":
        temps["x"] = float("nan")
    if fault == "joint_drift":
        q[0] += 3.1
    if fault == "tip_drift":
        pose[0, 3] += 0.0071
    assert p.tick(0.5, dict(zip(NAMES, q, strict=True)), pose, load, temps) is None
    assert p.status == "failed" and p.index == 0
    assert tick(p, 0.6, p.q[0]) is None, "a later cool/good reading cannot restart a failed trajectory"


def test_temperature_spike_does_not_freeze_path_but_sustained_heat_does():
    p = make_path([[0, 0, 0, 80], [0, 0, 0, 80]])
    hot = dict.fromkeys(NAMES, 30)
    hot["x"] = 88
    assert tick(p, 0.01, p.q[0], temps=hot) is not None
    assert p.status != "failed"
    assert tick(p, 0.02, p.q[0]) is not None
    assert p.high_temp_streak["x"] == 0
    for i in range(HOT_READS_TO_STOP - 1):
        assert tick(p, 0.03 + i * 0.01, p.q[0], temps=hot) is not None
        assert p.status != "failed"
    assert tick(p, 0.03 + (HOT_READS_TO_STOP - 1) * 0.01, p.q[0], temps=hot) is None
    assert p.status == "failed" and "temperature" in p.reason


def test_free_space_endpoint_can_accept_a_checked_region_while_contact_stays_strict():
    q = [[0, 0, 0, 80], [0.5, 0, 0, 80], [1, 0, 0, 80]]
    strict = make_path(q)
    relaxed = make_path(q, endpoint_tolerance_mm=5, endpoint_tolerance_deg=2.5)
    for path in (strict, relaxed):
        path.index = 2
        tick(path, 0.1, [0.2, 0, 0, 80])
    assert strict.status == "running"
    assert relaxed.status == "done"
    with pytest.raises(ValueError, match="strict endpoint"):
        make_path([[0, 0, 0, 80], [0, 0, 0, 81]], endpoint_tolerance_mm=5)
    with pytest.raises(ValueError, match="following bounds"):
        make_path(q, endpoint_tolerance_mm=8)


def test_contact_closing_can_stop_on_the_object_but_opening_must_reach_its_target():
    p = make_path([[0, 0, 0, k] for k in range(75, 82)], contact=True)
    for k in range(20):
        tick(p, k * 0.11, [0, 0, 0, 76])
    assert p.status == "done", p.reason
    p = make_path([[0, 0, 0, k] for k in range(81, 74, -1)])
    for k in range(40):
        tick(p, k * 0.11, [0, 0, 0, 81])
    assert p.status == "failed" and p.index < len(p.q) - 1


@pytest.mark.parametrize("bad", ["jump", "nan", "range", "mixed", "contact_opening", "contact_transit"])
def test_invalid_segments_are_rejected_before_execution(bad):
    q = np.array([[0, 0, 0, 80], [0.5, 0, 0, 80]], float)
    contact = False
    if bad == "jump":
        q[1, 0] = 2
    if bad == "nan":
        q[1, 0] = np.nan
    if bad == "range":
        q[:, 0] = 91
    if bad == "mixed":
        q[1, 3] = 79
    if bad == "contact_opening":
        q[1] = [0, 0, 0, 79]
        contact = True
    if bad == "contact_transit":
        contact = True
    with pytest.raises(ValueError):
        make_path(q, contact)


def test_path_start_uses_fresh_observed_joints_and_keeps_the_grasp_command():
    p = make_path()
    cmd = dict(zip(NAMES, p.q[0], strict=True))
    p.check_start({**cmd, "gripper": 76}, cmd, Kin())
    with pytest.raises(ValueError, match="moved since planning"):
        p.check_start({**cmd, "x": 0.3}, cmd, Kin())
    with pytest.raises(ValueError, match="gripper command"):
        p.check_start(cmd, {**cmd, "gripper": 81}, Kin())


def test_changed_calibration_and_workspace_are_rejected():
    p = make_path()
    poses = p.poses.copy()
    poses[:, 0, 3] += 0.001
    args = ("owner", NAMES, p.q, poses, Kin(), (np.full(4, -90.0), np.full(4, 90.0)))
    with pytest.raises(ValueError, match="calibration"):
        GuardedPath.prepare(*args, ((-1, -1, -1), (1, 1, 1)), 3, False, 0)
    with pytest.raises(ValueError, match="workspace"):
        GuardedPath.prepare(*args[:3], p.poses, *args[4:], ((-1, -1, -1), (0.001, 1, 1)), 3, False, 0)


def test_done_is_rechecked_if_the_arm_moves_away():
    p = make_path([[0, 0, 0, 80], [0, 0, 0, 80]])
    tick(p, 0.1, p.q[0])
    tick(p, 0.2, p.q[0])
    assert p.status == "done"
    q = p.q[0].copy()
    q[0] = 1
    tick(p, 0.3, q)
    assert p.status == "running"
    tick(p, 3, q)
    assert p.status == "failed"


@pytest.mark.parametrize("label", ["above_objects", "drift_abort"])
@pytest.mark.parametrize("stream_samples", [False, True])
def test_recorded_reset_short_arrival_cannot_start_the_next_descent(label, stream_samples):
    record = json.loads((Path(__file__).parent / "fixtures/reset_20261012_160640_arrival.json").read_text())
    sample = next(s for s in record["samples"] if s["label"] == label)
    names = tuple(record["names"])
    q = np.asarray(sample["q_cmd"])
    goal = np.asarray(sample["pose_cmd"])
    # Probe an in-progress arrival decision at an already commanded endpoint.
    # This fixture is telemetry, not a path that may be started on hardware.
    p = GuardedPath(
        "recorded-reset",
        names,
        np.stack([q, q]),
        np.stack([goal, goal]),
        np.array([0.0, 0.1]),
        names.index("gripper"),
        False,
        0.0,
        0.0,
        0.0,
    )
    p.stream_samples = stream_samples
    p.index = 1 if stream_samples else 0
    endpoint_index = p.index
    observed = dict(zip(names, sample["q_obs"], strict=True))
    for t in (0.1, ARRIVAL_TIMEOUT_S + 0.1):
        p.heartbeat_at = t
        command = p.tick(t, observed, np.asarray(sample["pose_obs"]), sample["load"], sample["temps"])
        assert p.index == endpoint_index, "a stopped-short reset must not advance to the next lower target"
        if label == "drift_abort":
            assert command is None, "the recorded 9.6 mm following error requires an immediate stop"
    assert p.status == "failed"
    assert "did not arrive" in p.reason if label == "above_objects" else "following error" in p.reason


def test_streaming_overcomes_recorded_elbow_deadband_without_skipping_collision_samples():
    record = json.loads((Path(__file__).parent / "fixtures/reset_20261012_elbow_retreat.json").read_text())
    q, poses = np.asarray(record["q"]), np.asarray(record["poses"])
    p = GuardedPath(
        "elbow-retreat",
        tuple(record["names"]),
        q,
        poses,
        np.full(len(q), 0.1),
        len(q[0]) - 1,
        False,
        0.0,
        0.0,
        0.0,
        stream_samples=True,
    )
    observed, pose = q[0].copy(), poses[0].copy()
    sent = []
    broken_free = False
    for t in np.arange(0, 6, 0.11):
        p.heartbeat_at = t
        command = p.tick(
            t,
            dict(zip(p.names, observed, strict=True)),
            pose,
            dict.fromkeys(p.names, 0),
            dict.fromkeys(p.names, 30),
        )
        assert command is not None, p.reason
        sent.append(p.index)
        # Real probe: the elbow never moved at a 0.5 degree goal difference.
        # A simple breakaway model follows once command lead reaches 1 degree.
        broken_free |= abs(command["elbow_flex"] - q[0, 2]) >= 1
        if broken_free:
            observed, pose = q[p.index].copy(), poses[p.index].copy()
        if p.status == "done":
            break
    assert broken_free and p.status == "done"
    assert sorted(set(sent)) == list(range(len(q)))


def test_streaming_cannot_send_a_sample_beyond_the_existing_following_limit():
    p = make_path()
    p.stream_samples = True
    for t in np.arange(0, 6, 0.11):
        command = tick(p, t, p.q[0])
        if command is None:
            break
        assert command["x"] <= 3, "next command must also obey the joint following bound"
    assert p.status == "failed" and "did not arrive" in p.reason
    assert p.index < len(p.q) - 1


def test_streamed_endpoint_still_requires_physical_arrival():
    p = make_path([[k * 0.5, 0, 0, 80] for k in range(6)])
    p.stream_samples = True
    for t in np.arange(0, 5, 0.11):
        tick(p, t, p.q[0])
        assert p.status != "done"
        if p.status == "failed":
            break
    assert p.index == len(p.q) - 1
    assert p.status == "failed" and "did not arrive" in p.reason
