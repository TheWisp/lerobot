"""No hardware: a planned command never counts as observed arrival."""

import numpy as np
import pytest

from lerobot.gui.api._guarded_path import ARRIVAL_TIMEOUT_S, LEASE_S, GuardedPath

NAMES = ("x", "y", "z", "gripper")


class Kin:
    def forward_kinematics(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q)[:3] / 1000
        return pose


def make_path(q=None, contact=False):
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


@pytest.mark.parametrize("fault", ["temperature", "load", "missing", "nan", "joint_drift", "tip_drift"])
def test_faults_stop_before_another_path_sample(fault):
    p = make_path()
    q = p.q[0].copy()
    pose = Kin().forward_kinematics(q)
    temps, load = dict.fromkeys(NAMES, 30), dict.fromkeys(NAMES, 0)
    if fault == "temperature":
        temps["x"] = 51
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
