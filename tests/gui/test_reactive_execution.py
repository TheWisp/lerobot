"""Live target/attachment corrections must reach the bus without replaying contact events."""

import asyncio
from collections import deque

import numpy as np
import pytest

from lerobot.gui.api import jog
from lerobot.gui.api.manipulation import live_execution, runtime
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
from lerobot.showservo import motion as core, reactive
from tests.gui.test_stream_objects import _TipKinematics, fake_playback


def pose(x):
    t = np.eye(4)
    t[0, 3] = x
    return t


@pytest.mark.parametrize("change", ["target", "attachment", "both"])
def test_update_during_contact_retargets_remaining_phase_and_preserves_release(monkeypatch, change):
    kin = _TipKinematics()
    state = {"q": np.array([0.0, 0, 0, 0, 0, 0, 80]), "motion": np.eye(4), "fix": np.eye(4)}
    commands, phases = [], []

    def send(q):
        state["q"] = np.array([q[m] for m in MOTOR_NAMES])
        commands.append(state["q"].copy())

    fake_playback(monkeypatch, send)
    monkeypatch.setattr(runtime, "ACT_TICK_S", 0.001)
    monkeypatch.setattr(jog, "servo_ranges", lambda: None)
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1, -1, -1), (1, 1, 1)))
    monkeypatch.setattr(
        jog,
        "current_tip_and_anchor",
        lambda: (
            kin.forward_kinematics(state["q"]),
            np.eye(4),
            dict(zip(MOTOR_NAMES, state["q"], strict=True)),
        ),
    )
    local = np.stack([pose(0), pose(0.002), pose(0.004), pose(0.006), pose(0.006), pose(0.008)])
    grips = np.array([80, 80, 80, 80, 20, 20])
    plan = {
        "q": np.tile(state["q"], (6, 1)),
        "times": np.arange(6) / jog.HZ,
        "stage": ["contact"] * 5 + ["withdraw"],
        "live_path": {
            "local": local,
            "frames": ["target"] * 6,
            "held": np.ones(6, bool),
            "grips": grips,
            "hints": np.zeros((6, 7)),
            "floor": np.full(6, -1.0),
        },
    }

    def observed(phase, q, at, goal):
        phases.append(phase)
        if phase == 2:
            if change in ("target", "both"):
                state["motion"] = pose(0.007)
            if change in ("attachment", "both"):
                state["fix"] = pose(-0.003)

    async def reselect(*args):
        pytest.fail("reachable correction must not search an alternative")

    reason, last = asyncio.run(
        live_execution.stream(
            plan,
            kin,
            lambda: {"target": state["motion"]},
            lambda: state["fix"],
            lambda: "",
            observed,
            reselect,
            (0.12, np.pi),
            1,
        )
    )
    assert not reason
    offset = state["motion"][0, 3] + state["fix"][0, 3]
    assert kin.forward_kinematics(last)[0, 3] == pytest.approx(0.008 + offset, abs=core.ACT_SOLVE_TOL_M)
    assert phases == sorted(phases) and set(phases) == set(range(6))
    first_open = next(q for q in commands if q[-1] < 80)
    assert kin.forward_kinematics(first_open)[0, 3] == pytest.approx(0.006 + offset, abs=core.ACT_SOLVE_TOL_M)
    assert commands[-1][-1] == 20, "release happens before withdrawal, even after inserted correction samples"
    assert max(np.linalg.norm(np.diff(np.array(commands)[:, :3], axis=0), axis=1)) <= 4.01


def test_cas_replaces_only_unsent_command_and_preserves_lag_guard(monkeypatch):
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    j = jog._Jog(robot=object(), mode="joints", q_target=q.copy(), q_obs=q.copy())
    monkeypatch.setattr(jog, "_jog", j)
    jog.play_joints([q], [0.0])
    j.playback_sent, j.playback_sent_at = 0, [10.0]
    j.playback_blocked, j.playback_held = 12.0, 11.0
    clock = j.playback_t0
    assert jog.update_playback_next(0, {**q, "gripper": 80}, 0.1)
    assert jog.update_playback_next(0, {**q, "gripper": 70}, 0.1)
    assert len(j.playback) == 2 and j.playback[-1]["gripper"] == 70
    assert (j.playback_t0, j.playback_blocked, j.playback_held) == (clock, 12.0, 11.0)
    j.playback_sent = 1
    assert not jog.update_playback_next(0, {**q, "gripper": 20}, 0.1)
    assert j.playback[-1]["gripper"] == 70


def test_empty_live_queue_cannot_reset_hard_lag_timeout():
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    target = {**q, "shoulder_lift": jog.PLAYBACK_STOP_LAG_DEG + 1}
    j = jog._Jog(q_obs=q, q_target=target, playback_live=True)
    j.playback, j.playback_due, j.playback_sent = [target], [0.0], 0
    jog._advance_playback(j, 10.0)
    jog._advance_playback(j, 10.0 + jog.PLAYBACK_HOLD_MAX_S)
    assert j.halted and "behind" in j.reason
    assert j.playback_sent == 0


def test_camera_fk_requires_bracketed_encoder_samples_and_rejects_serial_stall(monkeypatch):
    j = jog._Jog(pose_history=deque([(1.0, {"joint": 0.0}), (1.04, {"joint": 8.0}), (1.5, {"joint": 20.0})]))
    monkeypatch.setattr(jog, "_jog", j)
    assert jog.joints_at(1.02, 0.07)["joint"] == pytest.approx(4.0)
    assert jog.joints_at(0.99, 0.07) is None
    assert jog.joints_at(1.51, 0.07) is None
    assert jog.joints_at(1.3, 0.07) is None


def test_recomposition_removes_old_attachment_and_respects_world_and_unheld_frames():
    path = {
        "local": np.stack([pose(0.1)] * 3),
        "frames": ["target", "world", "target"],
        "held": [True, True, False],
    }
    out = reactive.targets(path, {"target": pose(0.02), "world": np.eye(4)}, pose(-0.006))
    np.testing.assert_allclose(out[:, 0, 3], [0.114, 0.094, 0.12])


def test_changed_target_can_select_only_a_permitted_turn_with_checked_entry():
    from tests.gui.test_stream_objects import _TipYawKinematics

    kin = _TipYawKinematics()
    q = np.array([0.0, 0, 0, 0, 0, 0, 80])
    path = {
        "local": np.stack([pose(0.01), pose(0.011)]),
        "frames": ["receiver"] * 2,
        "held": [True, True],
        "grips": np.full(2, 80.0),
        "hints": np.zeros((2, 7)),
        "floor": np.full(2, -1.0),
    }
    lo, hi = np.full(7, -200.0), np.full(7, 200.0)
    hi[0] = 5.0  # moving right is impossible, but the allowed half-turn goes left
    result = reactive.choose_turn(
        kin,
        path,
        {"receiver": np.eye(4)},
        np.eye(4),
        "receiver",
        np.zeros(3),
        [0.0, 180.0],
        np.tile(q, (2, 1)),
        q,
        6,
        (lo, hi),
        ((-1, -1, -1), (1, 1, 1)),
        (0.12, np.pi),
        30.0,
    )
    assert result["turn_deg"] == 180.0
    assert result["attempts"][-1]["ok"]
    assert result["attempts"][-1]["worst_residual_mm"] <= core.ACT_REACH_TOL_M * 1000


def test_moving_hold_uses_image_time_fk_and_drops_result_after_release(monkeypatch):
    from lerobot.gui.api.manipulation import camera, held_views

    kin = _TipKinematics()
    state = {"released": False}
    published = []
    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    stamp = 123.0
    q = dict(zip(MOTOR_NAMES, [12.0, 0, 0, 0, 0, 0, 80], strict=True))
    monkeypatch.setattr(jog, "joints_at", lambda at, max_gap: q if at == stamp else None)

    async def frame(after):
        return np.zeros((3, 3, 3)), np.ones((3, 3)), {}, stamp

    async def measure(demo, obj, bc, prediction, stopped, *, capture, publish):
        view = await capture(obj)
        assert view[3][0, 3] == pytest.approx(0.012)
        assert view[4] is None, "a stale mask may not be used on a new image"
        state["released"] = True
        publish({"fix": np.eye(4), "orientation": "measured"})
        return None, "stopped", []

    monkeypatch.setattr(camera, "timed_frame", frame)
    monkeypatch.setattr(held_views, "measure", measure)
    asyncio.run(
        held_views.watch(
            None,
            "arbitrary",
            np.eye(4),
            lambda: np.eye(4),
            lambda: state["released"],
            published.append,
            lambda *args: None,
        )
    )
    assert not published
