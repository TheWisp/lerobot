"""Live target/attachment corrections must reach the bus without replaying contact events."""

import asyncio
import threading
import time
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


def test_checked_queue_blend_keeps_old_prefix_and_new_endpoint_without_blending_grip():
    kin = _TipKinematics()
    start = np.array([0.0, 0, 0, 0, 0, 0, 80])
    old = [np.array([x, 0, 0, 0, 0, 0, 80.0]) for x in (1, 2, 3, 4)]
    new = [np.array([x, 0, 0, 0, 0, 0, 80.0]) for x in (0.5, 1, 1.5, 2)]
    box = ((-1, -1, -1), (1, 1, 1))
    blended = reactive.checked_queue_blend(kin, start, old, new, 6, box, -1.0, 2.0, 4)
    assert blended is not None
    assert blended[0][0] == pytest.approx(0.875)
    np.testing.assert_array_equal(blended[-1], new[-1])
    assert reactive.checked_queue_blend(kin, start, old, new, 6, box, -1.0, 0.5, 4) is None
    releasing = [q.copy() for q in old]
    releasing[1][6] = 20.0
    assert reactive.checked_queue_blend(kin, start, releasing, new, 6, box, -1.0, 2.0, 4) is None


def test_unverified_pregrasp_shift_uses_object_centre():
    centre = np.array([0.2, 0.0, 0.0])
    turn = np.eye(4)
    turn[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    turn[:3, 3] = centre - turn[:3, :3] @ centre
    points = np.stack([centre + [0.01, 0, 0], centre - [0.01, 0, 0]])
    assert reactive.unverified_object_shift(turn, points, 0.003) is None
    turn[1, 3] += 0.007
    assert reactive.unverified_object_shift(turn, points, 0.003) == pytest.approx(0.007)
    turn[1, 3] -= 0.002
    assert reactive.unverified_object_shift(turn, points, 0.003) is None


def test_pregrasp_motion_stops_only_near_the_observed_object():
    points = np.array([[0.19, 0, 0], [0.21, 0, 0]])
    moved = pose(0.007)
    assert reactive.nearby_pregrasp_shift(moved, points, [0.21, 0, 0], 0.003) == pytest.approx(0.007)
    assert reactive.nearby_pregrasp_shift(moved, points, [0.5, 0, 0], 0.003) is None
    assert reactive.nearby_object_extent(points, [0.21, 0, 0], 0.003)
    assert not reactive.nearby_object_extent(points, [0.5, 0, 0], 0.003)


@pytest.mark.parametrize("closing", [1.0, -1.0])
def test_withdrawal_does_not_reclose_on_the_just_released_object(closing):
    gi = MOTOR_NAMES.index("gripper")
    released = 65.0 if closing > 0 else 35.0
    q = np.zeros((4, len(MOTOR_NAMES)))
    q[:, gi] = [released, released + closing * 8, released - closing * 5, released + closing * 30]
    original = q.copy()
    plan = {"q": q, "live_path": {"grips": q[:, gi].copy(), "frames": ["arbitrary"] * len(q)}}
    guarded = reactive.keep_released(plan, gi, closing)
    np.testing.assert_array_equal(guarded["q"][:, gi], [released, released, released - closing * 5, released])
    np.testing.assert_array_equal(guarded["live_path"]["grips"], guarded["q"][:, gi])
    np.testing.assert_array_equal(plan["q"], original)
    assert guarded["live_path"]["frames"] == ["arbitrary"] * len(q)


@pytest.mark.parametrize("change", ["target", "attachment", "both", "during_solve"])
def test_update_during_contact_retargets_remaining_phase_and_preserves_release(monkeypatch, change):
    kin = _TipKinematics()
    state = {"q": np.array([0.0, 0, 0, 0, 0, 0, 80]), "motion": np.eye(4), "fix": np.eye(4), "revision": 0}
    commands, phases, evidence = [], [], []
    original_step = reactive.checked_step

    def checked_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if change == "during_solve" and state["revision"] == 0:
            state["fix"] = pose(-0.003)
            state["revision"] = 1
            # Make the first queued sample due immediately, so it is consumed
            # before the producer can replace it with a second solve.
            time.sleep(2 / jog.HZ)
        return result

    monkeypatch.setattr(reactive, "checked_step", checked_step)

    def send(q):
        state["q"] = np.array([q[m] for m in MOTOR_NAMES])
        commands.append(state["q"].copy())

    fake_playback(monkeypatch, send)
    replacements = []
    original_replace = jog.replace_playback_tail

    def replace_with_evidence(head, samples, due):
        replacements.append(len(samples))
        return original_replace(head, samples, due)

    monkeypatch.setattr(jog, "replace_playback_tail", replace_with_evidence)
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

    def observed(phase, q, at, goal, composition):
        phases.append(phase)
        if composition is not None:
            reconstructed = np.array(composition["motion"]) @ np.array(composition["local"])
            if composition["held"]:
                reconstructed = reconstructed @ np.array(composition["fix"])
            np.testing.assert_allclose(reconstructed, goal, atol=1e-12)
            evidence.append((composition, state["revision"]))
        if phase == 2:
            if change in ("target", "both"):
                state["motion"] = pose(0.007)
            if change in ("attachment", "both"):
                state["fix"] = pose(-0.003)
                state["revision"] = 1

    async def reselect(*args):
        pytest.fail("reachable correction must not search an alternative")

    diagnostics = {}
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
            revision=lambda: state["revision"],
            diagnostics=diagnostics,
        )
    )
    if change == "attachment":
        assert diagnostics["blended_replacements"] >= 1
        assert diagnostics["blended_samples"] >= 2
    assert not reason
    offset = state["motion"][0, 3] + state["fix"][0, 3]
    assert kin.forward_kinematics(last)[0, 3] == pytest.approx(0.008 + offset, abs=core.ACT_SOLVE_TOL_M)
    assert phases == sorted(phases) and set(phases) == set(range(6))
    first_open = next(q for q in commands if q[-1] < 80)
    assert kin.forward_kinematics(first_open)[0, 3] == pytest.approx(0.006 + offset, abs=core.ACT_SOLVE_TOL_M)
    assert commands[-1][-1] == 20, "release happens before withdrawal, even after inserted correction samples"
    assert (replacements or change == "during_solve") and all(replacements), (
        "a pose update must replace queued motion atomically, without an empty interval"
    )
    assert max(np.linalg.norm(np.diff(np.array(commands)[:, :3], axis=0), axis=1)) <= 4.01
    if change == "during_solve":
        first, current_revision = evidence[0]
        assert first["hold_revision"] == 0 and current_revision == 1
        np.testing.assert_array_equal(first["fix"], np.eye(4))
        assert evidence[-1][0]["hold_revision"] == 1
        np.testing.assert_array_equal(evidence[-1][0]["fix"], pose(-0.003))


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


def test_checked_horizon_replaces_only_unsent_future_without_resetting_guards(monkeypatch):
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    j = jog._Jog(robot=object(), mode="joints", q_target=q.copy(), q_obs=q.copy())
    monkeypatch.setattr(jog, "_jog", j)
    jog.play_joints([q], [0.0])
    j.playback_sent, j.playback_sent_at = 0, [10.0]
    j.playback_blocked, j.playback_held = 12.0, 11.0
    clock = j.playback_t0
    assert jog.append_playback_next(0, {**q, "shoulder_pan": 1.0}, 0.1)
    assert jog.append_playback_chunk(
        1,
        [{**q, "shoulder_pan": 1.5}, {**q, "shoulder_pan": 2.0, "gripper": 80.0}],
        [0.15, 0.2],
    )
    assert not jog.append_playback_next(0, q, 0.3)
    j.playback_sent = 1  # the first future sample, including its grip, has gone out
    assert not jog.replace_playback_tail(0, [q], [0.3])
    assert jog.replace_playback_tail(1, [{**q, "shoulder_pan": 1.5, "gripper": 20.0}], [0.2])
    assert [sample["shoulder_pan"] for sample in j.playback] == [0.0, 1.0, 1.5]
    assert j.playback[-1]["gripper"] == 20.0
    assert (j.playback_t0, j.playback_blocked, j.playback_held) == (clock, 12.0, 11.0)


def test_unheld_grasp_close_can_fill_the_checked_queue(monkeypatch):
    kin = _TipKinematics()
    gi = MOTOR_NAMES.index("gripper")
    q = np.zeros(len(MOTOR_NAMES))
    q[gi] = 80.0
    sent: list[np.ndarray] = []
    state = {"q": q.copy()}

    def consume(command):
        state["q"] = np.array([command[m] for m in MOTOR_NAMES])
        sent.append(state["q"].copy())

    fake_playback(monkeypatch, consume)
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
    plan = {
        "q": np.tile(q, (2, 1)),
        "times": np.array([0.0, 1.0]),
        "stage": ["start", "grasp"],
        "live_path": {
            "local": np.stack([pose(0.0), pose(0.0)]),
            "frames": ["world", "world"],
            "held": np.zeros(2, bool),
            "grips": np.array([80.0, 100.0]),
            "hints": np.zeros((2, len(q))),
            "floor": np.full(2, -1.0),
        },
    }
    diagnostics: dict = {}
    reason, last = asyncio.run(
        live_execution.stream(
            plan,
            kin,
            lambda: {"world": np.eye(4)},
            lambda: np.eye(4),
            lambda: "",
            lambda *_: None,
            lambda *_: pytest.fail("stationary grasp should not need another pose"),
            (0.12, np.pi),
            1,
            diagnostics=diagnostics,
        )
    )
    assert not reason and last[gi] == pytest.approx(100.0)
    assert diagnostics["max_queue_samples"] > 2, "grasp samples should be buffered ahead of the bus"
    assert np.all(np.diff(np.array(sent)[:, gi]) >= 0), "the queue still closes in order"


@pytest.mark.parametrize("solve_s", [0.02, 0.055])
def test_checked_horizon_keeps_motor_ticks_filled_while_solves_take_time(monkeypatch, solve_s):
    """Even a 55 ms IK solve should feed two interpolated 30 Hz commands."""
    kin = _TipKinematics()
    initial = np.zeros(len(MOTOR_NAMES))
    count = 24
    lock, stop = threading.Lock(), threading.Event()
    playback = {"samples": [], "due": [], "sent": -1, "at": [], "t0": 0.0, "q": initial.copy()}
    motor_ticks = []

    def motor():
        tick = time.monotonic()
        while not stop.is_set():
            now = time.monotonic()
            if now < tick:
                time.sleep(tick - now)
                continue
            with lock:
                k = playback["sent"] + 1
                if k < len(playback["samples"]) and now - playback["t0"] >= playback["due"][k]:
                    playback["sent"] = k
                    playback["q"] = np.array([playback["samples"][k][m] for m in MOTOR_NAMES])
                    playback["at"].append(time.time())
                motor_ticks.append((now, playback["q"][0]))
            tick += 1 / jog.HZ

    def play(samples, due):
        with lock:
            playback.update(samples=list(samples), due=list(due), sent=-1, at=[], t0=time.monotonic())

    def state():
        with lock:
            return playback["sent"], list(playback["at"])

    def append(tail, sample, due):
        with lock:
            if tail != len(playback["samples"]) - 1:
                return False
            playback["samples"].append(sample)
            playback["due"].append(due)
            return True

    def append_chunk(tail, samples, due):
        with lock:
            if tail != len(playback["samples"]) - 1:
                return False
            playback["samples"].extend(samples)
            playback["due"].extend(due)
            return True

    def replace(sent, samples, due):
        with lock:
            if sent != playback["sent"]:
                return False
            playback["samples"][sent + 1 :] = list(samples)
            playback["due"][sent + 1 :] = list(due)
            return True

    def current():
        with lock:
            q = playback["q"].copy()
        return kin.forward_kinematics(q), np.eye(4), dict(zip(MOTOR_NAMES, q, strict=True))

    def checked_step(_kin, previous, goal, *_args, **_kwargs):
        time.sleep(solve_s)
        q = previous.copy()
        q[0] = goal[0, 3] * 1000
        return {"reason": "", "q": q, "pose": goal, "complete": True}

    monkeypatch.setattr(jog, "play_joints", play)
    monkeypatch.setattr(jog, "playback_state", state)
    monkeypatch.setattr(jog, "append_playback_next", append)
    monkeypatch.setattr(jog, "append_playback_chunk", append_chunk)
    monkeypatch.setattr(jog, "replace_playback_tail", replace)
    monkeypatch.setattr(jog, "playback_stop", lambda: stop.set())
    monkeypatch.setattr(jog, "current_tip_and_anchor", current)
    monkeypatch.setattr(jog, "servo_ranges", lambda: None)
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1, -1, -1), (1, 1, 1)))
    monkeypatch.setattr(reactive, "checked_step", checked_step)
    local = np.stack([pose(i / 1000) for i in range(count + 1)])
    plan = {
        "q": np.tile(initial, (count + 1, 1)),
        "times": np.arange(count + 1) / jog.HZ,
        "stage": ["move"] * (count + 1),
        "live_path": {
            "local": local,
            "frames": ["world"] * (count + 1),
            "held": np.zeros(count + 1, bool),
            "grips": np.zeros(count + 1),
            "hints": np.zeros((count + 1, len(MOTOR_NAMES))),
            "floor": np.full(count + 1, -1.0),
        },
    }
    thread = threading.Thread(target=motor, daemon=True)
    thread.start()
    try:
        reason, last = asyncio.run(
            live_execution.stream(
                plan,
                kin,
                lambda: {"world": np.eye(4)},
                lambda: np.eye(4),
                lambda: "test timeout" if time.monotonic() - playback["t0"] > 3 else "",
                lambda *_: None,
                lambda *_: pytest.fail("all goals are reachable"),
                (0.12, np.pi),
                1,
            )
        )
    finally:
        stop.set()
        thread.join(timeout=1)
    assert not reason and last[0] == pytest.approx(count)
    active = np.array([q for _, q in motor_ticks if 2 <= q < count - 1])
    assert len(active) >= count - 4
    assert np.mean(np.diff(active) < 0.05) < 0.2, "slow IK must not make the bus hold every other tick"


def test_interpolated_joint_chord_must_follow_checked_cartesian_step():
    class BentKinematics(_TipKinematics):
        def forward_kinematics(self, q):
            out = super().forward_kinematics(q)
            out[2, 3] = 0.01 * np.sin(np.pi * q[0] / 10)
            return out

    start = np.zeros(len(MOTOR_NAMES))
    end = start.copy()
    end[0] = 10
    assert (
        reactive.checked_joint_samples(
            BentKinematics(),
            start,
            {"q": end, "pose": pose(0.01), "reason": ""},
            2,
            MOTOR_NAMES.index("gripper"),
            ((-1, -1, -1), (1, 1, 1)),
            -1,
            10,
        )
        is None
    )
    samples = reactive.checked_joint_samples(
        _TipKinematics(),
        start,
        {"q": end, "pose": pose(0.01), "reason": ""},
        2,
        MOTOR_NAMES.index("gripper"),
        ((-1, -1, -1), (1, 1, 1)),
        -1,
        10,
    )
    assert [q[0] for q in samples] == [5, 10]


def test_checked_motor_samples_keep_velocity_continuous_across_solve_chunks():
    kin = _TipKinematics()
    gi = MOTOR_NAMES.index("gripper")
    start = np.zeros(len(MOTOR_NAMES))
    first_end = start.copy()
    first_end[0] = 4
    args = (kin, 4, gi, ((-1, -1, -1), (1, 1, 1)), -1, 3)
    first = reactive.checked_joint_samples(
        kin,
        start,
        {"q": first_end, "pose": pose(0.004), "reason": ""},
        *args[1:],
        previous_sample=start,
        max_joint_accel_deg=1,
    )
    assert first is not None
    first_steps = np.diff(np.array([start, *first])[:, 0])
    np.testing.assert_allclose(first_steps, [0.4, 0.8, 1.2, 1.6], atol=1e-10)
    second_end = first_end.copy()
    second_end[0] = 8
    second = reactive.checked_joint_samples(
        kin,
        first_end,
        {"q": second_end, "pose": pose(0.008), "reason": ""},
        *args[1:],
        previous_sample=first[-2],
        max_joint_accel_deg=1,
    )
    assert second is not None
    steps = np.diff(np.array([first[-2], first[-1], *second])[:, 0])
    assert np.max(np.abs(np.diff(steps))) < 1
    assert second[-1][0] == pytest.approx(8)
    # A stale incoming velocity is not a reason to leave the checked path.
    stale = start.copy()
    stale[0] = -10
    fallback = reactive.checked_joint_samples(
        kin,
        start,
        {"q": first_end, "pose": pose(0.004), "reason": ""},
        *args[1:],
        previous_sample=stale,
        max_joint_accel_deg=1,
    )
    assert [q[0] for q in fallback] == [1, 2, 3, 4]


def test_empty_live_queue_cannot_reset_hard_lag_timeout():
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    target = {**q, "shoulder_lift": jog.PLAYBACK_STOP_LAG_DEG + 1}
    j = jog._Jog(q_obs=q, q_target=target, playback_live=True)
    j.playback, j.playback_due, j.playback_sent = [target], [0.0], 0
    jog._advance_playback(j, 10.0)
    jog._advance_playback(j, 10.0 + jog.PLAYBACK_HOLD_MAX_S)
    assert j.halted and "behind" in j.reason
    assert j.playback_sent == 0


def test_best_effort_stop_clears_pending_command_and_holds_fresh_encoders(monkeypatch):
    target = dict.fromkeys(MOTOR_NAMES, 0.0)
    target["shoulder_pan"] = 5.0
    observed = {**target, "shoulder_pan": 2.0}
    j = jog._Jog(
        robot=object(),
        mode="joints",
        q_target=target,
        q_obs=observed,
        last_read_at=time.monotonic(),
        playback=[target, {**target, "shoulder_pan": 6.0}],
    )
    monkeypatch.setattr(jog, "_jog", j)
    assert jog.playback_stop_at_observed() == observed
    assert j.playback is None and j.q_target == observed and j.goal_prev is None
    j.playback = [target]
    j.last_read_at -= 1.0
    with pytest.raises(RuntimeError, match="fresh checked joint read"):
        jog.playback_stop_at_observed()
    assert j.playback is not None, "stale telemetry must not claim a safe soft stop"


@pytest.mark.parametrize("grip_drift", [0.0, 0.05])
def test_live_best_effort_tries_incremental_steps_then_holds_observed_prefix(monkeypatch, grip_drift):
    kin = _TipKinematics()
    initial = np.array([0.0, 0, 0, 0, 0, 0, 80])
    state = {"q": initial.copy()}
    sent, stops = [], []

    def send(q):
        state["q"] = np.array([q[m] for m in MOTOR_NAMES])
        sent.append(state["q"].copy())

    def checked_step(_kin, previous, *_args, **_kwargs):
        if previous[0] >= 2:
            return {"reason": "updated pose is outside this arm configuration's reach"}
        q = previous.copy()
        q[0] += 1.0
        return {"reason": "", "q": q, "complete": False}

    fake_playback(monkeypatch, send)
    monkeypatch.setattr(reactive, "checked_step", checked_step)
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

    def stop_at_observed():
        stops.append(state["q"].copy())
        return dict(zip(MOTOR_NAMES, state["q"], strict=True))

    monkeypatch.setattr(jog, "playback_stop_at_observed", stop_at_observed)
    plan = {
        "q": np.stack([initial, initial]),
        "times": np.array([0.0, 0.03]),
        "stage": ["start", "hover"],
        "live_path": {
            "local": np.stack([pose(0), pose(0.08)]),
            "frames": ["world", "world"],
            "held": np.zeros(2, bool),
            "grips": np.array([80.0, 80.0 + grip_drift]),
            "hints": np.zeros((2, 7)),
            "floor": np.full(2, -1.0),
            "constraint": ["pose", "best_effort"],
        },
    }
    reported = []
    reason, last = asyncio.run(
        live_execution.stream(
            plan,
            kin,
            lambda: {"world": np.eye(4)},
            lambda: np.eye(4),
            lambda: "",
            lambda *_: None,
            lambda *_: pytest.fail("best effort must not search a landing turn"),
            (0.12, np.pi),
            1,
            best_effort_stopped=lambda *event: reported.append(event),
        )
    )
    assert not reason and stops and sent[-1][0] >= 2
    np.testing.assert_array_equal(last, stops[-1])
    assert reported[0][0] == "hover" and "reach" in reported[0][1]


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

    async def measure(demo, obj, bc, prediction, stopped, *, capture, publish, **kwargs):
        view = await capture(obj)
        assert view[3][0, 3] == pytest.approx(0.012)
        assert view[4] is None, "a stale mask may not be used on a new image"
        state["released"] = True
        publish({"fix": np.eye(4), "orientation": "measured"})
        return None, "stopped", []

    monkeypatch.setattr(camera, "timed_frame", frame)
    monkeypatch.setattr(held_views, "prepare_references", lambda *args: [object()])
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


def test_missing_encoder_bracket_is_logged_as_timing_failure_not_arm_disconnection(monkeypatch):
    from lerobot.gui.api.manipulation import camera, held_views, runtime

    state = {"done": False}
    logs, published = [], []
    monkeypatch.setattr(jog, "kinematics", _TipKinematics)
    monkeypatch.setattr(jog, "joints_at", lambda *args: None)
    monkeypatch.setattr(runtime, "HISTORY_MATCH_S", 0)
    monkeypatch.setattr(held_views, "prepare_references", lambda *args: [object()])

    async def frame(after):
        return np.zeros((3, 3, 3)), np.ones((3, 3)), {}, 123.0

    async def measure(*args, capture, **kwargs):
        await capture("arbitrary")
        pytest.fail("an unsynchronized frame must not reach registration")

    def log(problem, attempts):
        logs.append(problem)
        state["done"] = True

    monkeypatch.setattr(camera, "timed_frame", frame)
    monkeypatch.setattr(held_views, "measure", measure)
    asyncio.run(
        held_views.watch(
            None, "arbitrary", np.eye(4), lambda: np.eye(4), lambda: state["done"], published.append, log
        )
    )
    assert len(logs) == 1 and "encoder samples bracket" in logs[0]
    assert not published
