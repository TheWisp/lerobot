"""Object-independent recording and sequential rigid-object interactions."""

import asyncio
import dataclasses
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import jog, manipulation
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
from lerobot.showservo import motion as core
from tests.gui.test_stream_objects import INTR, _TipKinematics, fake_playback

GI = MOTOR_NAMES.index("gripper")


def interaction_demo(names=("tool-17", "part-42", "fixture-9")):
    """Two attachments, with an intermediate tool-use segment that does not release."""
    first, second, target = names
    t = np.arange(180) / 30.0
    q = np.zeros((len(t), len(MOTOR_NAMES)))
    q[:, 0] = np.interp(
        np.arange(len(t)),
        [0, 25, 35, 65, 85, 105, 115, 145, 179],
        [100, 110, 150, 160, 200, 210, 240, 250, 260],
    )
    q[:, 2] = 60.0
    closed = ((np.arange(len(t)) >= 15) & (np.arange(len(t)) < 60)) | (
        (np.arange(len(t)) >= 95) & (np.arange(len(t)) < 140)
    )
    q[:, GI] = np.where(closed, 82.0, 60.0)
    cmd = q.copy()
    cmd[:, GI] = np.where(closed, 85.0, 60.0)
    kin = _TipKinematics()
    demo = manipulation.runtime._Demo(
        name="interactions",
        concept="demo",
        fps=30.0,
        t=t,
        tips=np.stack([kin.forward_kinematics(x) for x in q]),
        grippers=q[:, GI],
        q_obs=q,
        q_cmd=cmd,
        deltas=np.tile(np.eye(4), (len(t), 1, 1)),
        seen=np.ones(len(t), bool),
        delta0=np.eye(4),
        intr=INTR,
    )
    specs = [
        (5, first, "line"),
        (25, first, "follow"),
        (35, target, "line"),
        (45, target, "follow"),
        (50, target, "line"),
        (65, target, "follow"),
        (85, second, "line"),
        (105, second, "follow"),
        (115, target, "line"),
        (145, target, "follow"),
        (160, "world", "line"),
    ]
    demo.keypoints = [
        {"kind": "key", "t": float(t[i]), "frame": name, "move": move, "grip": "first"}
        for i, name, move in specs
    ]
    for name in names:
        demo.objects[name] = {
            "status": "done",
            "frame": 0,
            "seen": np.ones(len(t), bool),
            "deltas": np.tile(np.eye(4), (len(t), 1, 1)),
            "mask": np.ones((4, 4), bool),
        }
    return demo, kin


def test_each_attachment_is_detected_without_a_task_or_object_class():
    demo, _ = interaction_demo()
    frames = core.key_frames(demo.keypoints)
    holds = manipulation.planning._holdings(demo)
    assert [(h["frame"], h["grip"], h["release"]) for h in holds] == [
        ("tool-17", 15, 60),
        ("part-42", 95, 140),
    ]
    assert core.interaction_phases(holds, demo.t, frames) == [(0, 6), (6, 11)]
    assert [
        None if (h := core.holding_at(holds, demo.t, frames, n)) is None else h["frame"]
        for n in range(len(frames))
    ] == [None, None, "tool-17", "tool-17", "tool-17", "tool-17", None, None, "part-42", "part-42", None]
    # The same mechanism handles a regrasp of the same object.
    again, _ = interaction_demo(("unlabelled-8", "unlabelled-8", "support-3"))
    assert len(manipulation.planning._holdings(again)) == 2


def test_attachment_corrections_do_not_leak_into_another_grasp():
    demo, _ = interaction_demo()
    fix = np.eye(4)
    fix[0, 3] = 0.006
    first = manipulation.planning._act_steps(demo, "all", 0, lambda _: np.eye(4), fix)
    second = manipulation.planning._act_steps(
        dataclasses.replace(demo, hold_index=1), "all", 0, lambda _: np.eye(4), fix
    )
    assert first[2]["held"] and second[8]["held"]
    assert np.allclose(first[2]["fix"], fix) and np.allclose(first[8]["fix"], np.eye(4))
    assert np.allclose(second[2]["fix"], np.eye(4)) and np.allclose(second[8]["fix"], fix)


def test_regrasp_uses_the_objects_pose_at_the_new_interaction(monkeypatch):
    demo, _ = interaction_demo(("item", "item", "support"))
    demo.recording = "recorded"
    monkeypatch.setattr(manipulation.recording, "_stream_times", lambda _: demo.t)
    demo.objects["item"]["deltas"][70:, 0, 3] = 0.1
    assert manipulation.planning._pose_frame(demo, "item") == 0
    later = dataclasses.replace(demo, hold_index=1, interaction_span=(6, 11))
    assert manipulation.planning._pose_frame(later, "item") == 85
    # It moved 100 mm in the demo and 120 mm live. The new approach shifts only 20 mm.
    live = np.eye(4)
    live[0, 3] = 0.12
    monkeypatch.setattr(manipulation.acquisition, "_located", lambda *_: {"ok": True, "delta": live})
    motion, problem = manipulation.planning._object_motion(later, "item", np.eye(4))
    assert problem == "" and motion[0, 3] == pytest.approx(0.02)


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize(
    "names", [("tool-17", "part-42", "fixture-9"), ("arbitrary-a", "arbitrary-a", "arbitrary-b")]
)
@pytest.mark.parametrize("empty_second", [False, True])
def test_runner_repeats_grasp_checks_and_keeps_a_tool_held_through_interaction(
    tmp_path, monkeypatch, names, empty_second
):
    demo, kin = interaction_demo(names)
    state = manipulation.runtime._State()
    monkeypatch.setattr(manipulation.runtime, "_state", state)
    state.demo = demo
    state.act = manipulation.runtime._Act(on=True, correct_hold=True, allow_grasp_prediction=True)
    state.groups_with_acts = True
    demo.recording = "synthetic"
    monkeypatch.setattr(manipulation.recording, "_stream_times", lambda _: demo.t)
    state.teach = manipulation.runtime._Teach(
        at="now",
        box=(0, 0, 0, 0),
        rgb=np.zeros((4, 4, 3), np.uint8),
        depth_m=np.ones((4, 4)),
        intr=INTR,
        keypoints={"concept": names[0], "mask": np.ones((4, 4), bool)},
    )
    state.test = manipulation.runtime._Test(
        at="now", rgb=None, result={"ok": True, "delta_cam": np.eye(4), "live_mask": np.ones((4, 4), bool)}
    )
    # Supply measured object/reference poses directly; exercise the actual runner and planner.
    monkeypatch.setattr(manipulation.planning, "_reference_motion", lambda *_args, **_kwargs: (np.eye(4), ""))
    monkeypatch.setattr(manipulation.planning, "_object_motion", lambda *_args, **_kw: (np.eye(4), ""))
    monkeypatch.setattr(
        manipulation.acquisition, "_located", lambda *_: {"ok": True, "mask": np.ones((4, 4), bool)}
    )

    find_before_motion = []

    async def found(*_args, **_kwargs):
        find_before_motion.append(len(sent))
        return ""

    monkeypatch.setattr(manipulation.acquisition, "_find_afresh", found)
    monkeypatch.setattr(manipulation.acquisition, "_locate_afresh", found)
    monkeypatch.setattr(manipulation.groups, "_carried", lambda *_: np.eye(4))
    monkeypatch.setattr(manipulation.runtime, "GROUPS_STALE_S", 1000)
    events, checks, sent = [], [], []

    async def designate(clicks, *_):
        stamp = time.time()
        for name in clicks:
            state.groups_feed.frames[name] = [(stamp, np.eye(4))]
        return "", stamp

    monkeypatch.setattr(manipulation.groups, "_designate_in_groups", designate)

    async def feed_held(name, *_):
        events.append(("attach", name, len(sent)))
        await asyncio.Event().wait()

    async def group_call(_method, path, body):
        if path == "/objects/held":
            events.extend(("release", name, len(sent)) for name, pose in body.items() if pose is None)
        return {}

    monkeypatch.setattr(manipulation.groups, "_feed_held", feed_held)
    monkeypatch.setattr(manipulation.groups, "_groups_call", group_call)

    async def demo_hold(d, name, *_args):
        checks.append((d.hold_index, name, _args[-1]))
        return None, "synthetic occlusion: use the explicitly enabled grasp prediction"

    monkeypatch.setattr(manipulation.holds, "_demo_hold", demo_hold)
    monkeypatch.setattr(manipulation.camera, "_t_base_cam", lambda: np.eye(4))
    sim = {"q": demo.q_obs[0].copy(), "grip": 60.0}

    def send(q):
        sim["q"] = np.array([q[m] for m in MOTOR_NAMES])
        sim["grip"] = min(float(sim["q"][GI]), 82.0)
        if empty_second and sim["q"][0] > 180:
            sim["grip"] = float(sim["q"][GI])
        sent.append(sim["q"].copy())

    async def start(q):
        send(q)

    async def stop():
        pass

    def target(p):
        sim["q"][:3] = np.asarray(p)[:3, 3] * 1000

    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    monkeypatch.setattr(
        jog,
        "current_tip_and_anchor",
        lambda: (kin.forward_kinematics(sim["q"]), np.eye(4), dict(zip(MOTOR_NAMES, sim["q"], strict=True))),
    )
    monkeypatch.setattr(jog, "current_status", lambda: {"connected": True, "halted": False, "holding": False})
    monkeypatch.setattr(jog, "current_gripper", lambda: sim["grip"])
    monkeypatch.setattr(jog, "set_gripper", lambda g: sim.update(grip=min(g, 82.0)))
    monkeypatch.setattr(jog, "set_target_pose", target)
    monkeypatch.setattr(jog, "walk_limits", lambda: (0.1, np.radians(60)))
    monkeypatch.setattr(jog, "set_walk_limits", lambda *_: None)
    monkeypatch.setattr(jog, "servo_ranges", lambda: None)
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1, -1, -1), (1, 1, 1)))
    monkeypatch.setattr(jog, "joints_start", start)
    monkeypatch.setattr(jog, "joints_stop", stop)
    fake_playback(monkeypatch, send)
    monkeypatch.setattr(manipulation.runtime, "ACT_TICK_S", 0.002)
    monkeypatch.setattr(manipulation.runtime, "ACT_STEP_TIMEOUT_S", 3)

    class Run:
        meta = {}
        arm_recording = False
        timeline = None

        def target(self, *_args, **_kwargs):
            pass

    run = Run()
    monkeypatch.setattr(manipulation.recording, "_begin_run", lambda *_: run)
    monkeypatch.setattr(manipulation.recording, "_finish_run", lambda *_: None)
    monkeypatch.setattr(manipulation.recording, "_record_trial", lambda *_: None)
    asyncio.run(asyncio.wait_for(manipulation.execution._act_task(3.0), 20))
    assert find_before_motion and all(n == 0 for n in find_before_motion)
    if empty_second:
        assert not state.act.ok and "grasp missed" in state.act.reason
        assert len(run.meta["hold_selections"]) == 1, "no reuse of the first grasp's prediction"
        assert [(a, n) for a, n, _ in events] == [("attach", names[0]), ("release", names[0])]
        return
    assert state.act.ok, state.act.reason
    assert checks == [
        (0, names[0], "grip"),
        (0, names[0], "carry"),
        (1, names[1], "grip"),
        (1, names[1], "carry"),
    ]
    assert [(a, n) for a, n, _ in events] == [
        ("attach", names[0]),
        ("release", names[0]),
        ("attach", names[1]),
        ("release", names[1]),
    ]
    # First tool-use follow ends at x=150 mm, still closed. Release must be after x=160, open.
    release_at = events[1][2]
    assert sent[release_at - 1][0] >= demo.q_obs[60, 0]
    assert sent[release_at - 1][GI] == 60
    assert [s["object"] for s in run.meta["hold_selections"]] == [names[0], names[1]]
    assert len(run.meta["holds"]) == 2
    # The reactive solve is re-seeded from the current command; assert the public
    # Cartesian solver tolerance, not an incidental tighter residual of batch IK.
    assert sim["q"][0] == pytest.approx(demo.q_obs[160, 0], abs=core.ACT_SOLVE_TOL_M * 1000)


def test_new_demo_clears_old_object_state_and_preserves_saved_data(tmp_path, monkeypatch):
    demo, _ = interaction_demo()
    demo.root = str(tmp_path / "previous")
    root = tmp_path / "previous"
    root.mkdir()
    (root / "keep").write_text("recorded evidence")
    state = manipulation.runtime._State(demo=demo)
    state.teach = object()
    state.located["old"] = {"ok": True}
    state.track.follow = True
    monkeypatch.setattr(manipulation.runtime, "_state", state)
    app = FastAPI()
    app.include_router(manipulation.router)
    with TestClient(app) as client:
        assert client.post("/api/pregrasp/demo/new").status_code == 200
        assert state.demo is None and state.teach is None and not state.located and not state.track.follow
        assert (root / "keep").read_text() == "recorded evidence"
        state.demo = dataclasses.replace(demo, root=None)
        assert client.post("/api/pregrasp/demo/new").status_code == 409
        assert state.demo is not None
        state.demo.root = str(root)
        state.act.on = True
        assert client.post("/api/pregrasp/demo/new").status_code == 409


def test_new_recording_cannot_overwrite_another_demo(tmp_path, monkeypatch):
    demo, _ = interaction_demo()
    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path)
    root = tmp_path / demo.name
    root.mkdir()
    (root / "keep").write_text("previous recording")
    from fastapi import HTTPException

    with pytest.raises(HTTPException, match="different demo"):
        manipulation.demos._write_demo(demo, None)
    assert (root / "keep").read_text() == "previous recording"


def test_a_hidden_regrasp_reference_cannot_reuse_the_original_pose(monkeypatch):
    demo, _ = interaction_demo(("item", "item", "support"))
    demo.recording = "recorded"
    monkeypatch.setattr(manipulation.recording, "_stream_times", lambda _: demo.t)
    demo.objects["item"]["seen"][60:] = False
    later = dataclasses.replace(demo, hold_index=1, interaction_span=(6, 11))
    assert manipulation.planning._pose_frame(later, "item") is None
    monkeypatch.setattr(manipulation.acquisition, "_located", lambda *_: {"ok": True, "delta": np.eye(4)})
    motion, problem = manipulation.planning._object_motion(later, "item", np.eye(4))
    assert motion is None and "not visible before this interaction" in problem


def test_held_inspection_uses_the_current_objects_mask(monkeypatch):
    demo, kin = interaction_demo()
    state = manipulation.runtime._State(demo=demo)
    monkeypatch.setattr(manipulation.runtime, "_state", state)
    primary = np.zeros((4, 4), bool)
    primary[0, 0] = True
    secondary = np.zeros((4, 4), bool)
    secondary[3, 3] = True
    state.test = manipulation.runtime._Test(at="now", rgb=None, result={"live_mask": primary})
    state.track.last = {"state": "tracking"}
    state.targets["part-42"] = manipulation.runtime._TargetTrack(last={"state": "tracking"})
    state.located["part-42"] = {"mask": secondary}
    monkeypatch.setattr(jog, "current_tip_and_anchor", lambda: (demo.tips[0], np.eye(4), {}))

    async def frame():
        return np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)), INTR

    monkeypatch.setattr(manipulation.camera, "_frame", frame)
    assert np.array_equal(asyncio.run(manipulation.holds._held_view("part-42"))[-1], secondary)
    assert np.array_equal(asyncio.run(manipulation.holds._held_view("tool-17"))[-1], primary)
    state.targets["part-42"].last = {"state": "lost"}
    assert asyncio.run(manipulation.holds._held_view("part-42"))[-1] is None
