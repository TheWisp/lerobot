"""The act with the point groups (src/lerobot/showservo/docs/act_loop.md): the objects it follows designated in the
view before the arm moves, and an object no view places moved on with what it rests on, from the frame of the view
that last placed it to the groups' newest frame.

The view is the real one's server (benchmarks/group_live.py, ``MjpegView``) with its live loop played here: requests
taken between frames, a pose per object on every frame, stamped with the time the frame was drawn."""

from __future__ import annotations

import asyncio
import pathlib
import socket
import sys
import threading
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.gui.api import _pregrasp_core as core, pregrasp
from tests.gui.test_pregrasp_core import INTR, _rect_scene, _YawKinematics

MIDDLE = np.array(
    [0.125, 0.0, 0.0]
)  # the gamepad's middle where the demo's fingertip comes down on it, base = camera


@pytest.fixture(scope="module")
def group_live():
    bench = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(bench))
    try:
        import group_live

        yield group_live
    finally:
        sys.path.remove(str(bench))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _turned(slide_m: float, yaw_deg: float, about: np.ndarray = MIDDLE) -> np.ndarray:
    """A turn about the vertical through ``about``, then a slide along x: how an object on a pushed tray moves."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    m[:3, 3] = about - m[:3, :3] @ about + [slide_m, 0.0, 0.0]
    return m


class _ViewLoop:
    """The view's live loop as an act sees it, on the real view's server: designations taken between frames, each
    answered by ``answer(name)``, and while ``drawing`` a frame every 20 ms with each designated object's ``pose()``."""

    def __init__(self, view, pose, answer=lambda name: {"ok": True, "own": 30, "around": 200}):
        self.view, self.pose, self.answer = view, pose, answer
        self.drawing = True
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            if self.drawing:
                names = [n for n, d in self.view.designated.items() if d.get("ok")]
                # Stamped finer than a microsecond, as time.time() is, in the way a stamp printed to microseconds
                # rounds down: an act asking for frames after that would be given its newest frame again and again.
                stamp = float(f"{time.time():.6f}") + 2.5e-7
                self.view.add_poses(stamp, {n: self.pose().ravel().tolist() for n in names})
            asked, taken = self.view.take_requests()
            if asked:
                self.view.designated.update({n: self.answer(n) for n in asked})
            self.view.served += taken
            time.sleep(0.02)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(2.0)
        self.view.server.shutdown()
        self.view.server.server_close()


def _run_act(
    tmp_path, monkeypatch, group_live, *, with_groups, pose, answer=None, arm_moves=None, first_target=None
):
    """One act on a fake arm with a yaw wrist: a demo comes down on the gamepad (pre-grasp at sample 10, grasp to 25),
    the act's tracker holds the view ``found`` (turned and slid from the demo's) and sees nothing newer, as under the
    wrist, and the point groups' view is up. ``pose(sim)`` is the gamepad's pose in the view on each frame. The walk
    moves the arm a third of the way on each target while ``arm_moves(sim)``; ``first_target(loop)`` runs on the
    first. Returns (act, sim, tips, found, view loop)."""
    from lerobot.gui.api import jog
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
    from tests.gui.test_stream_objects import fake_playback

    gi = MOTOR_NAMES.index("gripper")
    kin = _YawKinematics()
    n = 30
    t = np.arange(n) / 30.0
    q_obs = np.zeros((n, 7))
    q_obs[:, 0] = 100.0 + np.arange(n)
    q_obs[:, 2] = np.linspace(60.0, 20.0, n)
    q_obs[:, gi] = np.where(np.arange(n) < 20, 60.0, 85.0)
    tips = np.stack([kin.forward_kinematics(q) for q in q_obs])
    demo = pregrasp._Demo(
        name="d",
        concept="gamepad",
        fps=30.0,
        t=t,
        tips=tips,
        grippers=q_obs[:, gi],
        q_obs=q_obs,
        q_cmd=q_obs.copy(),
        deltas=np.tile(np.eye(4), (n, 1, 1)),
        seen=np.ones(n, dtype=bool),
        delta0=np.eye(4),
        taught=True,
    )
    demo.keypoints = [{"t": float(t[10]), "kind": "pregrasp"}, {"t": float(t[25]), "kind": "grasp_end"}]
    rgb, depth = _rect_scene(0.0)
    mask = np.zeros(depth.shape, bool)
    mask[225:275, 375:485] = True
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={
            "mode": "features",
            "concept": "gamepad",
            "n_points": 40,
            "xyz": np.zeros((40, 3)),
            "mask": mask,
        },
    )
    found = _turned(0.010, 5.0)  # where the act's own tracker last placed it, from the demo's view
    sim = {"q": np.array([80.0, -20.0, 90.0, 0, 0, 0, 60.0]), "grip": 60.0, "targets": [], "streamed": []}

    def set_target_pose(pose):
        sim["targets"].append(np.array(pose))
        if first_target is not None and len(sim["targets"]) == 1:
            first_target(loop)
        if arm_moves is not None and not arm_moves(sim):
            return  # the arm stays where it is
        sim["q"][:3] += (np.asarray(pose)[:3, 3] * 1000.0 - sim["q"][:3]) * 0.34
        want = np.degrees(np.arctan2(pose[1, 0], pose[0, 0]))
        sim["q"][4] += ((want - sim["q"][4] + 180.0) % 360.0 - 180.0) * 0.34

    async def joints_start(q_first):
        sim["q"] = np.array([q_first[m] for m in MOTOR_NAMES])

    async def joints_stop():
        pass

    def set_target_joints(q):
        sim["q"] = np.array([q[m] for m in MOTOR_NAMES])
        sim["streamed"].append(sim["q"].copy())

    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    monkeypatch.setattr(
        jog,
        "current_tip_and_anchor",
        lambda: (
            kin.forward_kinematics(sim["q"]),
            np.eye(4),
            {m: float(sim["q"][k]) for k, m in enumerate(MOTOR_NAMES)},
        ),
    )
    monkeypatch.setattr(jog, "set_target_pose", set_target_pose)
    monkeypatch.setattr(jog, "current_status", lambda: {"connected": True, "halted": False, "holding": False})
    monkeypatch.setattr(jog, "current_gripper", lambda: sim["grip"])
    monkeypatch.setattr(jog, "set_gripper", lambda g: sim.__setitem__("grip", g))
    monkeypatch.setattr(jog, "walk_limits", lambda: (0.04, np.radians(30)))
    monkeypatch.setattr(jog, "set_walk_limits", lambda lin, ang: None)
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    monkeypatch.setattr(jog, "joints_start", joints_start)
    monkeypatch.setattr(jog, "joints_stop", joints_stop)
    monkeypatch.setattr(jog, "set_target_joints", set_target_joints)
    fake_playback(monkeypatch, set_target_joints)
    monkeypatch.setattr(jog, "start_record", lambda: time.time())
    monkeypatch.setattr(jog, "stop_record", lambda: [])
    monkeypatch.setattr(jog, "fk_tip", lambda q: np.eye(4))
    monkeypatch.setattr(pregrasp, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(pregrasp, "ACT_TICK_S", 0.002)
    monkeypatch.setattr(pregrasp, "ACT_STEP_TIMEOUT_S", 5.0)
    monkeypatch.setattr(pregrasp, "TRIALS_PATH", tmp_path / "trials.jsonl")
    monkeypatch.setattr(pregrasp, "_trials", None)
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    monkeypatch.setattr(pregrasp._state, "groups", pregrasp._GroupsView())  # a view started elsewhere
    monkeypatch.setattr(pregrasp._state, "groups_with_acts", with_groups)
    port = _free_port()
    monkeypatch.setattr(pregrasp, "GROUPS_VIEW", ("127.0.0.1", port))
    view = group_live.MjpegView(port, tmp_path / "recordings")
    loop = _ViewLoop(view, lambda: pose(sim), **({} if answer is None else {"answer": answer}))
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach = demo, teach
        pregrasp._state.test = pregrasp._Test(at="now", rgb=rgb, result={"ok": True, "delta_cam": found})
        pregrasp._state.track.on = True
        pregrasp._state.track.history = [(time.time() - 1.0, True, found)]
        pregrasp._state.track.last = {"state": "tracking"}
        pregrasp._state.act = pregrasp._Act(on=True, speed=4.0)
    try:
        asyncio.run(asyncio.wait_for(pregrasp._act_task(4.0), timeout=20.0))
        return pregrasp._state.act, sim, tips, found, loop
    finally:
        loop.close()
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.track.on = False
            pregrasp._state.track.history = []
            pregrasp._state.track.last = {}
            pregrasp._state.act = pregrasp._Act()


@pytest.mark.parametrize("with_groups", [True, False])
def test_a_covered_object_moves_with_what_it_rests_on_and_the_grasp_follows(
    tmp_path, monkeypatch, group_live, with_groups
):
    """The wrist covers the gamepad as the arm comes in, so the act's tracker places it no more, and the tray it lies
    on is pushed: turned 10 degrees about the tray's middle and slid 20 mm. The point groups see the tray move and
    carry the gamepad with it; the act's pose is its last view moved on by that, and the walk and the grasp go where it
    now lies. Off, the act holds the last view, as it did before the groups were there."""
    resting = np.eye(4)
    resting[:3, 3] = MIDDLE  # the gamepad's pose in the view: its own frame at its middle
    pushed = _turned(0.020, 10.0, about=np.array([0.05, 0.08, 0.0]))  # about the tray's middle

    def pose(sim):
        return pushed @ resting if "pushed_at" in sim else resting

    def arm_moves(
        sim,
    ):  # the push comes with the third target; the arm, slower than the view, waits 0.3 s on it
        if len(sim["targets"]) >= 3:
            sim.setdefault("pushed_at", time.time())
        return time.time() - sim.get("pushed_at", np.inf) > 0.3

    act, sim, tips, found, loop = _run_act(
        tmp_path, monkeypatch, group_live, with_groups=with_groups, pose=pose, arm_moves=arm_moves
    )
    assert act.ok, act.reason
    lies = pushed @ found if with_groups else found
    m, deg = core.pose_residual(sim["targets"][0], found @ tips[10])
    assert m < 1e-6 and deg < 1e-3, "it first aims at the last view"
    m, deg = core.pose_residual(sim["targets"][-1], lies @ tips[10])
    assert m < 1e-6 and deg < 1e-3, "then where the object lies now"
    end = _YawKinematics().forward_kinematics(sim["streamed"][-1])
    m, deg = core.pose_residual(end, lies @ tips[25])
    assert m <= core.ACT_SOLVE_TOL_M and deg <= core.ACT_SOLVE_TOL_DEG, "the grasp, on it"
    assert set(loop.view.designated) == ({"gamepad"} if with_groups else set()), (
        "designated only with the groups"
    )


def test_an_act_whose_object_the_view_cannot_take_does_not_move(tmp_path, monkeypatch, group_live):
    act, sim, *_ = _run_act(
        tmp_path,
        monkeypatch,
        group_live,
        with_groups=True,
        pose=lambda sim: np.eye(4),
        answer=lambda name: {"ok": False, "reason": "3 own and 0 borrowed corners; click it again"},
    )
    assert not act.ok and act.reason == (
        "the point groups did not take gamepad: 3 own and 0 borrowed corners; click it again"
    )
    assert sim["targets"] == [], "nothing moved"


def test_the_act_stops_when_the_view_stops_drawing(tmp_path, monkeypatch, group_live):
    """A view that draws nothing more would hold every object it carries still, whatever happens to them."""
    monkeypatch.setattr(pregrasp, "GROUPS_STALE_S", 0.3)

    def stop_drawing(loop):
        loop.drawing = False

    act, sim, *_ = _run_act(
        tmp_path,
        monkeypatch,
        group_live,
        with_groups=True,
        pose=lambda sim: np.eye(4),
        arm_moves=lambda sim: False,  # the walk would never end on its own
        first_target=stop_drawing,
    )
    assert not act.ok and act.reason == "the point groups view drew no frame for 0.3 s", act.reason
