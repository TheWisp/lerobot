"""The act with the point groups (src/lerobot/showservo/docs/act_loop.md): the objects it follows designated in the
view before the arm moves, and an act whose object the view cannot take refused before anything moves.

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

from lerobot.gui.api import manipulation
from tests.gui.test_manipulation_motion import INTR, _rect_scene, _YawKinematics

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
    demo = manipulation.runtime._Demo(
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
    teach = manipulation.runtime._Teach(
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
    monkeypatch.setattr(manipulation.camera, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(manipulation.runtime, "ACT_TICK_S", 0.002)
    monkeypatch.setattr(manipulation.runtime, "ACT_STEP_TIMEOUT_S", 5.0)
    monkeypatch.setattr(manipulation.runtime, "TRIALS_PATH", tmp_path / "trials.jsonl")
    monkeypatch.setattr(manipulation.runtime, "_trials", None)
    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path / "demos")
    monkeypatch.setattr(
        manipulation.runtime._state, "groups", manipulation.runtime._GroupsView()
    )  # a view started elsewhere
    monkeypatch.setattr(manipulation.runtime._state, "groups_with_acts", with_groups)
    port = _free_port()
    monkeypatch.setattr(manipulation.runtime, "GROUPS_VIEW", ("127.0.0.1", port))
    view = group_live.MjpegView(port, tmp_path / "recordings")
    loop = _ViewLoop(view, lambda: pose(sim), **({} if answer is None else {"answer": answer}))
    with manipulation.runtime._state.lock:
        manipulation.runtime._state.demo, manipulation.runtime._state.teach = demo, teach
        manipulation.runtime._state.test = manipulation.runtime._Test(
            at="now", rgb=rgb, result={"ok": True, "delta_cam": found}
        )
        manipulation.runtime._state.track.on = True
        manipulation.runtime._state.track.history = [(time.time() - 1.0, True, found)]
        manipulation.runtime._state.track.last = {"state": "tracking"}
        manipulation.runtime._state.act = manipulation.runtime._Act(on=True, speed=4.0)
    try:
        asyncio.run(asyncio.wait_for(manipulation.execution._act_task(4.0), timeout=20.0))
        return manipulation.runtime._state.act, sim, tips, found, loop
    finally:
        loop.close()
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
                manipulation.runtime._state.test
            ) = None
            manipulation.runtime._state.track.on = False
            manipulation.runtime._state.track.history = []
            manipulation.runtime._state.track.last = {}
            manipulation.runtime._state.act = manipulation.runtime._Act()


@pytest.mark.usefixtures("act_simulation")
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
