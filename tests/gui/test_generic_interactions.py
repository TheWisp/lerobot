"""Object-independent recording and sequential rigid-object interactions."""

import dataclasses

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import manipulation
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
from lerobot.showservo import motion as core
from tests.gui.test_stream_objects import INTR, _TipKinematics

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
