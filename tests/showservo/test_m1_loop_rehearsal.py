# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The M1 loop rehearsed end to end with no rig: real HTTP, real state machine, real
probe/Broyden/PI/DLS, real server-side clamps — only perception and motors are fake.

The fake world is linear and honest: three joints move the held end through a fixed
(but unknown to the loop) 3x3 map, the target sits displaced by a fixed offset, and
the loop must discover the map from its own probe and drive the held end to the
taught relation. This is the flight check the first powered run rests on — the arm's
maiden move must not double as the state machine's first execution.

The arm endpoint validates every command with the REAL `check_arm_move`, so the
worker and the server's safety contract are exercised against each other.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))

import showservo_m1  # noqa: E402
from showservo_m1 import M1_JOINTS, Pair, m1_loop, plan_pairs  # noqa: E402

from lerobot.gui.api.showservo import check_arm_move  # noqa: E402
from lerobot.showservo.pose import CameraIntrinsics, Rigid3, RigidFit  # noqa: E402

# Units -> metres, deliberately cross-coupled and anisotropic: nothing in the loop
# may assume axis-aligned joints, because the real arm's certainly are not.
J_TRUE = np.array(
    [
        [0.0080, 0.0015, -0.0010],
        [-0.0020, 0.0090, 0.0012],
        [0.0010, -0.0018, 0.0070],
    ]
)
TARGET_MOVE = np.array([0.040, -0.030, 0.020])  # 54 mm of taught-relation error
FRAME_CAP = 120


class _World:
    """Joint state + what the fake perception derives from it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.q = np.zeros(3)
        self.start = dict.fromkeys(M1_JOINTS, 0.0)
        self.last = dict.fromkeys(M1_JOINTS, 0.0)
        self.moves = 0
        self.frames = 0

    def held_offset(self) -> np.ndarray:
        with self.lock:
            return J_TRUE @ self.q

    def apply_move(self, deltas: dict) -> dict:
        with self.lock:
            targets = check_arm_move(deltas, self.last, self.start)  # the real contract
            self.last.update(targets)
            for i, j in enumerate(M1_JOINTS):
                self.q[i] = self.last[j]
            self.moves += 1
            return targets


class _StubCard:
    """Duck-typed Card: a small cloud in front of the camera is all the loop reads."""

    def __init__(self, rng):
        self.xyz = np.array([0.0, 0.0, 0.5]) + rng.uniform(-0.02, 0.02, size=(20, 3))
        self.uv = np.zeros((20, 2))


class _StubDesignator:
    def mask(self, frame):
        return np.ones(frame.rgb.shape[:2], dtype=bool)


class _StubTier:
    """Raises on teach: Recruits.refresh/fallback treat that as 'nothing recruitable'
    and stay silent, which removes recruitment from the rehearsal without touching
    the loop's code path selection."""

    def teach(self, rgb, mask):
        raise AssertionError("no features in the rehearsal world")


def _fake_bind(world: _World, target_card, held_card):
    def bind(card, frame, mask, tier, intr):
        if card is target_card:
            return RigidFit(
                ok=True, transform=Rigid3(np.eye(3), TARGET_MOVE.copy()), inliers=np.ones(20, bool)
            ), None
        assert card is held_card, "the loop bound a card the rehearsal never taught"
        return RigidFit(
            ok=True, transform=Rigid3(np.eye(3), world.held_offset()), inliers=np.ones(20, bool)
        ), None

    return bind


def _serve(world: _World) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code: int, payload: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            assert self.path.endswith("/live/frame.npz")
            with world.lock:
                world.frames += 1
                over = world.frames > FRAME_CAP
            if over:
                self._reply(409, b"{}")
                return
            buf = io.BytesIO()
            np.savez_compressed(
                buf, rgb=np.zeros((8, 8, 3), np.uint8), depth=np.full((8, 8), 0.5, np.float32)
            )
            self._reply(200, buf.getvalue(), "application/octet-stream")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.endswith("/arm/move"):
                try:
                    targets = world.apply_move(json.loads(body)["deltas"])
                except ValueError as e:  # the safety contract refused: 409 like the GUI
                    self._reply(409, json.dumps({"detail": str(e)}).encode())
                    return
                self._reply(200, json.dumps({"positions": targets}).encode())
            else:
                assert self.path.endswith("/live/result")
                self._reply(200, b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_the_m1_loop_probes_learns_the_map_and_converges(monkeypatch):
    rng = np.random.default_rng(7)
    world = _World()
    target_card, held_card = _StubCard(rng), _StubCard(rng)
    monkeypatch.setattr(showservo_m1, "bind_rigid3d", _fake_bind(world, target_card, held_card))

    server = _serve(world)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=4.0, cy=4.0)
        m1_loop(url, [Pair(target_card, held_card)], _StubDesignator(), _StubDesignator(), _StubTier(), intr)
    finally:
        server.shutdown()

    residual_mm = float(np.linalg.norm(TARGET_MOVE - world.held_offset())) * 1000.0
    assert residual_mm < 4.0, f"loop ended {residual_mm:.1f} mm from the taught relation"
    assert world.moves >= len(M1_JOINTS) + 2, "a probe plus at least a couple of servo steps"
    assert world.frames > world.moves, "commands must be paced by frames, not free-running"


def test_the_teaching_rule_pairs_photos_deterministically():
    """The whole teaching contract, enumerated. T = target visible, H = held visible."""
    both, t_only, h_only, neither = (True, True), (True, False), (False, True), (False, False)

    # One photo with both ends is a complete demo by itself.
    assert plan_pairs([both]) == [(0, 0)]
    # The two-photo teach: object alone, then the goal pose with the object hidden.
    assert plan_pairs([t_only, h_only]) == [(0, 1)]
    # A goal photo always takes the MOST RECENT object photo before it.
    assert plan_pairs([t_only, t_only, h_only]) == [(1, 2)]
    # Several goal photos may share one object photo (several taught goals).
    assert plan_pairs([t_only, h_only, h_only]) == [(0, 1), (0, 2)]
    # A both-photo also serves as the object photo for a later goal photo.
    assert plan_pairs([both, h_only]) == [(0, 0), (0, 1)]
    # A goal photo with no object photo before it teaches nothing; order matters.
    assert plan_pairs([h_only, t_only]) == []
    # Photos where nothing designates change nothing.
    assert plan_pairs([neither, t_only, neither, h_only]) == [(1, 3)]
    # An object photo after the last goal photo is unused (no goal to serve).
    assert plan_pairs([t_only, h_only, t_only]) == [(0, 1)]


def test_the_loop_halts_when_the_server_refuses_a_move(monkeypatch):
    """Stop semantics: the first 409 from the arm is a hard halt — the loop must keep
    measuring (frames continue) but never command again."""
    rng = np.random.default_rng(11)
    world = _World()
    target_card, held_card = _StubCard(rng), _StubCard(rng)
    monkeypatch.setattr(showservo_m1, "bind_rigid3d", _fake_bind(world, target_card, held_card))

    refused_after: list[int] = []
    original = world.apply_move

    def stop_after_two(deltas):
        if world.moves >= 2:
            refused_after.append(world.frames)
            raise ValueError("stopped")
        return original(deltas)

    world.apply_move = stop_after_two
    server = _serve(world)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=4.0, cy=4.0)
        m1_loop(url, [Pair(target_card, held_card)], _StubDesignator(), _StubDesignator(), _StubTier(), intr)
    finally:
        server.shutdown()

    assert world.moves == 2, "no command may follow a refusal"
    assert len(refused_after) == 1, "one refusal is enough — a halted loop must not keep asking"
    assert world.frames >= FRAME_CAP, "the loop must keep annotating after the halt, not die"
