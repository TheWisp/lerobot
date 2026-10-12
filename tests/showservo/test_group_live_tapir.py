"""The point groups view's TAPIR (benchmarks/group_live.py): tracks retired for good leave TAPIR's state, near 1 MB
of GPU memory each, which the view otherwise kept for ever (14-18 GB after a morning of acts on 2026-10-12, until the
GPU ran out), while every track keeps its index for the groups. A stand-in tracker keeps its state as Point2Pose's
TapirTracker does: the queries (N, 3), each resolution's features and each causal state tensor along axis 1."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
group_live = pytest.importorskip("group_live")


class _Features(NamedTuple):
    lowres: tuple
    hires: tuple
    resolutions: tuple


class _Frame:
    def __init__(self, id: int, rgb: np.ndarray):
        self.id, self.rgb = id, rgb


class _Tracker:
    """Each query tracks on its own: it reads its point from its own features and its own causal state, moved by the
    frame number, so a row sliced wrong shows as the wrong point."""

    def __init__(self):
        self.query_points = self.query_features = self._causal_state = None

    def initialize(self, frame):
        return True

    def add_query_points(self, frame, uv):
        q = torch.tensor(np.c_[np.full(len(uv), float(frame.id)), uv[:, ::-1]], dtype=torch.float32)
        old = 0 if self.query_points is None else len(self.query_points)
        if self.query_points is None:
            self.query_points = q
            self.query_features = _Features((q[None].clone(),), (q[None].clone(),), ((256, 256),))
            self._causal_state = [{"state": q[None].clone(), "level": 1}]
        else:
            self.query_points = torch.cat([self.query_points, q])
            f = self.query_features
            self.query_features = _Features(
                (torch.cat([f.lowres[0], q[None]], 1),), (torch.cat([f.hires[0], q[None]], 1),), f.resolutions
            )
            self._causal_state = [
                {"state": torch.cat([self._causal_state[0]["state"], q[None]], 1), "level": 1}
            ]
        return np.arange(old, old + len(q))

    def track_once(self, frame):
        own = self.query_features.lowres[0][0][:, [2, 1]] + self._causal_state[0]["state"][0][:, [2, 1]]
        uv = (own / 2.0 + frame.id).numpy()
        return uv, np.zeros(len(uv)), np.ones(len(uv), bool)


def _tapir():
    t = object.__new__(group_live.Tapir)
    t.torch, t.Frame, t.tracker = torch, _Frame, _Tracker()
    t.ids, t.last_uv = np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    t.frame_id, t.n = 0, 0
    return t


def test_a_dropped_track_leaves_tapirs_state_and_every_track_keeps_its_index():
    rgb = np.zeros((4, 4, 3), np.uint8)
    t = _tapir()
    uv0 = np.array([[10, 1], [20, 2], [30, 3], [40, 4], [50, 5]], np.float32)
    assert list(t.add(rgb, uv0)) == [0, 1, 2, 3, 4]
    uv, seen = t.step(rgb)
    assert np.allclose(uv, uv0 + 1) and seen.all()
    assert t.drop([1, 3]) == 2 and t.drop([1, 3]) == 0, "dropped once"
    tr = t.tracker
    kept = uv0[[0, 2, 4]]
    assert np.allclose(tr.query_points[:, [2, 1]].numpy(), kept)
    assert np.allclose(tr.query_features.lowres[0][0][:, [2, 1]].numpy(), kept)
    assert np.allclose(tr.query_features.hires[0][0][:, [2, 1]].numpy(), kept)
    assert np.allclose(tr._causal_state[0]["state"][0][:, [2, 1]].numpy(), kept)
    assert tr._causal_state[0]["level"] == 1, "what is not per query is kept as it is"
    uv, seen = t.step(rgb)
    assert len(uv) == len(seen) == 5, "by track index, the dropped ones included"
    assert np.allclose(uv[[0, 2, 4]], kept + 2) and seen[[0, 2, 4]].all(), "the others track as before"
    assert np.allclose(uv[[1, 3]], uv0[[1, 3]] + 1) and not seen[[1, 3]].any(), "unseen where they were last"
    assert list(t.add(rgb, np.array([[60, 6], [70, 7]], np.float32))) == [5, 6], (
        "new tracks after every earlier one"
    )
    uv, seen = t.step(rgb)
    assert len(uv) == 7 and np.allclose(uv[[5, 6]], [[63, 9], [73, 10]]) and seen[[0, 2, 4, 5, 6]].all()
