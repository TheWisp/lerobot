"""Pause evidence must retain long stops that an active-motion window excludes."""

from collections import deque

import numpy as np
import pytest

from lerobot.gui.api.manipulation import groups, recording, runtime


def test_pause_durations_include_a_long_stationary_middle_and_distinguish_queue_state():
    t = np.arange(61) / 30
    q = np.zeros((61, 7))
    q[:, 0] = np.r_[np.arange(10), np.full(31, 9), np.arange(10, 30)]
    waiting = np.zeros(61, int)
    waiting[10:25] = 1
    depth = np.full(61, 10)
    depth[25:40] = 0
    arm = {"t": t, "q_cmd": q, "q_obs": q, "playback_waiting": waiting, "playback_queue_depth": depth}
    result = recording._control_cadence(arm, [])
    pauses = result["pauses"]
    assert pauses["command_stationary"]["total_s"] == pytest.approx(31 / 30)
    assert pauses["command_stationary"]["longest_s"] == pytest.approx(31 / 30)
    assert pauses["lag_wait"]["total_s"] == pytest.approx(0.5)
    assert pauses["queue_empty"]["total_s"] == pytest.approx(0.5)
    assert not pauses["playback_state_recorded"]
    # Legacy active windows omit the middle of this stop. The new duration
    # must not shrink to those few boundary ticks.
    assert result["active_ticks"] < 60


@pytest.mark.parametrize("name", ["pick_candidate", "placement_target"])
def test_group_evidence_has_the_same_authority_for_any_free_object(monkeypatch, name):
    monkeypatch.setattr(groups.time, "time", lambda: 100.0)
    context = {"group": 3, "own": False, "why": "partly occluded"}
    drawing = {"stamp": 99.9, "objects": {name: context}, "groups": {"3": {"supported": True}}}
    feed = runtime._GroupsFeed(frames={name: deque([(99.9, np.eye(4))])}, drawing=drawing)
    monkeypatch.setattr(runtime._state, "groups_feed", feed)
    monkeypatch.setattr(runtime._state, "groups_with_acts", True)
    assert groups._motion_supported(name), "a current support motion can carry an occluded free object"
    drawing["groups"]["3"]["supported"] = False
    assert not groups._motion_supported(name), "an unsupported group is not a measurement"
    context["own"] = True
    assert groups._motion_supported(name), "the object's own measured motion is sufficient"
    context["why"] = "held"
    assert not groups._motion_supported(name), "the arm prediction cannot verify itself"
    context["why"] = ""
    drawing["stamp"] -= runtime.GROUPS_DRAW_MAX_AGE_S
    assert not groups._motion_supported(name), "stale evidence cannot authorize following"


def test_motion_support_uses_evidence_from_the_pose_frame(monkeypatch):
    monkeypatch.setattr(groups.time, "time", lambda: 100.0)
    name = "target"
    feed = runtime._GroupsFeed(
        frames={name: deque([(99.9, np.eye(4))])},
        evidence={name: deque([(99.9, {"own": False, "why": "", "group_supported": False})])},
        drawing={"stamp": 100.0, "objects": {name: {"group": 3}}, "groups": {"3": {"supported": True}}},
    )
    monkeypatch.setattr(runtime._state, "groups_feed", feed)
    monkeypatch.setattr(runtime._state, "groups_with_acts", True)
    assert not groups._motion_supported(name), "a newer group fit cannot authorize the older unsupported pose"
    feed.evidence[name][-1][1]["own"] = True
    assert groups._motion_supported(name)
