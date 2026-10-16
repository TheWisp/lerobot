"""Delayed group catch-up must not repeat movement already accepted visually."""

from collections import deque

import numpy as np

from lerobot.gui.api.manipulation import groups, runtime


def test_visual_override_blocks_double_motion_until_new_consistent_observation(monkeypatch):
    feed = runtime._GroupsFeed()
    monkeypatch.setattr(runtime._state, "groups_feed", feed)
    monkeypatch.setattr(runtime._state, "groups_with_acts", True)
    old, moved = np.eye(4), np.eye(4)
    moved[0, 3] = 0.02
    feed.frames["object"] = deque([(1.0, old), (2.0, moved)])
    check = {
        "object": "object",
        "constraint": {"group_override": {"pose_support": 0.9}},
        "corrected_groups": {"state": "disagreement"},
    }
    groups._accept_transport(check, 1.0)
    np.testing.assert_array_equal(groups._carried("object", 1.0), np.eye(4))
    assert "withheld" in check["group_transport"]
    assert not groups._motion_supported("object")
    # An unrelated object still follows group motion.
    feed.frames["other"] = feed.frames["object"]
    np.testing.assert_allclose(groups._carried("other", 1.0), moved)
    check["constraint"] = {}
    check["corrected_groups"] = {"state": "unavailable"}
    groups._accept_transport(check, 1.5)
    np.testing.assert_array_equal(groups._carried("object", 1.5), np.eye(4))
    check["corrected_groups"] = {"state": "consistent"}
    groups._accept_transport(check, 2.0)
    later = moved.copy()
    later[1, 3] = 0.01
    feed.frames["object"].append((3.0, later))
    expected = np.eye(4)
    expected[1, 3] = 0.01
    np.testing.assert_allclose(groups._carried("object", 2.0), expected)
