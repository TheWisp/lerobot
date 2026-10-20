"""The sealed-demo adapter must preserve event meaning when clocks are sampled."""

import numpy as np

from benchmarks.flash_prepare_legacy import _interval_layout, _interval_reference, _nearest_index
from lerobot.policies.flash.skill import Relation


def test_lift_uses_grasp_anchor_until_nearest_recorded_boundary():
    # A 9.7-second event is represented by the 9.71-second arm sample. The
    # previous float comparison incorrectly made this entire lift target-relative.
    times = np.array([0.0, 7.64, 8.10, 9.68, 9.71, 11.45, 17.17, 19.66])
    keypoints = [
        {"t": 7.64, "move": "line"},
        {"t": 8.10, "move": "follow"},
        {"t": 9.70, "move": "lift"},
        {"t": 11.45, "move": "line"},
        {"t": 19.66, "move": "line"},
    ]
    grasp = _nearest_index(times, 7.64)
    release = _nearest_index(times, 17.17)
    anchor_end = _nearest_index(times, 9.70)
    layout = _interval_layout(times, keypoints, grasp, release, anchor_end)

    assert anchor_end == 4
    assert [
        (first, last, start_anchored, carrying) for first, last, _, _, start_anchored, carrying in layout
    ] == [
        (0, 1, False, False),
        (1, 2, True, False),
        (2, 4, True, False),
        (4, 5, False, True),
        (5, 6, False, True),
        (6, 7, False, False),
    ]


def test_world_relative_retreat_does_not_follow_target_after_release():
    role, relation, anchored = _interval_reference(
        first=552,
        grasp=228,
        end_key={"t": 19.66, "frame": "world", "move": "line"},
        start_anchored=False,
        carrying=False,
        held="held",
        target="target",
    )
    assert (role, relation, anchored) == ("__start__", Relation.EE_TO_START, True)
