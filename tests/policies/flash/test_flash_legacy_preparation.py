"""The sealed-demo adapter must preserve event meaning when clocks are sampled."""

import numpy as np
import pytest

from lerobot.policies.flash.sealed_demo_import import (
    _apply_reference_polygons,
    _endpoint_requirement,
    _interval_layout,
    _interval_reference,
    _nearest_index,
    _symmetry_from_annotation,
)
from lerobot.policies.flash.skill import EndpointRequirement, Relation


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


def test_best_effort_hover_is_a_preferred_endpoint():
    assert _endpoint_requirement({"constraint": "best_effort"}) == EndpointRequirement.PREFERRED
    assert _endpoint_requirement({"constraint": "pose"}) == EndpointRequirement.REQUIRED


def test_legacy_annotation_preserves_end_for_end_symmetry():
    points = np.array([[-0.04, 0.0, 0.0], [0.0, 0.001, 0.0], [0.04, 0.0, 0.0]])
    symmetry = _symmetry_from_annotation(
        "held", {"axis_kind": "long_axis", "ends_interchangeable": True}, points, np.eye(4)
    )
    assert symmetry.ends_interchangeable
    assert abs(symmetry.axis_object[0]) == pytest.approx(1, abs=0.001)
    assert np.isfinite(symmetry.equivalent_transform(exchange_ends=True)).all()

    without_swap = _symmetry_from_annotation("held", {"axis_kind": "long_axis"}, points, np.eye(4))
    with pytest.raises(ValueError, match="not annotated as interchangeable"):
        without_swap.equivalent_transform(exchange_ends=True)


def test_reference_polygon_replaces_only_its_role_and_checks_image_bounds():
    original = {"held": np.zeros((12, 12), dtype=bool), "target": np.ones((12, 12), dtype=bool)}
    replaced = _apply_reference_polygons(
        original, {"held": [[2, 2], [9, 2], [9, 9], [2, 9]]}, set(original), (12, 12)
    )
    assert replaced["held"][5, 5] and not replaced["held"][0, 0]
    assert np.array_equal(replaced["target"], original["target"])
    assert not original["held"].any()
    with pytest.raises(ValueError, match="in-frame"):
        _apply_reference_polygons(original, {"held": [[2, 2], [20, 2], [9, 9]]}, set(original), (12, 12))
