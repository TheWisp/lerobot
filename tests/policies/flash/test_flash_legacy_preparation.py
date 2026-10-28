"""The sealed-demo adapter must preserve event meaning when clocks are sampled."""

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.sealed_demo_import import (
    _apply_reference_polygons,
    _endpoint_requirement,
    _frame,
    _interval_layout,
    _interval_reference,
    _nearest_index,
    _reference_observations,
    _symmetry_from_annotation,
    _turn_for_interval,
    explicit_paths_from_prepared_skill,
)
from lerobot.policies.flash.skill import EndpointRequirement, Relation


def test_sealed_rgbd_frame_uses_recorded_camera_time(tmp_path):
    (tmp_path / "recording/rgb").mkdir(parents=True)
    (tmp_path / "recording/depth").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "recording/rgb/000007.jpg"), np.zeros((2, 2, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "recording/depth/000007.png"), np.full((2, 2), 1000, np.uint16))

    frame = _frame(tmp_path, 7, np.eye(3), np.eye(4), "calibration", 1.23)

    assert frame.frame_id == 7
    assert frame.source_time_s == frame.available_time_s == 1.23


def test_prepared_skill_auditor_imports_the_production_legacy_adapter():
    from benchmarks.flash_audit_legacy_skill import audit

    assert callable(audit)


def test_reference_observations_use_each_objects_clear_frame():
    from types import SimpleNamespace

    class Masks:
        def __init__(self):
            self.seen = []
            self.resets = 0

        def reset_tracking(self):
            self.resets += 1

        def masks(self, frame):
            self.seen.append(frame.frame_id)
            roles = ("held",) if frame.frame_id == 3 else ("target", "moved_obstacle")
            return {role: np.ones((5, 5), dtype=bool) for role in roles}

    masks = Masks()
    selected = _reference_observations(
        {"held": "dowel", "target": "ring", "moved_obstacle": "block"},
        {"held": 3, "target": 12, "moved_obstacle": 12},
        lambda index: SimpleNamespace(frame_id=index, depth=np.zeros((5, 5))),
        masks,
        {},
    )
    assert masks.seen == [3, 12]
    assert masks.resets == 1
    assert {role: frame.frame_id for role, (frame, _) in selected.items()} == {
        "held": 3,
        "target": 12,
        "moved_obstacle": 12,
    }


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


def test_grasp_event_does_not_override_the_authored_path():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    line = [{"t": 2.0, "move": "line"}, {"t": 3.0, "move": "line"}]
    follow = [{"t": 2.0, "move": "follow"}, {"t": 3.0, "move": "line"}]

    assert _interval_layout(times, line, grasp=2, release=3, start_anchor_end=2)[0][3] is False
    assert _interval_layout(times, follow, grasp=2, release=3, start_anchor_end=2)[0][3] is True


def test_legacy_compiled_motion_becomes_explicit_without_changing_other_intervals():
    times = np.arange(5, dtype=float)
    old = {
        "keypoints": [
            {"t": 1.0, "move": "line"},
            {"t": 3.0, "move": "line"},
            {"t": 4.0, "move": "line"},
        ]
    }
    skill = {
        "intervals": [
            {"mode": mode, "samples": [{"source_frame_index": first}, {"source_frame_index": last}]}
            for first, last, mode in [
                (0, 1, "connection"),
                (1, 2, "demonstrated"),
                (2, 3, "connection"),
                (3, 4, "connection"),
            ]
        ]
    }
    upgraded, migrated = explicit_paths_from_prepared_skill(old, skill, times)
    assert migrated
    assert old["keypoints"] == [
        {"t": 1.0, "move": "line"},
        {"t": 3.0, "move": "line"},
        {"t": 4.0, "move": "line"},
    ]
    assert upgraded["path_selection_version"] == 2
    assert [(point["t"], point["move"]) for point in upgraded["keypoints"]] == [
        (1.0, "line"),
        (2.0, "follow"),
        (3.0, "line"),
        (4.0, "line"),
    ]
    old_layout = _interval_layout(times, old["keypoints"], 2, 4, 2, implicit_grasp_follow=True)
    new_layout = _interval_layout(times, upgraded["keypoints"], 2, 4, 2)
    assert [(first, last, replay) for first, last, _, replay, _, _ in new_layout] == [
        (first, last, replay) for first, last, _, replay, _, _ in old_layout
    ]
    assert explicit_paths_from_prepared_skill(upgraded, skill, times) == (upgraded, False)


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


def test_authored_reference_wins_over_grasp_timing_when_not_start_anchored():
    before_grasp = _interval_reference(
        first=20,
        grasp=80,
        end_key={"frame": "fixture", "move": "line"},
        start_anchored=False,
        carrying=False,
        held="tool",
        target="fixture",
    )
    during_carry = _interval_reference(
        first=120,
        grasp=80,
        end_key={"frame": "tool", "move": "line"},
        start_anchored=False,
        carrying=True,
        held="tool",
        target="fixture",
    )
    assert before_grasp == ("fixture", Relation.EE_TO_OBJECT, False)
    assert during_carry == ("tool", Relation.EE_TO_OBJECT, False)


def test_additional_tracked_object_can_anchor_a_key_frame():
    roles = {"tool", "receiver", "moving_fixture"}
    before = _interval_reference(
        first=20,
        grasp=80,
        end_key={"frame": "moving_fixture", "move": "line"},
        start_anchored=False,
        carrying=False,
        held="tool",
        target="receiver",
        available_roles=roles,
    )
    while_carrying = _interval_reference(
        first=120,
        grasp=80,
        end_key={"frame": "moving_fixture", "move": "follow"},
        start_anchored=False,
        carrying=True,
        held="tool",
        target="receiver",
        available_roles=roles,
    )
    assert before == ("moving_fixture", Relation.EE_TO_OBJECT, False)
    assert while_carrying == ("moving_fixture", Relation.HELD_TO_TARGET, False)
    with pytest.raises(ValueError, match="not a prepared object"):
        _interval_reference(
            first=20,
            grasp=80,
            end_key={"frame": "unknown", "move": "line"},
            start_anchored=False,
            carrying=False,
            held="tool",
            target="receiver",
            available_roles=roles,
        )


def test_start_anchor_still_overrides_old_keypoint_reference():
    assert _interval_reference(
        first=80,
        grasp=80,
        end_key={"frame": "tool", "move": "follow"},
        start_anchored=True,
        carrying=False,
        held="tool",
        target="fixture",
    ) == ("__start__", Relation.EE_TO_START, True)


def test_best_effort_hover_is_a_preferred_endpoint():
    assert _endpoint_requirement({"constraint": "best_effort"}) == EndpointRequirement.PREFERRED
    assert _endpoint_requirement({"constraint": "pose"}) == EndpointRequirement.REQUIRED


def test_turn_rule_applies_to_each_complete_interval_of_the_same_reference():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    rule = {
        "rule_id": "turn-1",
        "reference_role": "fixture",
        "start_s": 0.0,
        "end_s": 2.0,
        "pivot_reference_m": [0.01, 0.02, 0.03],
        "axis_reference": [0, 0, 1],
        "min_angle_rad": -np.pi,
        "max_angle_rad": np.pi,
    }
    first = _turn_for_interval([rule], times, 0, 1, "fixture")
    second = _turn_for_interval([rule], times, 1, 2, "fixture")
    assert first == second
    assert _turn_for_interval([rule], times, 2, 3, "fixture") is None
    with pytest.raises(ValueError, match="different reference"):
        _turn_for_interval([rule], times, 1, 2, "other")
    with pytest.raises(ValueError, match="interval boundaries"):
        _turn_for_interval([rule], times, 1, 3, "fixture")


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
