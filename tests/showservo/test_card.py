# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The card is the only task-specific artifact, so it must reject malformed teaching
at load time and survive a round trip through the review screen's JSON unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.showservo.card import Card, Chapter, GoalRelation, Keypoint, Termination

from .conftest import TARGET_UV, make_team, unit_descriptor


def test_round_trip_preserves_every_field(d2_card, tmp_path):
    path = tmp_path / "card.json"
    d2_card.save(path)
    back = Card.load(path)

    assert back.name == d2_card.name
    assert len(back.chapters) == 1
    a, b = d2_card.chapters[0], back.chapters[0]
    assert (a.camera, a.name) == (b.camera, b.name)
    assert a.termination.type == b.termination.type and a.termination.params == b.termination.params
    assert (a.budget.seconds, a.budget.retries) == (b.budget.seconds, b.budget.retries)
    np.testing.assert_allclose(a.travel_dir, b.travel_dir)
    np.testing.assert_allclose(a.goal_relation.held_uv, b.goal_relation.held_uv)
    np.testing.assert_allclose(a.goal_relation.spread_uv, b.goal_relation.spread_uv)
    assert a.goal_relation.n_demos == b.goal_relation.n_demos
    for team in ("target", "held"):
        np.testing.assert_allclose(a.team_uv(team), b.team_uv(team))
        np.testing.assert_allclose(a.team_descriptors(team), b.team_descriptors(team), atol=1e-6)


def test_grasp_aperture_survives_round_trip(d1_card, tmp_path):
    path = tmp_path / "card.json"
    d1_card.save(path)
    assert Card.load(path).chapters[0].grasp_aperture_expected == pytest.approx(0.35)


def test_travel_dir_is_normalised_not_merely_accepted(d1_chapter):
    # A card written by hand in the review screen will carry un-normalised vectors;
    # the servo's back-off distance must not silently scale with the operator's typing.
    ch = Chapter(
        camera="top",
        teams={"target": make_team(TARGET_UV)},
        goal_relation=GoalRelation(held_uv=[[10.0, 10.0]]),
        travel_dir=[0.0, 0.0, -7.5],
        termination=Termination("contact"),
    )
    np.testing.assert_allclose(ch.travel_dir, [0.0, 0.0, -1.0])


def test_zero_travel_dir_is_refused():
    with pytest.raises(AssertionError, match="direction"):
        Chapter(
            camera="top",
            teams={"target": make_team(TARGET_UV)},
            goal_relation=GoalRelation(held_uv=[[10.0, 10.0]]),
            travel_dir=[0.0, 0.0, 0.0],
            termination=Termination("contact"),
        )


def test_unknown_termination_fails_at_load_not_at_second_forty():
    with pytest.raises(AssertionError, match="unknown termination"):
        Termination("wait_until_it_looks_right")


def test_chapter_without_target_team_is_refused():
    with pytest.raises(AssertionError, match="no target team"):
        Chapter(
            camera="top",
            teams={"target": []},
            goal_relation=GoalRelation(held_uv=[[1.0, 2.0]]),
            travel_dir=[0.0, 0.0, -1.0],
            termination=Termination("contact"),
        )


def test_goal_and_held_team_must_agree_in_length():
    # The servo indexes goal points against held points; a mismatch would servo to
    # the wrong feature rather than fail, which is the worst available outcome.
    with pytest.raises(AssertionError, match="held points"):
        Chapter(
            camera="top",
            teams={"target": make_team(TARGET_UV), "held": make_team(np.array([[5.0, 5.0]]))},
            goal_relation=GoalRelation(held_uv=[[1.0, 2.0], [3.0, 4.0]]),
            travel_dir=[0.0, 0.0, -1.0],
            termination=Termination("contact"),
        )


def test_unnormalised_descriptor_is_refused():
    with pytest.raises(AssertionError, match="L2-normalised"):
        Keypoint(uv=[1.0, 2.0], descriptor=np.array([3.0, 4.0], dtype=np.float32))


def test_empty_goal_is_refused():
    with pytest.raises(AssertionError, match="nothing to servo"):
        GoalRelation(held_uv=np.zeros((0, 2)))


def test_spread_is_ordinal_below_five_demos():
    few = GoalRelation(held_uv=[[1.0, 1.0]], spread_uv=[[0.5, 0.5]], n_demos=3)
    many = GoalRelation(held_uv=[[1.0, 1.0]], spread_uv=[[0.5, 0.5]], n_demos=5)
    assert not few.tolerance_is_calibrated
    assert many.tolerance_is_calibrated


def test_held_end_routes_by_card_shape_not_by_task_name(d1_chapter, d2_chapter):
    # This is the invariant that keeps the runtime task-agnostic: whether the moving
    # end is the gripper or a grasped object is read off the card's structure.
    assert d1_chapter.held_end == "gripper"
    assert d2_chapter.held_end == "held"


def test_team_descriptors_are_all_or_nothing():
    # A team where only some points carry descriptors cannot be matched coherently;
    # returning a short stack would silently misalign descriptors with points.
    mixed = [
        Keypoint(uv=[1.0, 1.0], descriptor=unit_descriptor(0)),
        Keypoint(uv=[2.0, 2.0]),
    ]
    ch = Chapter(
        camera="top",
        teams={"target": mixed},
        goal_relation=GoalRelation(held_uv=[[1.0, 2.0]]),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("contact"),
    )
    assert ch.team_descriptors("target") is None
    assert ch.team_uv("target").shape == (2, 2)


def test_absent_team_yields_an_empty_block_not_an_error(d1_chapter):
    assert d1_chapter.team_uv("held").shape == (0, 2)
    assert d1_chapter.team_descriptors("held") is None
