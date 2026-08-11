# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""§5's structural claim, made falsifiable: "if adding a task requires touching runtime
code, that is a bug in the runtime".

One driver runs a pick-shaped chapter (no held team, fission termination) and an
insertion-shaped one (held team, push-test termination) without naming either. Every
difference between the two runs is read off the card: which end moves comes from
``Chapter.held_end``, and how the chapter ends comes from ``Termination.type`` through
a registry. A third, never-before-seen card shape is run at the end to show the
registry is the extension point rather than the driver.

The plant is synthetic on purpose — this test is about the wiring, not about optics.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.fewshot.registration import Sim2
from lerobot.showservo.card import Budget, Chapter, GoalRelation, Termination
from lerobot.showservo.grouping import AttachmentMonitor, fit_team
from lerobot.showservo.monitor import ChapterMonitor, Event, State
from lerobot.showservo.servo import ConvergenceCertificate, JacobianEstimator, PIController, servo_error

from .conftest import TARGET_UV, make_team

J_TRUE = np.array([[2.4, -0.9, 0.3], [0.6, 1.8, -0.7]])


class Plant:
    """Two teams in one image. The held end moves with the joints; the target does not
    until it is grasped, after which it moves with the held end.

    ``grasp_at``/``seat_at`` are what the WORLD does, not what the driver knows: a
    block can be picked up and a hole cannot, and a peg can bottom out where a pure
    alignment never does. Setting them per scenario is the point — the asymmetry lives
    in the physics, and the driver below still never learns which task it is running.
    """

    def __init__(self, taught_target, taught_held, target_offset, *, grasp_at=None, seat_at=None):
        self.taught_target = np.asarray(taught_target, float)
        self.taught_held = np.asarray(taught_held, float)
        self.target_offset = np.asarray(target_offset, float)
        self.grasp_at, self.seat_at = grasp_at, seat_at
        self.q = np.zeros(J_TRUE.shape[1])
        self.attached = False
        self.seated = False
        self._grabbed_at = np.zeros(2)

    @property
    def held_offset(self) -> np.ndarray:
        return J_TRUE @ self.q

    def move(self, dq: np.ndarray) -> None:
        if self.seated:
            return  # a seated part does not move, however hard it is pushed
        self.q = self.q + dq

    def observe(self):
        held = self.taught_held + self.held_offset
        target = self.taught_target + self.target_offset
        if self.attached:
            target = target + (self.held_offset - self._grabbed_at)
        return target, held

    def latch(self, err_norm: float) -> None:
        if self.grasp_at is not None and not self.attached and err_norm < self.grasp_at:
            self.attached, self._grabbed_at = True, self.held_offset.copy()
        if self.seat_at is not None and not self.seated and err_norm < self.seat_at:
            self.seated = True


def _fission_detector(chapter, plant, taught_target):
    mon = AttachmentMonitor(sustain=int(chapter.termination.params.get("sustain", 3)))
    mon.reset(taught_target + plant.target_offset)

    def check(target_live, held_fit, _err):
        holder = Sim2.from_angle(0.0, t=tuple(plant.held_offset))
        ev = mon.update(target_live, np.ones(len(target_live), bool), holder)
        return ev is not None and ev.kind == "fission"

    return check


def _push_test_detector(chapter, plant, _taught_target):
    def check(_target_live, _held_fit, _err):
        return plant.seated

    return check


def _pose_hold_detector(chapter, _plant, _taught_target):
    tol = float(chapter.termination.params.get("tolerance_px", 2.0))
    hold = int(chapter.termination.params.get("frames", 5))
    run = {"n": 0}

    def check(_target_live, _held_fit, err):
        run["n"] = run["n"] + 1 if err < tol else 0
        return run["n"] >= hold

    return check


# The extension point. A new task shape adds a row here, never a branch in run_chapter.
DETECTORS = {
    "fission": _fission_detector,
    "push_test": _push_test_detector,
    "pose_hold": _pose_hold_detector,
}


def run_chapter(chapter: Chapter, plant: Plant, *, max_frames: int = 400):
    """The task-agnostic driver. Reads only card fields and certificates."""
    taught_target = chapter.team_uv("target")

    # Which points constitute the moving end is a property of the CARD's shape: a
    # chapter with a held team tracks the grasped object, one without tracks the
    # gripper, whose taught positions are the goal's own held points.
    taught_held = chapter.team_uv("held") if chapter.held_end == "held" else chapter.goal_relation.held_uv
    goal_held = chapter.goal_relation.held_uv

    monitor = ChapterMonitor(chapter)
    monitor.feed(Event.START)

    est = JacobianEstimator(n_joints=J_TRUE.shape[1], damping=1e-3)
    est.seed_from_probe(np.eye(3), np.eye(3) @ (J_TRUE * 0.6).T)  # deliberately wrong seed
    pi = PIController(kp=0.5, v_max=40.0)
    cert = ConvergenceCertificate(window=40, min_improvement=0.02)

    monitor.feed(Event.BIND_OK)
    terminated = DETECTORS[chapter.termination.type](chapter, plant, taught_target)

    for frame in range(max_frames):
        target_live, held_live = plant.observe()
        target_fit = fit_team(taught_target, target_live)
        held_fit = fit_team(taught_held, held_live)

        err = servo_error(goal_held, target_fit, held_fit)
        if not err.ok:
            monitor.feed(Event.TRACK_LOST, frame=frame)
            break

        cert.update(err.norm)
        monitor.log.err_norms.append(err.norm)
        plant.latch(err.norm)

        if terminated(target_live, held_fit, err.norm):
            monitor.feed(Event.TERMINATED, frame=frame)
            break
        if not cert.progressing:
            monitor.feed(Event.NO_PROGRESS, frame=frame)
            break

        u = pi.step(err.e_uv, dt=0.02)
        dq = est.solve(u)
        before = plant.held_offset.copy()
        plant.move(dq)
        est.update(dq, plant.held_offset - before)
        monitor.log.servo_energy += float(np.abs(u).sum())
    else:
        monitor.feed(Event.TIMEOUT, frame=max_frames)

    return monitor


def test_a_pick_shaped_card_runs_to_success(d1_chapter):
    plant = Plant(d1_chapter.team_uv("target"), d1_chapter.goal_relation.held_uv, [55.0, -30.0], grasp_at=6.0)
    monitor = run_chapter(d1_chapter, plant)

    assert monitor.state is State.SUCCEEDED, monitor.log.failure_class
    assert plant.attached, "the pick terminated on fission, so the object must be held"
    assert monitor.log.final_error <= 6.0


def test_an_insertion_shaped_card_runs_to_success_through_the_same_driver(d2_chapter):
    plant = Plant(d2_chapter.team_uv("target"), d2_chapter.team_uv("held"), [40.0, 35.0], seat_at=4.0)
    monitor = run_chapter(d2_chapter, plant)

    assert monitor.state is State.SUCCEEDED, monitor.log.failure_class
    assert plant.seated
    assert monitor.log.final_error <= 4.0


def test_a_task_shape_never_seen_before_needs_no_driver_change():
    # A pure alignment task: hold a pose, no grasp, no contact. It runs because the
    # card describes it, not because the driver was taught about it.
    chapter = Chapter(
        name="align-and-hold",
        camera="top",
        teams={"target": make_team(TARGET_UV)},
        goal_relation=GoalRelation(held_uv=np.array([[180.0, 70.0], [210.0, 92.0], [190.0, 110.0]])),
        travel_dir=[1.0, 0.0, 0.0],
        termination=Termination("pose_hold", {"tolerance_px": 2.0, "frames": 5}),
        budget=Budget(seconds=25.0, retries=2),
    )
    plant = Plant(chapter.team_uv("target"), chapter.goal_relation.held_uv, [-45.0, 20.0])
    monitor = run_chapter(chapter, plant)

    assert monitor.state is State.SUCCEEDED, monitor.log.failure_class
    assert monitor.log.final_error < 2.0


@pytest.mark.parametrize("offset", [[70.0, -50.0], [-60.0, 40.0], [25.0, 65.0], [-80.0, -20.0]])
def test_convergence_does_not_depend_on_which_way_the_target_moved(d2_chapter, offset):
    # The D1/D2 success curve is measured against displacement from the demo pose; a
    # loop that only converges from one quadrant would fake that curve.
    plant = Plant(d2_chapter.team_uv("target"), d2_chapter.team_uv("held"), offset, seat_at=4.0)
    monitor = run_chapter(d2_chapter, plant)
    assert monitor.state is State.SUCCEEDED, (offset, monitor.log.failure_class)


def test_the_certificate_trail_is_populated_for_analysis(d2_chapter):
    plant = Plant(d2_chapter.team_uv("target"), d2_chapter.team_uv("held"), [50.0, -40.0], seat_at=4.0)
    monitor = run_chapter(d2_chapter, plant)

    assert len(monitor.log.err_norms) >= 3
    assert monitor.log.servo_energy > 0.0
    assert monitor.log.err_norms[0] > monitor.log.err_norms[-1]
    assert [name for _, name in monitor.log.timeline][-1] == "terminated"
