# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the demo believed of the objects it held, by the rule an act believes its own (act_execution.md, "The demo's
beliefs"): how each sat in the demo's gripper, sample by sample, from the demo's perception (lerobot.showservo
.perception) through the engine's own World and hold.

As in an act: an object's pose is its views' (World); at the grasp, how it sits in the gripper is where it was against
the fingertip, and every view of it while held updates that (Steady, the hold an act keeps), its threshold the noise
its views showed at rest; between views it is carried as it sat. The act then puts its held object where the demo's
was by how it sat in the demo's gripper then (Engine._fix), not by how it lay at the grasp.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .scene import Measurement, SceneLog
from .timeline import Demo, Event
from .world import Steady, World


def grips(
    demo: Demo,
    events: list[Event],
    views: dict[str, list[tuple[float, np.ndarray]]],
    centres: dict[str, np.ndarray],
    symmetry: dict[str, tuple[dict, np.ndarray]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per object the demo held: how it sat in the gripper at each demo sample (motion since the demo, fingertip
    frame), from the grasp to its release and as it was let go after. ``views`` are the demo's taken views of each
    object, (demo time, motion since the demo, base frame), in time order; ``centres`` and ``symmetry`` as the act's
    World has them. With what it was made of, per object: views while held, and how far how it sat moved."""
    scene = SceneLog(None, keep_s=np.inf)
    world = World(scene, {n: np.eye(4) for n in centres}, centres, symmetry=symmetry)
    pending = sorted((t, n, m) for n, vs in views.items() for t, m in vs)
    fed = 0

    def feed(until: float) -> None:
        nonlocal fed
        while fed < len(pending) and pending[fed][0] <= until:
            t, n, m = pending[fed]
            if n in centres:
                scene.add_measurement(n, Measurement(t=t, motion=m))
            fed += 1
        world.update()

    out: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    for e in sorted((e for e in events if e.kind == "grasp"), key=lambda e: e.i):
        if e.obj not in centres:
            continue
        release = next((r.i for r in events if r.kind == "release" and r.grasp_i == e.i), len(demo.t) - 1)
        t_grasp = float(demo.t[e.i])
        feed(t_grasp)
        world.touched(e.obj, t_grasp)
        reference = np.linalg.inv(demo.tips[e.i])

        def as_held(i: int, motion: np.ndarray, obj: str = e.obj, ref: np.ndarray = reference) -> np.ndarray:
            return world.equivalent(obj, np.linalg.inv(demo.tips[i]) @ motion, ref)

        hold = Steady(as_held(e.i, world.motion(e.obj)), centres[e.obj])
        sat = np.tile(hold.estimate, (len(demo.t), 1, 1))
        held_views = [(t, m) for t, n, m in pending if n == e.obj and t_grasp < t <= float(demo.t[release])]
        j = 0
        for i in range(e.i, release + 1):
            while j < len(held_views) and held_views[j][0] <= float(demo.t[i]):
                t, m = held_views[j]
                k = int(np.argmin(np.abs(demo.t - t)))
                hold.add(t, as_held(k, m), world.move_threshold(e.obj))
                j += 1
            sat[i] = hold.estimate
        sat[release + 1 :] = hold.estimate
        sat[: e.i] = sat[e.i]
        out[e.obj] = sat
        centre = np.append(centres[e.obj], 1.0)
        moved = np.linalg.norm((sat[release] @ centre)[:3] - (sat[e.i] @ centre)[:3])
        info[e.obj] = {"views_held": len(held_views), "moved_mm": round(float(moved) * 1000.0, 1)}
    return out, info


def views_from(perception, origin: float, base_from_camera: np.ndarray, names: dict[str, str] | None = None):
    """The taken views of a recording's perception as :func:`grips` takes them: per object, (demo time, motion since
    its designation, base frame). ``origin`` is the demo's start (wall clock); ``names`` renames the perception's
    objects to the act's."""
    inv = np.linalg.inv(base_from_camera)
    out: dict[str, list[tuple[float, np.ndarray]]] = {}
    for row in perception.rows:
        for name, o in (row.get("objects") or {}).items():
            if not o.get("taken"):
                continue
            cam = np.asarray(o["candidates"][0]["motion"], dtype=float)
            out.setdefault((names or {}).get(name, name), []).append(
                (float(row["t"]) - origin, base_from_camera @ cam @ inv)
            )
    return out
