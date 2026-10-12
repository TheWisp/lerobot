"""Bounded bidirectional joint-space search, independent of the rig and hardware.

Every returned segment is checked at ``resolution`` in each joint. This is
discrete collision checking, not a continuous swept-volume guarantee. The
collision model must reserve clearance for interpolation and tracking error.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class NoPathError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchLimits:
    resolution: float = 0.5  # motor degrees
    extension: float = 8.0
    iterations: int = 1500
    checks: int = 60000
    seed: int = 0
    seconds: float = 20.0

    def __post_init__(self):
        if not (
            np.isfinite([self.resolution, self.extension]).all()
            and 0 < self.resolution <= self.extension
            and self.iterations > 0
            and self.checks > 0
            and 0 < self.seconds < float("inf")
        ):
            raise ValueError("positive resolution, extension and search budgets required")


def interpolate(start, goal, resolution):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if (
        start.shape != goal.shape
        or not np.isfinite([start, goal]).all()
        or not np.isfinite(resolution)
        or resolution <= 0
    ):
        raise ValueError("finite, equally shaped endpoints and positive resolution required")
    n = max(1, int(np.ceil(np.max(np.abs(goal - start)) / resolution)))
    return np.linspace(start, goal, n + 1)


class JointPlanner:
    """RRT-Connect with bounded work and deterministic sampling; no arm calls."""

    def __init__(self, lower, upper, valid: Callable[[np.ndarray], bool], limits=SearchLimits()):
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        if (
            self.lower.ndim != 1
            or self.lower.shape != self.upper.shape
            or not np.isfinite([self.lower, self.upper]).all()
            or (self.lower > self.upper).any()
        ):
            raise ValueError("finite ordered joint bounds required")
        self.valid = valid
        self.limits = limits
        self.checks = 0
        self.started = time.monotonic()

    def state(self, q):
        if time.monotonic() - self.started > self.limits.seconds:
            raise NoPathError("collision-search time budget exhausted")
        self.checks += 1
        if self.checks > self.limits.checks:
            raise NoPathError("collision-check budget exhausted")
        return bool(
            q.shape == self.lower.shape
            and np.isfinite(q).all()
            and (q >= self.lower).all()
            and (q <= self.upper).all()
            and self.valid(q)
        )

    def edge(self, a, b):
        return all(self.state(q) for q in interpolate(a, b, self.limits.resolution))

    def plan(self, start, goal):
        self.checks = 0
        self.started = time.monotonic()
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not self.state(start):
            raise NoPathError("start is outside limits or in collision")
        if not self.state(goal):
            raise NoPathError("goal is outside limits or in collision")
        if self.edge(start, goal):
            return interpolate(start, goal, self.limits.resolution)
        rng = np.random.default_rng(self.limits.seed)
        trees = [([start], [-1]), ([goal], [-1])]

        def extend(tree, target):
            points, parents = tree
            i = int(np.argmin(np.linalg.norm(np.asarray(points) - target, axis=1)))
            d = target - points[i]
            step = points[i] + d * min(1.0, self.limits.extension / max(np.max(np.abs(d)), 1e-12))
            if not self.edge(points[i], step):
                return None, False
            points.append(step)
            parents.append(i)
            return len(points) - 1, np.max(np.abs(step - target)) < 1e-9

        def backtrace(tree, index):
            points, parents = tree
            chain = []
            while index >= 0:
                chain.append(points[index])
                index = parents[index]
            return chain[::-1]

        for iteration in range(self.limits.iterations):
            a = iteration % 2
            target = trees[1 - a][0][0] if iteration % 10 == 0 else rng.uniform(self.lower, self.upper)
            ia, _ = extend(trees[a], target)
            if ia is None:
                continue
            for _ in range(100):
                ib, reached = extend(trees[1 - a], trees[a][0][ia])
                if ib is None:
                    break
                if reached:
                    first, second = backtrace(trees[a], ia), backtrace(trees[1 - a], ib)
                    waypoints = first + second[-2::-1]
                    if a == 1:
                        waypoints.reverse()
                    # Deterministic shortcuts are collision-checked too.
                    for _ in range(30):
                        if len(waypoints) < 3:
                            break
                        i, j = sorted(rng.choice(len(waypoints), 2, replace=False))
                        if j > i + 1 and self.edge(waypoints[i], waypoints[j]):
                            waypoints[i + 1 : j] = []
                    return np.concatenate(
                        [
                            interpolate(x, y, self.limits.resolution)[:-1]
                            for x, y in zip(waypoints, waypoints[1:], strict=False)
                        ]
                        + [np.asarray(waypoints[-1:])]
                    )
        raise NoPathError("no collision-free connection within search budget")
