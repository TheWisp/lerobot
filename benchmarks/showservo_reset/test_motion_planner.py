"""Offline planning regressions; no camera, server or motor connections."""

import unittest

import numpy as np
from motion_planner import JointPlanner, NoPathError, SearchLimits, interpolate


class JointSearchTests(unittest.TestCase):
    def test_routes_around_obstacle_that_blocks_straight_line(self):
        def clear(q):
            return not (abs(q[0]) < 1 and abs(q[1]) < 3)

        p = JointPlanner([-5, -5], [5, 5], clear, SearchLimits(0.1, 1, 1000, 30000))
        path = p.plan([-4, 0], [4, 0])
        np.testing.assert_allclose(path[[0, -1]], [[-4, 0], [4, 0]])
        self.assertTrue(all(clear(q) for q in path))
        self.assertGreater(np.max(np.abs(path[:, 1])), 3)
        self.assertLessEqual(np.max(np.abs(np.diff(path, axis=0))), 0.100001)

    def test_checks_between_clear_endpoints(self):
        p = JointPlanner([-2], [2], lambda q: abs(q[0]) >= 0.1, SearchLimits(0.05, 1, 20, 1000))
        self.assertFalse(p.edge(np.array([-1.0]), np.array([1.0])))
        with self.assertRaises(NoPathError):
            p.plan([-1], [1])

    def test_rejects_colliding_start_and_invalid_goal(self):
        p = JointPlanner([0, 0], [2, 2], lambda q: q.sum() > 0)
        for start, goal in [([0, 0], [1, 1]), ([1, 1], [3, 1]), ([1, 1], [np.nan, 1])]:
            with self.assertRaises(NoPathError):
                p.plan(start, goal)

    def test_locked_gripper_does_not_change_to_find_a_path(self):
        p = JointPlanner(
            [-5, -5, 65],
            [5, 5, 65],
            lambda q: not (abs(q[0]) < 1 and abs(q[1]) < 3),
            SearchLimits(0.1, 1, 1000, 30000),
        )
        path = p.plan([-4, 0, 65], [4, 0, 65])
        np.testing.assert_array_equal(path[:, 2], 65)

    def test_has_a_hard_work_budget(self):
        p = JointPlanner([-2], [2], lambda q: True, SearchLimits(0.001, 1, 20, 10))
        with self.assertRaisesRegex(NoPathError, "budget"):
            p.plan([-1], [1])

    def test_time_budget_is_checked(self):
        from unittest.mock import patch

        planner = JointPlanner([-2], [2], lambda q: True)
        with (
            patch("motion_planner.time.monotonic", side_effect=[0, 21]),
            self.assertRaisesRegex(NoPathError, "time budget"),
        ):
            planner.plan([-1], [1])

    def test_nonfinite_resolution_cannot_disable_edge_checks(self):
        for resolution in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                interpolate([0], [1], resolution)
            with self.assertRaises(ValueError):
                SearchLimits(resolution=resolution, extension=np.inf)

    def test_interpolation_rejects_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            interpolate([np.nan], [0], 1)


if __name__ == "__main__":
    unittest.main()
