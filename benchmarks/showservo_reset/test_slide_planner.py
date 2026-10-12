"""Recorded-scene, no-motion regression tests for the slide fallback."""

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import reset_policy as p
from slide_planner import SlidePlanner, hull


class Slides(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        q = p.Policy.__new__(p.Policy)
        q.vision = p.Vision()
        q.failed_yaws = {n: [] for n in p.NAMES}
        f = Path(__file__).parent / "fixtures"
        q.scene, q.frame = q.vision.see(
            cv2.cvtColor(cv2.imread(str(f / "slide_after.jpg")), cv2.COLOR_BGR2RGB),
            np.load(f / "slide_after_depth.npz")["depth"],
            False,
        )
        cls.q = q

    def test_reachable_corridor_after_blocker_is_vacated(self):
        with patch.object(p.rig, "move", side_effect=AssertionError("Motion in offline test")):
            planner = SlidePlanner(self.q, "gamepad", vacated=("cube",))
            plan = planner.find()
        self.assertIsNotNone(plan)
        for key in ("start", "end", "low_entry", "withdraw"):
            self.assertGreaterEqual(plan[key][2], p.rig.TRAY_Z + 2)
            self.assertLessEqual(np.linalg.norm(plan[key][:2]), planner.tip_radius_limit)
        for jaw in np.array(plan["jaw_polygons"], dtype=object):
            jaw = np.asarray(jaw, float)
            self.assertTrue(
                planner.safe_surface(hull(np.vstack([jaw + plan["start"][:2], jaw + plan["end"][:2]])))
            )
        self.assertGreater(planner.stats["reachable_paths"], 0)

    def test_unknown_floor_and_cube_are_rejected(self):
        planner = SlidePlanner(self.q, "gamepad")
        self.assertFalse(planner.safe_surface(planner.other[0]))
        planner.support[:] = 0
        self.assertFalse(planner.safe_surface(planner.object))

    def test_stack_is_never_slid(self):
        planner = SlidePlanner(self.q, "gamepad")
        planner.src = dict(planner.src, stacked=True)
        self.assertIsNone(planner.find())

    def test_counterfactual_plan_cannot_execute(self):
        q = self.q
        q.held = None
        with self.assertRaises(p.Unsafe):
            q.slide({"name": "gamepad", "vacated": ["cube"]})

    def test_actual_floor_error_aborts(self):
        q = self.q
        q.slide_floor_guard = True
        q.log = {}
        position = [*q.scene["gamepad"]["centre"], q.tray(q.scene["gamepad"]["centre"])]
        with (
            patch.object(
                p.rig, "state", return_value={"connected": True, "position": position, "max_motor_temp_c": 30}
            ),
            self.assertRaises(p.Unsafe),
        ):
            q.state()
        q.slide_floor_guard = False


if __name__ == "__main__":
    unittest.main()
