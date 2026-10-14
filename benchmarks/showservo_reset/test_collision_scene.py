"""Geometric regressions with small analytic meshes; never use the rig or server."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import coal
except ImportError:
    coal = None

from collision_scene import Body, CollisionScene, box, opposing_jaw_contacts
from motion_planner import JointPlanner, SearchLimits


def pose(x=0, y=0, z=0):
    t = np.eye(4)
    t[:3, 3] = [x, y, z]
    return t


class BoxRobot:
    """A moving wrist and jaw, with Cartesian coordinates as the test joints."""

    lower, upper = np.full(3, -1.0), np.full(3, 1.0)
    pairs = []
    geom = SimpleNamespace(geometryObjects=[SimpleNamespace(name="L5_wrist"), SimpleNamespace(name="L6_jaw")])

    def valid_limits(self, q):
        return (
            q.shape == (3,) and np.isfinite(q).all() and (q >= self.lower).all() and (q <= self.upper).all()
        )

    def placements(self, q):
        return [box([0.01] * 3, pose(q[0], q[1], q[2] + 0.06)), box([0.01] * 3, self.tip(q))]

    def tip(self, q):
        return pose(*q)


@unittest.skipIf(coal is None, "optional Coal collision library required")
class CollisionTests(unittest.TestCase):
    def scene(self, bodies=None, obstacles=None, floor=None):
        return CollisionScene(
            BoxRobot(), obstacles or [], floor or [], bodies or {}, [0, 0, 0], [[-1, -1], [1, 1]]
        )

    def test_opposing_jaws_require_side_contacts_away_from_edges(self):
        body = Body("bar", np.array([0.10, 0.02, 0.02]), pose())
        left = box([0.012, 0.004, 0.01], pose(0, -0.013, 0))
        right = box([0.012, 0.004, 0.01], pose(0, 0.013, 0))
        self.assertTrue(opposing_jaw_contacts([left, right], body, [0, 1, 0])["ok"])
        top = box([0.012, 0.004, 0.01], pose(0, -0.005, 0.015))
        self.assertFalse(opposing_jaw_contacts([top, right], body, [0, 1, 0])["ok"])
        end = box([0.012, 0.004, 0.01], pose(0.049, -0.013, 0))
        self.assertFalse(opposing_jaw_contacts([end, right], body, [0, 1, 0])["ok"])
        upper = box([0.012, 0.004, 0.01], pose(0, 0, 0.014))
        lower = box([0.012, 0.004, 0.01], pose(0, 0, -0.014))
        self.assertFalse(opposing_jaw_contacts([upper, lower], body, [0, 1, 0])["ok"])

    def test_target_contact_does_not_exempt_wrist_or_other_objects(self):
        object_ = Body("cube", np.array([0.02] * 3), pose(0, 0, 0.10))
        s = self.scene({"cube": object_})
        self.assertFalse(s.check(np.array([0, 0, 0.10])))
        self.assertTrue(s.check(np.array([0, 0, 0.10]), contact="cube"))
        self.assertFalse(s.check(np.array([0, 0, 0.04]), contact="cube"))
        self.assertIn("L5", s.reason)
        s.bodies["other"] = Body("other", np.array([0.02] * 3), pose(0, 0, 0.10))
        self.assertFalse(s.check(np.array([0, 0, 0.10]), contact="cube"))

    def test_near_allows_margin_entry_but_not_intersection(self):
        s = self.scene({"cube": Body("cube", np.array([0.02] * 3), pose(0, 0, 0.10))})
        q = np.array([0.018, 0, 0.10])  # 3mm gap, less than the normal 7mm reserve
        self.assertFalse(s.check(q))
        self.assertTrue(s.check(q, near="cube"))
        self.assertFalse(s.check(np.array([0.010, 0, 0.10]), near="cube"))

    def test_payload_collision_even_when_the_arm_is_clear(self):
        s = self.scene(
            {"cube": Body("cube", np.array([0.03] * 3), pose(0.5, 0, 0.10))},
            [box([0.02] * 3, pose(0.04, 0, 0.10))],
        )
        q = np.array([0, 0, 0.10])
        hold = pose(0.04, 0, 0)
        self.assertTrue(s.check(q))
        self.assertFalse(s.check(q, attached=("cube", hold)))
        self.assertEqual(s.reason, "held object touches clutter")

    def test_payload_table_contact_allowed_only_in_support_phase(self):
        s = self.scene({"cube": Body("cube", np.array([0.02] * 3), pose(0.5, 0, 0.01))})
        hold = pose(0.04, 0, 0)
        self.assertFalse(s.check(np.array([0, 0, 0.01]), attached=("cube", hold)))
        self.assertTrue(s.check(np.array([0, 0, 0.01]), attached=("cube", hold), support=True))
        self.assertFalse(s.check(np.array([0, 0, 0.009]), attached=("cube", hold), support=True))

    def test_held_source_is_removed_but_other_body_remains(self):
        body = Body("cube", np.array([0.02] * 3), pose(0, 0, 0.10))
        s = self.scene({"cube": body})
        q = np.array([0, 0, 0.10])
        hold = pose(0.04, 0, 0)
        self.assertFalse(s.check(q))
        self.assertTrue(s.check(q, attached=("cube", hold)))
        s.bodies["other"] = Body("other", np.array([0.02] * 3), pose(0.04, 0, 0.10))
        self.assertFalse(s.check(q, attached=("cube", hold)))
        self.assertEqual(s.reason, "held object touches other")

    def test_planner_checks_link_collision_between_waypoints(self):
        s = self.scene(obstacles=[box([0.01, 0.1, 0.1], pose(0, 0, 0.10))])
        p = JointPlanner([-0.1, 0, 0.1], [0.1, 0, 0.1], s.check, SearchLimits(0.001, 0.01, 20, 1000))
        a, b = np.array([-0.05, 0, 0.1]), np.array([0.05, 0, 0.1])
        self.assertTrue(s.check(a) and s.check(b))
        self.assertFalse(p.edge(a, b))

    def test_floor_checks_arm_and_target_contact_cannot_disable_it(self):
        s = self.scene(floor=[box([0.5, 0.5, 0.05], pose(0, 0, -0.025))])
        self.assertFalse(s.check(np.array([0, 0, 0.001]), contact="cube", support=True))
        self.assertTrue(s.check(np.array([0, 0, 0.03])))

    def test_nonfinite_configuration_is_rejected_before_geometry(self):
        self.assertFalse(self.scene().check(np.array([0, np.nan, 0.1])))

    def test_missing_depth_is_an_obstacle_not_empty_space(self):
        frame = {"pts": np.full((2, 2, 3), np.nan), "plane": [0, 0, 0]}
        with patch("rig.pixel", return_value=(0, 0)):
            scene = CollisionScene.from_observation(BoxRobot(), {}, frame)
        self.assertFalse(scene.check(np.array([0, -0.2, 0.1])))
        self.assertIn("unobserved scene space", scene.reason)

    def test_oblique_self_occlusion_applies_at_the_hidden_tray_cell(self):
        # The self return and the hidden tray are in different XY cells.
        frame = {
            "pts": np.array([[[100.0, -300.0, 200.0], [np.nan] * 3]]),
            "plane": [0, 0, 0],
            "self_mask": np.array([[True, False]]),
        }
        with patch("rig.pixel", return_value=(0, 0)):
            scene = CollisionScene.from_observation(BoxRobot(), {}, frame)
        self.assertTrue(scene.check(np.array([0, -0.2, 0.1])), scene.reason)

        # Seeing the arm along that ray cannot erase a real clutter return
        # already present in the tray cell, even if the arm occludes its centre.
        frame["pts"][0, 1] = [0, -200, 100]
        with patch("rig.pixel", return_value=(0, 0)):
            scene = CollisionScene.from_observation(BoxRobot(), {}, frame)
        self.assertFalse(scene.check(np.array([0, -0.2, 0.1])))
        self.assertIn("scene collision", scene.reason)

    def test_self_occlusion_requires_a_finite_raised_depth_return(self):
        for point in ([np.nan] * 3, [100.0, -300.0, 0.0]):
            with self.subTest(point=point):
                frame = {"pts": np.array([[point]]), "plane": [0, 0, 0], "self_mask": [[True]]}
                with patch("rig.pixel", return_value=(0, 0)):
                    scene = CollisionScene.from_observation(BoxRobot(), {}, frame)
                self.assertFalse(scene.check(np.array([0, -0.2, 0.1])))

    def test_body_corners_follow_full_rotation(self):
        t = pose(0.1, 0.2, 0.3)
        t[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        c = Body("cube", np.array([0.02, 0.04, 0.06]), t).corners()
        np.testing.assert_allclose(np.ptp(c, axis=0), [0.04, 0.02, 0.06])
        np.testing.assert_allclose(c.mean(axis=0), t[:3, 3])


if __name__ == "__main__":
    unittest.main()
