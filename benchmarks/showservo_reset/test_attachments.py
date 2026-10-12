"""Rigid accessory geometry must move with the arm and remain collision checked."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
from collision_scene import Body, CollisionScene, MeshRobot, box


@unittest.skipUnless(
    all(importlib.util.find_spec(n) for n in ("pinocchio", "coal", "pink")),
    "optional geometry dependencies required",
)
class AttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
        from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES, JointAlignment

        spec = json.loads((Path(__file__).parent / "fixtures/planner_calibration.json").read_text())
        alignment = {n: JointAlignment(**v) for n, v in spec["alignment"].items()}
        kin = make_so107_arm_kinematics(alignment, tip_offset=np.array(spec["tip_offset"]))
        cls.k = {
            "kin": kin,
            "names": list(MOTOR_NAMES),
            "lo": np.rad2deg(kin._inner._inner.q_lo),
            "hi": np.rad2deg(kin._inner._inner.q_hi),
        }
        cls.calibration = spec["motor_ranges"]
        cls.q = np.array([0, -45, 74, 2, -41, -12, 93.5])
        # Synthetic accessory outside native links; not a hardware calibration.
        cls.spec = {
            "name": "camera",
            "parent_geometry": "L6_1_0",
            "size": [0.02, 0.03, 0.04],
            "pose": np.eye(4).tolist(),
        }
        cls.spec["pose"][0][3] = 0.10
        cls.spec["pose"][1][3] = -0.34
        cls.spec["pose"][2][3] = 0.16

    def robot(self, specs=None):
        return MeshRobot(self.k, self.calibration, [self.spec] if specs is None else specs)

    def scene(self, robot, bodies=None, obstacles=None):
        return CollisionScene(robot, obstacles or [], [], bodies or {}, [0, 0, -1], [[-1, -1], [1, 1]])

    def test_accessory_follows_parent_without_mutating_cached_robot(self):
        original = self.k["kin"]._inner._inner.robot.collision_model.ngeoms
        r = self.robot()
        self.assertEqual(self.k["kin"]._inner._inner.robot.collision_model.ngeoms, original)
        self.assertEqual(self.robot().geom.ngeoms, original + 1)
        locations = []
        for turn in (-12, 30):
            q = self.q.copy()
            q[5] = turn
            r.placements(q)
            parent = r.gdata.oMg[r.geom.getGeometryId("L6_1_0")].homogeneous
            actual = r.gdata.oMg[-1].homogeneous
            np.testing.assert_allclose(actual, parent @ np.asarray(self.spec["pose"]), atol=1e-12)
            locations.append(actual[:3, 3].copy())
        self.assertGreater(np.linalg.norm(locations[1] - locations[0]), 0.03)

    def test_accessory_is_masked_but_farther_depth_is_preserved(self):
        r = self.robot()
        r.placements(self.q)
        t = r.gdata.oMg[-1]
        local = np.array([[0, 0, 0], [0.015, 0, 0], [0.04, 0, 0], [np.nan] * 3])
        points = (local @ t.rotation.T + t.translation)[None]
        np.testing.assert_array_equal(r.observed_mask(points, self.q), [[True, True, False, False]])

    def test_contact_never_exempts_accessory_from_objects_or_payload(self):
        r = self.robot()
        r.placements(self.q)
        at = r.gdata.oMg[-1].homogeneous.copy()
        body = Body("cube", np.full(3, 0.005), at)
        bare = self.robot([])
        self.assertTrue(self.scene(bare, {"cube": body}).check(self.q))
        s = self.scene(r, {"cube": body})
        self.assertFalse(s.check(self.q, contact="cube"))
        self.assertEqual(s.reason, "camera touches cube")
        self.assertFalse(s.check(self.q, attached=("cube", np.linalg.inv(r.tip(self.q)) @ at)))
        self.assertEqual(s.reason, "held object touches arm")
        s = self.scene(r, obstacles=[box(body.size, at)])
        self.assertFalse(s.check(self.q))
        self.assertEqual(s.reason, "scene collision camera")

    def test_accessory_keeps_adjacent_joint_collision_pairs(self):
        r = self.robot()
        camera = r.geom.getGeometryId("camera")
        names = {r.geom.geometryObjects[j if i == camera else i].name for i, j in r.pairs if camera in (i, j)}
        self.assertIn("L5_1_0", names)
        self.assertIn("L7_1_0", names)
        self.assertNotIn("L6_1_0", names)

    def test_invalid_geometry_and_contact_exemption_names_are_rejected(self):
        for key, value in [
            ("name", "L6_camera"),
            ("size", [0.1, 0, 0.1]),
            ("pose", np.zeros((4, 4)).tolist()),
        ]:
            spec = copy.deepcopy(self.spec)
            spec[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.robot([spec])
