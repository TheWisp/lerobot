"""End-to-end planning on an explicitly idealized tabletop, using frozen calibration.

No camera/GUI calls, serial port, or user's current calibration are used.
"""

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from collision_scene import CollisionScene, MeshRobot, box, collides
from motion_planner import NoPathError
from planned_reset import ResetPlanner


def _fixture_pinch(jaws, body, closing_axis):
    """Keep the old positive fixture for downstream route tests only."""
    target = box(body.size, body.pose)
    return {"ok": all(collides(jaw, target, 0.002) for jaw in jaws), "reason": "fixture proximity"}


@unittest.skipUnless(
    all(importlib.util.find_spec(n) for n in ("pinocchio", "coal", "pink")),
    "optional Pinocchio, Coal, and Pink dependencies required",
)
class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        import rig

        sys.path.insert(0, rig.SSW_SRC)
        from reset_policy import Vision

        from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
        from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES, JointAlignment

        spec = json.loads((Path(__file__).parent / "fixtures/planner_calibration.json").read_text())
        alignment = {n: JointAlignment(**v) for n, v in spec["alignment"].items()}
        kin = make_so107_arm_kinematics(alignment, tip_offset=np.array(spec["tip_offset"]))
        k = {
            "kin": kin,
            "names": list(MOTOR_NAMES),
            "lo": np.rad2deg(kin._inner._inner.q_lo),
            "hi": np.rad2deg(kin._inner._inner.q_hi),
        }
        with patch.object(rig, "_call", side_effect=AssertionError("offline planner called GUI")):
            cls.robot = MeshRobot(k, spec["motor_ranges"])
            target = rig.target()
            objects, frame = Vision().see(target["rgb"], target["depth"], True)
        recorded = CollisionScene.from_observation(cls.robot, objects, frame)
        # This positive fixture intentionally contains the tray and two objects
        # only. It does NOT certify clearance in the recorded cluttered scene.
        cls.scene = CollisionScene(
            cls.robot, [], recorded.floor, recorded.bodies, recorded.plane, recorded.bounds
        )
        cls.start = np.array([rig.READY_MOTOR[n] for n in cls.robot.names])

    def test_a_proximity_only_grasp_is_refused(self):
        goal = copy.deepcopy(self.scene.bodies["gamepad"])
        goal.pose[0, 3] += 0.020
        planner = ResetPlanner(self.scene)
        with self.assertRaisesRegex(NoPathError, "central grasps rejected"):
            planner.transfer("gamepad", goal, self.start)
        self.assertTrue(any("closure lacks opposing jaw contact" in r["reason"] for r in planner.rejected))

    def test_post_pinch_transfer_has_collision_checked_arm_payload_and_release(self):
        import rig

        goal = copy.deepcopy(self.scene.bodies["gamepad"])
        goal.pose[0, 3] += 0.020
        # This test isolates the rest of the route after an independently
        # verified pinch; the fixture's measured box does not verify one.
        with (
            patch.object(rig, "_call", side_effect=AssertionError("offline planner called GUI")),
            patch("planned_reset.opposing_jaw_contacts", side_effect=_fixture_pinch),
        ):
            result = ResetPlanner(self.scene).transfer("gamepad", goal, self.start)
        self.assertEqual(
            [s.phase for s in result.segments],
            [
                "open",
                "approach",
                "grasp_descent",
                "close",
                "lift",
                "carry",
                "place_descent",
                "release",
                "retreat",
            ],
        )
        self.assertTrue(result.rejected)
        previous = self.start
        for segment in result.segments:
            np.testing.assert_allclose(segment.joints[0], previous)
            self.assertLessEqual(np.max(np.abs(np.diff(segment.joints, axis=0))), 1.000001)
            scene = self.scene
            if segment.phase in ("release", "retreat"):
                scene = copy.copy(scene)
                scene.bodies = {**scene.bodies, "gamepad": result.destination}
            for q in segment.joints:
                self.assertTrue(
                    scene.check(
                        q,
                        attached=segment.attached,
                        contact=segment.contact,
                        near=segment.near,
                        support=segment.support,
                    ),
                    scene.reason,
                )
            previous = segment.joints[-1]
        carry = result.segments[5]
        held_pose = self.robot.tip(carry.joints[-1]) @ carry.attached[1]
        self.assertGreater(held_pose[2, 3] - result.destination.pose[2, 3], 0.045)
        place = result.segments[6]
        actual = self.robot.tip(place.joints[-1]) @ place.attached[1]
        np.testing.assert_allclose(actual[:3, 3], result.destination.pose[:3, 3], atol=0.002)
        corners = result.destination.corners()
        self.assertAlmostEqual(
            min(corners[:, 2] - np.c_[corners[:, :2], np.ones(8)] @ self.scene.plane), 0.002
        )
        grasp_tip = self.robot.tip(result.segments[2].joints[-1])[:3, 3]
        floor_z = self.scene.plane @ np.r_[grasp_tip[:2], 1]
        self.assertGreater(grasp_tip[2] - floor_z, 0.020)

    def test_measured_grasp_shift_changes_tool_goal_but_preserves_object_destination(self):
        goal = copy.deepcopy(self.scene.bodies["gamepad"])
        goal.pose[0, 3] += 0.020
        planner = ResetPlanner(self.scene)
        with patch("planned_reset.opposing_jaw_contacts", side_effect=_fixture_pinch):
            nominal = planner.transfer("gamepad", goal, self.start)
        lifted = nominal.segments[4].joints[-1]
        hold = nominal.segments[4].attached[1].copy()
        hold[0, 3] += 0.004  # measured four-millimetre grasp shift; not an encoder/target injection
        remaining = planner.place_held("gamepad", nominal.destination, lifted, hold)
        placed = self.robot.tip(remaining[1].joints[-1]) @ hold
        np.testing.assert_allclose(placed[:3, 3], nominal.destination.pose[:3, 3], atol=0.0005)
        np.testing.assert_allclose(placed[:3, :3], nominal.destination.pose[:3, :3], atol=0.001)
        old_tool = self.robot.tip(nominal.segments[6].joints[-1])
        old_object = old_tool @ hold
        self.assertGreater(np.linalg.norm(old_object[:3, 3] - nominal.destination.pose[:3, 3]), 0.0035)
        self.assertTrue(np.all(remaining[0].joints[:, -1] == lifted[-1]))
        np.testing.assert_array_equal(remaining[0].attached[1], hold)
        # The opposite hypothetical hold puts the model through the fixed jaw
        # after release. The complete remaining path must refuse before carry.
        hold[0, 3] -= 0.008
        with self.assertRaisesRegex(NoPathError, "retreat start"):
            planner.place_held("gamepad", nominal.destination, lifted, hold)

    def test_invalid_hold_measurement_is_refused_before_ik(self):
        planner = ResetPlanner(self.scene)
        for value in [np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3)]:
            with self.assertRaisesRegex(NoPathError, "rigid transform"):
                planner.place_held("gamepad", self.scene.bodies["gamepad"], self.start, value)

    def test_contact_line_rejects_a_blocked_initial_configuration(self):
        scene = copy.copy(self.scene)
        scene.check = lambda *a, **k: False
        scene.reason = "occupied initial pose"
        with self.assertRaisesRegex(NoPathError, "start: occupied"):
            ResetPlanner(scene)._line("lift", self.start, self.robot.tip(self.start), scene)

    def test_transit_cannot_open_gripper_to_squeeze_through(self):
        goal = self.start.copy()
        goal[-1] += 1
        with self.assertRaisesRegex(ValueError, "gripper"):
            ResetPlanner(self.scene)._segment("carry", self.start, goal, self.scene)


if __name__ == "__main__":
    unittest.main()
