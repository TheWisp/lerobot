import unittest
from unittest.mock import Mock, patch

import numpy as np
import reset_policy as p


class ContactControl(unittest.TestCase):
    def test_parking_an_already_disconnected_arm_is_a_noop(self):
        with (
            patch.object(p.rig, "state", return_value={"connected": False}),
            patch.object(p.rig, "_call") as call,
        ):
            p.rig.park()
        call.assert_not_called()

    def policy(self):
        q = p.Policy.__new__(p.Policy)
        q.phase = Mock()
        q.event = Mock()
        q.held = None
        return q

    def test_stalled_descent_cancels_deeper_target(self):
        q = self.policy()
        q.state = lambda *a: {"position": [0, -150, 8], "yaw": 0}
        q.move = Mock(
            side_effect=[{"position": [0, -150, 5]}, {"position": [0, -150, 2]}, {"position": [0, -150, 1.9]}]
        )
        q.rebase_at_observed = Mock()
        q.descend([0, -150, -8], 0)
        self.assertEqual([c.args[0][2] for c in q.move.call_args_list], [2, -1, -4])
        q.rebase_at_observed.assert_called_once()

    def test_rebase_forces_a_controller_command(self):
        q = self.policy()
        q.state = lambda *a: {"position": [0, -150, 2], "yaw": 0}
        q.path_ok = lambda p: True
        q.monitored_move = Mock(return_value={"ok": True})
        q.settle = Mock()
        q.rebase_at_observed(0)
        q.monitored_move.assert_called_once_with([0, -150, 2], 0, False)

    def test_descent_evaluates_only_targets_sent_to_controller(self):
        # Recorded 20261012_132417: from 40.5 mm the requested 30.5 mm
        # target was split, stopped at 35.5, then logged as 30.5. Repeated
        # partial steps falsely looked like contact 35 mm above the tray.
        q = self.policy()
        q.held = {"name": "cube"}
        q.motion_phase = "place_descent"
        q.state = lambda *a: {"position": [0, -150, 40.5], "yaw": 0}
        q.monitored_move = Mock(return_value={"ok": True, "held_short": True})
        with (
            patch.object(p.rig, "reachable", return_value={"ok": True, "seed_next": {}}),
            patch.object(p.rig, "state", return_value={}),
        ):
            q.move([0, -150, 30.5], 0, descent=True)
        self.assertEqual(len(q.monitored_move.call_args_list), 1)
        np.testing.assert_allclose(q.monitored_move.call_args.args[0], [0, -150, 30.5])

    def test_short_descent_is_not_skipped_by_arrival_tolerance(self):
        q = self.policy()
        q.state = lambda *a: {"position": [0, -150, 5], "yaw": 0}
        q.monitored_move = Mock(return_value={"ok": True, "held_short": True})
        with (
            patch.object(p.rig, "reachable", return_value={"ok": True, "seed_next": {}}),
            patch.object(p.rig, "state", return_value={}),
        ):
            q.move([0, -150, 2], 0, descent=True)
        q.monitored_move.assert_called_once()
        np.testing.assert_allclose(q.monitored_move.call_args.args[0], [0, -150, 2])

    def test_lift_waits_for_actual_height_and_uses_small_steps(self):
        q = self.policy()
        position = [0, -150, 1]
        q.state = lambda: {"position": position.copy(), "yaw": 0}
        q.path_ok = lambda points: True
        q.settle = Mock()
        commands = []

        def move(point, yaw):
            self.assertLessEqual(point[2] - (commands[-1][2] if commands else 1), 8.01)
            self.assertEqual(point[:2], position[:2])
            commands.append(point)
            position[2] = point[2] - 2
            return {"ok": True, "held_short": True}

        q.monitored_move = move
        q.lift(40, 0)
        self.assertGreaterEqual(position[2], 36)
        self.assertGreater(len(commands), 4)

    def test_held_short_does_not_accept_large_lateral_error(self):
        q = self.policy()
        q.minimum_transit_z = None
        self.assertFalse(
            q.arrived(
                {"position": [20, -150, 40], "yaw": 0}, np.array([0, -150, 40]), 0, 5, {"held_short": True}
            )
        )

    def test_freeze_does_not_resume_even_in_recovery(self):
        q = self.policy()
        q.log = {}
        with (
            patch.object(
                p.rig, "state", return_value={"connected": True, "halted": True, "reason": "overload"}
            ),
            patch.object(p.rig, "recover") as recover,
        ):
            with self.assertRaises(p.Unsafe):
                q.state(recovery=True)
            recover.assert_not_called()

    def test_thermal_recheck_needs_five_cool_readings_and_parks(self):
        for temp, required in ((34, False), (57, True)):
            q = self.policy()
            states = [{"connected": False}] + [
                {"connected": True, "temps": {str(n): temp for n in range(7)}}
            ] * 14
            with (
                patch.object(p.rig, "state", side_effect=states),
                patch.object(p.rig, "connect"),
                patch.object(p.rig, "_call", return_value=(200, {})) as call,
                patch.object(p.time, "sleep"),
            ):
                self.assertEqual(q.thermal_recheck(), required)
                call.assert_called_once_with("/api/jog/park", {}, timeout=120)

    def test_missing_temperature_telemetry_requires_cooldown(self):
        q = self.policy()
        with (
            patch.object(
                p.rig, "state", side_effect=[{"connected": False}] + [{"connected": True, "temps": {}}] * 14
            ),
            patch.object(p.rig, "connect"),
            patch.object(p.rig, "_call", return_value=(200, {})),
            patch.object(p.time, "sleep"),
        ):
            self.assertTrue(q.thermal_recheck())

    def test_transient_temperature_retries_without_cooldown(self):
        q = self.policy()
        q.started = 0
        q.log = {}
        q.thermal_recheck = Mock(return_value=False)
        attempts = []

        def attempt():
            attempts.append(True)
            q.log["thermal_stop"] = len(attempts) == 1
            return 1 if len(attempts) == 1 else 0

        q.attempt = attempt
        with patch.object(p.time, "sleep") as sleep:
            self.assertEqual(q._run(), 0)
            sleep.assert_not_called()
        self.assertEqual(len(attempts), 2)


if __name__ == "__main__":
    unittest.main()
