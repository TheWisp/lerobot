"""Experimental measured-pose reset through the GUI; default is planning only.

The arm must already be connected and stationary in a collision-valid pose.
This does not connect, ramp to ready, or automatically park. A failure freezes
an owned trajectory; the enclosing experiment must arrange a checked recovery.
"""

import argparse
import copy
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from collision_scene import CollisionScene, MeshRobot
from guarded_execution import ExecutionStoppedError, GuiExecutor
from motion_planner import NoPathError
from planned_reset import ResetPlanner
from reset_policy import Vision, clean


def request(path, body=None, binary=False):
    req = urllib.request.Request(
        "http://127.0.0.1:9100/api/reset/" + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        raw = response.read()
    return raw if binary else json.loads(raw)


def reference_hold(measurement, reference, reference_to_body):
    if measurement["reference"] != reference:
        raise ExecutionStoppedError("object reference changed during reset")
    return ResetPlanner._checked_hold(np.asarray(measurement["hold_from_reference"]) @ reference_to_body)


class MeasuredReset:
    def __init__(self, scene, executor, measure, log):
        self.scene, self.executor, self.measure, self.log = scene, executor, measure, log
        self.planner = ResetPlanner(scene)

    def transfer(self, planned, initial_measurement):
        """Measure at closure, after lift, and before placement; never replay a nominal hold blindly."""
        name = planned.name
        reference = initial_measurement["reference"]
        reference_to_body = (
            np.linalg.inv(np.asarray(initial_measurement["motion_base"])) @ planned.source.pose
        )
        try:
            for segment in planned.segments[:4]:
                self.executor.run(segment)
            measurement = self.measure(name, self.executor.identity, planned.source.pose)
            hold = reference_hold(measurement, reference, reference_to_body)
            q = self._configuration(measurement)
            lift = self.planner.lift_held(name, q, hold)
            self.executor.run(lift)
            measurement = self.measure(
                name, self.executor.identity, self.scene.robot.tip(lift.joints[-1]) @ hold
            )
            for attempt in range(3):
                hold = reference_hold(measurement, reference, reference_to_body)
                q = self._configuration(measurement)
                remaining = self.planner.place_held(name, planned.destination, q, hold)
                self.executor.run(remaining[0])
                # Motion itself may change the grip. Re-measure before lowering.
                check = self.measure(
                    name, self.executor.identity, self.scene.robot.tip(remaining[0].joints[-1]) @ hold
                )
                latest = reference_hold(check, reference, reference_to_body)
                difference = np.linalg.norm(latest[:3, 3] - hold[:3, 3])
                angle = np.degrees(
                    np.arccos(np.clip((np.trace(latest[:3, :3].T @ hold[:3, :3]) - 1) / 2, -1, 1))
                )
                self.log(
                    {
                        "hold_recheck": {
                            "attempt": attempt,
                            "translation_mm": difference * 1000,
                            "angle_deg": angle,
                        }
                    }
                )
                if difference <= 0.0005 and angle <= 0.5:
                    # Preserve the already checked hold within this measurement
                    # tolerance. A larger change requires a fresh complete plan.
                    break
                measurement = check
            else:
                raise ExecutionStoppedError("held object keeps shifting; no stable preplace hold")
            self.executor.run(remaining[1])
            placed_scene = copy.copy(self.scene)
            placed_scene.bodies = {**self.scene.bodies, name: copy.deepcopy(planned.destination)}
            self.executor.scene = placed_scene
            self.executor.run(remaining[2])
            released = self.measure(name, self.executor.identity, planned.destination.pose)
            if released["reference"] != reference:
                raise ExecutionStoppedError("object reference changed after release")
            actual = np.asarray(released["motion_base"]) @ reference_to_body
            if np.linalg.norm(actual[:3, 3] - planned.destination.pose[:3, 3]) > 0.005:
                raise ExecutionStoppedError("released object is more than 5 mm from the reset goal")
            vertical = actual[:3, 2] @ planned.destination.pose[:3, 2]
            if vertical < np.cos(np.deg2rad(5)):
                raise ExecutionStoppedError("released object is tilted more than 5 degrees")
            bottom = self.scene.bodies[name].corners(actual)[::2]
            gap = bottom[:, 2] - np.c_[bottom[:, :2], np.ones(4)] @ self.scene.plane
            if np.min(gap) < -0.002 or np.max(gap) > 0.005:
                raise ExecutionStoppedError("released object is not supported on the tray")
            # Recheck retreat against the observed object, not its requested pose.
            placed_scene.bodies[name].pose = actual
            q = self._configuration(released)
            target = placed_scene.robot.tip(q)
            target[2, 3] += 0.05
            retreat = ResetPlanner(placed_scene)._line("retreat", q, target, placed_scene, near=name)
            self.executor.run(retreat)
            self.scene = placed_scene
            self.planner = ResetPlanner(placed_scene)
            return actual
        except BaseException:
            self.executor.stop()
            raise

    def _configuration(self, measurement):
        q = np.array([measurement["q_obs"][n] for n in self.scene.robot.names])
        q[-1] = measurement["q_cmd"]["gripper"]
        return q


def main():
    import rig

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", choices=["gamepad", "cube"], default="gamepad")
    ap.add_argument("--destination-mm", nargs=2, type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--attachments", type=Path, help="JSON list of measured rigid accessory boxes")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    def log(row):
        with (args.out / "events.jsonl").open("a") as f:
            f.write(json.dumps(clean({"at": time.time(), **row})) + "\n")

    report = {"ok": False, "execution_requested": args.execute, "object": args.object}
    executor = None
    try:
        raw = request("frame.npz", binary=True)
        (args.out / "initial.npz").write_bytes(raw)
        snapshot = np.load(io.BytesIO(raw), allow_pickle=False)
        if not np.allclose(snapshot["t_base_cam"], rig.T_BC, atol=1e-6, rtol=0):
            raise NoPathError("saved reset calibration differs from the connected GUI")
        rgb, depth = snapshot["rgb"], snapshot["depth"]
        cv2.imwrite(str(args.out / "initial.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        attachments = json.loads(args.attachments.read_text()) if args.attachments else []
        report["attachments"] = attachments
        robot = MeshRobot(attachments=attachments)
        if list(snapshot["motor_names"]) != list(robot.names):
            raise NoPathError("joint order changed")
        q = snapshot["q_obs"].copy()
        q[-1] = snapshot["q_cmd"][-1]
        objects, frame = Vision().see(rgb, depth, True)
        frame["self_mask"] = robot.observed_mask(frame["pts"] / 1000, snapshot["q_obs"])
        for obj in objects.values():
            frame["self_mask"] &= ~cv2.dilate(obj["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(
                bool
            )
        scene = CollisionScene.from_observation(robot, objects, frame)
        if not scene.check(snapshot["q_obs"]):
            raise NoPathError("observed start: " + scene.reason)
        goal = copy.deepcopy(scene.bodies[args.object])
        goal.pose[:2, 3] = np.asarray(args.destination_mm) / 1000
        planned = ResetPlanner(scene).transfer(args.object, goal, q)
        report["plan"] = {
            "segments": [vars(s) for s in planned.segments],
            "destination": vars(planned.destination),
            "rejected": planned.rejected,
        }
        if args.execute:
            executor = GuiExecutor(scene, log, stopped=lambda: (args.out / "STOP").exists())

            def measure(name, token, predicted):
                centre = predicted[:3, 3].copy()
                centre += predicted[:3, 2] * scene.bodies[name].size[2] / 2
                u, v = rig.pixel(*(centre * 1000))
                measurement = request("object_pose", {"object": name, "click": [u, v], "path_id": token})
                log({"measurement": measurement})
                return measurement

            initial = measure(args.object, None, planned.source.pose)
            runner = MeasuredReset(scene, executor, measure, log)
            actual = runner.transfer(planned, initial)
            # Return along a newly checked path in the updated scene. Do not
            # substitute the legacy unchecked fold/park ramp on completion.
            state = executor.call("state")
            current, command = executor._read(state)
            current[-1] = command[-1]
            home = q.copy()
            home[-1] = current[-1]
            back = runner.planner._segment("return_to_start", current, home, runner.scene)
            executor.run(back)
            executor.finish()
            report["actual_object_pose"] = actual
        report["ok"] = True
    except Exception as e:
        if executor is not None:
            executor.stop()
        report["reason"] = str(e)
    finally:
        (args.out / "report.json").write_text(json.dumps(clean(report), indent=2))
    print(
        json.dumps({k: v for k, v in report.items() if k != "plan"}, default=lambda v: np.asarray(v).tolist())
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
