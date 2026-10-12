"""Plan a reset using calibrated meshes, grasp alternatives and joint-space search.

The command line is offline: it writes a reviewable plan and never connects to
the arm. Plans are experimental and are not accepted by the live Cartesian jog API.
"""

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from collision_scene import Body, CollisionScene, MeshRobot, box, collides
from motion_planner import JointPlanner, NoPathError, SearchLimits, interpolate


@dataclass
class Segment:
    phase: str
    joints: np.ndarray
    attached: tuple | None = None
    contact: str | None = None
    support: bool = False
    near: str | None = None


@dataclass
class Transfer:
    name: str
    source: Body
    destination: Body
    segments: list[Segment]
    rejected: list[dict] = field(default_factory=list)


class ResetPlanner:
    def __init__(self, scene, limits=SearchLimits()):
        self.scene, self.robot, self.limits = scene, scene.robot, limits
        self.rejected = []

    def _segment(self, phase, start, goal, scene, *, attached=None, contact=None, support=False):
        lower, upper = self.robot.lower.copy(), self.robot.upper.copy()
        if abs(start[-1] - goal[-1]) > 1e-8:
            raise ValueError("transit cannot change the gripper")
        lower[-1] = upper[-1] = start[-1]

        def valid(q):
            return scene.check(q, attached=attached, contact=contact, support=support)

        p = JointPlanner(lower, upper, valid, self.limits)
        return Segment(phase, p.plan(start, goal), attached, contact, support)

    def _line(self, phase, start, target, scene, *, attached=None, support=False, near=None):
        """Constrained approach/lift: straight tool translation at fixed orientation."""
        initial = self.robot.tip(start)
        if not np.allclose(initial[:3, :3], target[:3, :3], atol=0.04):
            raise NoPathError("contact motion would rotate the tool")
        n = max(1, int(np.ceil(np.linalg.norm(target[:3, 3] - initial[:3, 3]) / 0.002)))
        rows = [start]
        if not scene.check(start, attached=attached, support=support, near=near):
            raise NoPathError(f"{phase} start: {scene.reason}")
        for t in np.linspace(0, 1, n + 1)[1:]:
            pose = initial.copy()
            pose[:3, 3] += t * (target[:3, 3] - initial[:3, 3])
            q = self.robot.solve(pose, rows[-1], start[-1])
            if q is None:
                raise NoPathError(f"{phase}: IK failed")
            for between in interpolate(rows[-1], q, self.limits.resolution)[1:]:
                if not scene.check(between, attached=attached, support=support, near=near):
                    raise NoPathError(f"{phase}: {scene.reason}")
                # Joint interpolation must still remain in the tool corridor.
                tip = self.robot.tip(between)
                direction = target[:3, 3] - initial[:3, 3]
                alpha = np.clip(
                    (tip[:3, 3] - initial[:3, 3]) @ direction / max(direction @ direction, 1e-12), 0, 1
                )
                if np.linalg.norm(tip[:3, 3] - initial[:3, 3] - alpha * direction) > 0.003:
                    raise NoPathError(f"{phase}: IK leaves the straight corridor")
                rows.append(between)
        return Segment(phase, np.asarray(rows), attached, None, support, near)

    def _gripper(self, phase, q, opening, scene, name):
        goal = q.copy()
        goal[-1] = opening
        path = interpolate(q, goal, 1.0)
        for at in path:
            if not scene.check(at, contact=name):
                raise NoPathError(f"{phase}: {scene.reason}")
        return Segment(phase, path, None, name)

    def transfer(self, name, destination, start, *, free_yaw=True):
        """Search central grasps on both sides; no end-adjacent 22mm shifts."""
        import rig

        source = self.scene.bodies[name]
        yaw_source = np.degrees(np.arctan2(source.pose[1, 0], source.pose[0, 0]))
        yaw_goal = np.degrees(np.arctan2(destination.pose[1, 0], destination.pose[0, 0]))
        turn = 0.0 if free_yaw else (yaw_goal - yaw_source + 90) % 180 - 90
        goal_body = copy.deepcopy(destination)
        goal_body.pose[:3, :3] = rig._rz(turn) @ source.pose[:3, :3]
        # Set height from ALL bottom corners, including the tray slope.
        corners = goal_body.corners()
        gap = corners[:, 2] - np.c_[corners[:, :2], np.ones(8)] @ self.scene.plane
        goal_body.pose[2, 3] += 0.002 - min(gap)
        base_yaw = rig.grasp_yaw(yaw_source)
        failures = []
        candidates = [
            (edge, skew, up, lateral)
            for up in (0.0, 0.2, 0.4)
            for edge in (0, 180)
            for skew in (0, -12, 12)
            for lateral in (0.0, -0.002, -0.004, -0.006, -0.008)
        ]
        started = time.monotonic()
        for edge, skew, up, lateral in candidates:
            yaw = (base_yaw + edge + skew + 180) % 360 - 180
            if time.monotonic() - started > 120:
                self.rejected = failures
                raise NoPathError("transfer planning time budget exhausted")
            try:
                grasp = np.eye(4)
                grasp[:3, :3] = rig._rz(yaw) @ rig.DEMO_GRASP_R
                grasp[:2, 3] = source.pose[:2, 3] + rig._rz(yaw)[:2, :2] @ np.array(
                    [0.007, (0.022 if name == "gamepad" else 0.018) + lateral]
                )
                floor_z = self.scene.plane @ np.r_[grasp[:2, 3], 1]
                pre = grasp.copy()
                pre[2, 3] = floor_z + 0.080
                qpre = self.robot.solve(pre, start, 65)
                if qpre is None:
                    raise NoPathError("pregrasp IK failed")
                if not self.scene.check(qpre):
                    raise NoPathError(f"pregrasp: {self.scene.reason}")
                # The calibrated tool origin is not the lowest jaw point. Derive
                # contact height from the actual mesh through the closure range.
                gap = float("inf")
                for opening in (65, 75, 86 if name == "gamepad" else 94):
                    qprobe = qpre.copy()
                    qprobe[-1] = opening
                    self.robot.placements(qprobe)
                    for geom, placement in zip(
                        self.robot.geom.geometryObjects, self.robot.gdata.oMg, strict=True
                    ):
                        if geom.name.startswith(("L6_", "L7_")):
                            pts = (
                                np.asarray(geom.geometry.vertices()) @ placement.rotation.T
                                + placement.translation
                            )
                            gap = min(
                                gap, min(pts[:, 2] - np.c_[pts[:, :2], np.ones(len(pts))] @ self.scene.plane)
                            )
                grasp[2, 3] = pre[2, 3] - gap + 0.003 + up * source.size[2]
                if grasp[2, 3] >= pre[2, 3]:
                    raise NoPathError("no grasp below pregrasp with jaw/table clearance")
                # Reject impossible contact geometry before spending RRT budget.
                descend = self._line("grasp_descent", qpre, grasp, self.scene, near=name)
                close = self._gripper(
                    "close", descend.joints[-1], 86 if name == "gamepad" else 94, self.scene, name
                )
                # A candidate must reach both opposing jaw bodies at closure.
                meshes = self.robot.placements(close.joints[-1])
                obj = box(source.size, source.pose)
                fingers = [
                    i
                    for i, g in enumerate(self.robot.geom.geometryObjects)
                    if g.name.startswith(("L6_", "L7_"))
                ]
                if not all(collides(meshes[i], obj, 0.002) for i in fingers):
                    raise NoPathError("closure does not contact both jaws")
                opened = self._gripper("open", start, 65, self.scene, "")
                approach = self._segment("approach", opened.joints[-1], qpre, self.scene)
                hold = np.linalg.inv(self.robot.tip(close.joints[-1])) @ source.pose
                attached = (name, hold)
                lifted_pose = self.robot.tip(close.joints[-1])
                lifted_pose[2, 3] += 0.05
                lift = self._line(
                    "lift", close.joints[-1], lifted_pose, self.scene, attached=attached, support=True
                )
                drop = goal_body.pose @ np.linalg.inv(hold)
                above = drop.copy()
                above[2, 3] += 0.05
                qabove = self.robot.solve(above, lift.joints[-1], close.joints[-1][-1])
                if qabove is None:
                    raise NoPathError("preplace IK failed")
                carry = self._segment("carry", lift.joints[-1], qabove, self.scene, attached=attached)
                lower = self._line("place_descent", qabove, drop, self.scene, attached=attached, support=True)
                placed_scene = copy.copy(self.scene)
                placed_scene.bodies = {**self.scene.bodies, name: goal_body}
                release = self._gripper("release", lower.joints[-1], 65, placed_scene, name)
                retreat_pose = self.robot.tip(release.joints[-1])
                retreat_pose[2, 3] += 0.05
                retreat = self._line("retreat", release.joints[-1], retreat_pose, placed_scene, near=name)
                return Transfer(
                    name,
                    source,
                    goal_body,
                    [opened, approach, descend, close, lift, carry, lower, release, retreat],
                    failures,
                )
            except NoPathError as e:
                failures.append(
                    {"yaw_deg": yaw, "extra_height_fraction": up, "lateral_m": lateral, "reason": str(e)}
                )
        self.rejected = failures
        raise NoPathError(f"all {len(candidates)} central grasps rejected")


def main():
    from reset_policy import Vision, clean

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rgb", type=Path, required=True)
    ap.add_argument("--depth", type=Path, required=True, help="NPY or NPZ depth in metres")
    ap.add_argument(
        "--joints", type=Path, required=True, help="JSON motor-name to observed-degree map from this frame"
    )
    ap.add_argument("--object", choices=["gamepad", "cube"], default="gamepad")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rgb = cv2.cvtColor(cv2.imread(str(args.rgb)), cv2.COLOR_BGR2RGB)
    depth = np.load(args.depth, allow_pickle=False)
    if isinstance(depth, np.lib.npyio.NpzFile):
        depth = depth["depth"]
    vision, robot = Vision(), MeshRobot()
    observed, frame = vision.see(rgb, depth, True)
    state = json.loads(args.joints.read_text())
    q = np.array([state[n] for n in robot.names])
    frame["self_mask"] = robot.observed_mask(frame["pts"] / 1000, q)
    for o in observed.values():
        frame["self_mask"] &= ~cv2.dilate(o["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    scene = CollisionScene.from_observation(robot, observed, frame)
    goal = copy.deepcopy(scene.bodies[args.object])
    goal.pose[:2, 3] = np.array(vision.goals[args.object]["centre"]) / 1000
    goal.pose[2, 3] = scene.plane @ np.r_[goal.pose[:2, 3], 1] + goal.size[2] / 2
    planner = ResetPlanner(scene)
    started = time.monotonic()
    report = {
        "mode": "offline_only",
        "execution_supported": False,
        "units": {"geometry": "metres", "arm_joints": "motor degrees", "gripper": "0-100 units"},
        "calibration": {
            "motor_names": robot.names,
            "urdf_sign": robot.kin._sign,
            "urdf_offset_deg": robot.kin._offset,
            "tip_offset": robot.kin._inner._tip_offset,
            "lower": robot.lower,
            "upper": robot.upper,
        },
        "object": args.object,
        "source_rgb": str(args.rgb.resolve()),
        "source_depth": str(args.depth.resolve()),
        "source_joints": str(args.joints.resolve()),
        "start": q,
        "model_clearance_m": scene.clearance,
        "unknown_space": "missing cells in tray view blocked; flat tray assumed under masked robot/objects; outside view not modeled",
    }
    if not scene.check(q):
        report.update(ok=False, reason=f"start: {scene.reason}", collision_point_m=scene.contact_point)
    else:
        try:
            result = planner.transfer(args.object, goal, q)
            report.update(
                ok=True,
                segments=[vars(s) for s in result.segments],
                source=vars(result.source),
                destination=vars(result.destination),
                rejected=result.rejected,
            )
        except NoPathError as e:
            report.update(ok=False, reason=str(e), rejected=planner.rejected)
    report["seconds"] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean(report), indent=2))
    print(json.dumps({k: v for k, v in report.items() if k in ("ok", "reason", "seconds")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
