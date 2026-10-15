"""Calibrated SO-107 meshes against a conservative, recorded RGB-D scene.

Metres for geometry, motor degrees for configurations. A flat tray is assumed;
objects and other observed clutter occupy columns down to it. Missing cells
are blocked, except for the explicit flat-tray assumption under masked robot
and object pixels. Verified static observations from the same camera may fill
cells hidden by the arm in the current frame. The fixed base region is outside
this model's coverage.
"""

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Body:
    name: str
    size: np.ndarray
    pose: np.ndarray

    def corners(self, pose=None):
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        points = signs * self.size / 2
        t = self.pose if pose is None else pose
        return points @ t[:3, :3].T + t[:3, 3]


class MeshRobot:
    def __init__(self, kinematics=None, motor_calibration=None, attachments=()):
        import pinocchio as pin
        import rig

        self.pin, self.rig = pin, rig
        k = rig._kin() if kinematics is None else kinematics
        self.kin, self.names = k["kin"], k["names"]
        self.robot = self.kin._inner._inner.robot
        self.model, self.geom = self.robot.model, self.robot.collision_model
        # Own the geometry model: adding rig hardware must not mutate the
        # kinematics cache or accumulate copies when another planner is built.
        self.geom = pin.GeometryModel(self.geom)
        self.attachment_names = set()
        for attachment in attachments:
            self._add_attachment(attachment)
        self.data = self.model.createData()
        self.gdata = self.geom.createData()
        bounds = np.array([self.kin._urdf_to_motor(k["lo"]), self.kin._urdf_to_motor(k["hi"])])
        self.lower, self.upper = bounds.min(axis=0), bounds.max(axis=0)
        calibration = (
            json.loads(rig.pathlib.Path(rig.MOTOR_CAL).read_text())
            if motor_calibration is None
            else motor_calibration
        )
        for i, name in enumerate(self.names):
            if name == "gripper":
                self.lower[i], self.upper[i] = 0, 100
            else:
                c = calibration[name]
                half = (c["range_max"] - c["range_min"]) * 360 / 4096 / 2
                self.lower[i] = max(self.lower[i], -half)
                self.upper[i] = min(self.upper[i], half)
        self.upper[self.names.index("wrist_roll")] = min(
            self.upper[self.names.index("wrist_roll")], rig.WRIST_ROLL_MAX_MOTOR
        )
        self.pairs = []
        for i, a in enumerate(self.geom.geometryObjects):
            for j in range(i):
                b = self.geom.geometryObjects[j]
                if a.parentJoint == b.parentJoint:
                    continue
                # Native adjacent links overlap at their joint. An added
                # attachment gets no exemption from the neighbouring joint:
                # a wrist camera can obstruct the moving jaw or forearm.
                if a.name not in self.attachment_names and b.name not in self.attachment_names:
                    if self.model.parents[a.parentJoint] == b.parentJoint:
                        continue
                    if self.model.parents[b.parentJoint] == a.parentJoint:
                        continue
                self.pairs.append((i, j))

    def _add_attachment(self, spec):
        """A measured rigid box in a named CAD geometry's local coordinates.

        Attachment bodies participate in environment, payload and nonadjacent
        self-collision checks. They never inherit the jaws' contact exemption.
        Flexible cables require their own model, not a rigid box assertion.
        """
        import coal

        name = spec["name"]
        existing = {g.name: g for g in self.geom.geometryObjects}
        if not isinstance(name, str) or not name or name in existing or name.startswith(("L6_", "L7_")):
            raise ValueError("attachment needs a unique non-jaw name")
        parent = existing[spec["parent_geometry"]]
        size, pose = np.asarray(spec["size"], float), np.asarray(spec["pose"], float)
        if (
            size.shape != (3,)
            or not np.isfinite(size).all()
            or (size <= 0).any()
            or pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)
        ):
            raise ValueError("attachment needs positive metre dimensions and a rigid local pose")
        transform = parent.placement * self.pin.SE3(pose[:3, :3], pose[:3, 3])
        self.geom.addGeometryObject(
            self.pin.GeometryObject(name, parent.parentJoint, parent.parentFrame, transform, coal.Box(*size))
        )
        self.attachment_names.add(name)

    def valid_limits(self, q):
        return bool(
            q.shape == self.lower.shape
            and np.isfinite(q).all()
            and (q >= self.lower).all()
            and (q <= self.upper).all()
        )

    def placements(self, q):
        import coal

        urdf = np.deg2rad(self.kin._motor_to_urdf(q))
        self.pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, urdf)
        return [
            coal.CollisionObject(g.geometry, coal.Transform3s(t.rotation, t.translation))
            for g, t in zip(self.geom.geometryObjects, self.gdata.oMg, strict=True)
        ]

    def tip(self, q):
        return self.kin.forward_kinematics(q)

    def observed_mask(self, points, q, tolerance=0.012):
        """Depth returns near the measured robot mesh, not a colour-only mask.

        The tolerance covers depth/calibration error; protected object masks
        are subtracted by the caller. Cables and unmodeled attachments remain
        obstacles. This does not supply observations behind an occluding arm.
        """
        from scipy.spatial import cKDTree

        self.placements(q)
        surface = []
        valid = np.isfinite(points).all(axis=-1)
        mask = np.zeros(valid.shape, bool)
        for g, t in zip(self.geom.geometryObjects, self.gdata.oMg, strict=True):
            if g.name in self.attachment_names:
                local = (points - t.translation) @ t.rotation
                # Euclidean distance to the solid box, including its interior.
                delta = np.maximum(np.abs(local) - np.asarray(g.geometry.halfSide), 0)
                mask |= valid & (np.linalg.norm(delta, axis=-1) <= tolerance)
                continue
            vertices = np.asarray(g.geometry.vertices())
            surface.append(vertices @ t.rotation.T + t.translation)
        tree = cKDTree(np.concatenate(surface))
        distances, _ = tree.query(points[valid], distance_upper_bound=tolerance)
        mask[valid] |= distances <= tolerance
        return mask

    def solve(self, target, seed, gripper):
        q = np.asarray(seed, float).copy()
        q[-1] = gripper
        for _ in range(8):
            nxt = self.kin.inverse_kinematics(q, target)
            nxt[-1] = gripper
            if np.max(np.abs(nxt - q)) < 1e-3:
                q = nxt
                break
            q = nxt
        got = self.tip(q)
        angle = np.arccos(np.clip((np.trace(got[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1))
        if (
            not self.valid_limits(q)
            or np.linalg.norm(got[:3, 3] - target[:3, 3]) > 0.002
            or angle > np.deg2rad(2)
        ):
            return None
        return q


def box(size, pose):
    import coal

    return coal.CollisionObject(coal.Box(*size), coal.Transform3s(pose[:3, :3], pose[:3, 3]))


def collides(a, b, margin=0.0):
    import coal

    req = coal.CollisionRequest()
    req.security_margin = margin
    return bool(coal.collide(a, b, req, coal.CollisionResult()))


def opposing_jaw_contacts(jaws, body, closing_axis, *, tolerance=0.002, max_penetration=0.003):
    """Verify two jaw meshes meet opposite, non-edge faces of a measured rigid box.

    Proximity alone does not establish a pinch: both jaws can graze the top, or
    one can meet an end while the other misses the body. ``jaws`` are the two
    calibrated collision objects at the closing configuration. The result is a
    diagnostic dictionary so a rejected grasp can say what the geometry showed.
    """
    import coal

    if len(jaws) != 2:
        raise ValueError("a parallel-jaw pinch needs exactly two jaw bodies")
    closing_axis = np.asarray(closing_axis, float)
    if (
        closing_axis.shape != (3,)
        or not np.isfinite(closing_axis).all()
        or np.linalg.norm(closing_axis) < 1e-9
    ):
        raise ValueError("the jaw closing axis must be a finite 3-vector")
    local_axis = body.pose[:3, :3].T @ (closing_axis / np.linalg.norm(closing_axis))
    target = box(body.size, body.pose)
    points = []
    distances = []
    face_axes = []
    for jaw in jaws:
        result = coal.DistanceResult()
        distance = float(coal.distance(jaw, target, coal.DistanceRequest(), result))
        point = (np.asarray(result.getNearestPoint2()) - body.pose[:3, 3]) @ body.pose[:3, :3]
        distances.append(distance)
        points.append(point)
        if not -max_penetration <= distance <= tolerance:
            return {"ok": False, "reason": "a jaw misses or penetrates the object", "distances": distances}
        half = body.size / 2
        # A true face contact is near one boundary and inset from both edges.
        possible = [
            axis
            for axis in range(3)
            if abs(abs(point[axis]) - half[axis]) <= tolerance
            and all(
                abs(point[other]) <= half[other] - min(0.01, 0.1 * body.size[other])
                for other in range(3)
                if other != axis
            )
        ]
        face_axes.append(possible)
    shared = [axis for axis in face_axes[0] if axis in face_axes[1] and abs(local_axis[axis]) >= 0.75]
    ok = any(np.sign(points[0][axis]) != np.sign(points[1][axis]) for axis in shared)
    return {
        "ok": bool(ok),
        "reason": "" if ok else "the jaws do not meet opposing interior faces",
        "distances": distances,
        "points": points,
    }


class CollisionScene:
    def __init__(self, robot, obstacles, floor, bodies, plane, bounds, clearance=0.007, unknown=None):
        import coal

        self.robot, self.bodies = robot, bodies
        self.plane, self.bounds = np.asarray(plane), np.asarray(bounds)
        self.clearance = clearance
        # Keep Python owners alive for the broad-phase manager's pointers.
        self.obstacles, self.floor, self.unknown = obstacles, floor, unknown or []
        self.manager = coal.DynamicAABBTreeCollisionManager()
        for obstacle in obstacles:
            self.manager.registerObject(obstacle)
        self.manager.setup()
        self.unknown_manager = coal.DynamicAABBTreeCollisionManager()
        for obstacle in self.unknown:
            self.unknown_manager.registerObject(obstacle)
        self.unknown_manager.setup()
        self.floor_manager = coal.DynamicAABBTreeCollisionManager()
        for obstacle in floor:
            self.floor_manager.registerObject(obstacle)
        self.floor_manager.setup()
        self.reason = ""

    @classmethod
    def from_observation(cls, robot, objects, frame, resolution=0.005, clearance=0.007):
        import cv2
        import rig

        # The observed tray working area; the mount itself lies beyond y=-50mm.
        bounds = np.array([[-0.2, -0.35], [0.2, -0.05]])
        dims = np.ceil((bounds[1] - bounds[0]) / resolution).astype(int)
        plane = np.asarray(frame["plane"], float).copy()
        plane[2] /= 1000
        points = frame["pts"] / 1000
        ix = np.floor(
            np.nan_to_num((points[..., :2] - bounds[0]) / resolution, nan=-1e6, posinf=-1e6, neginf=-1e6)
        ).astype(int)
        valid = np.isfinite(points).all(axis=2) & (ix >= 0).all(axis=2) & (ix < dims).all(axis=2)
        own = np.zeros(valid.shape, np.uint8)
        bodies = {}
        for name, o in objects.items():
            if o.get("occluded"):
                raise ValueError(f"cannot plan around unobserved {name}")
            own |= np.asarray(o["mask"], np.uint8)
            x, y = np.asarray(o["centre"]) / 1000
            ground = plane @ np.array([x, y, 1])
            height = o["height"] / 1000
            t = np.eye(4)
            yaw = np.deg2rad(o["axis_deg"])
            normal = np.r_[-plane[:2], 1.0]
            normal /= np.linalg.norm(normal)
            along = np.array([np.cos(yaw), np.sin(yaw), plane[:2] @ [np.cos(yaw), np.sin(yaw)]])
            along /= np.linalg.norm(along)
            t[:3, :3] = np.column_stack((along, np.cross(normal, along), normal))
            t[:3, 3] = [x, y, ground + height / (2 * normal[2])]
            bodies[name] = Body(name, np.array([o["length"], o["width"], o["height"]]) / 1000, t)
        own = cv2.dilate(own, np.ones((5, 5), np.uint8)).astype(bool)
        robot_mask = np.asarray(frame.get("self_mask", np.zeros(valid.shape, bool)), bool)
        # Explicit rig assumption: the calibrated flat tray also continues under
        # the recognized objects and measured arm silhouette. These columns are
        # NOT observed free space. Unknown columns elsewhere remain blocked.
        seen = np.zeros(tuple(dims), bool)
        tops = np.full(tuple(dims), -np.inf)

        def accumulate(observation):
            observed = np.asarray(observation["pts"], float) / 1000
            if observed.shape != points.shape:
                raise ValueError("background observation must share the aligned camera grid")
            cell = np.floor(
                np.nan_to_num(
                    (observed[..., :2] - bounds[0]) / resolution,
                    nan=-1e6,
                    posinf=-1e6,
                    neginf=-1e6,
                )
            ).astype(int)
            measured = np.isfinite(observed).all(axis=2) & (cell >= 0).all(axis=2) & (cell < dims).all(axis=2)
            seen[cell[..., 0][measured], cell[..., 1][measured]] = True
            masked_robot = np.asarray(observation.get("self_mask", np.zeros(valid.shape, bool)), bool)
            if masked_robot.shape != valid.shape:
                raise ValueError("robot mask must share the aligned camera grid")
            clutter = measured & ~own & ~masked_robot
            np.maximum.at(tops, (cell[..., 0][clutter], cell[..., 1][clutter]), observed[..., 2][clutter])

        accumulate(frame)
        for background in frame.get("background_frames", ()):
            accumulate(background)
        obstacles, floor, unknown = [], [], []
        for i in range(dims[0]):
            for j in range(dims[1]):
                xy = bounds[0] + (np.array([i, j]) + 0.5) * resolution
                z = plane @ np.r_[xy, 1]
                u, v = rig.pixel(*(xy * 1000), z * 1000)
                # The base/mount lies outside the overhead camera's tray view.
                # No clearance claim is made outside this modeled region.
                if not (0 <= u < valid.shape[1] and 0 <= v < valid.shape[0]):
                    continue
                t = np.eye(4)
                # Finite columns avoid an infinite table intersecting the mount.
                bottom = -0.2
                top = z + 0.002
                t[:3, 3] = [*xy, (bottom + top) / 2]
                floor.append(box([resolution, resolution, top - bottom], t))
                # A raised self return occupies a different XY cell from the
                # tray behind it in an oblique camera. Apply the documented
                # flat-tray assumption along that ray too, not just vertically
                # under its measured surface point. Never erase observed
                # clutter, or fill missing depth without a measured occluder.
                ray_point = points[v, u]
                self_occluded = (
                    robot_mask[v, u]
                    and np.isfinite(ray_point).all()
                    and ray_point[2] > plane @ np.r_[ray_point[:2], 1] + 0.005
                )
                unobserved = not seen[i, j] and not self_occluded
                high = tops[i, j] if seen[i, j] else (-np.inf if self_occluded else 1.0)
                if high <= z + 0.005:
                    continue
                top = high + clearance
                t[:3, 3] = [*xy, (bottom + top) / 2]
                (unknown if unobserved else obstacles).append(
                    box([resolution + 2 * clearance, resolution + 2 * clearance, top - bottom], t)
                )
        return cls(robot, obstacles, floor, bodies, plane, bounds, clearance, unknown)

    def _manager_hit(self, manager, obj):
        import coal

        callback = coal.CollisionCallBackDefault()
        callback.data.request.enable_contact = True
        manager.collide(obj, callback)
        if callback.data.result.isCollision():
            self.contact_point = np.asarray(callback.data.result.getContact(0).pos).copy()
            return True
        return False

    def check(self, q, *, attached=None, contact=None, near=None, support=False):
        """Attachment = (object name, tip-to-object pose); contact permits finger contact only."""
        self.reason, self.contact_point = "", None
        q = np.asarray(q, float)
        if not self.robot.valid_limits(q):
            self.reason = "joint limit"
            return False
        links = self.robot.placements(q)
        for i, j in self.robot.pairs:
            if collides(links[i], links[j]):
                self.reason = f"self collision {self.robot.geom.geometryObjects[i].name}/{self.robot.geom.geometryObjects[j].name}"
                return False
        for i, link in enumerate(links):
            name = self.robot.geom.geometryObjects[i].name
            if self._manager_hit(self.manager, link):
                self.reason = f"scene collision {name}"
                return False
            if self._manager_hit(self.floor_manager, link):
                self.reason = f"tray collision {name}"
                return False
            if self._manager_hit(self.unknown_manager, link):
                at = self.contact_point * 1000
                self.reason = (
                    f"unobserved scene space at {name} near ({at[0]:.0f}, {at[1]:.0f}, {at[2]:.0f}) mm"
                )
                return False
            for key, body in self.bodies.items():
                if attached and key == attached[0]:
                    continue
                if key == contact and name.startswith(("L6_", "L7_")):
                    continue
                # Final grasp approach may enter the safety margin around the
                # selected object, but cannot intersect it until closing.
                margin = 0.0 if key == near and name.startswith(("L6_", "L7_")) else self.clearance
                if collides(link, box(body.size, body.pose), margin):
                    self.reason = f"{name} touches {key}"
                    return False
        if attached:
            key, hold = attached
            body = self.bodies[key]
            pose = self.robot.tip(q) @ hold
            payload = box(body.size + 2 * self.clearance, pose)
            if self._manager_hit(self.manager, payload):
                self.reason = "held object touches clutter"
                return False
            if self._manager_hit(self.unknown_manager, payload):
                self.reason = "held object enters unobserved scene space"
                return False
            corners = body.corners(pose)
            floor_z = np.c_[corners[:, :2], np.ones(len(corners))] @ self.plane
            if np.min(corners[:, 2] - floor_z) < (-0.0005 if support else self.clearance):
                self.reason = "held object lacks table clearance"
                return False
            for other, obstacle in self.bodies.items():
                if other != key and collides(payload, box(obstacle.size, obstacle.pose)):
                    self.reason = f"held object touches {other}"
                    return False
            for i, link in enumerate(links):
                if not self.robot.geom.geometryObjects[i].name.startswith(("L6_", "L7_")) and collides(
                    payload, link
                ):
                    self.reason = "held object touches arm"
                    return False
        return True
