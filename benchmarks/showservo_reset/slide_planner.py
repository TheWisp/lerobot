"""Conservative open-jaw slides, generated from observed scene geometry.

No hardware actions: all reach checks go through rig.reachable. Gripper XY
hulls come from the rig's calibrated tool transform and its CAD jaw meshes.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import rig

HERE = Path(__file__).resolve().parent


def rotation(a):
    a = np.radians(a)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def hull(points):
    return cv2.convexHull(np.asarray(points, np.float32)).reshape(-1, 2)


def expand(poly, margin):
    ring = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]) * margin
    return hull((poly[:, None, :] + ring).reshape(-1, 2))


def intersects(a, b):
    return cv2.intersectConvexConvex(np.asarray(a, np.float32), np.asarray(b, np.float32))[0] > 0.1


class SlidePlanner:
    def __init__(self, policy, name, current=None, vacated=()):
        self.current = current
        self.vacated = tuple(vacated)
        # Contact force at the extension limit produced >6 mm vertical error
        # despite valid static IK. Let the open jaw extend the working range;
        # keep the controlled tool point near the demonstrated work region.
        self.tip_radius_limit = max(np.linalg.norm(g["centre"]) for g in policy.vision.goals.values()) + 20.0
        self.q = policy
        self.name = name
        self.src = policy.scene[name]
        self.c = np.array(self.src["centre"])
        self.ref = policy.vision.goals[name]
        target = policy.vision.target
        points = rig._arm_points(target["depth"])
        m = target["masks"][name]
        pts = points[m]
        pts = pts[pts[:, 2] > self.ref["top_z"] - self.ref["height"] / 2]
        local = (hull(pts[:, :2]) - self.ref["centre"]) @ rotation(self.ref["axis_deg"])
        self.object = expand(local @ rotation(self.src["axis_deg"]).T + self.c, 2.0)
        self.jaws = []
        for link in ("L6_1", "L7_1"):
            pts = np.load(HERE / f"gripper_{link}.npz")["points"]
            low = pts[pts[:, 2] < pts[:, 2].min() + self.ref["height"] + 4]
            self.jaws.append(expand(hull(low[:, :2]), 2.0))
        self.other = []
        for n, o in policy.scene.items():
            if n != name and n not in vacated:
                if o.get("occluded"):
                    raise ValueError("Cannot slide beside an unlocated object")
                self.other.append(policy.footprint(o, n, padding=7))
        # A 2 mm tray grid encodes both obstacle heights and observed support.
        self.origin = np.array(rig.WORK_MIN, dtype=float)
        self.scale = 2.0
        self.shape = (176, 201)
        pts = policy.frame["pts"]
        xy = pts[:, :, :2]
        ij = np.rint(np.nan_to_num((xy - self.origin) / self.scale, nan=-1000)).astype(int)
        finite = np.isfinite(pts[:, :, 2])
        valid = (
            finite
            & (ij[:, :, 0] >= 0)
            & (ij[:, :, 0] < self.shape[1])
            & (ij[:, :, 1] >= 0)
            & (ij[:, :, 1] < self.shape[0])
        )
        own = np.zeros(policy.frame["depth"].shape, bool)
        # Include the moving object's depth boundary: mixed pixels around its
        # colour silhouette otherwise appear to be stationary obstacles.
        components = rig.objects(policy.frame["depth"], policy.frame["rgb"])
        for n in (name,) + self.vacated:
            mask = policy.scene[n]["mask"].copy()
            ref = policy.vision.goals[n]
            for component in components:
                m = component["mask"]
                if (
                    np.count_nonzero(m & mask) > 0.75 * np.count_nonzero(mask)
                    and np.count_nonzero(m) < 2.2 * np.count_nonzero(mask)
                    and component["length"] < ref["length"] * 1.5
                    and component["width"] < ref["width"] * 1.8
                ):
                    mask |= m
            own |= mask
        self_mask = policy.frame.get("self_mask", np.zeros(own.shape, bool))
        heights = policy.frame["heights"]
        rgb = policy.frame["rgb"]
        value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
        obstacle = (
            valid
            & (heights > 5)
            & ~cv2.dilate(own.astype("uint8"), np.ones((9, 9), np.uint8)).astype(bool)
            & ~self_mask
        )
        self.blocked = np.zeros(self.shape, np.uint8)
        self.blocked[ij[:, :, 1][obstacle], ij[:, :, 0][obstacle]] = 1
        floor = valid & ((abs(heights) < 5) & (value < 125) | own)
        support = np.zeros(self.shape, np.uint8)
        support[ij[:, :, 1][floor], ij[:, :, 0][floor]] = 1
        memory = policy.frame.get("floor_memory", getattr(policy, "floor_memory", None))
        if memory is not None and self_mask.any():
            # Project robot pixels onto the tray plane. Their world XY at arm
            # height is not the floor shadow because the camera is oblique.
            origin = rig.T_BC[:3, 3] * 1000
            direction = pts - origin
            plane = policy.frame["plane"]
            den = direction[:, :, 2] - direction[:, :, 0] * plane[0] - direction[:, :, 1] * plane[1]
            numerator = origin[0] * plane[0] + origin[1] * plane[1] + plane[2] - origin[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                ground = origin[:2] + direction[:, :, :2] * (numerator / den)[:, :, None]
            shadow_ij = np.rint(
                np.nan_to_num((ground - self.origin) / self.scale, nan=-1000, posinf=-1000, neginf=-1000)
            ).astype(int)
            shadow_ok = (
                self_mask
                & (shadow_ij[:, :, 0] >= 0)
                & (shadow_ij[:, :, 0] < 201)
                & (shadow_ij[:, :, 1] >= 0)
                & (shadow_ij[:, :, 1] < 176)
            )
            shadow = np.zeros(self.shape, np.uint8)
            shadow[shadow_ij[:, :, 1][shadow_ok], shadow_ij[:, :, 0][shadow_ok]] = 1
            shadow = cv2.dilate(shadow, np.ones((3, 3), np.uint8))
            support |= memory & shadow
        self.support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cv2.fillConvexPoly(self.support, self.grid(self.object), 1)
        self.stats = {
            "surface_paths": 0,
            "jaw_clear_contacts": 0,
            "geometry_candidates": 0,
            "reachable_contacts": 0,
            "reachable_paths": 0,
        }

    def grid(self, poly):
        return np.rint((np.asarray(poly) - self.origin) / self.scale).astype(np.int32)

    def safe_surface(self, poly):
        coords = self.grid(poly)
        if (
            (coords < 0).any()
            or (coords[:, 0] >= self.shape[1]).any()
            or (coords[:, 1] >= self.shape[0]).any()
        ):
            return False
        region = np.zeros(self.shape, np.uint8)
        cv2.fillConvexPoly(region, coords, 1)
        mask = region > 0
        if any(intersects(poly, other) for other in self.other):
            return False
        if np.count_nonzero(self.blocked[mask]) > 2:
            return False
        return np.count_nonzero(self.support[mask]) >= 0.97 * np.count_nonzero(mask)

    def jaw_clear(self, jaws, point, obj):
        return all(not intersects(j + point, obj) and self.safe_surface(j + point) for j in jaws)

    def corridors(self, plan):
        regions = [expand(hull(np.vstack([self.object, self.object + plan["expected_delta"]])), 5.0)]
        points = np.array([plan[k][:2] for k in ("low_entry", "start", "end", "withdraw")])
        for jaw in plan["jaw_polygons"]:
            regions.append(expand(hull((np.array(jaw)[:, None, :] + points).reshape(-1, 2)), 5.0))
        return regions

    def find(self, min_progress=1.0):
        if self.src.get("stacked") or abs(self.src["height"] - self.ref["height"]) > 8:
            return None
        poly = self.object
        # Inward directions include lateral ways around the other object.
        inward = -self.c / np.linalg.norm(self.c)
        choices = []
        for angle in (0, -20, 20, -40, 40, -60, 60, -65, 65, -70, 70, -75, 75, -80, 80):
            direction = rotation(angle) @ inward
            stroke = 24.0
            delta = direction * stroke
            if np.linalg.norm(self.c + delta) > np.linalg.norm(self.c) - min_progress:
                continue
            swept = expand(hull(np.vstack([poly, poly + delta])), 3.0)
            if not self.safe_surface(swept):
                continue
            self.stats["surface_paths"] += 1
            for i in range(len(poly)):
                a, b = poly[i], poly[(i + 1) % len(poly)]
                edge = b - a
                if np.linalg.norm(edge) < 5:
                    continue
                normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
                if normal @ (self.c - (a + b) / 2) < 0:
                    normal = -normal
                if normal @ direction < 0.55:
                    continue
                for fraction in (0.25, 0.5, 0.75):
                    contact = a + (b - a) * fraction
                    for yaw in range(-105, 106, 15):
                        jaws = [j @ rotation(yaw).T for j in self.jaws]
                        for jaw in jaws:
                            leading = jaw[np.argmax(jaw @ normal)]
                            start = contact - leading - direction * 6.0
                            if not self.jaw_clear(jaws, start, poly):
                                continue
                            end = start + delta
                            if max(np.linalg.norm(start), np.linalg.norm(end)) > self.tip_radius_limit:
                                continue
                            if not all(
                                self.safe_surface(hull(np.vstack([j + start, j + end]))) for j in jaws
                            ):
                                continue
                            self.stats["jaw_clear_contacts"] += 1
                            # Reserve clearance for the measured 6 mm low-pose
                            # tracking error; contact remains below the top face.
                            z = max(rig.TRAY_Z + 2, self.q.tray(start) + min(10.0, self.ref["height"] - 3.0))
                            low = rig.reachable(*start, z, yaw)
                            if not low["ok"]:
                                continue
                            self.stats["reachable_contacts"] += 1
                            # Enter the gap from either side, outside the object.
                            tangent = np.array([-direction[1], direction[0]])
                            for approach_delta in (tangent * 45, -tangent * 45, direction * 50):
                                entry = start + approach_delta
                                if np.linalg.norm(entry) > self.tip_radius_limit:
                                    continue
                                if not self.jaw_clear(jaws, entry, poly):
                                    continue
                                if not all(
                                    self.jaw_clear(jaws, start + approach_delta * t, poly)
                                    for t in np.linspace(0, 1, 12)
                                ):
                                    continue
                                high = max(40.0, self.src["top_z"] + 18)
                                if self.current:
                                    high = max(
                                        high,
                                        self.q.clear_height(self.current["position"][:2], entry, self.name),
                                    )
                                retreat = end - direction * 7
                                z_end = max(
                                    rig.TRAY_Z + 2, self.q.tray(end) + min(10.0, self.ref["height"] - 3.0)
                                )
                                exit_high = max(40.0, self.src["top_z"] + 18)
                                path = [
                                    ([*entry, high], yaw),
                                    ([*entry, z], yaw),
                                    ([*start, z], yaw),
                                    ([*end, z_end], yaw),
                                    ([*retreat, z_end], yaw),
                                    ([*retreat, exit_high], yaw),
                                ]
                                if self.current:
                                    path = [
                                        (self.current["position"], self.current["yaw"]),
                                        ([*self.current["position"][:2], high], self.current["yaw"]),
                                    ] + path
                                # Check densely, chaining the same IK branch across
                                # approach, contact, withdrawal and lift.
                                dense = []
                                for point, angle in path:
                                    if dense:
                                        prev, pa = dense[-1]
                                        count = max(
                                            1,
                                            math.ceil(np.linalg.norm(np.array(point) - prev) / 6),
                                            math.ceil(abs((angle - pa + 180) % 360 - 180) / 10),
                                        )
                                        dense.extend(
                                            [
                                                (
                                                    np.array(prev) + (np.array(point) - prev) * k / count,
                                                    pa + ((angle - pa + 180) % 360 - 180) * k / count,
                                                )
                                                for k in range(1, count + 1)
                                            ]
                                        )
                                    else:
                                        dense.append((np.array(point), angle))
                                self.stats["geometry_candidates"] += 1
                                if not self.q.path_ok(dense):
                                    continue
                                self.stats["reachable_paths"] += 1
                                choices.append(
                                    {
                                        "name": self.name,
                                        "start": [*start, z],
                                        "end": [*end, z_end],
                                        "entry": [*entry, high],
                                        "low_entry": [*entry, z],
                                        "withdraw": [*retreat, z_end],
                                        "exit": [*retreat, exit_high],
                                        "yaw": yaw,
                                        "expected_delta": delta.tolist(),
                                        "score": np.linalg.norm(self.c + delta) + 0.1 * abs(yaw),
                                        "vacated": list(self.vacated),
                                        "jaw_polygons": [j.tolist() for j in jaws],
                                    }
                                )
                                return min(choices, key=lambda p: p["score"])
        return None
