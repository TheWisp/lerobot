#!/usr/bin/env python3
"""Closed-loop scene reset for the left SO-107. All hardware access is through rig.

Run with the lerobot Python environment. --scramble --seed N selects a new,
collision-free, reachable goal. Unknown/ambiguous scenes fail closed, with logs.
No live object coordinates are stored in this policy.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import datetime as dt
import hashlib
import json
import math
import signal
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import rig
from load_monitor import LoadMonitor
from slide_planner import SlidePlanner, intersects

HERE = Path(__file__).resolve().parent
NAMES = ("gamepad", "cube")


def wrap(a):
    return (float(a) + 180) % 360 - 180


def rot(a):
    a = np.deg2rad(a)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def clean(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items() if k != "mask"}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


class Unsafe(RuntimeError):  # noqa: N818 - retain the reset API name
    pass


class ThermalLimit(Unsafe):
    pass


# Honor the session's 50 C park/cool/reconnect limit, including during moves.
THERMAL_LIMIT_C = 50.0


def too_hot(temp):
    return THERMAL_LIMIT_C is not None and temp > THERMAL_LIMIT_C


class Vision:
    def __init__(self):
        self.target = rig.target()
        self.goals = {
            n: rig.measure(m, self.target["depth"], self.target["rgb"])
            for n, m in self.target["masks"].items()
        }
        g = self.goals["gamepad"]
        self.plus(g, self.target["masks"]["gamepad"], self.target["rgb"], self.target["depth"])

    def plus(self, o, mask, rgb, depth):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # The white controls fail the shell's colour filter and leave holes in its mask. A cable or its shadow across
        # the shell can open such a hole to the outside, cutting the D-pad out of the filled outline: search the hull.
        hull = cv2.convexHull(cv2.findNonZero(mask.astype("uint8")))
        inner = (
            cv2.erode(cv2.fillConvexPoly(np.zeros(mask.shape, np.uint8), hull, 1), np.ones((5, 5), np.uint8))
            > 0
        )
        parts = []
        # Local shadow can make the D-pad only slightly brighter than the
        # mint shell. Retry stricter saturation with a lower brightness gate.
        for saturation, brightness in ((0.60, 1.05), (0.55, 1.03), (0.50, 1.02), (0.55, 1.01)):
            white = (
                inner
                & (hsv[:, :, 1] < max(25, o["hsv"][1] * saturation))
                & (hsv[:, :, 2] > o["hsv"][2] * brightness)
            ).astype("uint8")
            _, labels, stats, centres = cv2.connectedComponentsWithStats(white)
            parts = sorted(
                [
                    (int(s[4]), i, c)
                    for i, (s, c) in enumerate(zip(stats, centres, strict=True))
                    if i and s[4] >= 12
                ],
                reverse=True,
                key=lambda p: p[0],
            )
            if len(parts) >= 2 and parts[0][0] >= 1.6 * parts[1][0] and parts[0][0] >= o["pixels"] * 0.025:
                break
        else:
            raise Unsafe("Cannot distinguish the D-pad from the round buttons")
        area, label, uv = parts[0]
        xyz = np.nanmedian(rig._arm_points(depth)[labels == label], axis=0)
        along = rot(o["axis_deg"])[:, 0]
        delta = xyz[:2] - o["centre"]
        if abs(delta @ along) < self.goals["gamepad"]["length"] * 0.16:
            raise Unsafe("D-pad component is not near a gamepad end")
        o["plus_direction_deg"] = wrap(o["axis_deg"] + (0 if delta @ along > 0 else 180))
        o["plus_pixel"] = uv.tolist()
        o["plus_component_area"] = area
        o["plus_confidence_ratio"] = round(area / parts[1][0], 2)

    def see(self, rgb, depth, require_cube=True, allow_partial=False):
        pts = rig._arm_points(depth)
        heights = rig._tray_height(pts)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        # The arm stays inside the workspace box; an object centred in it may reach past its sides by half its length,
        # and is seen whole (a gamepad that slipped across x = -200 measured 53 mm long, its D-pad cut off).
        reach = max(g["length"] for g in self.goals.values()) / 2
        inbox = (
            (pts[:, :, 0] > rig.WORK_MIN[0] + 5 - reach)
            & (pts[:, :, 0] < rig.WORK_MAX[0] - 5 + reach)
            & (pts[:, :, 1] > rig.WORK_MIN[1] + 5)
            & (pts[:, :, 1] < -65)
            & (heights > 4)
            & (heights < 85)
        )
        # Wide hue bands allow daylight changes; saturation, dimensions and
        # the white controls distinguish these objects from the other clutter.
        filters = {
            "gamepad": (h > 40) & (h < 96) & (s > 22) & (s < 123) & (v > 65),
            "cube": (h > 23) & (h < 58) & (s > 95) & (v > 70),
        }
        found = {}
        candidates = {}
        # A thin thing lying across an object (the wrist board's cable, 2026-10-08) cuts its colour mask in two, and
        # the halves measured as two small blobs, neither the object. Under it the object is still one raised body,
        # the thing on top raised with it: an object is the pixels of its colour on one raised body. The depth camera
        # sees nothing on the black cable, so the strip it leaves (16 px across the gamepad) is filled in from around
        # it first; a body is not cut where the height is merely unknown.
        known = ~np.isnan(heights)
        filled = np.where(known, heights, 0.0).astype(np.float32)
        for _ in range(10):
            weight = cv2.blur(known.astype(np.float32), (3, 3))
            grown = cv2.blur(filled, (3, 3)) / np.maximum(weight, 1e-6)
            take = ~known & (weight > 0)
            filled[take], known[take] = grown[take], True
        _, bodies = cv2.connectedComponents(((filled > 4) & (filled < 85)).astype("uint8"))
        for name in NAMES:
            binary = (inbox & filters[name]).astype("uint8")
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            # A neighbour's edge in blended colour can pass this filter as a strand a pixel or two wide that the
            # closing joins on (the gamepad against the cube measured 102 mm long): thinner than the kernel, it goes.
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            # Each piece of the colour mask: its body, its place, its size, its colour. A piece of 300 px seeds an
            # object and takes the pieces on its body within one object length of it that share its hue and
            # saturation (a shadow, the cable's, dims only the value). The tray's rim is a raised body too and joins
            # what stands along it, and a neighbour's edge in blended colour passes the filter: the cube's, 47 mm
            # from the gamepad, hue 43 against its 72.
            n_pieces, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            pieces = []
            for k in range(1, n_pieces):
                px = labels == k
                if stats[k, cv2.CC_STAT_AREA] < 20:
                    continue
                body = np.bincount(bodies[px]).argmax()
                pieces.append(
                    (
                        int(stats[k, cv2.CC_STAT_AREA]),
                        k,
                        int(body),
                        np.nanmean(pts[px][:, :2], axis=0),
                        np.median(hsv[px][:, :2], axis=0),
                    )
                )
            pieces.sort(reverse=True, key=lambda piece: piece[0])
            reach = self.goals[name]["length"]
            taken = set()
            matches = []
            for area, k, body, at, colour in pieces:
                if area < 300 or k in taken:
                    continue
                mine = [
                    j
                    for a, j, b, where, c in pieces
                    if j not in taken
                    and b == body
                    and np.linalg.norm(where - at) < reach
                    and abs(c[0] - colour[0]) <= 12
                    and abs(c[1] - colour[1]) <= 35
                ]
                taken.update(mine)
                m = np.isin(labels, mine) & inbox
                try:
                    o = rig.measure(m, depth, rgb)
                except (ValueError, cv2.error):
                    continue
                ref = self.goals[name]
                if name == "gamepad":
                    valid = (
                        ref["length"] * 0.75 < o["length"] < ref["length"] * 1.28
                        and ref["width"] * 0.65 < o["width"] < ref["width"] * 1.35
                        and o["length"] / o["width"] > 1.4
                    )
                    if not valid:
                        continue
                    try:
                        self.plus(o, m, rgb, depth)
                    except Unsafe:
                        continue
                else:
                    valid = (
                        ref["length"] * 0.65 < o["length"] < ref["length"] * 1.35
                        and ref["width"] * 0.60 < o["width"] < ref["length"] * 1.3
                        and ref["height"] * 0.60 < o["height"] < ref["height"] * 1.35
                    )
                    if not valid:
                        continue
                o["mask"] = m
                matches.append(o)
            candidates[name] = matches
            if len(matches) == 1:
                found[name] = matches[0]
            elif len(matches) > 1:
                raise Unsafe(f"Ambiguous {name}: {clean(matches)}")
        if "gamepad" not in found:
            if not allow_partial:
                raise Unsafe("Gamepad not confidently recognised")
            found["gamepad"] = {"occluded": True, "centre": None, "axis_deg": None, "stacked": False}
        g = found["gamepad"]
        if not g.get("occluded"):
            g["stacked"] = g["height"] > self.goals["gamepad"]["height"] + 9
        if "cube" not in found:
            if not allow_partial and (not g["stacked"] or require_cube):
                raise Unsafe("Cube not visible; no safe independent cube grasp")
            found["cube"] = {"occluded": True, "centre": None, "axis_deg": None}
        else:
            found["cube"]["occluded"] = False
        # Cache a tray plane from the actual depth; no image-layout constants.
        flat = (abs(heights) < 1.5) & np.isfinite(heights)
        p = pts[flat]
        plane = np.linalg.lstsq(np.c_[p[:, :2], np.ones(len(p))], p[:, 2], rcond=None)[0]
        return found, {"pts": pts, "heights": heights, "plane": plane, "rgb": rgb, "depth": depth}


class Policy:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.folder = HERE / "runs" / dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.folder.mkdir(parents=True)
        source = Path(__file__).read_bytes()
        (self.folder / "policy.py").write_bytes(source)
        (self.folder / "slide_planner.py").write_bytes((HERE / "slide_planner.py").read_bytes())
        for dependency in ("load_monitor.py", "rig.py"):
            (self.folder / dependency).write_bytes((HERE / dependency).read_bytes())
        self.log = {
            "mode": "scramble" if args.scramble else "reset",
            "seed": args.seed,
            "policy_sha256": hashlib.sha256(source).hexdigest(),
            "events": [],
            "success": False,
            "tolerances": {
                "position_mm": getattr(args, "tol_mm", 10.0),
                "angle_deg": getattr(args, "tol_deg", 10.0),
            },
        }
        self.vision = Vision()
        self.log["target"] = clean(self.vision.goals)
        self.scene = None
        self.frame = None
        self.held = None
        self.grasp_failures = dict.fromkeys(NAMES, 0)
        self.failed_yaws = {n: [] for n in NAMES}
        self.active_goals = self.vision.goals
        self.carry_route = []
        self.minimum_transit_z = None
        self.monitor = None
        self.monitor_enabled = True
        self.motion_phase = "idle"
        self.phase_object = "none"
        self.seq = 0
        self.slides = dict.fromkeys(NAMES, 0)
        self.slide_stalls = dict.fromkeys(NAMES, 0)
        self.deferred_restores = {}
        self.slide_views = dict.fromkeys(NAMES, 0)

    def event(self, kind, **data):
        e = {"time_s": round(time.monotonic() - self.started, 2), "kind": kind, **clean(data)}
        self.log["events"].append(e)
        (self.folder / "report.json").write_text(json.dumps(clean(self.log), indent=2))
        print(json.dumps(e), flush=True)

    def phase(self, name):
        self.motion_phase = name
        if getattr(self, "monitor", None):
            self.monitor.phase = f"{self.seq:03d}_{self.phase_object}_{name}"

    def settle(self, seconds=0.35):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.state()
            if getattr(self, "monitor", None):
                self.monitor.check()
            time.sleep(0.05)

    def state(self, recovery=False):
        s = rig.state()
        self.log["max_observed_motor_temp_c"] = max(
            self.log.get("max_observed_motor_temp_c", 0), s.get("max_motor_temp_c", 0)
        )
        if not s.get("connected"):
            raise Unsafe("Arm disconnected")
        # A servo freeze ends this attempt, including a set-down recovery.
        # Only rig.park() may clear it to put the arm safely at rest.
        if too_hot(s.get("max_motor_temp_c", 0)) and not recovery:
            raise ThermalLimit(f"Motor temperature {s['max_motor_temp_c']} C exceeds 50 C")
        if s.get("halted"):
            raise Unsafe("Arm halted: " + str(s.get("reason")))
        if (
            getattr(self, "slide_floor_guard", False)
            and not recovery
            and s["position"][2] < self.tray(s["position"][:2]) + 1
        ):
            raise Unsafe("Actual slide fingertip fell below tray clearance; parking")
        return s

    def capture(self, tag, require_cube=False, allow_partial=False):
        connected = rig.state().get("connected", False)
        if connected:
            self.state()
        rgb, depth = rig.look()
        cv2.imwrite(str(self.folder / f"{tag}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(self.folder / f"{tag}_depth.npy", depth)
        scene, frame = self.vision.see(rgb, depth, require_cube, allow_partial)
        # The observation pose projects the wrist/camera board into the tray.
        # Identify that known moving body from its measured tip pose, height,
        # and neutral materials; preserve every recognised object mask.
        state = self.state() if connected else {"position": [0, -100, 25]}
        pts = frame["pts"]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # The forearm extends from the base toward the tip, so a tip-only
        # radius leaves part of the arm masquerading as a tall tray obstacle.
        tip = np.array(state["position"][:2], float)
        along = np.clip(np.sum(pts[:, :, :2] * tip, axis=2) / max(tip @ tip, 1), 0, 1)
        near = np.linalg.norm(pts[:, :, :2] - along[..., None] * tip, axis=2) < 100
        neutral = (hsv[:, :, 1] < 65) | (hsv[:, :, 2] < 85)
        robot = near & neutral & (pts[:, :, 2] > max(35, state["position"][2] - 45))
        robot = cv2.dilate(robot.astype("uint8"), np.ones((7, 7), np.uint8)) > 0
        for obj in scene.values():
            if "mask" in obj:
                robot &= ~cv2.dilate(obj["mask"].astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
        frame["self_mask"] = robot
        self.remember_floor(rgb, pts, frame["heights"])
        frame["floor_memory"] = self.floor_memory.copy()
        cv2.imwrite(str(self.folder / f"{tag}_robot_mask.png"), robot.astype("uint8") * 255)
        self.scene, self.frame = scene, frame
        self.event("perception", image=f"{tag}.jpg", objects=scene)
        return scene

    def remember_floor(self, rgb, pts, heights):
        """Retain measured bare tray, never inferred empty space."""
        if not hasattr(self, "floor_memory"):
            self.floor_memory = np.zeros((176, 201), np.uint8)
        ij = np.rint(np.nan_to_num((pts[:, :, :2] - np.array(rig.WORK_MIN)) / 2, nan=-1000)).astype(int)
        valid = (
            np.isfinite(heights) & (abs(heights) < 4) & (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2] < 125)
        )
        valid &= (ij[:, :, 0] >= 0) & (ij[:, :, 0] < 201) & (ij[:, :, 1] >= 0) & (ij[:, :, 1] < 176)
        self.floor_memory[ij[:, :, 1][valid], ij[:, :, 0][valid]] = 1

    def tray(self, xy):
        return float(np.r_[xy, 1] @ self.frame["plane"])

    def offset(self, name, yaw):
        # Jaw point at one face; rotate the demonstrated pinch geometry.
        held = getattr(self, "held", None)
        local = held.get("grasp_offset") if held and held["name"] == name else None
        return rot(yaw) @ np.array(local if local is not None else [7.0, 22.0 if name == "gamepad" else 18.0])

    def move(self, xyz, yaw=None, recovery=False, tolerance=5.0, descent=False):
        xyz = np.asarray(xyz, float)
        s = self.state(recovery)
        yaw = s["yaw"] if yaw is None else wrap(yaw)
        if xyz[2] < rig.TRAY_Z + 2 - 0.01:
            raise Unsafe("Requested descent below API floor")
        distance = np.linalg.norm(xyz - s["position"])
        turn = wrap(yaw - s["yaw"])
        if not descent and distance <= tolerance and abs(turn) <= 5:
            self.event("already_at_pose", target=xyz, yaw=yaw, position=s["position"])
            return s
        gentle = bool(self.held) or getattr(self, "motion_phase", "") in ("pick_lift", "empty_lift")
        # descend() already limits each change to its last commanded height.
        # Splitting again can stop at an intermediate held-short target while
        # its contact detector measures the unsent final target as a stall.
        # Even the last 3 mm step must reach the controller, not the 5 mm
        # already-at-pose shortcut.
        steps = (
            1
            if descent
            else max(
                1, math.ceil(distance / (8 if gentle else 25)), math.ceil(abs(turn) / (5 if gentle else 15))
            )
        )
        start = np.array(s["position"])
        seed = None
        path = []
        for i in range(1, steps + 1):
            p = start + (xyz - start) * i / steps
            a = s["yaw"] + turn * i / steps
            reach = rig.reachable(*p, a, seed=seed)
            if not reach["ok"]:
                raise Unsafe(f"Unreachable motion {p.tolist()}, yaw {a}: {reach['why']}")
            seed = reach["seed_next"]
            path.append((p, a))
        commands = []
        last_position, last_yaw = start, s["yaw"]
        for index, (p, a) in enumerate(path):
            if (
                gentle
                or index == len(path) - 1
                or np.linalg.norm(p - last_position) >= 60
                or abs(wrap(a - last_yaw)) >= 30
            ):
                commands.append((p, a))
                last_position, last_yaw = p, a
        for p, a in commands:
            self.state(recovery)
            result = self.monitored_move(p, a, recovery)
            self.event(
                "move", target=p, yaw=a, result=result, max_motor_temp_c=rig.state().get("max_motor_temp_c")
            )
            if result.get("held_short") and descent:
                # Return contact evidence immediately; retrying the same lower
                # target would maintain pressure before descend() can react.
                obs = self.state(recovery)
                if np.linalg.norm(np.array(obs["position"])[:2] - p[:2]) > 7:
                    self.rebase_at_observed(a, recovery)
                    raise Unsafe("Descent drifted more than 7 mm sideways")
                return obs
            # A timeout alone is not a failed lift. Reissue the same point,
            # never move sideways until actual clearance has been reached.
            for _retry in range(2):
                obs = self.state(recovery)
                if self.arrived(obs, p, a, tolerance, result):
                    break
                result = self.monitored_move(p, a, recovery)
                self.event("arrival_retry", target=p, result=result)
                obs = self.state(recovery)
            if not self.arrived(obs, p, a, tolerance, result):
                raise Unsafe(f"Arm did not arrive: {obs}, target {p.tolist()}")
            if self.held and not recovery:
                self.carry_route.append((obs["position"], obs["yaw"]))
            if gentle and not recovery:
                self.settle(0.3)
        return obs

    def monitored_move(self, point, yaw, recovery=False):
        if getattr(self, "monitor", None):
            return self.monitor.call(
                lambda: rig.move(*point, yaw), target={"position": list(point), "yaw": yaw}, recovery=recovery
            )
        if recovery:
            return rig.move(*point, yaw)

        def check_temperature(_signum, _frame):
            try:
                state = rig.state()
            except Exception:
                return
            temp = state.get("max_motor_temp_c", 0)
            self.log["max_observed_motor_temp_c"] = max(self.log.get("max_observed_motor_temp_c", 0), temp)
            if too_hot(temp):
                raise ThermalLimit(f"Motor temperature {temp} C exceeds 50 C during motion")
            if (self.folder / "STOP").exists():
                raise Unsafe("Requested stop; parking")
            if getattr(self, "slide_floor_guard", False) and state.get("connected"):
                position = state["position"]
                if position[2] < self.tray(position[:2]) + 1:
                    self.event(
                        "slide_clearance_abort",
                        position=position,
                        minimum_z=self.tray(position[:2]) + 1,
                        target=point,
                    )
                    raise Unsafe("Actual slide fingertip fell below tray clearance; parking")

        previous = signal.signal(signal.SIGALRM, check_temperature)
        signal.setitimer(signal.ITIMER_REAL, 0.5, 0.5)
        try:
            return rig.move(*point, yaw)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def arrived(self, obs, desired, yaw, tolerance, result=None):
        error = np.array(obs["position"]) - desired
        if abs(wrap(obs["yaw"] - yaw)) > 5:
            return False
        if np.linalg.norm(error) <= tolerance:
            return True
        minimum = self.held.get("minimum_tip_z") if self.held else getattr(self, "minimum_transit_z", None)
        # Held short by the servos (rig.move): commanded onto the target and stopped within the servos' band of it,
        # which waiting and reissuing do not close. Arrived when that leaves the fingertip above the target (a
        # descent), or below it no further than the tolerance or than the swept corridor allows.
        if result and result.get("held_short"):
            z = obs["position"][2]
            if np.linalg.norm(error[:2]) <= 6 and (
                z >= desired[2] - tolerance or (minimum is not None and z >= minimum)
            ):
                return True
        # Loaded motion may sag while staying well above every surface in the
        # swept corridor. Only use this allowance in transit, never on descent.
        if self.held and desired[2] > 15 and 0 <= error[2] <= 10 and np.linalg.norm(error[:2]) <= 6:
            return True
        if minimum is not None and desired[2] >= minimum + 4:
            return np.linalg.norm(error[:2]) <= 6 and minimum <= obs["position"][2] <= desired[2] + 5
        return False

    def descend(self, xyz, yaw, recovery=False):
        p = np.asarray(xyz, float)
        s = self.state(recovery)
        if np.linalg.norm(np.array(s["position"])[:2] - p[:2]) > 7:
            raise Unsafe("Descent is not vertically above the grasp/drop")
        self.phase("recovery" if recovery else ("place_descent" if self.held else "pick_descent"))
        # Servo deadband occurs in free air too: detect lost downward progress,
        # not held_short alone. Never accumulate deeper targets after contact.
        z = s["position"][2]
        previous_z = z
        stalls = 0
        for index in range(40):
            next_z = max(p[2], z - (3 if z < p[2] + 15 else 6))
            if next_z >= z - 0.1:
                break
            obs = self.move([*p[:2], next_z], yaw, recovery, descent=True)
            progress = previous_z - obs["position"][2]
            gap = obs["position"][2] - next_z
            self.event(
                "descent_progress",
                command_z=next_z,
                actual_z=obs["position"][2],
                progress_mm=progress,
                short_mm=gap,
                load=obs.get("load"),
            )
            stalled = gap > 3 and progress < 0.5 * (z - next_z)
            stalls = stalls + 1 if stalled else 0
            if stalled and index > 0 and (obs["position"][2] <= p[2] + 12 or stalls >= 2):
                self.event("contact_stop", position=obs["position"], uncompleted_target=p)
                if obs["position"][2] > p[2] + 12:
                    self.rebase_at_observed(yaw, recovery)
                    raise Unsafe("Descent stalled unexpectedly far above contact; no grip or release")
                break
            z = next_z
            previous_z = obs["position"][2]
            if z <= p[2] + 0.1:
                break
            time.sleep(0.15)
        return self.rebase_at_observed(yaw, recovery)

    def rebase_at_observed(self, yaw, recovery=False):
        s = self.state(recovery)
        point = list(s["position"])
        point[2] = max(point[2], rig.TRAY_Z + 2)
        if not self.path_ok([(s["position"], s["yaw"]), (point, s["yaw"])]):
            raise Unsafe("Cannot safely clear residual descent target")
        self.phase("recovery" if recovery else ("place_relax" if self.held else "pick_relax"))
        # Deliberately bypass move()'s already-at-pose shortcut: the controller
        # must receive this target to cancel its old, lower one.
        result = self.monitored_move(point, s["yaw"], recovery)
        self.event("contact_rebase", target=point, result=result)
        if not recovery:
            self.settle(0.35)
        return self.state(recovery)

    def lift(self, height, yaw):
        self.phase("pick_lift" if self.held else "empty_lift")
        initial = self.state()
        anchor = initial["position"][:2]
        command_z = initial["position"][2]
        for _ in range(18):
            s = self.state()
            if s["position"][2] >= height - 4:
                self.settle(0.4)
                if self.state()["position"][2] >= height - 4:
                    return
            next_z = min(height + 8, command_z + 8)
            if next_z <= command_z + 0.1:
                raise Unsafe("Lift remained short after bounded upward compensation")
            point = [*anchor, next_z]
            if not self.path_ok([(s["position"], s["yaw"]), (point, yaw)]):
                raise Unsafe("No reachable gentle lift step")
            result = self.monitored_move(point, yaw)
            self.event("lift_step", target=point, result=result)
            observed = self.state()
            if np.linalg.norm(np.array(observed["position"][:2]) - anchor) > 7:
                raise Unsafe("Lift drifted outside its vertical corridor")
            if abs(wrap(observed["yaw"] - yaw)) > 5:
                raise Unsafe("Lift yaw did not settle")
            if self.held:
                self.carry_route.append((observed["position"], observed["yaw"]))
            command_z = next_z
            self.settle(0.35)
        raise Unsafe("Lift failed to reach and hold clearance height")

    def grip(self, p, recovery=False):
        self.phase("grip" if p > 65 else ("release" if self.held else "open"))
        self.state(recovery)
        r = (
            self.monitor.call(lambda: rig.grip(p), target={"gripper": p}, recovery=recovery)
            if getattr(self, "monitor", None)
            else rig.grip(p)
        )
        self.event("grip", command=p, result=r)
        return r

    def clear_height(self, start, end, name, raw=False):
        # Swept corridor, enlarged for the object and open jaws. This avoids
        # demanding a height above distant clutter that the path never crosses.
        pts = self.frame["pts"]
        xy = pts[:, :, :2]
        a, b = np.asarray(start), np.asarray(end)
        v = b - a
        t = np.clip(np.sum((xy - a) * v, axis=2) / max(v @ v, 1), 0, 1)
        dist = np.linalg.norm(xy - (a + t[..., None] * v), axis=2)
        valid = (dist < (48 if name == "gamepad" else 35)) & np.isfinite(pts[:, :, 2])
        valid &= ~self.frame.get("self_mask", np.zeros(valid.shape, bool))
        own = self.scene[name].get("mask")
        if own is not None:
            moving = cv2.dilate(own.astype("uint8"), np.ones((9, 9), np.uint8)) > 0
            for other, obj in self.scene.items():
                if other != name and "mask" in obj:
                    moving &= ~obj["mask"]
            valid &= ~moving
        # Fixed arms and clutter remain obstacles.
        top = float(np.percentile(pts[:, :, 2][valid], 99)) if valid.any() else 20
        if self.scene[name].get("stacked"):
            top = max(top, self.tray(self.scene[name]["centre"]) + self.vision.goals["cube"]["height"])
        return top if raw else max(30.0, top + 18.0)

    def path_ok(self, points):
        seed = None
        previous = None
        for point, yaw in points:
            point = np.array(point, float)
            if previous is None:
                samples = [(point, yaw)]
            else:
                start, a = previous
                delta = wrap(yaw - a)
                count = max(1, math.ceil(np.linalg.norm(point - start) / 25), math.ceil(abs(delta) / 15))
                samples = [
                    (start + (point - start) * t / count, a + delta * t / count) for t in range(1, count + 1)
                ]
            for xyz, angle in samples:
                # rig.move clamps a target into the workspace box: a path outside it would not be the path flown.
                if not (
                    rig.WORK_MIN[0] <= xyz[0] <= rig.WORK_MAX[0]
                    and rig.WORK_MIN[1] <= xyz[1] <= rig.WORK_MAX[1]
                ):
                    return False
                r = rig.reachable(*xyz, angle, seed=seed)
                if not r["ok"]:
                    return False
                seed = r["seed_next"]
            previous = (point, yaw)
        return True

    def plan(self, name, src, dst):
        # A kinematically reachable high lift can still overload a loaded arm.
        # Retract first outside the inner demonstration work region.
        if (
            name == "gamepad"
            and np.linalg.norm(src["centre"]) > np.linalg.norm(self.vision.goals[name]["centre"]) + 40
        ):
            return self.edge_plan(name, src, dst)
        choices = []
        base = rig.grasp_yaw(src["axis_deg"])
        turns = (
            [wrap(dst["plus_direction_deg"] - src["plus_direction_deg"])]
            if name == "gamepad"
            else [wrap(dst["axis_deg"] - src["axis_deg"] + k * 90) for k in range(4)]
        )
        for yaw in [base, wrap(base + 180)] if name == "gamepad" else [wrap(base + k * 90) for k in range(4)]:
            if abs(yaw) > 120:
                continue
            pickxy = np.array(src["centre"]) + self.offset(name, yaw)
            # For an elevated gamepad, grasp above its support; never pinch the cube.
            elevation = max(0, src["height"] - self.vision.goals[name]["height"]) if name == "gamepad" else 0
            # The tall cube can be pinched at mid-face; the thin gamepad must
            # be grasped near its underside as in the demonstration.
            jaw_height = 2 if name == "gamepad" else self.vision.goals[name]["height"] * 0.35
            pickz = max(
                rig.TRAY_Z + 2, self.tray(pickxy) + jaw_height + elevation + (3 if elevation > 9 else 0)
            )
            for turn in turns:
                dyaw = wrap(yaw + turn)
                if abs(dyaw) > 120:
                    continue
                dropxy = np.array(dst["centre"]) + self.offset(name, dyaw)
                bottom_offset = pickz - self.tray(src["centre"]) - elevation
                dropz = max(rig.TRAY_Z + 2, self.tray(dst["centre"]) + bottom_offset + 2)
                separation = np.linalg.norm(np.array(src["centre"]) - dst["centre"])
                # An in-place correction needs an excursion to uncover its source.
                if separation >= 65:
                    probes = [(np.array(dst["centre"]), dropxy)]
                else:
                    probes = []
                    for delta in (
                        [0, 75],
                        [75, 0],
                        [-75, 0],
                        [50, 55],
                        [-50, 55],
                        [50, -55],
                        [-50, -55],
                        [0, -75],
                    ):
                        centre = np.array(src["centre"]) + delta
                        pose = dict(dst, centre=centre.tolist())
                        if self.free(pose, name):
                            probes.append((centre, centre + self.offset(name, dyaw)))
                for probe_centre, probe in probes:
                    obstacle_top = max(
                        self.clear_height(src["centre"], probe_centre, name, raw=True),
                        self.clear_height(probe_centre, dst["centre"], name, raw=True),
                    )
                    clear = max(
                        self.clear_height(src["centre"], probe_centre, name),
                        self.clear_height(probe_centre, dst["centre"], name),
                    )
                    clear += max(0, bottom_offset - 3)
                    points = [([*pickxy, pickz], yaw), ([*pickxy, clear + 4], yaw)]
                    for t in np.linspace(0, 1, 6):
                        points.append(([*((1 - t) * pickxy + t * probe), clear + 4], yaw + t * turn))
                    points.extend([([*dropxy, clear + 4], dyaw), ([*dropxy, dropz], dyaw)])
                    if self.path_ok(points):
                        choices.append(
                            {
                                "pick": [*pickxy, pickz],
                                "drop": [*dropxy, dropz],
                                "probe": [*probe, clear + 4],
                                "yaw": yaw,
                                "drop_yaw": dyaw,
                                "clear": clear,
                                "minimum_tip_z": max(8, obstacle_top + bottom_offset + 8),
                                "score": abs(turn)
                                + 0.05 * (np.linalg.norm(pickxy - probe) + np.linalg.norm(probe - dropxy))
                                + 0.1 * abs(yaw)
                                + 500 * sum(abs(wrap(yaw - f)) < 25 for f in self.failed_yaws[name]),
                            }
                        )
                        break
        return min(choices, key=lambda q: q["score"]) if choices else self.edge_plan(name, src, dst)

    def edge_plan(self, name, src, dst, turn=None):
        """Reach an end-adjacent pinch, retract at local clearance, then turn.

        Candidate offsets are relative to the measured outline, never to a
        particular scene. The offset rotates with the held object at placement.
        With ``turn`` (deg) the object is set down at ``dst`` turned only that much.
        """
        if name != "gamepad":
            return None
        base = rig.grasp_yaw(src["axis_deg"])
        full = wrap(dst["plus_direction_deg"] - src["plus_direction_deg"])
        turn = full if turn is None else turn
        elevation = max(0, src["height"] - self.vision.goals[name]["height"])
        length = self.vision.goals[name]["length"]
        shift_limit = min(22.0, length / 2 - 12.0)
        choices = []
        for skew in (0, 12, -12, 24, -24, 36, -36):
            for edge in (0, 180):
                yaw = wrap(base + edge + skew)
                dyaw = wrap(yaw + turn)
                if max(abs(yaw), abs(dyaw)) > 120:
                    continue
                for shift in (0, shift_limit, -shift_limit):
                    offset = rot(yaw) @ np.array([7.0, 22.0]) + rot(src["axis_deg"]) @ np.array([shift, 0.0])
                    # Oblique jaws must still start outside the measured long
                    # edge; rotating the demo offset alone can put a jaw on top.
                    normal = rot(src["axis_deg"])[:, 1]
                    outward = 1.0 if offset @ normal >= 0 else -1.0
                    outside = self.vision.goals[name]["width"] / 2 + 3
                    offset += normal * outward * max(0.0, outside - abs(offset @ normal))
                    local = rot(-yaw) @ offset
                    pickxy = np.array(src["centre"]) + offset
                    pickz = max(
                        rig.TRAY_Z + 2, self.tray(pickxy) + 2 + elevation + (3 if elevation > 9 else 0)
                    )
                    low = rig.reachable(*pickxy, pickz, yaw)
                    if not low["ok"]:
                        continue
                    bottom = pickz - self.tray(src["centre"]) - elevation
                    dropxy = np.array(dst["centre"]) + rot(dyaw) @ local
                    dropz = max(rig.TRAY_Z + 2, self.tray(dst["centre"]) + bottom + 2)
                    for delta in ([35, 35], [-35, 35], [0, 50], [50, 0], [-50, 0], [35, -35], [-35, -35]):
                        centre = np.array(src["centre"]) + delta
                        retreat = centre + offset
                        top = self.clear_height(src["centre"], centre, name, raw=True)
                        minimum = max(8.0, top + bottom + 8.0)
                        lift = max(20.0, minimum + 14, pickz + 12)  # reserve for measured loaded sag
                        if not rig.reachable(*pickxy, lift, yaw, seed=low["seed_next"])["ok"]:
                            continue
                        far_top = self.clear_height(centre, dst["centre"], name, raw=True)
                        high = max(lift, far_top + bottom + 20, 34.0)
                        transit_min = max(8.0, far_top + bottom + 8.0)
                        # Verification must uncover the source before release.
                        if np.linalg.norm(np.array(dst["centre"]) - src["centre"]) >= 65:
                            probe = dropxy
                        else:
                            verify_centre = np.array(src["centre"]) + np.array(delta) * 1.8
                            probe = verify_centre + rot(dyaw) @ local
                            extra = max(
                                self.clear_height(centre, verify_centre, name, raw=True),
                                self.clear_height(verify_centre, dst["centre"], name, raw=True),
                            )
                            high = max(high, extra + bottom + 20)
                            transit_min = max(transit_min, extra + bottom + 8)
                        approach = [([*retreat, high], yaw), ([*retreat, lift], yaw), ([*pickxy, lift], yaw)]
                        escape = [([*pickxy, lift], yaw), ([*retreat, lift], yaw), ([*retreat, high], yaw)]
                        points = (
                            approach
                            + [([*pickxy, pickz], yaw)]
                            + escape
                            + [([*probe, high], dyaw), ([*dropxy, high], dyaw), ([*dropxy, dropz], dyaw)]
                        )
                        if not self.path_ok(points):
                            continue
                        choices.append(
                            {
                                "pick": [*pickxy, pickz],
                                "drop": [*dropxy, dropz],
                                "probe": [*probe, high],
                                "yaw": yaw,
                                "drop_yaw": dyaw,
                                "clear": high - 4,
                                "minimum_tip_z": minimum,
                                "transit_minimum_tip_z": transit_min,
                                "grasp_offset": local.tolist(),
                                "approach": approach,
                                "escape": escape,
                                "score": 40
                                + abs(turn)
                                + abs(skew)
                                + abs(shift) * 0.3
                                + np.linalg.norm(delta) * 0.05
                                + 0.8
                                * max(
                                    0,
                                    np.linalg.norm(pickxy)
                                    - np.linalg.norm(self.vision.goals[name]["centre"]),
                                )
                                + 500 * sum(abs(wrap(yaw - f)) < 25 for f in self.failed_yaws[name]),
                            }
                        )
                        break
        if choices:
            return min(choices, key=lambda p: p["score"])
        # The wrist turns at most 120 deg either way at the pick and at the drop. A gamepad turned further from its
        # goal than that allows is set down at the goal turned part of the way; the next correction turns the rest.
        if turn == full:
            for part in reversed(range(15, int(abs(full)), 15)):
                plan = self.edge_plan(name, src, dst, turn=math.copysign(part, full))
                if plan:
                    return {**plan, "partial_turn": math.copysign(part, full)}
        return None

    def diagnose_reach(self, name):
        """Explain a refusal without moving or weakening any safety limit.

        A finer search distinguishes unreachable contact/lift poses from a
        blocked transfer. This is a report about this grasp family, not a proof
        that every possible manipulation or wrist orientation is impossible.
        """
        src = self.scene[name]
        if name != "gamepad" or src.get("occluded"):
            return None
        base = rig.grasp_yaw(src["axis_deg"])
        normal = rot(src["axis_deg"])[:, 1]
        along = rot(src["axis_deg"])[:, 0]
        limit = min(22.0, self.vision.goals[name]["length"] / 2 - 12.0)
        elevation = max(0, src["height"] - self.vision.goals[name]["height"])
        rows = []
        for edge in (0, 180):
            for skew in range(-36, 37, 6):
                yaw = wrap(base + edge + skew)
                for shift in np.linspace(-limit, limit, 9):
                    offset = rot(yaw) @ np.array([7.0, 22.0]) + along * shift
                    outward = 1.0 if offset @ normal >= 0 else -1.0
                    outside = self.vision.goals[name]["width"] / 2 + 3
                    offset += normal * outward * max(0.0, outside - abs(offset @ normal))
                    xy = np.array(src["centre"]) + offset
                    z = max(rig.TRAY_Z + 2, self.tray(xy) + 2 + elevation + (3 if elevation > 9 else 0))
                    low = rig.reachable(*xy, z, yaw)
                    row = {"pick": [*xy, z], "yaw": yaw, "skew": skew, "shift": shift, "contact": low}
                    if low["ok"]:
                        lift_z = max(20.0, z + 12)
                        row["lift_z"] = lift_z
                        row["lift"] = rig.reachable(*xy, lift_z, yaw, seed=low["seed_next"])
                    rows.append(row)
        filename = f"reachability_{name}.json"
        (self.folder / filename).write_text(json.dumps(clean(rows), indent=2))
        summary = {
            "name": name,
            "candidates": len(rows),
            "reachable_contacts": sum(r["contact"]["ok"] for r in rows),
            "reachable_lifts": sum(r.get("lift", {}).get("ok", False) for r in rows),
            "details": filename,
            "scope": "long-edge pinches; fixed rig wrist pitch; yaw deviations up to 36 degrees",
        }
        self.event("reachability_refusal", **summary)
        return summary

    def footprint(self, pose, name, padding=5):
        ref = self.vision.goals[name]
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
        corners *= [ref["length"] / 2 + padding, ref["width"] / 2 + padding]
        return (corners @ rot(pose["axis_deg"]).T + pose["centre"]).astype(np.float32)

    def overlaps(self, a, aname, b, bname):
        return cv2.intersectConvexConvex(self.footprint(a, aname), self.footprint(b, bname))[0] > 1

    def free(self, dest, name, scene=None):
        scene = scene or self.scene
        c = np.array(dest["centre"])
        for other in NAMES:
            if other == name:
                continue
            if scene[other].get("occluded"):
                # A hidden cube could be under either end of the gamepad.
                if self.overlaps(dest, name, scene["gamepad"], "gamepad"):
                    return False
                continue
            if self.overlaps(dest, name, scene[other], other):
                return False
        # Check the full destination footprint against all non-target surfaces.
        p = self.frame["pts"]
        xy = p[:, :, :2]
        local = (xy - c) @ rot(dest["axis_deg"])
        ref = self.vision.goals[name]
        region = (abs(local[:, :, 0]) < ref["length"] / 2 + 9) & (abs(local[:, :, 1]) < ref["width"] / 2 + 9)
        own = self.scene[name].get("mask", np.zeros(region.shape, bool))
        own = cv2.dilate(own.astype("uint8"), np.ones((11, 11), np.uint8)) > 0
        # The colour mask misses faces in shadow, and the depth smeared off the object's edges stands raised where no
        # surface is: the object is also everything inside its outline in the image and all raised depth joined to it.
        now = self.scene[name]
        if now.get("centre") is not None and now.get("axis_deg") is not None:
            base = self.tray(now["centre"])
            top = base + max(now.get("height") or 0, ref["height"])
            uv = np.array(
                [rig.pixel(x, y, z) for x, y in self.footprint(now, name, padding=3) for z in (base, top)],
                np.int32,
            )
            own |= cv2.fillConvexPoly(np.zeros(region.shape, np.uint8), cv2.convexHull(uv), 1) > 0
        raised = (self.frame["heights"] > 7).astype("uint8")
        _, labels = cv2.connectedComponents(raised)
        joined = np.unique(labels[own & (raised > 0)])
        own |= np.isin(labels, joined[joined > 0])
        occupied = region & (raised > 0) & ~own
        occupied &= ~self.frame.get("self_mask", np.zeros(occupied.shape, bool))
        return np.count_nonzero(occupied) < 35

    def stage(self, name, src, final=None, forbidden=()):
        # Generate staging points from the demonstration work area, with no
        # dependence on the initial live layout. These also enable half turns.
        rng = np.random.default_rng(947)
        middle = np.mean([self.vision.goals[n]["centre"] for n in NAMES], axis=0)
        candidates = []
        proposed = []
        for delta in ([0, 0], [25, 0], [-25, 0], [0, 25], [0, -25]):
            proposed.append(
                {
                    "centre": (np.array(self.active_goals[name]["centre"]) + delta).tolist(),
                    "axis_deg": src["axis_deg"],
                    "plus_direction_deg": src.get("plus_direction_deg", src["axis_deg"]),
                }
            )
        for _ in range(160):
            c = middle + rng.uniform([-65, -15], [70, 65])
            a = rng.uniform(0, 180)
            proposed.append(
                {"centre": c.tolist(), "axis_deg": a, "plus_direction_deg": wrap(a + rng.choice([0, 180]))}
            )
        for d in proposed:
            if not self.free(d, name):
                continue
            if any(intersects(self.footprint(d, name, padding=7), region) for region in forbidden):
                continue
            # A temporary placement must not occupy either final destination.
            if any(n != name and self.overlaps(d, name, g, n) for n, g in self.active_goals.items()):
                continue
            plan = self.plan(name, src, d)
            if not plan:
                continue
            if final:
                future = dict(src, **d, height=self.vision.goals[name]["height"])
                if not self.plan(name, future, final):
                    continue
            candidates.append((plan["score"], d, plan))
            if len(candidates) >= 5:
                break
        if not candidates:
            raise Unsafe(f"No safe reachable staging pose for {name}")
        _, d, plan = min(candidates, key=lambda p: p[0])
        return d, plan

    def verify_lift(self, name, source, before_tip):
        """Inspect the vacated footprint, since the camera board hides a load."""
        state = self.state()
        if not state["holding"]:
            return False, "gripper no longer reports holding"
        rgb, depth = rig.look()
        cv2.imwrite(
            str(self.folder / f"{self.seq:03d}_grasp_check_{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        np.save(self.folder / f"{self.seq:03d}_grasp_check_{name}_depth.npy", depth)
        pts = rig._arm_points(depth)
        plane = self.frame["plane"]
        heights = pts[:, :, 2] - (pts[:, :, 0] * plane[0] + pts[:, :, 1] * plane[1] + plane[2])
        self.remember_floor(rgb, pts, heights)
        polygon = self.footprint(source, name, padding=-4)
        pixels = np.array([rig.pixel(*xy, self.tray(xy)) for xy in polygon], np.int32)
        footprint = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(footprint, pixels, 1)
        footprint = footprint.astype(bool)
        valid = footprint & np.isfinite(heights)
        bare = valid & (abs(heights) < 4.5)
        fraction = float(bare.sum() / max(1, footprint.sum()))
        valid_fraction = float(valid.sum() / max(1, footprint.sum()))
        support_ok = True
        support_pixels = 0
        if name == "gamepad" and source.get("stacked"):
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            lime = (hsv[:, :, 0] > 23) & (hsv[:, :, 0] < 58) & (hsv[:, :, 1] > 95)
            h = self.vision.goals["cube"]["height"]
            support = valid & lime & (abs(heights - h) < 8)
            support_pixels = int(support.sum())
            support_ok = support_pixels > 100
            required = 0.25
        else:
            required = 0.72
        ok = fraction >= required and valid_fraction > 0.85 and support_ok
        self.event(
            "grasp_check",
            name=name,
            holding=state["holding"],
            bare_tray_fraction=fraction,
            valid_depth_fraction=valid_fraction,
            cube_support_pixels=support_pixels,
            source_vacated=ok,
            tip=state["position"],
        )
        return (
            ok,
            "source footprint is occupied or obscured"
            if support_ok
            else "cube not left at source after unstack",
        )

    def carry(self, name, dst, plan):
        self.phase_object = name
        src = self.scene[name]
        if self.grasp_failures[name] >= 3:
            raise Unsafe(f"Three failed grasps of {name}")
        self.seq += 1
        self.event("pick_place_plan", name=name, source=src, destination=dst, plan=plan)
        self.grip(65)
        self.phase("approach")
        s = self.state()
        # Lift vertically before any XY or wrist movement.
        # Return from the observation height vertically, then use the height
        # actually checked by the planner. Do not carry 70 mm into a far reach.
        high = plan["clear"] + 4
        self.minimum_transit_z = plan["minimum_tip_z"]
        self.move([*s["position"][:2], high], s["yaw"])
        if plan.get("approach"):
            for point, angle in plan["approach"]:
                self.move(point, angle)
        else:
            self.move([*plan["pick"][:2], high], plan["yaw"])
        self.descend(plan["pick"], plan["yaw"])
        tip = self.state()["position"]
        bottom = tip[2] - (
            self.tray(src["centre"]) + max(0, src["height"] - self.vision.goals[name]["height"])
        )
        height_delta = tip[2] - plan["pick"][2]
        self.event("measured_grasp_height", planned_z=plan["pick"][2], actual_z=tip[2], bottom_offset=bottom)
        self.held = {
            "name": name,
            "source": src,
            "pick": tip,
            "yaw": plan["yaw"],
            "grasp_offset": plan.get("grasp_offset", [7.0, 22.0 if name == "gamepad" else 18.0]),
            "minimum_tip_z": plan["minimum_tip_z"] + height_delta,
            "bottom_offset": bottom,
        }
        self.carry_route = [(tip, plan["yaw"])]
        g = self.grip(82 if name == "gamepad" else 94)
        if name == "gamepad":
            # 82 is the demonstrated contact position, not a reliable contact
            # command: at 81.4 the API's 0.7 threshold still says not holding.
            # Narrow end-adjacent contacts need a little more closure to retain
            # the gamepad through a turn; camera verification remains mandatory.
            g = self.grip(90 if plan.get("grasp_offset") else 86)
        if not g["holding"]:
            self.grasp_failures[name] += 1
            self.failed_yaws[name].append(plan["yaw"])
            self.event(
                "failed_grasp",
                name=name,
                reason="gripper did not report holding",
                count=self.grasp_failures[name],
            )
            self.recover(return_source=True)
            return False
        if plan.get("escape"):
            for index, (point, angle) in enumerate(plan["escape"]):
                if index == 2:
                    self.held["minimum_tip_z"] = plan["transit_minimum_tip_z"] + height_delta
                if index in (0, 2):
                    self.lift(max(point[2], self.held["minimum_tip_z"] + 4), angle)
                else:
                    self.phase("transfer")
                    self.move(point, angle, tolerance=5)
        else:
            self.lift(max(plan["clear"] + 4, self.held["minimum_tip_z"] + 4), plan["yaw"])
        if self.state()["position"][2] < self.held["minimum_tip_z"]:
            raise Unsafe("Insufficient clearance for verification shift")
        self.phase("transfer")
        # Move along the prechecked transfer path to expose the source footprint.
        # Do not lower/release until both holding and bare-tray evidence agree.
        if plan.get("escape"):
            ok, reason = self.verify_lift(name, src, tip)
            if not ok and reason == "source footprint is occupied or obscured":
                self.move(plan["probe"], plan["drop_yaw"])
                ok, reason = self.verify_lift(name, src, tip)
        else:
            self.move(plan["probe"], plan["drop_yaw"])
            ok, reason = self.verify_lift(name, src, tip)
        if not ok and reason == "source footprint is occupied or obscured":
            # Use low side views; a high pose near the base can exceed wrist
            # limits. Check clearance of the held object's entire extra path.
            for point, _ in self.observation_candidates():
                centre = np.array(point[:2]) - self.offset(name, plan["drop_yaw"])
                current = self.state()
                current_centre = np.array(current["position"][:2]) - self.offset(name, current["yaw"])
                clear = max(
                    plan["clear"] + 4,
                    self.clear_height(current_centre, centre, name)
                    + max(0, self.held["bottom_offset"] - 3)
                    + 4,
                    self.clear_height(centre, dst["centre"], name)
                    + max(0, self.held["bottom_offset"] - 3)
                    + 4,
                )
                point = [*point[:2], clear]
                above = [*current["position"][:2], clear]
                if self.path_ok(
                    [
                        (current["position"], current["yaw"]),
                        (above, current["yaw"]),
                        (point, plan["drop_yaw"]),
                        ([*plan["drop"][:2], clear], plan["drop_yaw"]),
                    ]
                ):
                    self.move(above, current["yaw"])
                    self.move(point, plan["drop_yaw"])
                    ok, reason = self.verify_lift(name, src, tip)
                    if ok:
                        break
        if not ok:
            self.grasp_failures[name] += 1
            self.failed_yaws[name].append(plan["yaw"])
            self.event("failed_grasp", name=name, reason=reason, count=self.grasp_failures[name])
            self.recover(return_source=True)
            return False
        self.move([*plan["drop"][:2], plan["clear"] + 4], plan["drop_yaw"])
        # Preserve the held object's lower surface above the tray; the source
        # stacking height is not carried over into the destination height.
        drop = list(plan["drop"])
        drop[2] = max(drop[2], self.tray(dst["centre"]) + self.held["bottom_offset"] + 2)
        self.descend(drop, plan["drop_yaw"])
        self.grip(65)
        self.held = None
        self.carry_route = []
        self.lift(plan["clear"] + 4, plan["drop_yaw"])
        self.phase("retreat")
        self.retreat()
        self.observe(f"{self.seq:03d}_placed_{name}")
        return True

    def observation_candidates(self, expected=None):
        known = {}
        for scene in (expected or {}, self.scene or {}):
            for name, obj in scene.items():
                if obj.get("centre") is not None:
                    known[name] = obj
        centres = np.array([o["centre"] for o in known.values()])
        if not len(centres):
            centres = np.array([self.vision.goals[n]["centre"] for n in NAMES])
        mid = centres.mean(axis=0)
        xy_points = [
            [float(centres[:, 0].max() + 65), float(mid[1])],
            [float(centres[:, 0].max() + 50), float(mid[1] + 25)],
            [float(centres[:, 0].min() - 60), float(mid[1])],
            [float(mid[0]), float(centres[:, 1].min() - 60)],
        ]
        clear = max(38, max((o.get("top_z", 0) for o in known.values()), default=15) + 15)
        for xy in xy_points:
            if not (
                rig.WORK_MIN[0] + 10 < xy[0] < rig.WORK_MAX[0] - 10 and rig.WORK_MIN[1] + 10 < xy[1] < -70
            ):
                continue
            for yaw in (0, -45, 45):
                yield [*xy, clear], yaw

    def missing_uncovered(self):
        return self.scene["gamepad"].get("occluded") or (
            self.scene["cube"].get("occluded") and not self.scene["gamepad"].get("stacked")
        )

    def observe(self, tag):
        expected = self.scene
        self.capture(tag, allow_partial=True)
        if not self.missing_uncovered():
            return self.scene
        for index, (point, yaw) in enumerate(self.observation_candidates(expected)):
            if index >= 9:
                break
            state = self.state()
            high = max(point[2], state["position"][2])
            above = [*state["position"][:2], high]
            destination = [*point[:2], high]
            if not self.path_ok([(above, state["yaw"]), (destination, yaw)]):
                continue
            self.move(above, state["yaw"])
            self.move(destination, yaw)
            self.capture(f"{tag}_reveal_{index}", allow_partial=True)
            if not self.missing_uncovered():
                return self.scene
        raise Unsafe("Objects remain obscured after reachable observation views")

    def retreat(self):
        state = self.state()
        for point, yaw in self.observation_candidates():
            high = max(point[2], state["position"][2])
            above = [*state["position"][:2], high]
            destination = [*point[:2], high]
            if self.path_ok([(above, state["yaw"]), (destination, yaw)]):
                self.move(above, state["yaw"])
                self.move(destination, yaw)
                return
        raise Unsafe("No reachable side observation pose")

    def recover(self, return_source=False):
        if not self.held:
            return
        h = self.held
        s = self.state(recovery=True)
        xy = np.array(s["position"][:2])
        # Before carry, the original support remains available. A suspected
        # double grasp is returned to its original contact height, not pushed down.
        if return_source and np.linalg.norm(xy - np.array(h["pick"][:2])) < 8:
            z = h["pick"][2]
        else:
            centre = xy - self.offset(h["name"], s["yaw"])
            z = max(rig.TRAY_Z + 2, self.tray(centre) + h["bottom_offset"] + 2)
            # Do not lower onto an obstacle during fault recovery.
            if self.scene and not self.free(
                {"centre": centre, "axis_deg": h["source"]["axis_deg"]}, h["name"]
            ):
                if np.linalg.norm(xy - np.array(h["pick"][:2])) < 8:
                    z = h["pick"][2]
                else:
                    # Retrace verified waypoints if a failure occurs above an
                    # occupied region. Never open the jaws over that region.
                    for p, a in reversed(self.carry_route):
                        if p[2] < s["position"][2] - 5:
                            continue
                        self.move(p, a, recovery=True)
                    s = self.state(recovery=True)
                    xy = np.array(s["position"][:2])
                    if np.linalg.norm(xy - np.array(h["pick"][:2])) > 8:
                        raise Unsafe("Cannot return load to its original support")
                    z = h["pick"][2]
        self.descend([*xy, z], s["yaw"], recovery=True)
        self.grip(65, recovery=True)
        self.held = None
        self.carry_route = []
        self.event("recovery_release", position=[*xy, z])

    def errors(self, goals):
        out = {}
        for n in NAMES:
            o = self.scene[n]
            t = goals[n]
            if o.get("occluded"):
                out[n] = {"position_mm": None, "angle_deg": None, "within_tolerance": False}
                continue
            a = (
                abs(wrap(o["plus_direction_deg"] - t["plus_direction_deg"]))
                if n == "gamepad"
                else abs((o["axis_deg"] - t["axis_deg"] + 45) % 90 - 45)
            )
            d = float(np.linalg.norm(np.array(o["centre"]) - np.array(t["centre"])))
            flat = abs(o["height"] - self.vision.goals[n]["height"]) < 9
            args = getattr(self, "args", None)
            tol_mm, tol_deg = (
                getattr(args, "tol_mm", 10.0),
                getattr(args, "tol_deg", 10.0),
            )  # the reset's strictness
            out[n] = {
                "position_mm": round(d, 2),
                "angle_deg": round(a, 2),
                "flat": flat,
                "within_tolerance": d <= tol_mm and a <= tol_deg and flat,
            }
        return out

    def choose_scramble(self):
        rng = np.random.default_rng(self.args.seed)
        middle = np.mean([self.vision.goals[n]["centre"] for n in NAMES], axis=0)
        for _ in range(400):
            goals = {}
            for n in NAMES:
                a = float(rng.uniform(0, 180))
                goals[n] = {
                    "centre": (middle + rng.uniform([-15, 5], [55, 45])).tolist(),
                    "axis_deg": a,
                    "plus_direction_deg": wrap(a + int(rng.integers(2)) * 180),
                }
            if np.linalg.norm(np.array(goals["gamepad"]["centre"]) - goals["cube"]["centre"]) < 65:
                continue
            if all(self.free(goals[n], n) and self.plan(n, self.scene[n], goals[n]) for n in NAMES):
                return goals
        raise Unsafe("Could not sample two safe, reachable random goals")

    def attempt(self):
        status = 1
        self.slide_floor_guard = False
        try:
            tag = "before" if not getattr(self, "cooldowns", 0) else f"cooldown_{self.cooldowns}_before"
            self.capture(tag, allow_partial=True)
            rig.connect()
            # A reconnect initially reports 0 before motor telemetry arrives.
            if getattr(self, "after_cooldown", False):
                for _ in range(20):
                    st = rig.state()
                    if st.get("max_motor_temp_c", 0) > 0:
                        break
                    time.sleep(0.5)
                if too_hot(st.get("max_motor_temp_c", 0)):
                    self.hot_after_wait = True
                    raise ThermalLimit("Motor still above 50 C immediately after ten-minute cooldown")
                if st.get("max_motor_temp_c", 0) <= 0:
                    raise Unsafe("No temperature telemetry after reconnect")
                self.after_cooldown = False
            self.event("connected", state=self.state())
            self.observe("connected_view")
            goals = self.vision.goals
            # Always peel off an elevated gamepad first. The hidden cube is
            # remeasured only after this operation, never moved from an inferred pose.
            if self.scene["gamepad"].get("stacked"):
                for _ in range(3):
                    dst, plan = self.stage("gamepad", self.scene["gamepad"])
                    if self.carry("gamepad", dst, plan):
                        break
                    self.retreat()
                    self.observe(f"{self.seq:03d}_retry")
                if self.scene["gamepad"].get("stacked"):
                    raise Unsafe("Could not unstack gamepad")
            if self.args.scramble:
                goals = self.log.get("goals") or self.choose_scramble()
            self.log["goals"] = clean(goals)
            self.event("goals", objects=goals)
            self.active_goals = goals
            self.execute_goals(goals)
            self.observe("after")
            self.log["final_errors"] = self.errors(self.vision.goals)
            self.log["goal_errors"] = self.errors(goals)
            status = 0 if all(x["within_tolerance"] for x in self.log["goal_errors"].values()) else 1
            self.log["success"] = status == 0
        except (Exception, KeyboardInterrupt) as e:
            self.log["thermal_stop"] = isinstance(e, ThermalLimit) or too_hot(
                rig.state().get("max_motor_temp_c", 0)
            )
            self.event("error", message=str(e), traceback=traceback.format_exc())
        finally:
            try:
                if self.held and not rig.state().get("halted"):
                    self.recover(return_source=True)
            except Exception as e:
                self.event("recovery_error", message=str(e))
            try:
                self.phase("park")
                rig.park()
                self.event("parked")
            except Exception as e:
                status = 1
                self.log["success"] = False
                self.event("park_error", message=str(e))
            try:
                rgb, depth = rig.look()
                cv2.imwrite(str(self.folder / "after.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                np.save(self.folder / "after_depth.npy", depth)
                self.scene, self.frame = self.vision.see(rgb, depth, False)
                self.log["final_perception"] = clean(self.scene)
                self.log["final_errors"] = self.errors(self.vision.goals)
                if "goals" in self.log:
                    self.log["goal_errors"] = self.errors(self.log["goals"])
                    if not all(x["within_tolerance"] for x in self.log["goal_errors"].values()):
                        status = 1
            except Exception as e:
                status = 1
                self.event("final_measurement_error", message=str(e))
            self.log["success"] = status == 0
            self.log["duration_s"] = round(time.monotonic() - self.started, 2)
            self.event(
                "attempt_finished",
                exit_code=status,
                duration_s=self.log["duration_s"],
                final_errors=self.log.get("final_errors"),
            )
        return status

    def telemetry_guard(self, s):
        if too_hot(s.get("max_motor_temp_c", 0)):
            raise ThermalLimit("Motor temperature above 50 C")
        if (self.folder / "STOP").exists():
            raise Unsafe("Requested stop; parking")
        if (
            getattr(self, "slide_floor_guard", False)
            and s.get("connected")
            and s["position"][2] < self.tray(s["position"][:2]) + 1
        ):
            raise Unsafe("Actual slide fingertip below tray clearance")

    def run(self):
        if not getattr(self, "monitor_enabled", False):
            return self._run()
        self.monitor = LoadMonitor(self.folder)
        self.monitor.started = self.started
        self.monitor.extra_check = self.telemetry_guard
        self.monitor.start()
        try:
            return self._run()
        finally:
            self.log["load_summary"] = self.monitor.close()
            self.event("load_log_saved", path="motor_loads.jsonl")

    def _run(self):
        self.cooldowns = 0
        self.thermal_transients = 0
        self.hot_after_wait = False
        self.after_cooldown = False
        try:
            while True:
                self.log["thermal_stop"] = False
                status = self.attempt()
                if not self.log.get("thermal_stop"):
                    break
                # A one-read spike (34 -> 57 -> 34 C on 20261012) still
                # interrupts motion and parks. Recheck while parked before
                # spending ten minutes cooling an already cool arm.
                if not self.thermal_recheck():
                    self.thermal_transients += 1
                    self.event("thermal_transient", number=self.thermal_transients)
                    if self.thermal_transients >= 3:
                        break
                    self.hot_after_wait = False
                    self.after_cooldown = False
                    continue
                if self.hot_after_wait:
                    break
                self.cooldowns += 1
                self.event("cooldown_start", seconds=600, number=self.cooldowns)
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    time.sleep(min(30, max(0, deadline - time.monotonic())))
                    self.event("cooldown_wait", seconds_remaining=round(max(0, deadline - time.monotonic())))
                self.after_cooldown = True
                self.event("cooldown_complete", number=self.cooldowns)
        except KeyboardInterrupt:
            status = 1
            rig.park()
            self.event("parked_after_interrupt")
        self.log["duration_s"] = round(time.monotonic() - self.started, 2)
        self.log["cooldown_seconds"] = 600 * self.cooldowns
        self.event(
            "finished",
            exit_code=status,
            duration_s=self.log["duration_s"],
            final_errors=self.log.get("final_errors"),
        )
        return status

    def thermal_recheck(self):
        """After parking, require five populated, cool readings to skip cooldown.

        Reconnecting holds the parked pose; no task motion is requested. Any
        hot reading, incomplete telemetry, or API failure keeps the cooldown.
        """
        readings = []
        try:
            if rig.state().get("connected"):
                raise Unsafe("Temperature recheck requires a parked arm")
            rig.connect()
            for _ in range(12):
                time.sleep(0.5)
                s = rig.state()
                temps = s.get("temps") or {}
                if not s.get("connected") or s.get("halted"):
                    break
                if len(temps) == 7 and min(temps.values()) > 0:
                    readings.append(dict(temps))
                    if too_hot(max(temps.values())) or len(readings) >= 5:
                        break
        except Exception as exc:
            self.event("temperature_recheck_error", message=str(exc))
        finally:
            if rig.state().get("connected"):
                code, out = rig._call("/api/jog/park", {}, timeout=120)
                if code != 200:
                    raise Unsafe(f"Cannot park after temperature recheck: {out}")
        confirmed = len(readings) < 5 or any(too_hot(max(r.values())) for r in readings)
        self.event("temperature_recheck", readings=readings, cooldown_required=confirmed)
        return confirmed

    def execute_goals(self, goals):
        if not hasattr(self, "placements"):
            self.placements = dict.fromkeys(NAMES, 0)
            self.stages = dict.fromkeys(NAMES, 0)
            self.command_goals = json.loads(json.dumps(clean(goals)))
        placements = self.placements
        stages = self.stages
        for _operation in range(18):
            errors = self.errors(goals)
            pending = [n for n in NAMES if not errors[n]["within_tolerance"]]
            if not pending:
                return
            for n in pending:
                if placements[n] >= 3:
                    raise Unsafe(f"{n} still outside tolerance after two corrections")
                if self.grasp_failures[n] >= 3:
                    raise Unsafe(f"Three failed grasps of {n}")
            # Prefer any unobstructed final placement. Only stage when neither
            # object can go directly to its goal (including cyclic obstruction).
            options = []
            for n in pending:
                if self.deferred_restores.get(n) in pending:
                    continue
                command = self.command_goals[n]
                if self.free(command, n):
                    plan = self.plan(n, self.scene[n], command)
                    if plan:
                        options.append((plan["score"], n, command, plan))
            final = bool(options)
            # Do not restore a staged blocker ahead of an object that still
            # needs a slide. This also works after a fresh process reconnects.
            slid_first = False
            if options:
                offered = {item[1] for item in options}
                for n in pending:
                    if n in offered or self.slides[n] >= 4 or self.slide_stalls[n] >= 2:
                        continue
                    if not self.free(self.command_goals[n], n):
                        continue
                    planner = SlidePlanner(self, n, self.state())
                    slide = planner.find(min_progress=6.0)
                    if slide:
                        regions = planner.corridors(slide)
                        if any(
                            any(intersects(self.footprint(item[2], item[1], padding=7), r) for r in regions)
                            for item in options
                        ):
                            self.event("slide_before_blocking_placement", name=n, plan=slide)
                            self.slide(slide)
                            slid_first = True
                            break
            if slid_first:
                continue
            if options:
                _, name, dst, plan = min(options, key=lambda x: x[0])
            else:
                possible = []
                needed = set()
                for n in pending:
                    if self.deferred_restores.get(n) in pending:
                        continue
                    if self.free(goals[n], n):
                        needed.add(n)
                    else:
                        for other in NAMES:
                            if (
                                other != n
                                and not self.scene[other].get("occluded")
                                and self.overlaps(goals[n], n, self.scene[other], other)
                            ):
                                needed.add(other)
                for name in NAMES:
                    if name not in needed:
                        continue
                    if stages[name] >= 2 or self.grasp_failures[name] >= 3:
                        continue
                    try:
                        dst, plan = self.stage(name, self.scene[name], goals[name])
                        possible.append((plan["score"], name, dst, plan))
                    except Unsafe:
                        continue
                if not possible:
                    slid = False
                    for n in NAMES:
                        if n not in needed or self.slides[n] >= 4 or self.slide_stalls[n] >= 2:
                            continue
                        planner = SlidePlanner(self, n, self.state())
                        slide = planner.find(min_progress=6.0)
                        self.event("slide_search", name=n, stats=planner.stats, plan=slide)
                        if (
                            not slide
                            and planner.stats["reachable_contacts"] == 0
                            and self.slide_views[n] < 2
                            and self.reveal_slide_corridor(n)
                        ):
                            slid = True
                            break
                        if not slide:
                            # A goal-satisfied object may block the only useful
                            # slide. Move it aside, then reobserve/replan; never
                            # execute a counterfactual slide against live objects.
                            for other in NAMES:
                                if other == n or stages[other] >= 2 or self.scene[other].get("occluded"):
                                    continue
                                hypothetical = SlidePlanner(self, n, self.state(), vacated=(other,))
                                future_slide = hypothetical.find(min_progress=6.0)
                                if not future_slide:
                                    continue
                                try:
                                    temporary, grasp = self.stage(
                                        other,
                                        self.scene[other],
                                        goals[other],
                                        forbidden=hypothetical.corridors(future_slide),
                                    )
                                except Unsafe:
                                    continue
                                self.event(
                                    "clear_slide_corridor",
                                    name=n,
                                    blocker=other,
                                    destination=temporary,
                                    hypothetical_slide=future_slide,
                                )
                                if self.carry(other, temporary, grasp):
                                    stages[other] += 1
                                    self.deferred_restores[other] = n
                                else:
                                    self.retreat()
                                    self.observe(f"{self.seq:03d}_retry")
                                slid = True
                                break
                            if slid:
                                break
                        if slide:
                            self.slide(slide)
                            slid = True
                            break
                    if slid:
                        continue
                    diagnostics = [self.diagnose_reach(n) for n in pending]
                    blocked = [d["name"] for d in diagnostics if d and d["reachable_lifts"] == 0]
                    if blocked:
                        raise Unsafe(
                            "No safe reachable grasp-and-lift for "
                            + ", ".join(blocked)
                            + "; no safe slide remains within clearance, reach and attempt limits; see diagnostics"
                        )
                    raise Unsafe("No safe sequence resolves the blocked destinations or wrist limits")
                _, name, dst, plan = min(possible, key=lambda x: x[0])
            if self.carry(name, dst, plan):
                if final:
                    placements[name] += 1
                else:
                    stages[name] += 1
                self.event(
                    "placement_error",
                    name=name,
                    placement=placements[name],
                    staging=not final,
                    errors=self.errors(goals)[name],
                )
                if final and not self.errors(goals)[name]["within_tolerance"]:
                    measured = self.scene[name]
                    goal = goals[name]
                    command = self.command_goals[name]
                    command["centre"] = (
                        np.array(command["centre"]) + np.array(goal["centre"]) - measured["centre"]
                    ).tolist()
                    if name == "gamepad":
                        error = wrap(measured["plus_direction_deg"] - goal["plus_direction_deg"])
                        command["plus_direction_deg"] = wrap(command["plus_direction_deg"] - error)
                    else:
                        error = (measured["axis_deg"] - goal["axis_deg"] + 45) % 90 - 45
                    command["axis_deg"] = (command["axis_deg"] - error) % 180
                    self.event("correction_goal", name=name, command=command)
            else:
                self.retreat()
                self.observe(f"{self.seq:03d}_retry")
        raise Unsafe("Operation budget exhausted")

    def slide(self, plan):
        """A flat object stays supported; both open jaws follow checked corridors."""
        name = plan["name"]
        before = self.scene[name]
        others = {n: np.array(o["centre"]) for n, o in self.scene.items() if n != name}
        if self.held or before.get("stacked") or plan.get("vacated"):
            raise Unsafe("Sliding requires an unheld, flat object and an observed clear path")
        self.seq += 1
        self.slides[name] += 1
        self.event("slide_plan", name=name, source=before, plan=plan, attempt=self.slides[name])
        self.grip(65)
        self.minimum_transit_z = None
        s = self.state()
        high = plan["entry"][2]
        self.move([*s["position"][:2], high], s["yaw"])
        self.move(plan["entry"], plan["yaw"])
        self.slide_floor_guard = True
        self.descend(plan["low_entry"], plan["yaw"])
        self.settle_slide()
        # No closure, vertical loading, or unchecked long stroke at contact.
        for key in ("start", "end", "withdraw"):
            begin = np.array(self.state()["position"])
            end = np.array(plan[key])
            count = max(1, math.ceil(np.linalg.norm(end - begin) / 6))
            for k in range(1, count + 1):
                point = begin + (end - begin) * k / count
                if key == "end":
                    point[2] = max(begin[2], point[2])
                self.move(point, plan["yaw"], tolerance=4.0)
                self.settle_slide()
        self.move(plan["exit"], plan["yaw"])
        self.slide_floor_guard = False
        self.retreat()
        self.observe(f"{self.seq:03d}_slid_{name}")
        measured = self.scene[name]
        distance = float(np.linalg.norm(np.array(measured["centre"]) - before["centre"]))
        self.slide_stalls[name] = self.slide_stalls[name] + 1 if distance < 3 else 0
        self.event("slide_result", name=name, distance_mm=distance, before=before, after=measured)
        for n, centre in others.items():
            if (
                self.scene[n].get("occluded")
                or np.linalg.norm(np.array(self.scene[n]["centre"]) - centre) > 8
            ):
                raise Unsafe("Another object moved or became hidden during slide")

    def settle_slide(self):
        # Arrival within 4 mm can precede joint settling. Avoid sending the
        # next contact target while the preceding move is still overshooting.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.state()
            time.sleep(0.1)

    def reveal_slide_corridor(self, name):
        """Move the empty hand/cable away and require a fresh scene check."""
        if self.held:
            return False
        s = self.state()
        candidates = sorted(self.observation_candidates(), key=lambda p: np.linalg.norm(p[0][:2]))
        for point, yaw in candidates:
            if np.linalg.norm(np.array(point[:2]) - s["position"][:2]) < 30:
                continue
            high = max(
                point[2],
                s["position"][2],
                self.clear_height(s["position"][:2], point[:2], name),
                max(o.get("top_z", 0) for o in self.scene.values()) + 18,
            )
            above = [*s["position"][:2], high]
            destination = [*point[:2], high]
            if not self.path_ok([(s["position"], s["yaw"]), (above, s["yaw"]), (destination, yaw)]):
                continue
            self.slide_views[name] += 1
            self.event("reveal_slide_corridor", name=name, destination=destination, yaw=yaw)
            self.move(above, s["yaw"])
            self.move(destination, yaw)
            self.observe(f"{self.seq:03d}_slide_view_{self.slide_views[name]}")
            return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scramble", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--tol-mm", type=float, default=10.0, help="how near each object must end to its demo spot, mm"
    )
    parser.add_argument(
        "--tol-deg", type=float, default=10.0, help="how near its demo turn, deg (the cube modulo 90)"
    )
    args = parser.parse_args()
    raise SystemExit(Policy(args).run())
