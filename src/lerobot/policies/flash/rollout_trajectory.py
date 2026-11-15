"""Dispatched Cartesian intent in the run's saved camera frame.

Adapted demo targets are retained before IK and motor compensation. Reset and
approach use their requested joint transitions. Undispatched plans are excluded.
"""

import json

import numpy as np
from scipy.interpolate import PchipInterpolator

from lerobot.robots.cartesian_model import CartesianModel


def read_trajectory(directory):
    commands, publications = [], {}
    origin, requested, pending = None, None, None
    segment = 0
    path = directory / "events.jsonl"
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break  # A live writer or interrupted write; never invent a point.
                event = json.loads(line)
                kind = event["kind"]
                if kind == "run.started":
                    origin = event["time_s"]
                elif kind == "plan.published":
                    publications[event["revision"]] = event.get("evidence")
                elif kind == "robot.command_requested":
                    requested, pending = event["requested"], None
                elif kind == "robot.command_failed":
                    requested, pending = None, None
                    segment += 1
                elif kind == "robot.command_accepted":
                    pending = None
                    if requested and event.get("accepted"):
                        at = event.get("started_s", event["time_s"])
                        origin = at if origin is None else origin
                        pending = {
                            "time_s": at,
                            "relative_s": at - origin,
                            "sequence": event["sequence"],
                            "segment": segment,
                            "action": requested,
                            "accepted": event["accepted"],
                            "receipt": None,
                        }
                        commands.append(pending)
                    requested = None
                elif kind == "control.tick":
                    receipt = event["tick"].get("receipt")
                    if (
                        pending is not None
                        and receipt
                        and receipt["requested"]["action"] == pending["action"]
                        and receipt["accepted"] == pending["accepted"]
                    ):
                        pending["receipt"] = receipt
                        pending = None
    points = []
    if commands:
        plan = json.loads((directory / "plan.json").read_text())
        camera_from_base = np.linalg.inv(np.asarray(plan["base_from_camera"]))
        keys = {key["frame"]: key for key in plan["keyframes"]}
        model, gaps = None, 0
        for command in commands:
            receipt = command["receipt"]
            queued = receipt["requested"] if receipt else {}
            target = queued.get("tool_target")
            interval = queued.get("interval", "")
            if target is not None:
                tool = np.asarray(target)
                source = "adapted_target"
            elif interval.startswith("demo:"):
                # Older journals retain the dispatched revision and its exact
                # reference pose. Recover its pre-IK target, never today's pose
                # or FK of the compensated motor goals. A publication may be
                # journaled just after its first receipt, hence this second pass.
                key = keys[int(interval.removeprefix("demo:"))]
                evidence = publications.get(receipt["revision"])
                if key["reference"] != "world" and not evidence:
                    gaps += 1
                    continue
                reference = (
                    np.eye(4) if key["reference"] == "world" else np.asarray(evidence["base_from_reference"])
                )
                tool = reference @ np.asarray(key["reference_from_tool"])
                source = "adapted_target"
            else:
                # Reset and approach are joint-space transitions. Their intended
                # path is the request before gravity compensation / motor limits.
                if model is None:
                    model = CartesianModel.from_description(plan["robot_cartesian"], plan["robot_geometry"])
                if not all(name in command["action"] for name in model.joint_names):
                    gaps += 1
                    continue
                tool = model.forward(command["action"])
                source = "joint_transition"
            points.append(
                {key: command[key] for key in ("time_s", "relative_s", "sequence")}
                | {
                    "position": (camera_from_base @ tool)[:3, 3].tolist(),
                    "segment": command["segment"] + gaps,
                    "source": source,
                }
            )
    return {"origin_s": origin, "points": points, "spans": fit_trajectory(points)}


def fit_trajectory(points):
    # Shape-preserving interpolation passes through every issued point without
    # cubic overshoot. Repeated timestamps start a new span rather than sorting
    # commands into a different order. Points remain separate for inspection.
    curves = []
    for point in points:
        if (
            not curves
            or point["segment"] != curves[-1][-1]["segment"]
            or point["time_s"] <= curves[-1][-1]["time_s"]
        ):
            curves.append([])
        curves[-1].append(point)
    spans = []
    for curve in curves:
        times = np.asarray([point["time_s"] for point in curve])
        positions = np.asarray([point["position"] for point in curve])
        samples = (
            times
            if len(curve) < 2
            else np.r_[
                np.concatenate(
                    [np.linspace(a, b, 4, endpoint=False) for a, b in zip(times[:-1], times[1:], strict=True)]
                ),
                times[-1],
            ]
        )
        fitted = positions if len(curve) < 2 else PchipInterpolator(times, positions)(samples)
        spans.append({"times": samples.tolist(), "positions": fitted.tolist()})
    return spans
