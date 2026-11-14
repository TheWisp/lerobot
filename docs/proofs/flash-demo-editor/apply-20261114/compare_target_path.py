"""Offline captured-reference controller comparison; never connect motors."""

import json
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.gui.api.robot import _make_robot_from_profile
from lerobot.policies.flash.demo_live_worker import ExecutionLoop

D = Path("docs/proofs/flash-demo-editor/apply-20261114")
p = json.loads((D / "compiled-plan.json").read_text())
e = json.loads((D / "live-execution.json").read_text())
r = _make_robot_from_profile(
    json.loads(Path("/home/feit/.cache/flash-demo-editor-preview/config/robots/white_left.json").read_text())
)
r.connect = Mock(side_effect=AssertionError("No motors"))
r.send_action = Mock(side_effect=AssertionError("No motors"))
a1 = np.array(
    [
        [-0.9992393055258567, -0.035027346002018574, -0.017143375521168066, -0.17230311004117477],
        [-0.027711368482377603, 0.9470686729162621, -0.31983278261845616, -0.08094086091001063],
        [0.027438847443702757, -0.3191144211919702, -0.9473188987021619, 0.4050805254648025],
        [0, 0, 0, 1],
    ]
)
a2 = np.array(e["evidence"]["base_from_reference"])
res = []
for label, ow in [
    ("existing cost", 1.0),
    ("normalized by existing 3mm and 3deg tolerances", 0.003 / np.deg2rad(3)),
]:
    loop = ExecutionLoop(p, r, Path(__file__).parent, Event(), simulate=True)
    pink = loop.model.kinematics.inner._inner._inner
    original = pink.inverse_kinematics
    pink.inverse_kinematics = lambda q, t, solve=original, weight=ow: solve(
        q, t, position_weight=1.0, orientation_weight=weight
    )
    done = []
    stalls = 0
    for i in range(1800):
        if loop.executor.index == len(p["keyframes"]):
            break
        now = 1 + i / 30
        k = p["keyframes"][loop.executor.index]
        anchor = a1 if k["reference"] == "object_1" else a2
        loop.observations = {
            k["reference"]: {
                "status": "observed_candidate",
                "pose": np.linalg.inv(p["base_from_camera"]) @ anchor,
                "source_time_s": now,
                "available_time_s": now,
            }
        }
        idx = loop.executor.index
        loop.tick(now, 1 / 30)
        if loop.executor.index > idx:
            f = loop.model.forward(loop.state)
            t = loop.executor.target
            done.append(
                {
                    "frame": k["frame"],
                    "position_mm": float(np.linalg.norm(f[:3, 3] - t[:3, 3]) * 1000),
                    "angle_deg": float(np.rad2deg(Rotation.from_matrix(t[:3, :3].T @ f[:3, :3]).magnitude())),
                }
            )
            stalls = 0
        else:
            stalls += 1
        if stalls >= 300:
            break
    result = {
        "variant": label,
        "completed": loop.executor.index,
        "next_frame": p["keyframes"][loop.executor.index]["frame"]
        if loop.executor.index < len(p["keyframes"])
        else None,
        "state": loop.executor.status,
        "ticks": i + 1,
        "completed_keys": done,
    }
    res.append(result)
    print({k: v for k, v in result.items() if k != "completed_keys"}, flush=True)
(D / "target-path-comparison.json").write_text(json.dumps(res, indent=2))
r.connect.assert_not_called()
r.send_action.assert_not_called()
