import json
from pathlib import Path
from threading import Event

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.gui.api.robot import _make_robot_from_profile
from lerobot.policies.flash.demo_live_worker import ExecutionLoop

p = json.loads((Path(__file__).parent / "compiled-plan.json").read_text())
r = _make_robot_from_profile(
    json.loads(Path("/home/feit/.cache/flash-demo-editor-preview/config/robots/white_left.json").read_text())
)
e = json.loads((Path(__file__).parent / "live-execution.json").read_text())
a1 = np.array(
    [
        [-0.9992393055258567, -0.035027346002018574, -0.017143375521168066, -0.17230311004117477],
        [-0.027711368482377603, 0.9470686729162621, -0.31983278261845616, -0.08094086091001063],
        [0.027438847443702757, -0.3191144211919702, -0.9473188987021619, 0.4050805254648025],
        [0, 0, 0, 1],
    ]
)
a2 = np.array(e["evidence"]["base_from_reference"])
loop = ExecutionLoop(p, r, Path(__file__).parent, Event(), simulate=True)
for i in range(3000):
    if loop.executor.index >= len(p["keyframes"]):
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
    previous = loop.model.degrees(loop.state)
    loop.tick(now, 1 / 30)
    q = loop.model.degrees(loop.state)
    if (
        loop.executor.index > 131
        or i > 350
        and loop.executor.status.startswith("waiting")
        and np.array_equal(q, previous)
    ):
        print(
            "tick",
            i,
            "key",
            loop.executor.index,
            "frame",
            k["frame"],
            "status",
            loop.executor.status,
            "q",
            q,
            "delta",
            q - previous,
            flush=True,
        )
        if loop.executor.index == 131 or i > 2000:
            break
m = loop.model
target = loop.executor.target
print("limits lower upper", m.lower_deg, m.upper_deg)
print("recorded reference", p["keyframes"][131]["demo_base_from_reference"])
print("current reference", a2.tolist())
for name, state in [("current", loop.state), ("recorded", p["keyframes"][131]["state"])]:
    q = m.kinematics.inverse_kinematics(m.degrees(state), target)
    achieved = m.kinematics.forward_kinematics(q)
    print(
        name,
        "full-goal IK q",
        q,
        "error_mm",
        np.linalg.norm(achieved[:3, 3] - target[:3, 3]) * 1000,
        "angle",
        np.rad2deg(Rotation.from_matrix(achieved[:3, :3].T @ target[:3, :3]).magnitude()),
        flush=True,
    )
print("state", loop.state)
print("target", target.tolist())
