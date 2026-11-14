"""Compile the real stacking demo and exercise its controller without motor I/O.

Run with the editor's GUI config directory and PYTHONPATH=src:.
Recorded reference poses stand in for live perception, and actions update
simulated joint state immediately. This does not test contacts or perception.
"""

import json
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.api.robot import _make_robot_from_profile
from lerobot.policies.flash.demo_document import atomic_json, document_path, load_document
from lerobot.policies.flash.demo_execution import compile_demo
from lerobot.policies.flash.demo_live_worker import ExecutionLoop

CONFIG = Path.home() / ".cache/flash-demo-editor-preview/config"
OUT = Path(__file__).parent
library = json.loads((CONFIG / "flash/editor/demos.json").read_text())
entry = next(e for e in library["demos"] if e["id"] == "b025c44d6f5c4341a078a0ee2d57fde0")
profile = json.loads((CONFIG / "robots/white_left.json").read_text())
robot = _make_robot_from_profile(profile)
robot.connect = Mock(side_effect=AssertionError("No motors in this diagnostic"))
robot.send_action = Mock(side_effect=AssertionError("No motors in this diagnostic"))
dataset = LeRobotDataset(Path(entry["root"]).name, root=entry["root"])
document, _ = load_document(dataset, entry["episode"])
folder = document_path(dataset, entry["episode"]).with_suffix("")
tracked = folder / json.loads((folder / "latest.json").read_text())["directory"]
plan = compile_demo(dataset, entry["episode"], document, tracked, robot, "overhead")
atomic_json(OUT / "compiled-plan.json", plan)
print(f"Compiled {len(plan['keyframes'])} keyframes", flush=True)
model = robot.cartesian_model()
base_camera = np.asarray(plan["base_from_camera"])
starts = {key["reference_frame"]: key for key in plan["keyframes"]}
results = []
for name, shift, from_first_row in (
    ("recorded placement", np.zeros(3), False),
    ("references shifted 10mm robot X", np.array([0.01, 0, 0]), False),
    ("approach from recording start", np.zeros(3), True),
):
    loop = ExecutionLoop(plan, robot, OUT, Event(), simulate=True)
    if from_first_row:
        row = dataset[0]
        loop.state = dict(
            zip(
                dataset.meta.features["observation.state"]["names"],
                row["observation.state"].tolist(),
                strict=True,
            )
        )
    completed = []
    no_progress = 0
    for i in range(9000):
        now = 1 + i / 30
        if loop.executor.index == len(plan["keyframes"]):
            break
        key = plan["keyframes"][loop.executor.index]
        anchor = np.array(key["demo_base_from_reference"])
        anchor[:3, 3] += shift
        # Supply evidence only at entry. Interior frames intentionally have no
        # observation, including the recorded ring loss at frame 601.
        loop.observations = (
            {}
            if loop.executor.reference_frame == key["reference_frame"]
            else {
                key["reference"]: {
                    "status": key["reference_status"],
                    "pose": (np.linalg.inv(base_camera) @ anchor).tolist(),
                    "source_time_s": now,
                    "available_time_s": now,
                }
            }
        )
        previous = loop.executor.index
        action = loop.tick(now, 1 / 30)
        if loop.executor.index > previous:
            actual = model.forward(loop.state)
            goal = loop.executor.target
            completed.append(
                {
                    "frame": key["frame"],
                    "position_error_mm": float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]) * 1000),
                    "rotation_error_deg": float(
                        np.rad2deg(Rotation.from_matrix(actual[:3, :3].T @ goal[:3, :3]).magnitude())
                    ),
                    "gripper": action[model.gripper],
                }
            )
            no_progress = 0
        else:
            no_progress += 1
        if no_progress > 900:
            break
    result = {
        "name": name,
        "state": loop.executor.status,
        "completed": len(completed),
        "total": len(plan["keyframes"]),
        "simulated_seconds": (i + 1) / 30,
        "commands": loop.commands_sent,
        "next_frame": plan["keyframes"][loop.executor.index]["frame"]
        if loop.executor.index < len(plan["keyframes"])
        else None,
        "max_position_error_mm": max((r["position_error_mm"] for r in completed), default=None),
        "max_rotation_error_deg": max((r["rotation_error_deg"] for r in completed), default=None),
        "completed_keyframes": completed,
    }
    results.append(result)
    print({k: v for k, v in result.items() if k != "completed_keyframes"}, flush=True)
    atomic_json(
        OUT / "result.json",
        {
            "demo": entry,
            "tracking_directory": str(tracked),
            "ranges": [
                {"frame": f, "reference": k["reference"], "status": k["reference_status"]}
                for f, k in starts.items()
            ],
            "runs": results,
            "physical_execution": False,
            "scope": "Real recorded joints/actions, calibrated FK/IK and controller; recorded entry poses, ideal motor response, no contacts or live perception.",
        },
    )
robot.connect.assert_not_called()
robot.send_action.assert_not_called()
