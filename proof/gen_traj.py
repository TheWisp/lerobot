"""Generate IK shape trajectories for the browser E2E recording.

3 shapes x 2 arms = 6 trajectories. Shapes are oriented toward the base
(matching tests/model/test_pink_ik_trajectory.py) so they stay reachable.
Dumps joint angles (radians, keyed by URDF joint name) + pinocchio FK EE
position per frame.
"""

import importlib
import json
import math
from pathlib import Path

import numpy as np

from lerobot.model.pink_kinematics import PinkKinematics

OUT = Path("/tmp/ikproof")  # nosec B108 - one-off proof harness, scratch dir
N = 256
Z = np.array([0.0, 0.0, 1.0])

ARMS = [
    {"rid": "so101", "ee": "gripper_frame_link", "seed": [0, 45, -90, 45, 0, 0], "ow": 0.0},
    {"rid": "so107", "ee": "L7_1", "seed": [0, -90, 60, 0, -40, 0, 0], "ow": 1.0},
]
SHAPES = [
    ("circle-30mm", "circle", 0.030),
    ("circle-60mm", "circle", 0.060),
    ("square-50mm", "square", 0.050),
]


def basis(t0):
    p = t0[:3, 3].copy()
    flat = np.array([p[0], p[1], 0.0])
    inward = -flat / np.linalg.norm(flat)
    return p, inward, np.cross(inward, Z)


def shape_positions(kind, size, t0, n):
    p, inward, perp = basis(t0)
    if kind == "circle":
        c = p + size * inward
        return [
            c + size * (math.cos(2 * math.pi * i / n) * -inward + math.sin(2 * math.pi * i / n) * perp)
            for i in range(n)
        ]
    corners = [p, p + size * inward, p + size * inward + size * perp, p + size * perp]
    per_edge = n // 4
    pts = []
    for e in range(4):
        a, b = corners[e], corners[(e + 1) % 4]
        pts += [a + (b - a) * (k / per_edge) for k in range(per_edge)]
    return pts


for arm in ARMS:
    mod = importlib.import_module(f"lerobot.robots.{arm['rid']}_description")
    urdf = mod.get_urdf_path()
    kin = PinkKinematics(urdf_path=str(urdf), target_frame_name=arm["ee"])
    seed = np.array(arm["seed"], dtype=float)
    t0 = kin.forward_kinematics(seed)
    for sname, skind, ssize in SHAPES:
        positions = shape_positions(skind, ssize, t0, N)
        q = seed.copy()
        frames, fk = [], []
        for pos in positions:
            target = t0.copy()
            target[:3, 3] = pos
            q = kin.inverse_kinematics(q, target, position_weight=1.0, orientation_weight=arm["ow"])
            tg = kin.forward_kinematics(q)
            frames.append({name: math.radians(float(q[j])) for j, name in enumerate(kin.joint_names)})
            fk.append([float(x) for x in tg[:3, 3]])
        data = {
            "robot": arm["rid"],
            "shape": sname,
            "urdf": f"{arm['rid']}/urdf/{urdf.name}",
            "ee_link": arm["ee"],
            "joint_names": list(kin.joint_names),
            "frames": frames,
            "fk_ee": fk,
        }
        (OUT / f"traj_{arm['rid']}_{sname}.json").write_text(json.dumps(data))
        print(f"{arm['rid']} {sname}: {len(frames)} frames")
