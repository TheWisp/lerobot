"""Measured, fixed camera poses owned by a robot, separate from motor calibration.

``T_base_cam`` maps camera optical coordinates (metres; x right, y down,
z forward) into this robot's base frame. A geometry file is explicitly bound
to a robot ID and camera configuration. Moving either mount requires a new
measurement; a matching serial number cannot detect a moved camera.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class RobotGeometry:
    def __init__(self, data: dict[str, Any], *, robot_id: str | None):
        if not robot_id or data.get("robot_id") != robot_id:
            raise ValueError("Robot geometry belongs to a different robot ID")
        cameras = data.get("cameras")
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError("Robot geometry needs at least one calibrated camera")
        for name, camera in cameras.items():
            if (
                not isinstance(camera, dict)
                or not isinstance(camera.get("identity"), dict)
                or not camera["identity"]
            ):
                raise ValueError(f"Camera {name!r} needs its identifying configuration")
            if "type" not in camera["identity"]:
                raise ValueError(f"Camera {name!r} identity needs its type")
            transform = np.asarray(camera.get("T_base_cam"), dtype=float)
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5, rtol=0)
            ):
                raise ValueError(f"Camera {name!r} needs a rigid 4x4 T_base_cam in metres")
        # Keep the actual measurements, fit errors, tool offset and joint-zero
        # corrections together. Snapshots cannot mutate the robot's geometry.
        self._data = copy.deepcopy(data)

    @classmethod
    def load(cls, path: str | Path | None, *, robot_id: str | None) -> RobotGeometry | None:
        if path is None:
            return None
        data = json.loads(Path(path).expanduser().read_text())
        if not isinstance(data, dict):
            raise ValueError("Robot geometry must be a JSON object")
        return cls(data, robot_id=robot_id)

    def camera_pose(self, name: str, camera_config: Any) -> np.ndarray:
        """Require this specific camera and return a copy of its measured pose."""

        def configured(key: str, default: Any = None) -> Any:
            value = (
                camera_config.get(key, default)
                if isinstance(camera_config, Mapping)
                else getattr(camera_config, key, default)
            )
            return value.value if isinstance(value, Enum) else value

        camera = self._data["cameras"].get(name)
        if camera is None:
            raise ValueError(f"Camera {name!r} has no camera-to-base calibration")
        for key, expected in camera["identity"].items():
            actual = configured(key)
            if actual != expected:
                raise ValueError(f"Camera {name!r} differs from its calibration ({key})")
        # Pixel rotation changes optical axes; it is not just a display setting.
        rotation = configured("rotation", 0)
        if rotation != camera["identity"].get("rotation", 0):
            raise ValueError(f"Camera {name!r} rotation differs from its calibration")
        return np.asarray(camera["T_base_cam"], dtype=float).copy()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)
