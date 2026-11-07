"""Execute the Robots-tab ZED camera form test under Node when available, and
hand the camera configs it leaves after cross-type reassignments to the parser
the launch uses."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import draccus
import pytest

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,  # noqa: F401  registers the type
)
from lerobot.cameras.zed.configuration_zed import ZEDCameraConfig


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_robot_zed_camera_js() -> None:
    test_js = Path(__file__).parent / "robot_zed_camera.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

    # The form's own assertions say what it wrote; the parser says whether the
    # launch would take it. A key kept from the role's previous type is refused
    # there ("The fields `enable_hole_filling` are not valid for ZEDCameraConfig")
    # and used to break the next record or teleop, not the save.
    line = next(line for line in result.stdout.splitlines() if line.startswith("cameras "))
    states = json.loads(line[len("cameras ") :])
    parsed = [draccus.decode(dict[str, CameraConfig], cameras) for cameras in states]
    assert [type(c["top"]) for c in parsed] == [ZEDCameraConfig, OpenCVCameraConfig, OpenCVCameraConfig]
    assert parsed[0]["top"].use_depth is True  # a setting both types have crosses with the role
    assert type(parsed[1]["wrist"]) is ZEDCameraConfig and type(parsed[2]["wrist"]) is OpenCVCameraConfig

    # The parser does refuse a foreign key; otherwise the checks above prove nothing.
    with pytest.raises(draccus.utils.DecodingError, match="enable_hole_filling"):
        draccus.decode(dict[str, CameraConfig], {"top": dict(states[0]["top"], enable_hole_filling=False)})
