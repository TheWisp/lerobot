"""Click-to-goto bimanual EE teleop (experimental).

Emits the same Cartesian-EE action shape as ``ScriptedBimanualEETeleop``
/ ``QuestVRTeleop`` so a robot's ``attach_teleop`` recognises it and
installs the Cartesian-IK adapter. Adds ``use_world_target`` +
``target_world_x/y/z`` keys that the extended ``CartesianIKController``
consumes as absolute world targets — populated when the GUI posts a
mailbox request unprojected from a top-camera pixel click.

Calibration (Kabsch fit of ``T_base_camera`` from
``(camera_frame_xyz, base_frame_gripper_xyz)`` pairs) lives in
:mod:`calibration`; the file-based IPC mailbox in :mod:`mailbox`.
"""

from .calibration import (
    kabsch_se3,
    load_extrinsics,
    save_extrinsics,
    unproject_pixel,
)
from .click_target_teleop import ClickTargetBimanualEETeleop
from .configuration_click_target import ClickTargetBimanualEETeleopConfig
from .mailbox import ClickMailbox
from .service import ClickCalibrationService

__all__ = [
    "ClickCalibrationService",
    "ClickMailbox",
    "ClickTargetBimanualEETeleop",
    "ClickTargetBimanualEETeleopConfig",
    "kabsch_se3",
    "load_extrinsics",
    "save_extrinsics",
    "unproject_pixel",
]
