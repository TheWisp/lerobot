"""SO-107 motor-space -> URDF-joint-space alignment (the "layer-2" delta).

The SO-107 URDF was exported from CAD, so its joint-zero does not coincide
with the motors' calibrated zero, and some joint axes are inverted. The
per-joint ``(sign, offset_deg)`` below realigns motor readings to the URDF:

    urdf_angle_deg = sign * motor_angle_deg + offset_deg

This is calibration data, not a model constant — a different physical SO-107
build would measure slightly different values. It lives here only as the
*default* the GUI seeds a new SO-107 robot profile with; the profile's
``urdf_calibration`` section is the editable source of truth. A well-authored
URDF (e.g. the vendored SO-101) needs only the identity alignment and carries
no file like this.

The left and right arms of a bimanual SO-107 have *different* alignments
(some axes are mirror-mounted) — which is exactly why this cannot be baked
into a single shared URDF.
"""

from __future__ import annotations

from dataclasses import dataclass

# Motor order matches the SO-107 follower motor list and the URDF joints S1..S7.
MOTOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
URDF_JOINT_NAMES: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
URDF_TIP_FRAME: str = "L7_1"


@dataclass(frozen=True)
class JointAlignment:
    """Per-joint ``(sign, offset_deg)``. ``urdf_deg = sign * motor_deg + offset_deg``."""

    sign: float
    offset_deg: float


# Discovered empirically per physical arm (the two arms are partly mirror-mounted).
RIGHT_ARM_ALIGNMENT: dict[str, JointAlignment] = {
    "shoulder_pan": JointAlignment(sign=-1.0, offset_deg=+0.00),
    "shoulder_lift": JointAlignment(sign=+1.0, offset_deg=-90.00),
    "elbow_flex": JointAlignment(sign=+1.0, offset_deg=+78.87),
    "forearm_roll": JointAlignment(sign=-1.0, offset_deg=-3.71),
    "wrist_flex": JointAlignment(sign=-1.0, offset_deg=+0.00),
    "wrist_roll": JointAlignment(sign=-1.0, offset_deg=+90.00),
    "gripper": JointAlignment(sign=-1.0, offset_deg=+0.00),
}
LEFT_ARM_ALIGNMENT: dict[str, JointAlignment] = {
    "shoulder_pan": JointAlignment(sign=-1.0, offset_deg=+0.00),
    "shoulder_lift": JointAlignment(sign=+1.0, offset_deg=-90.00),
    "elbow_flex": JointAlignment(sign=+1.0, offset_deg=+78.87),
    "forearm_roll": JointAlignment(sign=-1.0, offset_deg=+0.00),
    "wrist_flex": JointAlignment(sign=+1.0, offset_deg=+0.00),
    "wrist_roll": JointAlignment(sign=-1.0, offset_deg=+90.00),
    "gripper": JointAlignment(sign=+1.0, offset_deg=-90.00),
}
