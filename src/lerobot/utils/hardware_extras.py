# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What a chosen robot needs installed, answerable before it is launched.

Hardware dependencies are optional extras, guarded where they are used —
``require_package`` inside a motor bus or a camera's constructor. That is the
right place to stop, and the wrong place to *ask*: by then a subprocess has
been spawned, hardware is being opened, and the answer reaches a log rather
than the person who chose the robot.

The declaration lives on the config classes (``required_extras``) because
that is what the choice registry already resolves a type name to, and the
same question can then be asked from anywhere with no hardware attached.
This does not replace the guards. A type that declares nothing still gets
stopped at construction, as before; what it does not get is a warning first.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Mapping

#: What to import to tell whether an extra is installed here. Keyed by the
#: extra's name in ``pyproject.toml``; the values are what its packages are
#: imported as, which is often not what they are installed as.
#:
#: Every extra a hardware guard can raise about belongs here — a test reads
#: the guards out of the tree and fails if one is missing, so this cannot
#: quietly fall behind them.
IMPORT_FOR_EXTRA: dict[str, tuple[str, ...]] = {
    "damiao": ("can",),
    "deepdiff-dep": ("deepdiff",),
    "dynamixel": ("dynamixel_sdk",),
    "feetech": ("scservo_sdk",),
    "gamepad": ("hid",),
    "intelrealsense": ("pyrealsense2",),
    "openarm-ff": ("mujoco", "openarm_mujoco"),
    "phone": ("hebi", "teleop"),
    "pynput-dep": ("pynput",),
    "pyserial-dep": ("serial",),
    "pyzmq-dep": ("zmq",),
    "reachy2": ("reachy2_sdk",),
    "rebot": ("motorbridge",),
    "robstride": ("can",),
    "unitree_g1": ("unitree_sdk2py",),
}


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A package whose parent is itself missing raises rather than
        # answering; either way it is not usable here.
        return False


def missing_extras(extras: Iterable[str]) -> set[str]:
    """Those of ``extras`` whose packages are not importable in this process.

    An extra this module does not know is never reported missing: it cannot
    be checked, and refusing a launch over a question we did not ask would be
    worse than letting the guard speak.
    """
    missing = set()
    for extra in extras:
        imports = IMPORT_FOR_EXTRA.get(extra)
        if imports and not all(_importable(name) for name in imports):
            missing.add(extra)
    return missing


def install_command(extras: Iterable[str]) -> str:
    """The command that fixes it, for the message a person will read."""
    return "uv sync " + " ".join(f"--extra {extra}" for extra in sorted(extras))


def _declared_by(registry, type_name: str | None) -> set[str]:
    if not type_name:
        return set()
    try:
        cls = registry.get_choice_class(type_name)
    except Exception:
        # An unregistered type is somebody else's error to report, and a
        # clearer one than anything this could say about it.
        return set()
    return set(getattr(cls, "required_extras", ()) or ())


def extras_for_robot(
    robot_type: str | None,
    cameras: Mapping[str, Mapping] | None = None,
    teleop_type: str | None = None,
) -> set[str]:
    """Every extra the chosen hardware declares, arm and cameras together.

    The camera set is per profile rather than per robot, so it is asked for
    here too — a depth camera added to an arm brings its own driver with it.

    Precondition: the config modules have been imported, so the choice
    registry knows these type names. The GUI does that in
    ``_ensure_configs_loaded``. A type the registry has not heard of returns
    nothing rather than guessing, so forgetting this makes the check silent
    rather than wrong.
    """
    from lerobot.cameras.configs import CameraConfig
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    extras = _declared_by(RobotConfig, robot_type)
    extras |= _declared_by(TeleoperatorConfig, teleop_type)
    for camera in (cameras or {}).values():
        extras |= _declared_by(CameraConfig, (camera or {}).get("type"))
    return extras
