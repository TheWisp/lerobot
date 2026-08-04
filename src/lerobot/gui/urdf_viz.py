# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observation -> URDF joint angles for the URDF state visualization.

Pure Python: no rendering, no pinocchio, no meshcat. The in-browser viewer
(``gui/static/urdf_viz.html``) fetches the joint angles computed here and
draws them with three.js / urdf-loader. Keeping the motor->URDF conversion
in Python (not JS) puts the correctness-critical logic under pytest and
lets it carry asserts.

This module is **robot-agnostic** — it names no specific robot. Each robot
ships a ``*_description`` package that declares a ``VIZ_SPEC`` (motor names,
URDF joint names, optional per-arm calibration); :func:`resolve_robot`
discovers those packages and matches one to a live robot by its motor set.
Adding a robot means adding a ``_description`` with a ``VIZ_SPEC`` — no
change here.

A ``VIZ_SPEC`` is a plain dict (kept plain so a ``_description`` package
need not import this GUI module):

    {
        "name":        str,               # human-facing display name
        "motors":      tuple[str, ...],   # motor names this robot exposes
        "urdf_joints": tuple[str, ...],   # URDF joint name per motor (parallel)
        "urdf_file":   str,               # URDF filename within urdf/
        "alignment":   {prefix: {motor: (sign, offset_deg)}} | None,
        # optional:
        "urdf_file_right": str,           # right-arm URDF when it differs (mirrored hardware)
        "base_offsets":    {prefix: (x, y, z)},  # per-arm base offsets, URDF world frame
    }

``alignment`` is layer 2 of the motor->URDF mapping
(``urdf_deg = sign * pos + offset_deg``); ``None`` means identity — correct
for a URDF authored to match the motor calibration.

``urdf_file_right`` is for bimanual robots whose two arms are mirrored
rather than identical (one URDF per side); when absent, both arms load
``urdf_file``. ``base_offsets`` places each arm's base in the scene (the
viewer maps the URDF-world (x, y, z) to (x, z, -y)); when absent, the
viewer falls back to its default side-by-side spacing.
"""

from __future__ import annotations

import functools
import importlib
import logging
import math
import pkgutil
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JointCalibration:
    """Per-joint motor->URDF delta: ``urdf_deg = sign * pos + offset_deg``."""

    sign: float = 1.0
    offset_deg: float = 0.0


def obs_to_urdf_rad(pos: float, calib: JointCalibration) -> float:
    """Convert one observed ``<motor>.pos`` value to a URDF joint angle (radians).

    Precondition: ``pos`` is finite. Postcondition: the result is finite.
    """
    assert math.isfinite(pos), f"non-finite observed position: {pos!r}"
    rad = math.radians(calib.sign * pos + calib.offset_deg)
    assert math.isfinite(rad), f"conversion produced non-finite angle from pos={pos!r}"
    return rad


@dataclass
class ArmSpec:
    """How to drive one arm of a URDF from an observation.

    Attributes:
        obs_prefix: Observation-key prefix — ``""`` unimanual, or ``"left_"``
            / ``"right_"`` for one arm of a bimanual robot.
        joints: Ordered ``(urdf_joint_name, motor_name)`` pairs.
        calibration: ``motor_name -> JointCalibration`` (layer 2); a motor
            absent from the map uses the identity calibration.
    """

    obs_prefix: str
    joints: list[tuple[str, str]]
    calibration: dict[str, JointCalibration] = field(default_factory=dict)


@dataclass
class RobotVizSpec:
    """Resolved viz spec for a live robot.

    TODO(viz-chains): generalize this to a set of independent URDF instances
    instead of arm-specific fields. The viewer only needs, per chain: URDF
    URL, base offset, and which observation keys drive which joints — whether
    a chain is an arm, a leg, or a base is irrelevant to rendering. Replace
    ``urdf_url_path`` + ``urdf_url_path_right`` + the ``bimanual`` flag and
    the arm-shaped ``base_offsets`` semantics with::

        chains: list[ChainSpec]  # {urdf_url_path, base_offset, joints, calibration}

    (two arms = 2 chains; arm+leg+base = 3). The joint-to-observation mapping
    stays per-description-package in VIZ_SPEC. Do this when a third kinematic
    chain appears (full-body OpenArm, quadruped, mobile base); until then the
    current shape is a tolerable wart. Also touch: resolve_robot, both
    urdf-viz meta endpoints (run.py / datasets.py), urdf_viz.html chain
    loading, and tests/gui/test_urdf_viz.py.

    Attributes:
        name: Human-facing display name (from the description's ``VIZ_SPEC``).
        urdf_url_path: Path of the URDF *within* the ``/urdf-assets/`` mount,
            e.g. ``so107_description/urdf/SO107.urdf``. For a bimanual robot
            with mirrored arms this is the *left* arm's URDF.
        urdf_url_path_right: Right-arm URDF path when the arms are mirrored
            (the description declared ``urdf_file_right``); ``None`` when both
            arms share ``urdf_url_path``.
        arms: One :class:`ArmSpec` (unimanual) or two (bimanual: left, right).
        ee_link: URDF link name to read the world position from for
            trajectory visualization. ``None`` when the description package
            didn't declare one — no polyline will be drawn for this robot.
        base_offsets: ``obs_prefix -> (x, y, z)`` base offsets in the URDF
            world frame, when the description declares them. ``None`` lets
            the viewer fall back to its default side-by-side arm spacing.
    """

    name: str
    urdf_url_path: str
    arms: list[ArmSpec]
    ee_link: str | None = None
    urdf_url_path_right: str | None = None
    base_offsets: dict[str, tuple[float, float, float]] | None = None


@functools.cache
def _discover_descriptions() -> list[tuple[str, dict]]:
    """Find every ``lerobot.robots.*_description`` package exposing a ``VIZ_SPEC``.

    Returns ``[(package_name, viz_spec), ...]``. Cached — the set of vendored
    description packages is fixed at runtime.
    """
    import lerobot.robots

    found: list[tuple[str, dict]] = []
    for _finder, modname, ispkg in pkgutil.iter_modules(
        lerobot.robots.__path__, lerobot.robots.__name__ + "."
    ):
        if not (ispkg and modname.endswith("_description")):
            continue
        try:
            module = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001 - a broken package must not break the rest
            logger.warning("urdf_viz: skipping %s (import failed: %s)", modname, e)
            continue
        viz_spec = getattr(module, "VIZ_SPEC", None)
        if viz_spec is not None:
            found.append((modname.rsplit(".", 1)[-1], viz_spec))
    return found


def resolve_robot(obs_keys: Iterable[str]) -> RobotVizSpec | None:
    """Resolve a robot's viz spec from its observation feature keys.

    Matches the robot's motor set against the ``VIZ_SPEC`` of each vendored
    ``_description`` package; the most specific match wins (so a 7-motor
    SO-107 is preferred over a 6-motor SO-arm whose motors are a subset).
    Unimanual vs bimanual is taken from the presence of ``left_*`` keys.

    Returns ``None`` when no description matches — a robot the viz can't
    show (no vendored URDF). No robot is named here; the knowledge lives
    in the description packages.
    """
    pos_keys = [k for k in obs_keys if isinstance(k, str) and k.endswith(".pos")]
    if not pos_keys:
        return None
    bimanual = any(k.startswith("left_") for k in pos_keys)
    motors_present = {k[:-4].removeprefix("left_").removeprefix("right_") for k in pos_keys}

    best: tuple[str, dict] | None = None
    for pkg_name, viz_spec in _discover_descriptions():
        motors = set(viz_spec["motors"])
        if motors <= motors_present and (best is None or len(motors) > len(set(best[1]["motors"]))):
            best = (pkg_name, viz_spec)
    if best is None:
        return None
    pkg_name, viz_spec = best

    joints = list(zip(viz_spec["urdf_joints"], viz_spec["motors"], strict=True))
    alignment = viz_spec.get("alignment")
    prefixes = ["left_", "right_"] if bimanual else [""]
    arms: list[ArmSpec] = []
    for prefix in prefixes:
        calibration: dict[str, JointCalibration] = {}
        if alignment:
            # A unimanual robot reuses the right-arm alignment.
            per_arm = alignment.get(prefix) or alignment.get("right_") or {}
            calibration = {m: JointCalibration(sign, off) for m, (sign, off) in per_arm.items()}
        arms.append(ArmSpec(obs_prefix=prefix, joints=joints, calibration=calibration))

    for arm in arms:
        assert arm.joints, f"resolved arm {arm.obs_prefix!r} has no joints"
    urdf_file_right = viz_spec.get("urdf_file_right")
    base_offsets = viz_spec.get("base_offsets")
    return RobotVizSpec(
        name=viz_spec.get("name") or pkg_name.removesuffix("_description"),
        urdf_url_path=f"{pkg_name}/urdf/{viz_spec['urdf_file']}",
        arms=arms,
        ee_link=viz_spec.get("ee_link"),
        urdf_url_path_right=f"{pkg_name}/urdf/{urdf_file_right}" if urdf_file_right else None,
        base_offsets={p: tuple(xyz) for p, xyz in base_offsets.items()} if base_offsets else None,
    )


def compute_joint_angles(spec: RobotVizSpec, observation: dict) -> dict[str, dict[str, float]]:
    """Compute URDF joint angles (radians) per arm from an observation.

    Returns ``{obs_prefix: {urdf_joint_name: radians}}``. A joint whose
    observation key is absent on this tick (early / partial observation) is
    omitted — the viewer holds its previous angle for that joint.
    """
    out: dict[str, dict[str, float]] = {}
    for arm in spec.arms:
        angles: dict[str, float] = {}
        for urdf_joint, motor in arm.joints:
            pos = observation.get(f"{arm.obs_prefix}{motor}.pos")
            if pos is None:
                continue
            calib = arm.calibration.get(motor, JointCalibration())
            angles[urdf_joint] = obs_to_urdf_rad(float(pos), calib)
        out[arm.obs_prefix] = angles
    return out
