#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""OpenArm 2.0 robot description (per-arm URDFs) for kinematics and visualization.

Two URDF flavours live under ``urdf/``:

- ``openarm_{left,right}.urdf`` — per-arm, kinematics-only extractions from
  the official enactic/openarm_description v2.0 bimanual model (Apache-2.0 —
  see ``LICENSE.txt``). Visual/collision geometry was stripped: pinocchio
  refuses to parse a URDF whose mesh files are missing, and FK/IK do not
  need geometry.
- ``openarm_{left,right}_viz.urdf`` — same kinematics (verbatim joint
  origins/axes/limits, FK-verified identical), plus ``<visual>`` geometry
  from the upstream .dae meshes and the two-finger gripper subtree, for the
  in-browser URDF viewer (``lerobot.gui``). Names are unprefixed
  (``joint1..7`` …) because the viewer loads one URDF copy per arm and
  addresses each copy by its own tree — one ``VIZ_SPEC`` joint list then
  serves both arms, exactly like the SO descriptions. The left copy mirrors
  its meshes (``scale="1 -1 1"``; link3/link4 are symmetric parts).

Unlike the SO-107 description, no motor<->URDF joint alignment is needed:
the OpenArm 2.0 motor factory zero (arm hanging straight down, gripper
closed) coincides with the URDF joint zero — verified by FK at q=0 placing
the gripper flange at (0, ±0.1225, -0.436) m with identity orientation,
matching the openarm_bimanual.xml MuJoCo model's qpos=0 convention. The
left/right mirror is baked into the URDF itself (mirrored joint axes), so
one identity mapping serves both arms. The gripper motor (degrees; left
0..45, right -45..0) maps 1:1 onto ``finger_joint1`` (radians deg->rad;
left 0..+0.7854, right -0.7854..0); ``finger_joint2`` mimics it.

See ``README.md`` for provenance and regeneration.
"""

from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent

SIDES = ("left", "right")


def get_urdf_path(side: str) -> Path:
    """Absolute path to the per-arm OpenArm 2.0 URDF (kinematics-only)."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    p = _PKG_DIR / "urdf" / f"openarm_{side}.urdf"
    assert p.exists(), f"OpenArm {side} URDF missing at {p}"
    return p


def get_viz_urdf_path(side: str = "left") -> Path:
    """Absolute path to the per-arm visualization URDF (visuals + gripper)."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    p = _PKG_DIR / "urdf" / f"openarm_{side}_viz.urdf"
    assert p.exists(), f"OpenArm {side} viz URDF missing at {p} — see {_PKG_DIR / 'README.md'}"
    return p


def get_meshes_dir() -> Path:
    """Absolute path to the directory containing the OpenArm link meshes."""
    p = _PKG_DIR / "meshes"
    assert p.is_dir(), f"OpenArm meshes dir missing at {p}"
    return p


# URDF-visualization metadata, discovered by ``lerobot.gui.urdf_viz``. Motor
# and URDF zeros coincide and axes already agree, so no per-joint alignment
# is needed (identity) — including the gripper, whose degree range maps 1:1
# onto finger_joint1 (finger_joint2 follows via URDF <mimic>).
VIZ_SPEC = {
    "name": "OpenArm 2.0",
    "motors": (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "gripper",
    ),
    "urdf_joints": (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
    ),
    # The arms are mirrored hardware: one viz URDF per side. ``urdf_file``
    # stays the default (left); ``urdf_file_right`` is loaded for the right
    # arm of a bimanual setup.
    "urdf_file": "openarm_left_viz.urdf",
    "urdf_file_right": "openarm_right_viz.urdf",
    "alignment": None,
    # Physical arm-base offsets (metres, URDF world frame: +X forward, +Y
    # left, +Z up) from the upstream body config — the arms mount at
    # y=±0.031, so side-by-side spacing is only 0.062 m, unlike the SO
    # layout the viewer assumes by default. z=0.45 is cosmetic: the
    # physical mount is a 0.698 m pedestal we don't render, and this puts
    # the hanging zero-pose gripper tips (~0.436 below the mount) at the
    # ground grid.
    "base_offsets": {
        "left_": (0.0, 0.031, 0.45),
        "right_": (0.0, -0.031, 0.45),
    },
    # EE link for trajectory visualization. The gripper flange: downstream
    # of joint7 but upstream of the finger joints, so the trace reads as
    # "where the gripper is" without open/close jiggle.
    "ee_link": "ee_base_link",
}

__all__ = ["get_urdf_path", "get_viz_urdf_path", "get_meshes_dir", "VIZ_SPEC", "SIDES"]
