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

"""OpenArm 2.0 robot description (per-arm URDFs) for kinematics.

The URDFs under ``urdf/`` are per-arm, kinematics-only extractions from the
official enactic/openarm_description v2.0 bimanual model (Apache-2.0 — see
``LICENSE.txt`` in this directory). Visual/collision geometry was stripped:
pinocchio refuses to parse a URDF whose mesh files are missing, and FK/IK
do not need geometry. See ``README.md`` for provenance and regeneration.

Unlike the SO-107 description, no motor<->URDF joint alignment is needed:
the OpenArm 2.0 motor factory zero (arm hanging straight down, gripper
closed) coincides with the URDF joint zero — verified by FK at q=0 placing
the gripper flange at (0, ±0.1225, -0.436) m with identity orientation,
matching the openarm_bimanual.xml MuJoCo model's qpos=0 convention. The
left/right mirror is baked into the URDF itself (mirrored joint axes), so
one identity mapping serves both arms.
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


__all__ = ["get_urdf_path", "SIDES"]
