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

from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.openarm_description import MJCFGravityCompensator


def _installed_model_path() -> Path:
    openarm_mujoco_v2 = pytest.importorskip("openarm_mujoco.v2")
    pytest.importorskip("mujoco")
    package_root = Path(openarm_mujoco_v2.__file__).resolve().parents[2]
    model = package_root / "share" / "openarm_mujoco" / "v2" / "openarm_bimanual.xml"
    if not model.is_file():
        pytest.skip(f"OpenArm MJCF data was not installed beside the package: {model}")
    return model


@pytest.mark.parametrize(
    ("side", "expected_dofs"),
    [("left", tuple(range(0, 7))), ("right", tuple(range(9, 16)))],
)
def test_installed_openarm_v2_model_resolves_named_arm_joints(side, expected_dofs):
    ff = MJCFGravityCompensator(side, xml=_installed_model_path(), gain=0.1)

    assert ff._dofs == expected_dofs
    assert ff._limits == pytest.approx([10.0, 10.0, 6.75, 6.75, 1.75, 1.75, 1.75])
    assert np.all(np.isfinite(ff.raw_torque(np.zeros(7))))


def test_gravity_feedforward_starts_at_zero_and_fades_in():
    ff = MJCFGravityCompensator("left", xml=_installed_model_path(), gain=0.1, fade_seconds=2.0)

    first = ff.torque(np.zeros(7), now=10.0)
    settled = ff.torque(np.zeros(7), now=12.0)

    assert first == pytest.approx(np.zeros(7))
    assert np.all(np.isfinite(settled))
    assert np.any(np.abs(settled) > 0.0)
