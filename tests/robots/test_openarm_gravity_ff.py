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

"""Offline tests for gravity feedforward (no CAN hardware needed).

Reference values come from the physical droop measurements on the OpenArm 2.0
(dora stack, 2026-07-18): at the front-mid pose the right shoulder drooped
6.07 deg under kp=70, i.e. the real gravity torque is ~7.4 Nm; the model
prediction was 7.1 deg (~8.7 Nm).
"""

import numpy as np
import pytest

from lerobot.utils.import_utils import _mujoco_available, _openarm_mujoco_available

if not (_mujoco_available and _openarm_mujoco_available):
    pytest.skip("mujoco / openarm-mujoco not available (extra: openarm-ff)", allow_module_level=True)

from lerobot.robots.openarm_follower.gravity_ff import (
    DOFS,
    GravityFF,
    _find_distribution_model,
    default_bimanual_xml,
)

FRONT_MID = {
    "right": [0.9007, -0.1745, 0.1079, 0.0, 0.1078, 0.7854, -0.2766],
    "left": [-0.9007, 0.1745, -0.1079, 0.0, -0.1078, -0.7854, 0.2766],
}


@pytest.fixture(params=["right", "left"])
def ff(request):
    return GravityFF(request.param, fade_secs=0.0)


class TestModelResolution:
    def test_distribution_model_resolves_independently_of_interpreter_prefix(self, tmp_path):
        module_file = tmp_path / "venv/lib/python3.12/site-packages/openarm_mujoco/v2/__init__.py"
        module_file.parent.mkdir(parents=True)
        module_file.touch()
        expected = tmp_path / "venv/share/openarm_mujoco/v2/openarm_bimanual.xml"
        expected.parent.mkdir(parents=True)
        expected.touch()

        assert _find_distribution_model(module_file) == expected

    def test_default_xml_resolves_from_package(self):
        xml = default_bimanual_xml()
        assert xml.endswith("openarm_bimanual.xml")
        assert GravityFF("right", xml=xml) is not None

    def test_dof_mapping(self):
        assert DOFS["left"] == list(range(0, 7))
        assert DOFS["right"] == list(range(9, 16))


class TestRawTau:
    def test_arms_down_is_torque_free(self, ff):
        tau = ff.raw_tau(np.zeros(7))
        assert np.all(np.abs(tau) < 0.2), tau

    def test_front_mid_shoulder_torque(self, ff):
        tau = ff.raw_tau(FRONT_MID[ff.side])
        # Physically validated band: ~7.4 Nm measured, ~8.7 Nm predicted.
        expected_sign = 1.0 if ff.side == "right" else -1.0
        assert expected_sign * tau[0] == pytest.approx(8.0, abs=2.5)
        # Elbow (J4) torque is much smaller at this pose.
        assert abs(tau[3]) < 0.5 * abs(tau[0])

    def test_left_right_mirror(self):
        tau_r = GravityFF("right", fade_secs=0.0).raw_tau(FRONT_MID["right"])
        tau_l = GravityFF("left", fade_secs=0.0).raw_tau(FRONT_MID["left"])
        assert tau_l[0] == pytest.approx(-tau_r[0], rel=1e-2)


class TestFiltering:
    def test_first_call_returns_unfiltered_value(self, ff):
        ff.gain = 1.0
        out = ff.torque(FRONT_MID[ff.side], now=1.0)
        np.testing.assert_allclose(out, ff.raw_tau(FRONT_MID[ff.side]))

    def test_lowpass_smooths_step(self, ff):
        ff.gain = 1.0
        ff.torque(np.zeros(7), now=1.0)
        full = ff.raw_tau(FRONT_MID[ff.side])
        out1 = ff.torque(FRONT_MID[ff.side], now=1.02)
        out5 = ff.torque(FRONT_MID[ff.side], now=1.10)
        # After a pose step the output approaches but does not reach the
        # new value immediately, and later samples get closer.
        assert abs(out1[0]) < 0.8 * abs(full[0])
        assert abs(out1[0]) < abs(out5[0]) < abs(full[0])

    def test_non_finite_pose_gives_zero(self, ff):
        q = np.full(7, np.nan)
        np.testing.assert_array_equal(ff.torque(q), np.zeros(7))

    def test_gain_scales_output(self, ff):
        ff.gain = 0.5
        out = ff.torque(FRONT_MID[ff.side])
        np.testing.assert_allclose(out, 0.5 * ff.raw_tau(FRONT_MID[ff.side]))


class TestClamps:
    def test_torque_frac_clamps(self):
        ff = GravityFF("right", torque_frac=0.01, fade_secs=0.0)
        out = ff.torque(FRONT_MID["right"])
        # 1% of the 40 Nm shoulder range -> clamp at 0.4 Nm.
        assert abs(out[0]) == pytest.approx(0.4)

    def test_limits_come_from_model(self, ff):
        # 50% of joint actuatorfrcrange: J1/J2 DM8009=40, J3/J4 DM4340=27,
        # J5/J6 DM4310=7, J7 DM3507=7.
        np.testing.assert_allclose(ff._limits, 0.5 * np.array([40, 40, 27, 27, 7, 7, 7], dtype=float))

    def test_gain_above_one_rejected(self):
        with pytest.raises(ValueError, match="gain"):
            GravityFF("right", gain=1.2)

    def test_residual_droop_estimate(self, ff):
        # With a perfect model, residual droop = (1 - gain) * original.
        ff.gain = 0.9
        tff = ff.torque(FRONT_MID[ff.side])
        tau = ff.raw_tau(FRONT_MID[ff.side])
        residual_rad = (tau[0] - tff[0]) / 70.0  # J1 kp from openarm_standard
        assert abs(np.degrees(residual_rad)) == pytest.approx(0.71, abs=0.25)


class TestFadeIn:
    def test_fade_ramps_linearly(self):
        ff = GravityFF("right", gain=1.0, fade_secs=2.0)
        q = FRONT_MID["right"]
        full = ff.raw_tau(q)
        np.testing.assert_allclose(ff.torque(q, now=0.0), np.zeros(7))
        np.testing.assert_allclose(ff.torque(q, now=1.0), 0.5 * full, rtol=1e-6)
        np.testing.assert_allclose(ff.torque(q, now=2.0), full, rtol=1e-6)
        np.testing.assert_allclose(ff.torque(q, now=3.0), full, rtol=1e-6)

    def test_fade_disabled_with_zero_secs(self):
        ff = GravityFF("right", gain=1.0, fade_secs=0.0)
        out = ff.torque(FRONT_MID["right"], now=0.0)
        np.testing.assert_allclose(out, ff.raw_tau(FRONT_MID["right"]))
